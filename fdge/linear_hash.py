from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np

from .decomposition import mdec
from .mathutils import (add_mod, bits_from_ints_le, matmul_mod, require_memory,
                        _FLOAT_EXACT_MAX, _matmul_mod_prepared_left, _prepared_product_plan)
from .params import FDGEParams


_F_DISK_ROW_BLOCK = 1
_F_DISK_COLUMN_BLOCK = 131072


def _same_key_decomposition(parts, expected_bits):
    if len(parts) != 2:
        return False
    size = expected_bits.size // 2
    for index, value in enumerate(parts):
        array = np.asarray(value)
        if array.shape != (size,) or array.dtype.kind not in 'iub':
            return False
        expected = expected_bits[index * size:(index + 1) * size]
        for start in range(0, size, 131072):
            if not np.array_equal(array[start:start + 131072], expected[start:start + 131072]):
                return False
    return True


def _canonical_digest_header(q, shape):
    digest = hashlib.sha256(b'FDGE immutable F canonical int64 v1\0')
    digest.update(q.to_bytes(8, 'little'))
    for size in shape:
        digest.update(int(size).to_bytes(8, 'little'))
    return digest


def _storage_dtype(q, storage_dtype):
    dtype = np.dtype(storage_dtype)
    if dtype not in (np.dtype('int64'), np.dtype('uint32'), np.dtype('float64')):
        raise ValueError('F dtype must be int64/uint32/float64')
    maximum = _FLOAT_EXACT_MAX if dtype.kind == 'f' else np.iinfo(dtype).max
    if q - 1 > maximum or (dtype.kind == 'f' and (q - 1)**2 > _FLOAT_EXACT_MAX):
        raise ValueError(f'canonical Z_q entries do not fit {dtype.name} losslessly')
    return dtype


def linear_hash_storage_plan(params: FDGEParams, *, storage_dtype='int64',
                             disk_backed=True, columns=None, read_only=False) -> dict:
    params.validate(require_distributions=False)
    if not isinstance(read_only, bool):
        raise ValueError('read_only must be a bool')
    dtype = _storage_dtype(params.q, storage_dtype)
    width = 2 * params.n * params.m * params.k
    entries = params.n * width

    input_workspace = 4 * params.n * params.m * params.k + 32 * params.n * params.m
    fixed_workspace = input_workspace + 2 * 1024 * 1024

    sample_chunk_entries = min(131072, max(1, params.max_memory_bytes // 64))
    sample_workspace = 8 * min(entries, sample_chunk_entries)
    if columns is not None and (not isinstance(columns, int) or isinstance(columns, bool) or columns < 1):
        raise ValueError('F column block size must be a positive integer')
    matrix_cast_workspace = 0
    vector_cast_workspace = 0
    freeze_workspace = 0
    metal_plan = None
    if disk_backed:
        available = params.max_memory_bytes - fixed_workspace
        default_columns = max(1, min(_F_DISK_COLUMN_BLOCK, width,
                                    available // (64 * min(_F_DISK_ROW_BLOCK, params.n))))
        actual_columns = min(width, default_columns if columns is None else columns)

        block_workspace = 64 * min(_F_DISK_ROW_BLOCK, params.n) * actual_columns
        working = fixed_workspace + max(sample_workspace, block_workspace)
    else:

        freeze_workspace = dtype.itemsize * entries if read_only else 0
        actual_columns = None if columns is None else min(width, columns)
        if actual_columns is None:

            matrix_cast_workspace = 8 * entries if dtype == np.dtype('uint32') else 0
            vector_cast_workspace = 8 * width
            block_workspace = matrix_cast_workspace + vector_cast_workspace
        else:
            block_workspace = 64 * params.n * actual_columns
        if read_only and dtype == np.dtype(np.uint32) and entries >= 8 * 1024**2:
            from . import metal_matrix
            if metal_matrix.available(params.q):
                metal_plan = metal_matrix.linear_product_plan((params.n, width), params.q,
                                                              columns=actual_columns)
                actual_columns = metal_plan['column_block_size']
                block_workspace = metal_plan['peak_workspace_bound_bytes']
                matrix_cast_workspace = vector_cast_workspace = 0
        working = dtype.itemsize * entries + fixed_workspace + max(sample_workspace, block_workspace,
                                                                   freeze_workspace)
    resident_matrix = 0 if disk_backed else dtype.itemsize * entries
    construction_working = resident_matrix + fixed_workspace + max(sample_workspace, freeze_workspace)
    application_working = resident_matrix + fixed_workspace + block_workspace
    prepared_field_plan = (_prepared_product_plan(
        min(_F_DISK_ROW_BLOCK, params.n) if disk_backed else params.n,
        width if actual_columns is None else actual_columns, 1, params.q)
        if dtype.kind == 'f' else None)
    return {'shape': [params.n, width], 'entries': entries, 'storage_dtype': dtype.name,
            'application_backend': ('metal_streamed_uint32' if metal_plan is not None else
                                    'prepared_exact_float64' if prepared_field_plan is not None else 'cpu'),
            'metal_workspace_plan': metal_plan,
            'prepared_field_product_plan': prepared_field_plan,
            'matrix_bytes': dtype.itemsize * entries, 'disk_backed': bool(disk_backed),
            'application_row_block_size': (metal_plan['row_block_size'] if metal_plan is not None else
                                           min(_F_DISK_ROW_BLOCK, params.n) if disk_backed else params.n),
            'column_block_size': actual_columns, 'sample_chunk_entries': sample_chunk_entries,
            'input_decomposition_workspace_bytes': input_workspace,
            'unblocked_matrix_int64_cast_bytes': matrix_cast_workspace,
            'unblocked_vector_int64_cast_bytes': vector_cast_workspace,
            'read_only': read_only, 'freeze_workspace_bytes': freeze_workspace,
            'resident_matrix_bytes': resident_matrix,
            'construction_explicit_buffer_bytes': construction_working,
            'application_explicit_buffer_bytes': application_working,
            'conservative_explicit_buffer_bytes': working,
            'max_memory_bytes': params.max_memory_bytes,
            'within_memory_budget': working <= params.max_memory_bytes,
            'OS_mapped_page_cache_included': False,
            'memory_scope': 'explicit_arrays/workspace; excludes_OS_cache'}


class LinearPublicKeyHash:
    canonical_chunk_entries = 131072
    canonical_chunk_bytes = 8 * canonical_chunk_entries

    def __init__(self, params: FDGEParams, rng: np.random.Generator,
                 matrix_path: str | Path | None = None, *, storage_dtype='int64',
                 min_disk_free_bytes: int = 0, on_progress=None, read_only=False):
        params.validate(require_distributions=False)
        self.params = params
        self.input_length = 2 * params.n * params.m * params.k
        self.shape = (params.n, self.input_length)
        self.storage_dtype = _storage_dtype(params.q, storage_dtype)
        if self.storage_dtype.kind == 'f' and not read_only:
            raise ValueError('prepared F storage requires immutable publication')
        self.matrix_bytes = self.storage_dtype.itemsize * params.n * self.input_length
        self.storage = "memory" if matrix_path is None else "memmap"
        self.matrix_path = None if matrix_path is None else Path(matrix_path)
        self.closed = False
        self._read_only = False
        self._matrix = None
        self._readonly_storage = None
        self._canonical_digest = None
        canonical_hash = _canonical_digest_header(params.q, self.shape) if read_only and matrix_path is None else None
        self.resource_plan = linear_hash_storage_plan(params, storage_dtype=self.storage_dtype,
                                                      disk_backed=matrix_path is not None,
                                                      read_only=read_only)
        self.column_block_size = self.resource_plan['column_block_size']
        if not isinstance(min_disk_free_bytes, int) or isinstance(min_disk_free_bytes, bool) or min_disk_free_bytes < 0:
            raise ValueError('min_disk_free_bytes must be >= 0')
        self.min_disk_free_bytes = min_disk_free_bytes
        if matrix_path is None:
            require_memory(self.resource_plan['conservative_explicit_buffer_bytes'],
                           params.max_memory_bytes, "random linear F and application workspace")
            self._matrix = np.empty(self.shape, dtype=self.storage_dtype)
        else:
            require_memory(self.resource_plan['conservative_explicit_buffer_bytes'], params.max_memory_bytes,
                           "disk-backed F input/decomposition workspace")
            path = self.matrix_path
            if not path.parent.is_dir():
                raise FileNotFoundError("F storage parent directory must already exist")
            if shutil.disk_usage(path.parent).free < self.matrix_bytes + min_disk_free_bytes:
                raise OSError(f"random F requires {self.matrix_bytes:,} disk bytes plus "
                              f"{min_disk_free_bytes:,} reserved free bytes")

            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                identity = os.fstat(fd)
                self._file_identity = (identity.st_dev, identity.st_ino)
                try:
                    os.ftruncate(fd, self.matrix_bytes)
                    self._matrix = np.memmap(path, mode="r+", dtype=self.storage_dtype, shape=self.shape)
                except BaseException:
                    self._remove_failed_file()
                    raise
            finally:
                os.close(fd)

        flat = self.matrix.reshape(-1)
        chunk = self.resource_plan['sample_chunk_entries']
        report_stride = max(chunk, flat.size // 100)
        next_report = 0

        space_stride = max(chunk, (64 * 1024**2) // self.storage_dtype.itemsize)
        next_space_check = 0
        try:
            for start in range(0, flat.size, chunk):
                count = min(chunk, flat.size - start)
                if self.storage == 'memmap' and start >= next_space_check:
                    upcoming = min(space_stride, flat.size - start) * self.storage_dtype.itemsize
                    if shutil.disk_usage(self.matrix_path.parent).free < min_disk_free_bytes + upcoming:
                        raise OSError('Insufficient F disk reserve')
                    next_space_check = start + space_stride
                values = rng.integers(0, params.q, size=count, dtype=np.int64)
                flat[start:start + count] = values
                if canonical_hash is not None:
                    canonical_hash.update(memoryview(np.ascontiguousarray(values, dtype='<i8')).cast('B'))
                if start >= next_report or start + count == flat.size:
                    if on_progress:
                        on_progress({'phase': 'sample_full_F', 'entries_written': start + count,
                                     'total_entries': int(flat.size),
                                     'bytes_written': (start + count) * self.storage_dtype.itemsize,
                                     'total_bytes': self.matrix_bytes})
                    next_report = start + report_stride
            if isinstance(self.matrix, np.memmap):
                if on_progress:
                    on_progress({'phase': 'flush_full_F', 'total_bytes': self.matrix_bytes})
                self.matrix.flush()
                if shutil.disk_usage(self.matrix_path.parent).free < min_disk_free_bytes:
                    raise OSError('F disk reserve exhausted')

            flat = None
            if read_only:
                if self.storage == 'memmap':
                    self._matrix._mmap.close()
                    self._matrix = None
                    self._matrix = np.memmap(self.matrix_path, mode='r', dtype=self.storage_dtype,
                                             shape=self.shape)
                else:
                    immutable_storage = self._matrix.tobytes(order='C')
                    self._matrix = np.frombuffer(immutable_storage, dtype=self.storage_dtype).reshape(self.shape)
                self._readonly_layout = (self.shape, self.storage_dtype, self._matrix.strides)
                self._readonly_params = params
                self._readonly_storage = self._matrix
                self._canonical_digest = None if canonical_hash is None else canonical_hash.digest()
                self._read_only = True
        except BaseException:
            flat = None
            self.close()
            if self.storage == 'memmap':
                self._remove_failed_file()
            raise

    def _ensure_open(self):
        if self.closed:
            raise ValueError('F storage is closed')
        if self._read_only:
            shape, dtype, strides = self._readonly_layout

            if (self._matrix is not self._readonly_storage
                    or self._matrix.shape != shape or self._matrix.dtype != dtype
                    or self._matrix.strides != strides or self._matrix.flags.writeable
                    or self.shape != shape or self.storage_dtype != dtype
                    or self.input_length != shape[1] or self.params is not self._readonly_params):
                raise ValueError('Read-only F metadata changed')
            if isinstance(self._matrix, np.memmap) and self._matrix._mmap.closed:
                raise ValueError('F storage is closed')

    @property
    def matrix(self):
        self._ensure_open()
        return self._matrix.view() if self._read_only else self._matrix

    @property
    def is_read_only(self):
        return self._read_only

    @property
    def frozen(self):
        return self._read_only

    @property
    def canonical_digest(self):
        self._ensure_open()
        if not self.is_read_only or self.storage != 'memory' or self._canonical_digest is None:
            raise ValueError('Immutable F storage required')
        return self._canonical_digest

    def iter_canonical_chunks(self, *, chunk_entries=canonical_chunk_entries):
        self._ensure_open()
        if not isinstance(chunk_entries, int) or isinstance(chunk_entries, bool) or chunk_entries < 1:
            raise ValueError('chunk_entries must be > 0')
        size = self.params.n * self.input_length
        require_memory(8 * min(size, chunk_entries), self.params.max_memory_bytes,
                       'canonical F chunk')

        def chunks():
            for start in range(0, size, chunk_entries):
                self._ensure_open()

                yield np.array(self._matrix.reshape(-1)[start:start + chunk_entries],
                               dtype='<i8', copy=True, order='C')

        return chunks()

    def _remove_failed_file(self):
        try:
            current = self.matrix_path.stat()
        except FileNotFoundError:
            return False
        if (current.st_dev, current.st_ino) == self._file_identity:
            self.matrix_path.unlink()
            return True
        return False

    def discard_owned_file(self):
        self.close()
        return False if self.matrix_path is None else self._remove_failed_file()

    def close(self):
        if not self.closed:
            if isinstance(self._matrix, np.memmap):
                self._matrix._mmap.close()
            self._matrix = None
            self._readonly_storage = None
            self.closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_exc):
        self.close()

    def _apply_bits_metal(self, vector, *, columns=None, on_progress=None):
        from . import metal_matrix

        self._ensure_open()
        vector = np.asarray(vector)
        if vector.shape != (self.input_length,) or vector.dtype.kind not in 'iub':
            raise ValueError(f'F input must be an integer vector of length {self.input_length}')
        resident = self.matrix_bytes if self.storage == 'memory' else 0
        fixed = self.resource_plan['input_decomposition_workspace_bytes'] + 2 * 1024 * 1024
        available = self.params.max_memory_bytes - resident - fixed - metal_matrix._WORKSPACE_LIMIT
        if available < 1024 * 1024:
            raise MemoryError('F Metal multiplication exceeds the memory budget')
        return metal_matrix.linear_product(
            self.matrix, vector, self.params.q,
            max_workspace_bytes=min(available, metal_matrix._WORKSPACE_LIMIT),
            columns=columns, on_progress=on_progress)

    def apply_bits(self, vector: np.ndarray, *, columns=None, on_progress=None) -> np.ndarray:
        self._ensure_open()
        vector = np.asarray(vector)
        if vector.shape != (self.input_length,) or vector.dtype.kind not in "iub":
            raise ValueError(f"F input must be an integer vector of length {self.input_length}")

        if (self.storage == 'memory' and self.is_read_only
                and self.storage_dtype == np.dtype(np.uint32)
                and self.params.n * self.input_length >= 8 * 1024**2):
            from . import metal_matrix
            if metal_matrix.available(self.params.q):
                return self._apply_bits_metal(vector, columns=columns, on_progress=on_progress)

        plan = linear_hash_storage_plan(self.params, storage_dtype=self.storage_dtype,
                                        disk_backed=self.storage == 'memmap', columns=columns,
                                        read_only=self.is_read_only)
        require_memory(plan['application_explicit_buffer_bytes'], self.params.max_memory_bytes,
                       'F multiplication workspace')
        if self.storage == "memory" and columns is None:
            if self.storage_dtype.kind == 'f':
                return _matmul_mod_prepared_left(self.matrix, vector, self.params.q)
            return matmul_mod(self.matrix, vector, self.params.q)
        result = np.zeros(self.params.n, dtype=np.int64)
        columns = plan['column_block_size']
        row_block = plan['application_row_block_size']
        matrix = self.matrix
        for row in range(0, self.params.n, row_block):
            stop = min(row + row_block, self.params.n)
            subtotal = np.zeros(stop - row, dtype=np.int64)
            for start in range(0, self.input_length, columns):
                product = _matmul_mod_prepared_left if self.storage_dtype.kind == 'f' else matmul_mod
                block = product(matrix[row:stop, start:start + columns],
                                vector[start:start + columns], self.params.q)
                subtotal = add_mod(subtotal, block, q=self.params.q)
            result[row:stop] = subtotal
            if on_progress:
                on_progress({'phase': 'apply_full_F', 'rows_completed': stop,
                             'total_rows': self.params.n,
                             'entries_processed': stop * self.input_length,
                             'total_entries': self.params.n * self.input_length})
        return result

    def apply(self, U1: np.ndarray, U2: np.ndarray, *, columns=None, on_progress=None) -> np.ndarray:
        self._ensure_open()
        p = self.params
        if U1.shape != (p.n, p.m) or U2.shape != U1.shape:
            raise ValueError("F expects two (n,m) public matrices")

        a_concat = np.concatenate((mdec(U1.T, p.q), mdec(U2.T, p.q)))
        syndrome = self.apply_bits(a_concat, columns=columns, on_progress=on_progress)

        return bits_from_ints_le(syndrome, p.k)

    def _prepare_key(self, U1, U2, *, columns=None):
        return _PreparedPublicKeyHash(self, U1, U2, columns=columns)

    def to_dict(self) -> dict:
        self._ensure_open()
        return {"shape": list(self.shape), "bytes": self.matrix_bytes,
                "storage": self.storage, "storage_dtype": self.storage_dtype.name,
                "sampling_dtype": 'int64', "storage_cast_lossless": True,
                "column_block_size": self.column_block_size,
                "is_read_only": self.is_read_only, "frozen": self.frozen,
                "read_only_scope": 'read_only; external_file_mutation_possible',
                "closed": self.closed, "distribution": "independent uniform entries in Z_q",
                "surrogate": False}


@dataclass(frozen=True, eq=False, init=False)
class _PreparedPublicKeyHash:
    _owner: LinearPublicKeyHash
    _params: object
    _params_state: dict
    _matrix: np.ndarray
    _matrix_layout: tuple
    _digest: bytes
    _sources: tuple[np.ndarray, np.ndarray]
    _input_bindings: tuple
    _bits: np.ndarray
    _syndrome: np.ndarray
    _leaf: np.ndarray

    def __init__(self, owner, U1, U2, *, columns=None):
        if (type(owner) is not LinearPublicKeyHash or not owner.is_read_only
                or owner.storage != 'memory'):
            raise ValueError('key preparation requires an immutable resident F')
        owner._ensure_open()
        p = owner.params
        sources = []
        bindings = []
        from .relations import _immutable_snapshot_binding
        for value in (U1, U2):
            array = np.asarray(value)
            if array.shape != (p.n, p.m) or array.dtype.kind not in 'iub':
                raise ValueError('F expects two canonical (n,m) public matrices')
            snapshot = np.frombuffer(np.asarray(array, dtype=np.int64).tobytes(order='C'),
                                     dtype=np.int64).reshape(p.n, p.m)
            if np.any(snapshot < 0) or np.any(snapshot >= p.q):
                raise ValueError('F expects two canonical (n,m) public matrices')
            sources.append(snapshot)
            bindings.append(_immutable_snapshot_binding(array))
        object.__setattr__(self, '_owner', owner)
        object.__setattr__(self, '_params', p)
        object.__setattr__(self, '_params_state', deepcopy(p.to_dict()))
        object.__setattr__(self, '_matrix', owner._matrix)
        object.__setattr__(self, '_matrix_layout', (owner.shape, owner.storage_dtype,
                                                   owner._matrix.strides, owner.input_length))
        object.__setattr__(self, '_digest', owner.canonical_digest)
        object.__setattr__(self, '_sources', tuple(sources))
        object.__setattr__(self, '_input_bindings', tuple(bindings))
        bits = np.concatenate(tuple(mdec(matrix.T, p.q) for matrix in sources))
        object.__setattr__(self, '_bits', np.frombuffer(bits.tobytes(), dtype=bits.dtype))
        del bits
        self._check_binding()
        syndrome = owner.apply_bits(self._bits, columns=columns)
        self._check_binding()
        object.__setattr__(self, '_syndrome', np.frombuffer(
            np.asarray(syndrome, dtype=np.int64).tobytes(), dtype=np.int64))
        leaf = bits_from_ints_le(self._syndrome, p.k)
        object.__setattr__(self, '_leaf', np.frombuffer(leaf.tobytes(), dtype=leaf.dtype))

    def _check_binding(self):
        owner = self._owner
        owner._ensure_open()
        if (owner.storage != 'memory' or not owner.is_read_only
                or owner.params is not self._params or owner.params.to_dict() != self._params_state
                or owner._matrix is not self._matrix
                or (owner.shape, owner.storage_dtype, owner._matrix.strides,
                    owner.input_length) != self._matrix_layout
                or owner.canonical_digest != self._digest):
            raise ValueError('Cached key/F parameters mismatch')

    def _validate_key(self, owner, U1, U2):
        self._check_binding()
        values = tuple(np.asarray(value) for value in (U1, U2))
        if owner is not self._owner:
            raise ValueError('Cached decomposition/key/F mismatch')
        from .relations import _immutable_snapshot_binding
        for value, source, saved in zip(values, self._sources, self._input_bindings):
            current = _immutable_snapshot_binding(value)
            matched = (saved is not None and current is not None
                       and current[0] is saved[0] and current[1] is saved[1]
                       and current[2:] == saved[2:])
            if (value.shape != source.shape or value.dtype.kind not in 'iub'
                    or (not matched and not np.array_equal(value, source))):
                raise ValueError('Cached decomposition/key/F mismatch')

    def _decomposition(self, index):
        self._check_binding()
        if index not in (0, 1):
            raise ValueError('public key component must be 0 or 1')
        size = self._bits.size // 2
        return self._bits[index * size:(index + 1) * size]

    def _matched_syndrome(self, parts):
        self._check_binding()
        if not _same_key_decomposition(parts, self._bits):
            return None
        self._check_binding()
        return self._syndrome.view()

    @property
    def leaf(self):
        self._check_binding()
        return self._leaf.view()
