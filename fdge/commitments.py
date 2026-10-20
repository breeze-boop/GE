from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
import hashlib
import math
import os

import numpy as np

from .mathutils import (_FLOAT_EXACT_MAX, _matmul_mod_canonical_left,
                        _matmul_mod_prepared_left, _add_canonical_mod, mod_q,
                        require_memory)
from . import metal_matrix


def _update_canonical_int64(digest, array):
    array = np.asarray(array)
    if not array.size:
        return
    if array.flags.c_contiguous and array.dtype == np.dtype('<i8'):
        digest.update(memoryview(array).cast('B'))
        return
    if array.flags.c_contiguous:
        flat = array.reshape(-1)
        workspace = np.empty(min(262144, flat.size), dtype='<i8')
        for start in range(0, flat.size, workspace.size):
            stop = min(start + workspace.size, flat.size)
            piece = workspace[:stop - start]
            np.copyto(piece, flat[start:stop], casting='unsafe')
            digest.update(memoryview(piece).cast('B'))
        return
    with np.nditer(array, flags=['external_loop', 'buffered', 'zerosize_ok', 'refs_ok'],
                   op_flags=[['readonly', 'contig']], op_dtypes=[np.dtype('<i8')],
                   casting='unsafe', order='C', buffersize=262144) as chunks:
        for piece in chunks:
            digest.update(memoryview(piece).cast('B'))


def framed_hash_state(domain: bytes, *parts):
    digest = hashlib.sha256(domain)
    for part in parts:
        if isinstance(part, np.ndarray):
            prefix = str(part.shape if part.ndim else (1,)).encode() + b":i8:"
            digest.update((len(prefix) + 8*part.size).to_bytes(8, "little"))
            digest.update(prefix)
            _update_canonical_int64(digest, part)
        else:
            data = part if isinstance(part, bytes) else str(part).encode()
            digest.update(len(data).to_bytes(8, "little"))
            digest.update(data)
    return digest


def framed_hash(domain: bytes, *parts) -> bytes:
    return framed_hash_state(domain, *parts).digest()


def _auxiliary_digest_state(context, q, vector_count):
    if (type(q) is not int or not 1 < q <= np.iinfo(np.int64).max
            or not isinstance(context, bytes) or len(context) != 32
            or type(vector_count) is not int or vector_count < 0):
        raise ValueError('invalid auxiliary commitment encoding')
    digest = hashlib.sha256(b'FDGE-aCommit-CRH-v1\0')
    digest.update(q.to_bytes(8, 'little'))
    digest.update(vector_count.to_bytes(8, 'little'))
    digest.update(context)
    return digest


def _update_auxiliary_payload(digest, piece, dtype, workspace=None):
    if piece.dtype == dtype and piece.flags.c_contiguous:
        digest.update(memoryview(piece).cast('B'))
        return workspace
    if workspace is None or workspace.size < piece.size:
        workspace = np.empty(piece.size, dtype=dtype)
    canonical = workspace[:piece.size]
    np.copyto(canonical, piece, casting='unsafe')
    digest.update(memoryview(canonical).cast('B'))
    return workspace


def _extend_auxiliary_digest(digest, vectors, q, *, chunk_entries=262144):
    if type(chunk_entries) is not int or chunk_entries < 1:
        raise ValueError('invalid auxiliary commitment encoding')
    dtype = np.dtype('<u4' if q <= 2**32 else '<u8')
    workspace = None
    for vector in vectors:
        vector = np.asarray(vector)
        if vector.ndim != 1 or vector.dtype.kind not in 'iu':
            raise ValueError('auxiliary encoding requires integer vectors')
        digest.update(vector.size.to_bytes(8, 'little'))
        for start in range(0, vector.size, chunk_entries):
            piece = vector[start:start+chunk_entries]
            outside = (int(piece.view(np.uint64).max()) >= q
                       if piece.dtype == np.dtype(np.int64)
                       else int(piece.min()) < 0 or int(piece.max()) >= q)
            if outside:
                raise ValueError('auxiliary vectors must be canonical field elements')
            workspace = _update_auxiliary_payload(digest, piece, dtype, workspace)


def auxiliary_digest(vectors, context, q, *, chunk_entries=262144):
    vectors = tuple(vectors)
    digest = _auxiliary_digest_state(context, q, len(vectors))
    _extend_auxiliary_digest(digest, vectors, q, chunk_entries=chunk_entries)
    return digest.digest()


def integer_vector(value, size: int, name: str, q: int | None = None) -> np.ndarray:
    a = np.asarray(value)
    if a.shape != (size,) or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer vector of length {size}")
    if a.dtype.kind == "u" and a.size and int(a.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds signed int64")
    a = a.astype(np.int64, copy=False)
    if q is not None and (np.any(a < 0) or np.any(a >= q)):
        raise ValueError(f"{name} must use canonical representatives in [0,q)")
    return a


def _freeze_public_matrix(value, shape, q):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'iu':
        raise ValueError('Invalid public matrix shape or dtype')
    if np.any(a < 0) or np.any(a >= q):
        raise ValueError('Public matrix must be canonical mod q')

    return np.frombuffer(a.astype(np.int64, copy=False).tobytes(order='C'),
                         dtype=np.int64).reshape(shape)


def _check_public_matrix(value, shape):
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype != np.dtype(np.int64)
            or value.strides != (8 * shape[1], 8) or value.flags.writeable):
        raise ValueError('Public matrix storage changed')


def _lattice_storage_dtype(q):
    if metal_matrix.available(q):
        return np.dtype(np.uint32)
    return np.dtype(np.float64 if (q-1)**2 <= _FLOAT_EXACT_MAX else np.int64)


def _freeze_lattice_matrix(value, shape, q):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'iuf':
        raise ValueError('Invalid commitment matrix shape or dtype')
    dtype = _lattice_storage_dtype(q)
    if dtype == np.dtype(np.uint32):
        return metal_matrix.prepare(a)
    if (np.any(a < 0) or np.any(a >= q)
            or (a.dtype.kind == 'f' and (not np.all(np.isfinite(a)) or np.any(a != np.rint(a))))):
        raise ValueError('Commitment coefficients must be canonical mod q')
    return np.frombuffer(a.astype(dtype, copy=False).tobytes(order='C'), dtype=dtype).reshape(shape)


def _check_lattice_matrix(value, shape, q):
    dtype = _lattice_storage_dtype(q)
    if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype != dtype
            or value.strides != (dtype.itemsize*shape[1], dtype.itemsize) or value.flags.writeable
            or (dtype == np.dtype(np.uint32) and not metal_matrix.is_prepared(value))):
        raise ValueError('prepared lattice matrix storage metadata changed')


def _check_key_dimensions(q, *dimensions):
    if (not isinstance(q, int) or isinstance(q, bool) or not 1 < q <= np.iinfo(np.int64).max
            or any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in dimensions)):
        raise ValueError('Invalid commitment dimensions or modulus')


def _commitment_tail(randomness, message, product, q, *, out=None):
    if out is None:
        result = np.empty_like(randomness, dtype=np.int64)
    else:
        result = np.asarray(out)
        if (result.shape != randomness.shape or result.dtype != np.dtype(np.int64)
                or not result.flags.c_contiguous or not result.flags.writeable
                or any(np.may_share_memory(result, value) for value in (randomness, message, product))):
            raise ValueError('Tail needs a separate writable int64 vector')
    if not randomness.size:
        randomness.min()
    int64 = np.iinfo(np.int64)
    for start in range(0, len(randomness), 65536):
        stop = min(start + 65536, len(randomness))
        r, m, p = randomness[start:stop], message[start:stop], product[start:stop]
        message_max = (int(m.view(np.uint64).max())
                       if (r.dtype == np.dtype(np.int64) and m.dtype == np.dtype(np.int64)
                           and r.shape == m.shape) else None)
        if message_max is not None and message_max <= int64.max:
            safe_sum = int(r.max()) + message_max + q - 1 <= int64.max
        else:
            lower = int(r.min()) + int(m.min())
            upper = int(r.max()) + int(m.max()) + q - 1
            safe_sum = int64.min <= lower and upper <= int64.max
        output = result[start:stop]
        if safe_sum:
            np.add(r, m, out=output)
            np.add(output, p, out=output)
            np.remainder(output, q, out=output)
        else:
            output[:] = _add_canonical_mod(_add_canonical_mod(mod_q(r, q), p, q),
                                           mod_q(m, q), q)
    return result


def _commitment_output(upper, randomness, message, product, q):
    result = np.empty(upper.size + randomness.size, dtype=np.int64)
    result[:upper.size] = upper
    _commitment_tail(randomness, message, product, q, out=result[upper.size:])
    return result


@dataclass(frozen=True)
class LatticeCommitmentKey:
    q: int
    l1: int
    l2: int
    width: int
    top: np.ndarray
    bottom: np.ndarray

    def __post_init__(self):
        _check_key_dimensions(self.q, self.l1, self.l2, self.width)
        object.__setattr__(self, 'top', _freeze_lattice_matrix(self.top, (self.l1, self.width + self.l2), self.q))
        object.__setattr__(self, 'bottom', _freeze_lattice_matrix(self.bottom, (self.width, self.l2), self.q))

    def _check_storage(self):
        _check_lattice_matrix(self.top, (self.l1, self.width + self.l2), self.q)
        _check_lattice_matrix(self.bottom, (self.width, self.l2), self.q)

    def _matrix_product(self, matrix, right):
        if metal_matrix.is_prepared(matrix):
            return metal_matrix.matmul(matrix, right, self.q)
        if matrix.dtype == np.dtype(np.float64):
            return _matmul_mod_prepared_left(matrix, right, self.q)
        if isinstance(right, tuple):
            right = np.column_stack(right)
        return _matmul_mod_canonical_left(matrix, right, self.q)

    def _matrix_products(self, top_right, bottom_right):
        if metal_matrix.is_prepared(self.top) and metal_matrix.is_prepared(self.bottom):
            return metal_matrix.matmul_many(((self.top, top_right),
                                             (self.bottom, bottom_right)), self.q)
        return (self._matrix_product(self.top, top_right),
                self._matrix_product(self.bottom, bottom_right))

    @property
    def randomness_size(self) -> int:
        return self.l1 + self.width + self.l2

    @property
    def commitment_size(self) -> int:
        return self.l1 + self.width

    @classmethod
    def generate(cls, q: int, l1: int, l2: int, width: int, rng,
                 max_memory_bytes: int = 512 * 1024 * 1024):
        if any(not isinstance(v, int) or v < 1 for v in (q, l1, l2, width)):
            raise ValueError('Invalid commitment dimensions or modulus')
        _check_key_dimensions(q, l1, l2, width)
        copies = 24 if _lattice_storage_dtype(q) == np.dtype(np.float64) else 16
        require_memory(copies * (l1 * (width + l2) + width * l2), max_memory_bytes,
                       'Yang commitment matrices')
        top = rng.integers(0, q, size=(l1, width + l2), dtype=np.int64)
        bottom = rng.integers(0, q, size=(width, l2), dtype=np.int64)
        top.flags.writeable = bottom.flags.writeable = False
        return cls(q, l1, l2, width, top, bottom)

    def apply(self, randomness: np.ndarray) -> np.ndarray:
        self._check_storage()
        s = integer_vector(randomness, self.randomness_size, "commitment randomness")
        top, bottom = self._matrix_products(s[self.l1:], s[-self.l2:])
        upper = _add_canonical_mod(mod_q(s[:self.l1], self.q), top, self.q)
        lower = _add_canonical_mod(mod_q(s[self.l1:self.l1 + self.width], self.q), bottom, self.q)
        return np.concatenate((upper, lower))

    def commit(self, message: np.ndarray, randomness: np.ndarray) -> np.ndarray:
        self._check_storage()
        message = integer_vector(message, self.width, "committed message")
        s = integer_vector(randomness, self.randomness_size, "commitment randomness")
        if type(self) is LatticeCommitmentKey and metal_matrix._can_fuse_commitment(self.top, self.bottom, self.q):
            return metal_matrix._commitment_products(self.top, self.bottom, (message,), (s,), self.q)[0]
        top, bottom = self._matrix_products(s[self.l1:], s[-self.l2:])
        upper = _add_canonical_mod(mod_q(s[:self.l1], self.q), top, self.q)
        return _commitment_output(upper, s[self.l1:self.l1+self.width], message,
                                  bottom, self.q)

    def commit_pair(self, first, second):
        self._check_storage()
        messages = [integer_vector(pair[0], self.width, 'committed message') for pair in (first, second)]
        randomness = [integer_vector(pair[1], self.randomness_size, 'commitment randomness') for pair in (first, second)]
        if type(self) is LatticeCommitmentKey and metal_matrix._can_fuse_commitment(self.top, self.bottom, self.q):
            return metal_matrix._commitment_products(self.top, self.bottom, messages, randomness, self.q)
        top_right = tuple(s[self.l1:] for s in randomness)
        bottom_right = tuple(s[-self.l2:] for s in randomness)
        if self.top.dtype.kind == 'f' or metal_matrix.is_prepared(self.top):
            upper_pair, lower = self._matrix_products(top_right, bottom_right)
        else:
            upper_pair = None
            lower = self._matrix_product(self.bottom, bottom_right)
        result = []
        for index, (message, s) in enumerate(zip(messages, randomness)):
            product = upper_pair[:, index] if upper_pair is not None else self._matrix_product(self.top, s[self.l1:])
            upper = _add_canonical_mod(mod_q(s[:self.l1], self.q), product, self.q)
            result.append(_commitment_output(upper, s[self.l1:self.l1+self.width], message,
                                             lower[:, index], self.q))
        return tuple(result)

    @property
    def digest(self) -> bytes:
        self._check_storage()
        dimensions = self.q, self.l1, self.l2, self.width
        cached = self.__dict__.get('_digest_cache')
        if (cached is not None and cached[0] == dimensions
                and cached[1] is self.top and cached[2] is self.bottom):
            return cached[3]
        value = framed_hash(b"Yang2019 commitment key v1", *dimensions, self.top, self.bottom)
        object.__setattr__(self, '_digest_cache', (dimensions, self.top, self.bottom, value))
        return value


@lru_cache(maxsize=32)
def _gaussian_alias_table(sigma: float, bound: int):
    support = np.arange(-bound, bound + 1, dtype=np.float64)
    masses = np.exp(-0.5 * (support / sigma) ** 2)
    scaled = masses * (masses.size / masses.sum())
    probabilities = np.ones(masses.size, dtype=np.float64)
    aliases = np.arange(masses.size, dtype=np.int64)
    small = list(np.flatnonzero(scaled < 1.0))
    large = list(np.flatnonzero(scaled >= 1.0))
    while small and large:
        left, right = small.pop(), large.pop()
        probabilities[left] = scaled[left]
        aliases[left] = right
        scaled[right] -= 1.0 - scaled[left]
        (small if scaled[right] < 1.0 else large).append(right)
    return (np.frombuffer(probabilities.tobytes(), dtype=np.float64),
            np.frombuffer(aliases.tobytes(), dtype=np.int64))


def sample_standard_discrete_gaussian(rng, size: int, sigma: float,
                                     tail: int = 12) -> np.ndarray:
    if not isinstance(size, int) or size < 0 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("invalid Gaussian size or standard deviation")
    if not isinstance(tail, int) or tail < 1:
        raise ValueError("tail must be a positive integer")
    bound = math.ceil(tail * sigma)
    if bound >= np.iinfo(np.int64).max // 4:
        raise ValueError("Gaussian support exceeds supported integer range")
    result = np.empty(size, dtype=np.int64)
    from .randomness import CryptographicRNG
    from .fast_uniform import uniform_source, _native_library
    native = isinstance(rng, CryptographicRNG) and _native_library() is not None
    workers = min(4, os.cpu_count() or 1) if native and size >= 1048576 else 1
    with ExitStack() as stack:
        sources = [stack.enter_context(uniform_source(rng)) for _ in range(workers)]
        if workers == 1:
            _fill_standard_discrete_gaussian(sources[0], result, sigma, bound)
        else:
            def fill(index):
                part = result[size*index//workers:size*(index+1)//workers]
                _fill_standard_discrete_gaussian(sources[index], part, sigma, bound)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(fill, range(workers)))
    return result


def _fill_standard_discrete_gaussian(rng, result, sigma, bound):
    size = result.size
    if bound <= 4096:
        probabilities, aliases = _gaussian_alias_table(float(sigma), bound)
        for offset in range(0, size, 65536):
            count = min(65536, size - offset)
            indices = rng.integers(0, probabilities.size, size=count, dtype=np.int64)
            alternate = rng.random(count) >= probabilities[indices]
            np.copyto(indices, aliases[indices], where=alternate)
            np.subtract(indices, bound, out=result[offset:offset + count])
        return
    use_laplace = 1.0 <= sigma <= 1e8
    if use_laplace:
        from .randomness import CryptographicRNG
        from .fast_uniform import _AES256CTRBytes
        probability = -math.expm1(-1.0 / sigma)
        centre = -sigma ** 2 * math.log1p(-probability)
        divisor = math.log1p(-probability)
        inplace_geometric = (getattr(type(rng), 'geometric', None) is CryptographicRNG.geometric
                             and getattr(type(rng), 'random', None)
                             in (CryptographicRNG.random, _AES256CTRBytes.random))
    filled = 0
    while filled < size:
        if use_laplace:
            count = min(65536, max(64, (3 * (size - filled) + 1) // 2))
            if inplace_geometric:
                candidates = _gaussian_geometric_inplace(rng, divisor, count)
                candidates -= _gaussian_geometric_inplace(rng, divisor, count)
            else:
                candidates = rng.geometric(probability, size=count)
                candidates -= rng.geometric(probability, size=count)
            probabilities = candidates.astype(np.float64)
            np.abs(probabilities, out=probabilities)
            np.subtract(probabilities, centre, out=probabilities)
            np.divide(probabilities, sigma, out=probabilities)
            np.square(probabilities, out=probabilities)
            np.multiply(probabilities, -0.5, out=probabilities)
            np.exp(probabilities, out=probabilities)
            accepted = candidates[(candidates >= -bound) & (candidates <= bound)
                                  & (rng.random(count) < probabilities)]
        else:
            count = min(65536, max(64, (size - filled) * (math.ceil(bound / sigma) + 1)))
            candidates = rng.integers(-bound, bound + 1, size=count, dtype=np.int64)
            probabilities = candidates.astype(np.float64)
            np.divide(probabilities, sigma, out=probabilities)
            np.square(probabilities, out=probabilities)
            np.multiply(probabilities, -0.5, out=probabilities)
            np.exp(probabilities, out=probabilities)
            accepted = candidates[rng.random(count) < probabilities]
        take = min(accepted.size, size - filled)
        result[filled:filled + take] = accepted[:take]
        filled += take


def _gaussian_geometric_inplace(rng, divisor, count):
    values = rng.random(count)
    np.negative(values, out=values)
    np.log1p(values, out=values)
    np.divide(values, divisor, out=values)
    np.floor(values, out=values)
    np.add(values, 1.0, out=values)
    return values.astype(np.int64)


@dataclass(frozen=True)
class SISBitCommitmentKey:
    q: int
    rows: int
    message_bits: int
    randomness_bits: int
    matrix: np.ndarray

    def __post_init__(self):
        _check_key_dimensions(self.q, self.rows, self.message_bits, self.randomness_bits)
        object.__setattr__(self, 'matrix', _freeze_public_matrix(
            self.matrix, (self.rows, self.message_bits + self.randomness_bits), self.q))

    def _check_storage(self):
        _check_public_matrix(self.matrix, (self.rows, self.message_bits + self.randomness_bits))

    @classmethod
    def generate(cls, q: int, rows: int, message_bits: int, hiding_bits: int, rng,
                 max_memory_bytes: int = 512 * 1024 * 1024):
        if any(not isinstance(v, int) or v < 1 for v in (q, rows, message_bits, hiding_bits)):
            raise ValueError("invalid auxiliary commitment parameters")
        randomness_bits = rows * (q - 1).bit_length() + 2 * hiding_bits
        _check_key_dimensions(q, rows, message_bits, hiding_bits)
        require_memory(16 * rows * (message_bits + randomness_bits), max_memory_bytes,
                       'auxiliary SIS matrix')
        matrix = rng.integers(0, q, size=(rows, message_bits + randomness_bits), dtype=np.int64)
        matrix.flags.writeable = False
        return cls(q, rows, message_bits, randomness_bits, matrix)

    def commit(self, message: np.ndarray, randomness: np.ndarray) -> np.ndarray:
        self._check_storage()
        m = integer_vector(message, self.message_bits, "auxiliary message bits")
        r = integer_vector(randomness, self.randomness_bits, "auxiliary randomness bits")
        if np.any((m != 0) & (m != 1)) or np.any((r != 0) & (r != 1)):
            raise ValueError("auxiliary message and randomness must be binary")
        return _matmul_mod_canonical_left(self.matrix, np.concatenate((m, r)), self.q)

    @property
    def hiding_distance_bound(self) -> float:
        return 0.5 * 2 ** ((self.rows * math.log2(self.q) - self.randomness_bits) / 2)

    @property
    def binding_sis_euclidean_bound(self) -> float:
        return math.sqrt(self.message_bits + self.randomness_bits)
