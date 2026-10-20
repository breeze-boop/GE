from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from io import BytesIO
from types import MappingProxyType
from typing import Callable
from weakref import ref

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .decomposition import idec, mdec, vdec, weights
from .mathutils import _add_canonical_mod, _sub_canonical_mod, add_mod, matmul_mod, require_memory, scale_mod
from .linear_hash import LinearPublicKeyHash, linear_hash_storage_plan


_CANONICAL_VECTORS = {}


def _remember_canonical_vector(array, q):
    binding = _immutable_snapshot_binding(array)
    if binding is None or array.dtype != np.dtype(np.int64):
        return
    key = id(array), q
    def discard(reference):
        current = _CANONICAL_VECTORS.get(key)
        if current is not None and current[0] is reference:
            _CANONICAL_VECTORS.pop(key, None)
    _CANONICAL_VECTORS[key] = (ref(array, discard), binding[1], binding[2:])
    while len(_CANONICAL_VECTORS) > 64:
        _CANONICAL_VECTORS.pop(next(iter(_CANONICAL_VECTORS)))


def _known_canonical_vector(array, q):
    saved = _CANONICAL_VECTORS.get((id(array), q))
    if saved is None or saved[0]() is not array:
        return False
    binding = _immutable_snapshot_binding(array)
    return (binding is not None and binding[1] is saved[1]
            and binding[2:] == saved[2])


def _binary(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "iub" or np.any((a != 0) & (a != 1)):
        raise ValueError(f"{name} must be binary with shape {shape}")
    return a.astype(np.int64, copy=False)


def _canonical(value, shape: tuple[int, ...], q: int, name: str) -> np.ndarray:
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in "iub" or np.any(a < 0) or np.any(a >= q):
        raise ValueError(f"{name} must be a canonical Z_q array with shape {shape}")
    return a.astype(np.int64, copy=True)


def _mm(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    return matmul_mod(a, b, q)


def _accumulator_linear_operands(node, jnode, sibling, jsibling, q):
    difference = _sub_canonical_mod(jnode, jsibling, q)
    return (_sub_canonical_mod(node, difference, q),
            _add_canonical_mod(sibling, difference, q))


def _scale(a: np.ndarray, coefficient: int, q: int) -> np.ndarray:
    return scale_mod(a, coefficient, q)


def _weighted(bits: np.ndarray, beta: int, length: int, q: int) -> np.ndarray:
    w = weights(beta)
    if not w.size:
        return np.zeros(length, dtype=np.int64)
    return _mm(bits.reshape(length, w.size), w, q)


def _ordinary_decode(bits: np.ndarray, n: int, k: int, q: int) -> np.ndarray:
    return _mm(bits.reshape(n, k), 1 << np.arange(k, dtype=np.int64), q)


def _ordinary_encode(values: np.ndarray, k: int) -> np.ndarray:
    return ((np.asarray(values, dtype=np.int64)[:, None] >> np.arange(k)) & 1).reshape(-1)


@dataclass(frozen=True)
class NonzeroCertificate:
    slack_bits: np.ndarray


def nonzero_certificate(p: np.ndarray) -> NonzeroCertificate:
    a = np.asarray(p)
    p = _binary(a, a.shape, "p")
    if p.ndim != 1 or not p.size or not p.any():
        raise ValueError('Nonzero bit vector required')
    return NonzeroCertificate(idec(int(p.sum()) - 1, p.size - 1))


@dataclass(frozen=True)
class PolicyCertificate:
    slack_bits: np.ndarray
    bound: int


def _policy_inputs(patterns, message, d):
    message = np.asarray(message)
    patterns = np.asarray(patterns)
    if message.ndim != 1 or patterns.ndim != 2:
        raise ValueError('Require message.ndim=1, patterns.ndim=2')
    message = _binary(message, message.shape, "message")
    patterns = _binary(patterns, patterns.shape, "patterns")
    e, t = patterns.shape
    if e < 1 or t < 1 or message.size < t or not 0 <= d <= t:
        raise ValueError("policy requires e>=1 and m>=t>=d>=0")
    windows = sliding_window_view(message, t)
    distances = np.array([np.count_nonzero(windows != s, axis=1) for s in patterns])
    return patterns, message, distances


def policy_certificate(patterns: np.ndarray, message: np.ndarray, d: int) -> PolicyCertificate:
    _, message, distances = _policy_inputs(patterns, message, d)
    if np.any(distances < d):
        raise ValueError("message violates the prohibitive policy")

    bound = message.size - d
    return PolicyCertificate(vdec((distances - d).reshape(-1), bound), bound)


@dataclass(frozen=True)
class LinearBlock:
    name: str
    size: int
    apply: Callable[[np.ndarray], np.ndarray]


def _canonical_matvec_input(value, width, q):
    array = np.asarray(value)
    if array.shape != (width,) or array.dtype.kind not in 'iub':
        raise ValueError(f'operator input must be an integer vector of length {width}')
    if _known_canonical_vector(array, q):
        return array
    if (array.dtype == np.dtype(np.int64) and array.flags.c_contiguous
            and not array.flags.writeable and type(q) is int
            and 1 < q <= np.iinfo(np.int64).max):
        view = np.frombuffer(array, dtype=np.int64)
        owner = view
        while isinstance(owner, (np.ndarray, memoryview)):
            if isinstance(owner, np.ndarray):
                owner = owner.base
            else:
                if not owner.readonly:
                    break
                owner = owner.obj
        if (isinstance(owner, bytes) and view.shape == (width,)
                and int(view.view(np.uint64).max(initial=0)) < q):
            return view
    if (array.dtype == np.dtype(np.int64) and type(q) is int
            and 1 < q <= np.iinfo(np.int64).max):
        result = array.copy()
        if int(result.view(np.uint64).max(initial=0)) >= q:
            np.remainder(result, q, out=result)
        return result
    return (array % q).astype(np.int64, copy=False)


@dataclass(frozen=True)
class ExactLinearOperator:
    blocks: tuple[LinearBlock, ...]
    width: int
    q: int

    @property
    def shape(self) -> tuple[int, int]:
        return (sum(b.size for b in self.blocks), self.width)

    def matvec(self, x: np.ndarray) -> np.ndarray:
        x = _canonical_matvec_input(x, self.width, self.q)
        return np.concatenate([np.asarray(b.apply(x), dtype=np.int64).reshape(b.size) % self.q
                               for b in self.blocks])

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        return self.matvec(x)


def _immutable_integer_snapshot(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'iu':
        raise TypeError(f'{name} must contain integers')
    if array.flags.c_contiguous and not array.flags.writeable:
        owner = array
        while isinstance(owner, (np.ndarray, memoryview)):
            if isinstance(owner, np.ndarray):
                owner = owner.base
            else:
                if not owner.readonly:
                    break
                owner = owner.obj
        if isinstance(owner, bytes):
            return np.frombuffer(array, dtype=array.dtype).reshape(array.shape)
    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)


def _immutable_snapshot_binding(array):
    if (type(array) is not np.ndarray or array.dtype.kind not in 'iu'
            or array.flags.writeable or not array.flags.c_contiguous):
        return None
    owner = array
    while isinstance(owner, (np.ndarray, memoryview)):
        if isinstance(owner, np.ndarray):
            owner = owner.base
        else:
            if not owner.readonly:
                return None
            owner = owner.obj
    if not isinstance(owner, bytes):
        return None
    return (array, owner, array.__array_interface__['data'][0], array.dtype.str,
            array.shape, array.strides)


@dataclass(frozen=True)
class CompiledRelation:
    M: ExactLinearOperator
    y: np.ndarray
    triples: np.ndarray
    layout: dict[str, slice]
    statement_digest: str
    x: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, 'layout', MappingProxyType(dict(self.layout)))
        object.__setattr__(self, 'y', _immutable_integer_snapshot(self.y, 'public target'))
        object.__setattr__(self, 'triples', _immutable_integer_snapshot(self.triples, 'multiplication triples'))
        object.__setattr__(self, '_CompiledRelation__yang_context_binding',
                           (self.statement_digest, _immutable_snapshot_binding(self.y),
                            _immutable_snapshot_binding(self.triples)))
        object.__setattr__(self, '_CompiledRelation__yang_context_state', None)
        object.__setattr__(self, '_CompiledRelation__validated_dimensions', None)
        from .yang import _attach_native_constraint_layout
        _attach_native_constraint_layout(self.triples, self.M.shape[1])

    def _check_context_binding(self):
        statement_digest, target_binding, triple_binding = self.__yang_context_binding
        if self.statement_digest != statement_digest:
            raise ValueError('Statement digest changed')
        for array, expected in ((self.y, target_binding), (self.triples, triple_binding)):
            current = _immutable_snapshot_binding(array)
            if (current is None or current[0] is not expected[0]
                    or current[1] is not expected[1] or current[2:] != expected[2:]):
                raise ValueError('Relation array layout changed')

    def _with_witness(self, witness, *, operator=None):
        self._check_context_binding()
        result = replace(self, x=witness, M=self.M if operator is None else operator)
        state = self.__yang_context_state
        if state is not None:
            object.__setattr__(result, '_CompiledRelation__yang_context_state', state.copy())
        dimensions = self._public_dimensions_cached
        if (dimensions is not None and result.M.shape == self.M.shape
                and result.M.q == self.M.q):
            result._remember_public_dimensions(dimensions)
        return result

    @property
    def _public_dimensions_cached(self):
        self._check_context_binding()
        saved = self.__validated_dimensions
        if (saved is not None and saved[0] is self.M
                and saved[1] == self.M.shape and saved[2] == self.M.q):
            return saved[3]
        return None

    def _remember_public_dimensions(self, dimensions):
        self._check_context_binding()
        object.__setattr__(self, '_CompiledRelation__validated_dimensions',
                           (self.M, self.M.shape, self.M.q, dimensions))

    @property
    def _context_prepared(self):
        self._check_context_binding()
        return self.__yang_context_state is not None

    @property
    def _yang_context_prefix(self):
        self._check_context_binding()
        state = self.__yang_context_state
        if state is not None:
            return state.copy()
        from .commitments import framed_hash_state
        state = framed_hash_state(b'Yang P1 FDGE statement context v1',
                                  bytes.fromhex(self.statement_digest), self.y, self.triples)
        object.__setattr__(self, '_CompiledRelation__yang_context_state', state)
        return state.copy()

    @property
    def target(self) -> np.ndarray:
        return self.y

    @property
    def witness(self) -> np.ndarray | None:
        return self.x

    def check(self, x: np.ndarray | None = None) -> bool:
        candidate = self.x if x is None else x
        if candidate is None:
            return False
        candidate = np.asarray(candidate)
        if candidate.shape != (self.M.width,) or candidate.dtype.kind not in "iub":
            return False
        if np.any((candidate != 0) & (candidate != 1)):
            return False
        out, left, right = self.triples.T
        return bool(np.array_equal(self.M @ candidate, self.y)
                    and np.all(candidate[out] == candidate[left] * candidate[right]))


def _framed_hash(items) -> str:
    h = hashlib.sha256(b"FDGE exact public relation v2; triples=out,left,right; immutable-F-digest")
    for item in items:
        if _streamed_F(item) and item.storage == 'memory':
            data = repr(('canonical-F-int64-sha256-v1', item.params.q,
                         item.shape, item.canonical_digest.hex())).encode('utf8')
            h.update(len(data).to_bytes(8, 'little'))
            h.update(data)
            continue
        if _streamed_F(item):

            prefix = (str(item.shape) + ":i8:").encode()
            byte_count = 8 * item.shape[0] * item.shape[1]
            h.update((len(prefix) + byte_count).to_bytes(8, "little"))
            h.update(prefix)
            seen = 0
            for chunk in item.iter_canonical_chunks():
                if (not isinstance(chunk, np.ndarray) or chunk.ndim != 1
                        or chunk.dtype != np.dtype('<i8') or not chunk.flags.c_contiguous
                        or not chunk.size or np.any(chunk < 0) or np.any(chunk >= item.params.q)):
                    raise ValueError('F chunks must be canonical C-order int64')
                seen += chunk.nbytes
                if seen > byte_count:
                    raise ValueError('F stream exceeds its declared shape')
                h.update(memoryview(chunk).cast('B'))
            if seen != byte_count:
                raise ValueError('F stream does not fill its declared shape')
            continue
        elif isinstance(item, np.ndarray):
            from .commitments import _update_canonical_int64

            prefix = (str(item.shape) + ":i8:").encode()
            h.update((len(prefix) + 8 * item.size).to_bytes(8, "little"))
            h.update(prefix)
            _update_canonical_int64(h, item)
            continue
        else:
            data = repr(item).encode("utf8")
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    return h.hexdigest()


def _streamed_F(F) -> bool:
    return isinstance(F, LinearPublicKeyHash) and F.is_read_only


def _immutable_multiplication_triples(width, products, index_dtype):
    index_dtype = np.dtype(index_dtype)
    if (not isinstance(width, (int, np.integer)) or isinstance(width, (bool, np.bool_))
            or width < 1 or index_dtype not in (np.dtype(np.uint32), np.dtype(np.int64))
            or width - 1 > np.iinfo(index_dtype).max):
        raise ValueError('Invalid triple width or dtype')
    products = tuple(products)
    total = int(width)
    for count, output, left, right, variable_left, divisor in products:
        if (any(not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                for value in (count, output, left, right, divisor))
                or type(variable_left) is not bool or count < 1 or divisor < 1
                or min(output, left, right) < 0 or output + count > width
                or left + (count if variable_left else 1) > width
                or right + (count - 1) // divisor >= width):
            raise ValueError('Triple index exceeds witness')
        total += int(count)
    chunk_size = min(65536, int(width))
    indices = np.arange(chunk_size, dtype=index_dtype)
    block = np.empty((chunk_size, 3), dtype=index_dtype)
    with BytesIO() as buffer:
        for start in range(0, int(width), chunk_size):
            count = min(chunk_size, int(width) - start)
            current = block[:count]
            np.add(indices[:count, None], start, out=current)
            buffer.write(memoryview(current).cast('B'))
        for count, output, left, right, variable_left, divisor in products:
            for start in range(0, int(count), chunk_size):
                size = min(chunk_size, int(count) - start)
                current = block[:size]
                np.add(indices[:size], int(output) + start, out=current[:, 0])
                if variable_left:
                    np.add(indices[:size], int(left) + start, out=current[:, 1])
                else:
                    current[:, 1].fill(left)
                if divisor == 1:
                    np.add(indices[:size], int(right) + start, out=current[:, 2])
                else:
                    np.add(indices[:size], start, out=current[:, 2])
                    np.floor_divide(current[:, 2], divisor, out=current[:, 2])
                    np.add(current[:, 2], right, out=current[:, 2])
                buffer.write(memoryview(current).cast('B'))
        result = np.frombuffer(buffer.getvalue(), dtype=index_dtype).reshape(total, 3)
        from .yang import _remember_native_constraint_layout
        _remember_native_constraint_layout(result, int(width))
        return result


def relation_resource_plan(params, *, F=None, F_block_columns=None) -> dict:
    n, m, k, nk, ell, e, t, d = (params.n, params.m, params.k, params.nk,
                                params.depth, params.e, params.t, params.d)
    noise_delta = (2 * params.error_bound).bit_length()
    policy_rows = e * (m - t + 1)
    width = (ell + 4 * ell * nk + (nk - 1).bit_length() + 4 * n * m * k + m
             + policy_rows * (m - d).bit_length() + 4 * n + (6 * m + 2 * ell) * noise_delta)
    triples = width + 2 * ell * nk + 2 * n * m * k
    rows = ell * n + 1 + n + 6 * m + 2 * ell + policy_rows
    f_bytes = 8 * n * 2 * n * m * k
    public_entries = 3 * n * m + 2 * n * ell + nk + e * t + 6 * m + 2 * ell

    index_dtype = np.dtype(np.uint32 if width <= 2**32 else np.int64)
    triples_bytes = index_dtype.itemsize * 3 * triples
    index_bytes = index_dtype.itemsize * min(width, 65536)
    triple_block_bytes = 3 * index_bytes
    triple_growth_reserve = 2 * triples_bytes
    other_arrays = 8 * (3 * public_entries + 11 * width + 3 * rows)
    other_arrays += triples_bytes + index_bytes + triple_block_bytes + triple_growth_reserve
    freeze_reserve = 8 * rows
    other_arrays += freeze_reserve
    canonical_hash_workspace = 8 * min(262144, max(3 * triples, rows, public_entries))
    other_arrays += canonical_hash_workspace + 2 * 1024 * 1024
    if F_block_columns is not None and (not isinstance(F_block_columns, int)
            or isinstance(F_block_columns, bool) or F_block_columns < 1):
        raise ValueError('F_block_columns must be a positive integer')
    if _streamed_F(F):
        if F.closed:
            raise ValueError('F storage is closed')
        if F.shape != (n, 2 * n * m * k) or F.params.q != params.q:
            raise ValueError('F shape/modulus mismatch')
        columns = F.column_block_size if F_block_columns is None else F_block_columns
        if columns is not None:
            columns = min(F.input_length, columns)
        backend_plan = linear_hash_storage_plan(F.params, storage_dtype=F.storage_dtype,
            disk_backed=F.storage == 'memmap', columns=columns, read_only=True)
        columns = backend_plan['column_block_size']
        resident = F.matrix_bytes if F.storage == 'memory' else 0

        digest_workspace = 4 * F.canonical_chunk_bytes
        workspace = max(backend_plan['application_explicit_buffer_bytes'] - resident,
                        digest_workspace)
        f_copy = 0
        backend = 'read_only_streamed_' + F.storage
        storage_dtype = F.storage_dtype.name
        peak = other_arrays + resident + workspace
    else:
        if F_block_columns is not None:
            raise ValueError('F_block_columns requires read-only F')

        resident, f_copy, workspace = f_bytes, f_bytes, 0
        backend, storage_dtype, columns = 'copied_canonical_int64', 'int64', None
        backend_plan = None
        peak = other_arrays + resident + f_copy
    return {"witness_coordinates": width, "triples": triples, "rows": rows,
            "F_copy_bytes": f_copy, "F_resident_matrix_bytes": resident,
            "F_workspace_bytes": workspace, "F_backend": backend,
            "F_storage_dtype": storage_dtype, "F_block_columns": columns,
            "F_backend_resource_plan": backend_plan,
            "F_mathematical_int64_bytes": f_bytes,
            "other_relation_arrays_bytes": other_arrays,
            "triples_storage_dtype": index_dtype.name,
            "triples_storage_bytes": triples_bytes,
            "construction_index_bytes": index_bytes,
            "triples_generation_block_bytes": triple_block_bytes,
            "triples_growth_reserve_bytes": triple_growth_reserve,
            "triples_freeze_copy_bytes": 0,
            "triples_builder_memory_scope": 'includes_growth/final_copy; RSS_unbounded',
            "canonical_hash_workspace_bytes": canonical_hash_workspace,
            "public_snapshot_reserve_bytes": freeze_reserve,
            "OS_mapped_page_cache_included": False,
            "conservative_working_bytes": peak,
            "budget_bytes": getattr(params, "max_memory_bytes", 512 * 1024 * 1024)}


def _validate_parameters(params) -> None:
    if hasattr(params, "validate"):
        params.validate()
    from .params import _is_prime_64
    for name in ("n", "q", "t", "e", "depth", "k", "m", "nk"):
        value = getattr(params, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("d", "error_bound"):
        value = getattr(params, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if params.q > np.iinfo(np.int64).max or params.q == 2 or not _is_prime_64(params.q):
        raise ValueError('q must be an odd int64 prime')
    if params.k != (params.q - 1).bit_length():
        raise ValueError("k must equal ceil(log2(q))")


def public_statement(*, params, Aacc, Arec, Aoa, F, oa_pk, root, patterns,
                     ciphertext, F_block_columns=None) -> CompiledRelation:
    _validate_parameters(params)
    n, m, k, nk, ell, q = params.n, params.m, params.k, params.nk, params.depth, params.q
    bound, d, e, t = params.error_bound, params.d, params.e, params.t
    if ell < 1 or nk != n * k or m != 2 * nk or q <= max(m, t):
        raise ValueError('Require depth>=1, m=2nk, q>max(m,t)')
    if not 0 <= d <= t <= m or e < 1 or not 0 <= bound < q // 2:
        raise ValueError("invalid policy or encryption noise bound")
    resource_plan = relation_resource_plan(params, F=F, F_block_columns=F_block_columns)
    require_memory(resource_plan["conservative_working_bytes"], resource_plan["budget_bytes"],
                   'compiled relation workspace')
    A = _canonical(Aacc, (n, 2 * nk), q, "Aacc")
    Ar = _canonical(Arec, (n, m), q, "Arec")
    Ao = _canonical(Aoa, (n, m), q, "Aoa")
    if _streamed_F(F):

        f = F
        def apply_F(vector):
            return f.apply_bits(vector, columns=resource_plan['F_block_columns'])
    else:
        f = _canonical(F if isinstance(F, np.ndarray) else F.matrix, (n, 2 * n * m * k), q, "F")
        def apply_F(vector):
            return _mm(f, vector, q)
    Uo = [_canonical(u, (n, ell), q, "OA key") for u in (oa_pk.U1, oa_pk.U2)]
    root = _binary(root, (nk,), "root").copy()
    patterns = _binary(patterns, (e, t), "patterns").copy()
    rec_pairs = (ciphertext.c_rec.c1, ciphertext.c_rec.c2)
    oa_pairs = (ciphertext.c_oa.c1, ciphertext.c_oa.c2)
    rec_ct = [(_canonical(c.c1, (m,), q, "recipient c1"),
               _canonical(c.c2, (m,), q, "recipient c2")) for c in rec_pairs]
    oa_ct = [(_canonical(c.c1, (m,), q, "OA c1"),
              _canonical(c.c2, (ell,), q, "OA c2")) for c in oa_pairs]
    layout, width = {}, 0

    def allocate(name, size):
        nonlocal width
        layout[name] = slice(width, width + size)
        width += size

    allocate("identity", ell)
    allocate("p", nk)
    allocate("nonzero_slack", weights(nk - 1).size)
    for i in range(2):
        allocate(f"a{i}", n * m * k)
    allocate("message", m)
    policy_rows = e * (m - t + 1)
    allocate("policy_slack", policy_rows * weights(m - d).size)
    for level in range(ell):
        if level:
            allocate(f"node{level}", nk)
        for prefix in ("sibling", "jnode", "jsibling"):
            allocate(f"{prefix}{level}", nk)
    delta_noise = weights(2 * bound).size
    for i in range(2):
        allocate(f"s_rec{i}", n)
        allocate(f"x_rec{i}", m * delta_noise)
        allocate(f"y_rec{i}", m * delta_noise)
        allocate(f"a_times_s{i}", n * m * k)
        allocate(f"s_oa{i}", n)
        allocate(f"x_oa{i}", m * delta_noise)
        allocate(f"y_oa{i}", ell * delta_noise)

    layout = MappingProxyType(layout)
    blocks, targets = [], []

    def equation(name, target, apply):
        target = np.asarray(target, dtype=np.int64).reshape(-1) % q
        blocks.append(LinearBlock(name, target.size, apply))
        targets.append(target)

    def part(x, name):
        return x[layout[name]]

    A0, A1 = A[:, :nk], A[:, nk:]
    for level in range(ell):
        node_name = "p" if level == 0 else f"node{level}"

        def accumulator_equation(x, level=level, node_name=node_name):
            left, right = _accumulator_linear_operands(
                part(x, node_name), part(x, f"jnode{level}"),
                part(x, f"sibling{level}"), part(x, f"jsibling{level}"), q)
            value = add_mod(_mm(A0, left, q), _mm(A1, right, q), q=q)
            if level + 1 < ell:
                value -= _ordinary_decode(part(x, f"node{level + 1}"), n, k, q)
            return value % q

        target = _ordinary_decode(root, n, k, q) if level + 1 == ell else np.zeros(n, dtype=np.int64)
        equation(f"accumulator_level_{level}", target, accumulator_equation)
    equation("nonzero_p", [1], lambda x: np.array([
        (int(_mm(np.ones(nk, dtype=np.int64), part(x, "p"), q))
         - int(_mm(weights(nk - 1), part(x, "nonzero_slack"), q))) % q]))
    key_coordinates = slice(layout['a0'].start, layout['a1'].stop)
    equation("key_binding_F", np.zeros(n, dtype=np.int64), lambda x:
             (_ordinary_decode(part(x, "p"), n, k, q)
              - apply_F(x[key_coordinates])) % q)
    half = q // 2
    for i in range(2):
        equation(f"recipient_{i}_c1", add_mod(rec_ct[i][0], np.array(bound), q=q), lambda x, i=i:
                 add_mod(_mm(Ar.T, part(x, f"s_rec{i}"), q),
                         _weighted(part(x, f"x_rec{i}"), 2 * bound, m, q), q=q))

        def recipient_second(x, i=i):
            products = part(x, f"a_times_s{i}").reshape(n, m, k)

            decoded = _mm(products.reshape(n * m, k), weights(q - 1), q).reshape(n, m)
            return add_mod(_mm(np.ones(n, dtype=np.int64), decoded, q),
                           _weighted(part(x, f"y_rec{i}"), 2 * bound, m, q),
                           _scale(part(x, "message"), half, q), q=q)

        equation(f"recipient_{i}_c2", add_mod(rec_ct[i][1], np.array(bound), q=q), recipient_second)
        equation(f"oa_{i}_c1", add_mod(oa_ct[i][0], np.array(bound), q=q), lambda x, i=i:
                 add_mod(_mm(Ao.T, part(x, f"s_oa{i}"), q),
                         _weighted(part(x, f"x_oa{i}"), 2 * bound, m, q), q=q))
        equation(f"oa_{i}_c2", add_mod(oa_ct[i][1], np.array(bound), q=q), lambda x, i=i:
                 add_mod(_mm(Uo[i].T, part(x, f"s_oa{i}"), q),
                         _weighted(part(x, f"y_oa{i}"), 2 * bound, ell, q),
                         _scale(part(x, "identity"), half, q), q=q))

    def policy_equations(x):
        windows = sliding_window_view(part(x, "message"), t)
        linear = np.concatenate([_mm(windows, 1 - 2 * s, q) for s in patterns])
        return (linear - _weighted(part(x, "policy_slack"), m - d, policy_rows, q)) % q

    equation("prohibitive_policy", np.repeat(d - patterns.sum(axis=1), m - t + 1), policy_equations)
    index_dtype = np.dtype(resource_plan["triples_storage_dtype"])
    products = []
    for level in range(ell):

        identity_index = layout["identity"].start + ell - 1 - level
        node_name = "p" if level == 0 else f"node{level}"
        for output_name, input_name in ((f"jnode{level}", node_name), (f"jsibling{level}", f"sibling{level}")):
            products.append((nk, layout[output_name].start, identity_index,
                             layout[input_name].start, False, 1))
    for i in range(2):
        count = n * m * k
        products.append((count, layout[f"a_times_s{i}"].start, layout[f"a{i}"].start,
                         layout[f"s_rec{i}"].start, True, m * k))
    triples = _immutable_multiplication_triples(width, products, index_dtype)
    assert len(triples) == resource_plan["triples"] and width == resource_plan["witness_coordinates"]
    y = np.concatenate(targets)
    digest = _framed_hash([
        (n, m, k, nk, ell, q, bound, d, e, t), A, Ar, Ao, f, *Uo, root, patterns,
        *[v for pair in rec_ct + oa_ct for v in pair],
        getattr(ciphertext, "epoch", None), getattr(ciphertext, "label", None),
        getattr(ciphertext, "context_id", None),
        tuple((name, s.start, s.stop) for name, s in layout.items()), triples, y,
    ])
    relation = CompiledRelation(ExactLinearOperator(tuple(blocks), width, q), y, triples, layout, digest)
    layout = relation.layout
    return relation


def _build_relation_with_key_hash(*, params, Aacc, Arec, Aoa, F, oa_pk, root, patterns,
                                  ciphertext, user_pk, witness, message, coins,
                                  F_block_columns=None, _prepared_key=None,
                                  _public_relation=None) -> CompiledRelation:
    prepared = _prepared_key
    if prepared is not None:
        leaf = prepared.leaf
    elif type(F) is LinearPublicKeyHash and F.is_read_only and F.storage == 'memory':
        if any(getattr(F.params, name) != getattr(params, name) for name in ('n', 'm', 'k', 'nk', 'q')):
            raise ValueError('F/relation dimension mismatch')
        plan = relation_resource_plan(params, F=F, F_block_columns=F_block_columns)
        retained = 16 * params.n * params.m + 2 * params.n * params.m * params.k
        retained += 8 * params.n + params.nk + 131072
        require_memory(plan['conservative_working_bytes'] + retained, plan['budget_bytes'],
                       'relation/key workspace')
        prepared = F._prepare_key(user_pk.U1, user_pk.U2, columns=F_block_columns)
        leaf = prepared.leaf
    else:
        leaf = (F.apply(user_pk.U1, user_pk.U2) if F_block_columns is None
                else F.apply(user_pk.U1, user_pk.U2, columns=F_block_columns))
    return _build_relation(params=params, Aacc=Aacc, Arec=Arec, Aoa=Aoa, F=F,
                           oa_pk=oa_pk, root=root, patterns=patterns, ciphertext=ciphertext,
                           user_pk=user_pk, user_p=leaf, witness=witness, message=message,
                           coins=coins, F_block_columns=F_block_columns, _prepared_key=prepared,
                           _public_relation=_public_relation)


def _build_relation(*, params, Aacc, Arec, Aoa, F, oa_pk, root, patterns,
                    ciphertext, user_pk, user_p, witness, message, coins,
                    F_block_columns=None, _prepared_key=None,
                    _public_relation=None) -> CompiledRelation:
    if _prepared_key is not None:
        from .linear_hash import _PreparedPublicKeyHash
        if type(_prepared_key) is not _PreparedPublicKeyHash:
            raise TypeError('Cached key/F binding mismatch')
        _prepared_key._validate_key(F, user_pk.U1, user_pk.U2)
    relation = (_public_relation if _public_relation is not None else
                public_statement(params=params, Aacc=Aacc, Arec=Arec, Aoa=Aoa, F=F,
                                oa_pk=oa_pk, root=root, patterns=patterns, ciphertext=ciphertext,
                                F_block_columns=F_block_columns))
    if type(relation) is not CompiledRelation or relation.x is not None:
        raise ValueError('Public relation must have x=None')
    relation._check_context_binding()
    operator = relation.M
    n, m, k, nk, ell, q, bound = (params.n, params.m, params.k, params.nk,
                                 params.depth, params.q, params.error_bound)
    if _prepared_key is not None:
        original = next(block for block in relation.M.blocks if block.name == 'key_binding_F')
        layout = relation.layout

        def key_binding(candidate):
            syndrome = _prepared_key._matched_syndrome(
                (candidate[layout['a0']], candidate[layout['a1']]))
            if syndrome is None:
                return original.apply(candidate)
            return (_ordinary_decode(candidate[layout['p']], n, k, q) - syndrome) % q

        blocks = tuple(LinearBlock(block.name, block.size, key_binding)
                       if block is original else block for block in relation.M.blocks)
        operator = ExactLinearOperator(blocks, relation.M.width, q)
    x = np.zeros(relation.M.width, dtype=np.int64)

    def put(name, value):
        value = np.asarray(value).reshape(-1)
        slot = relation.layout[name]
        if value.size != slot.stop - slot.start:
            raise ValueError(f"wrong witness length for {name}")
        x[slot] = value

    if not isinstance(witness.index, (int, np.integer)) or not 0 <= witness.index < 2 ** ell:
        raise ValueError("identity is outside the accumulator")
    if len(witness.siblings_bottom_up) != ell:
        raise ValueError("wrong accumulator path length")
    identity = np.array([(witness.index >> i) & 1 for i in range(ell)], dtype=np.int64)
    p = _binary(user_p, (nk,), "p")
    message = _binary(message, (m,), "message")
    put("identity", identity)
    put("p", p)
    put("nonzero_slack", nonzero_certificate(p).slack_bits)
    put("message", message)
    put("policy_slack", policy_certificate(patterns, message, params.d).slack_bits)
    A = _canonical(Aacc, (n, 2 * nk), q, "Aacc")
    node = p
    for level, sibling in enumerate(witness.siblings_bottom_up):
        sibling = _binary(sibling, (nk,), "sibling")
        if level:
            put(f"node{level}", node)
        put(f"sibling{level}", sibling)
        path_bit = identity[ell - 1 - level]
        put(f"jnode{level}", path_bit * node)
        put(f"jsibling{level}", path_bit * sibling)
        children = np.concatenate((node, sibling) if path_bit == 0 else (sibling, node))
        node = _ordinary_encode(_mm(A, children, q), k)
    for i, U in enumerate((user_pk.U1, user_pk.U2)):
        if _prepared_key is None:
            U = _canonical(U, (n, m), q, "recipient key")
            a = mdec(U.T, q)
        else:
            a = _prepared_key._decomposition(i)
        put(f"a{i}", a)
        s_rec = _binary(coins.s_rec[i], (n,), "s_rec")
        s_oa = _binary(coins.s_oa[i], (n,), "s_oa")
        put(f"s_rec{i}", s_rec)
        put(f"s_oa{i}", s_oa)
        np.multiply(a.reshape(n, m, k), s_rec[:, None, None],
                    out=x[relation.layout[f"a_times_s{i}"]].reshape(n, m, k), dtype=np.int64)
        for prefix, length in (("x_rec", m), ("y_rec", m), ("x_oa", m), ("y_oa", ell)):
            noise = np.asarray(getattr(coins, prefix)[i])
            if noise.shape != (length,) or noise.dtype.kind not in "iu" or np.any(noise < -bound) or np.any(noise > bound):
                raise ValueError(f"{prefix} is outside the prescribed error range")
            put(f"{prefix}{i}", vdec(noise.astype(np.int64) + bound, 2 * bound))
    return relation._with_witness(x, operator=operator)
