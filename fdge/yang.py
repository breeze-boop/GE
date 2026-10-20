from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from weakref import ref
import math
import secrets

import numpy as np

from .commitments import (LatticeCommitmentKey, SISBitCommitmentKey, framed_hash,
                          auxiliary_digest, integer_vector, sample_standard_discrete_gaussian,
                          _lattice_storage_dtype, _update_auxiliary_payload)
from .mathutils import (require_memory, scale_mod,
                        _add_canonical_mod, _sub_canonical_mod, _affine_canonical_mod,
                        _affine_canonical_equal)
from .params import _is_prime_64


@dataclass(frozen=True)
class YangProtocolParams:
    l1: int
    l2: int
    p: int
    sigma1: float
    auxiliary_rows: int
    auxiliary_hiding_bits: int
    gaussian_tail: int
    max_memory_bytes: int = 512 * 1024 * 1024

    def validate(self, q: int):
        for name, value in asdict(self).items():
            if name == "sigma1":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('sigma1 must be finite and > 0')
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if q > np.iinfo(np.int64).max or not _is_prime_64(q) or q <= 2 * self.p:
            raise ValueError("Yang requires prime q > 2p")
        if self.sigma1 < math.sqrt(2 * self.l2 / math.pi):
            raise ValueError("Yang requires sigma1 >= sqrt(2*l2/pi)")


def resource_plan(q: int, rows: int, width: int, count: int,
                  settings: YangProtocolParams | None = None) -> dict[str, int]:
    if settings is None:
        raise ValueError('Yang parameters required')
    settings.validate(q)
    if any(not isinstance(v, int) or v < 1 for v in (rows, width, count)):
        raise ValueError('rows, width, count must be positive integers')
    message_fields = rows + 4 * settings.l1 + 2 * width + 2 * count
    message_bits = 256
    rho_bits = settings.auxiliary_rows * (q - 1).bit_length() + 2 * settings.auxiliary_hiding_bits
    lattice_entries = (settings.l1 + settings.l2) * (width + count) + 2 * settings.l1 * settings.l2
    auxiliary_entries = settings.auxiliary_rows * (message_bits + rho_bits)
    itemsize = _lattice_storage_dtype(q).itemsize
    matrix_bytes = itemsize * lattice_entries + 8 * auxiliary_entries
    workspace_cache_bytes = 64 * 1024**2 if itemsize == 4 else 0

    response_snapshot_bytes = 8 * (rows + 6 * settings.l1 + 2 * settings.l2
                                   + 4 * width + 3 * count + rho_bits)
    return {"auxiliary_message_bits": message_bits, "auxiliary_randomness_bits": rho_bits,
            "auxiliary_payload_field_elements": message_fields,
            "auxiliary_payload_bytes": (4 if q <= 2**32 else 8) * message_fields,
            "auxiliary_encoding_workspace_bytes": 4 * 1024**2,
            "auxiliary_construction": "CRH_then_SIS",
            "dense_auxiliary_message_bits": 256 + message_fields * (q - 1).bit_length(),
            "public_matrix_storage_bytes": matrix_bytes,
            "public_matrix_entries": lattice_entries + auxiliary_entries,
            "matrix_workspace_cache_bytes": workspace_cache_bytes,
            "lattice_matrix_itemsize": itemsize,
            "public_matrix_freeze_reserve_bytes": 16 * (lattice_entries + auxiliary_entries),
            "response_snapshot_bytes": response_snapshot_bytes,
            "conservative_working_bytes": matrix_bytes + 16 * (lattice_entries + auxiliary_entries) + 200 * (width + count)
                                          + 24 * message_bits + response_snapshot_bytes + 4 * 1024**2 + workspace_cache_bytes,
            "budget_bytes": settings.max_memory_bytes}


def _constraint_indices_in_range(triples, width):
    if not triples.size:
        return False
    if triples.dtype.kind == 'i' and int(triples.min()) < 0:
        return False
    return int(triples.max()) < width


def _public_dimensions(relation):
    if getattr(relation, "x", None) is not None:
        raise ValueError('Verifier statement requires x=None')
    from .relations import CompiledRelation
    if type(relation) is CompiledRelation:
        relation._check_context_binding()
        cached = relation._public_dimensions_cached
        if cached is not None:
            return cached
    rows, width = relation.M.shape
    q = relation.M.q
    integer_vector(relation.y, rows, "public target", q)
    triples = np.asarray(relation.triples)
    if triples.ndim != 2 or triples.shape[1] != 3 or triples.dtype.kind not in "iu":
        raise ValueError("triples must have shape (L,3)")
    if not _constraint_indices_in_range(triples, width):
        raise ValueError("invalid multiplication triple index")
    if not isinstance(relation.statement_digest, str) or len(relation.statement_digest) != 64:
        raise ValueError('Canonical SHA256 digest required')
    bytes.fromhex(relation.statement_digest)
    dimensions = q, rows, width, len(triples)
    if type(relation) is CompiledRelation:
        relation._remember_public_dimensions(dimensions)
    return dimensions


@dataclass(frozen=True)
class YangPublicParameters:
    settings: YangProtocolParams
    witness_key: LatticeCommitmentKey
    quadratic_key: LatticeCommitmentKey
    auxiliary_key: SISBitCommitmentKey
    relation_rows: int
    sigma2: float
    rejection_M: float
    digest: bytes

    @classmethod
    def generate(cls, public_relation, settings: YangProtocolParams | None = None, *, rng=None):
        q, rows, width, count = _public_dimensions(public_relation)
        return cls.generate_dimensions(q=q, rows=rows, width=width, count=count,
                                       settings=settings, rng=rng)

    @classmethod
    def generate_dimensions(cls, *, q: int, rows: int, width: int, count: int,
                            settings: YangProtocolParams | None = None, rng=None):
        if settings is None:
            raise ValueError('Yang parameters required')
        plan = resource_plan(q, rows, width, count, settings)
        rng = np.random.default_rng() if rng is None else rng
        l1, l2 = settings.l1, settings.l2
        length = 2 * l1 + 2 * l2 + width + count
        logarithm = math.log2(length)
        sigma2 = 2 * settings.p * math.sqrt(length) * logarithm * settings.sigma1
        rejection_M = math.exp(13.3 / logarithm + 1 / (2 * logarithm ** 2))
        if (settings.p * math.ceil(settings.gaussian_tail * settings.sigma1)
                + math.ceil(settings.gaussian_tail * sigma2) >= np.iinfo(np.int64).max // 4):
            raise ValueError('Yang response exceeds integer range')

        message_bits = plan["auxiliary_message_bits"]
        require_memory(plan["conservative_working_bytes"],
                       settings.max_memory_bytes, "Yang P1 public matrices and working memory")
        from concurrent.futures import ThreadPoolExecutor
        B1 = LatticeCommitmentKey.generate(q, l1, l2, width, rng, settings.max_memory_bytes)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_digest = pool.submit(getattr, B1, 'digest')
            B2 = LatticeCommitmentKey.generate(q, l1, l2, count, rng, settings.max_memory_bytes)
            second_digest = pool.submit(getattr, B2, 'digest')
            auxiliary = SISBitCommitmentKey.generate(q, settings.auxiliary_rows, message_bits,
                                                    settings.auxiliary_hiding_bits, rng,
                                                    settings.max_memory_bytes)
            digest = framed_hash(b"Yang P1 public parameters v1", str(asdict(settings)),
                                 first_digest.result(), second_digest.result(), auxiliary.matrix, rows,
                                 repr(sigma2), repr(rejection_M))
        return cls(settings, B1, B2, auxiliary, rows, sigma2, rejection_M, digest)

    def diagnostics(self) -> dict:
        settings, B1, B2 = self.settings, self.witness_key, self.quadratic_key
        length = B1.randomness_size + B2.randomness_size
        explicit_q_factor = (16 * settings.p * math.sqrt(max(B1.randomness_size, B2.randomness_size))
                             * (self.sigma2 + settings.p * settings.sigma1))
        return {
            "protocol": "Yang2019_P1_interactive_Fig2",
            "auxiliary_commitment": "CRH_then_uniform_matrix_SIS_bit_commitment",
            "auxiliary_security_assumptions": 'SHA256 collision resistance; SIS binding/hiding',
            "B1_B2_storage": B1.top.dtype.name,
            "B1_B2_arithmetic": ('exact_uint32_mod_q; signed_responses_retained'
                                  if B1.top.dtype == np.dtype(np.uint32)
                                  else 'exact_mod_q; signed_responses_retained'),
            "q": B1.q, "witness_coordinates": B1.width, "quadratic_constraints": B2.width,
            "l1": settings.l1, "l2": settings.l2, "p": settings.p,
            "sigma1_standard_deviation": settings.sigma1,
            "sigma2_standard_deviation": self.sigma2, "rejection_M": self.rejection_M,
            "log_base": 2,
            "theorem_single_round_soundness_expression": 2 / (2 * settings.p + 1),
            "theorem_soundness_bound_is_validated": False,
            "theorem_single_attempt_completeness_error_expression": 1 - 1 / self.rejection_M,
            "theorem_q_explicit_factor_before_unspecified_soft_O": explicit_q_factor,
            "q_exceeds_explicit_factor": B1.q >= explicit_q_factor,
            "security_validated": False,
            "security_limitation": 'finite_security_unvalidated; soft_O_unspecified',
            "gaussian_sampling": 'finite_PMF; alias/rejection; not_constant_time',
            "public_matrix_randomness": 'runtime_rng; full_matrices',
            "gaussian_tail_standard_deviations": settings.gaussian_tail,
            "gaussian_union_truncation_bound_per_attempt": min(1.0, 4 * length * math.exp(-settings.gaussian_tail ** 2 / 2)),
            "auxiliary_randomness_bits": self.auxiliary_key.randomness_bits,
            "auxiliary_hiding_distance_bound": self.auxiliary_key.hiding_distance_bound,
            "auxiliary_SIS_collision_l2_bound": self.auxiliary_key.binding_sis_euclidean_bound,
            "public_matrix_storage_bytes": B1.top.nbytes + B1.bottom.nbytes + B2.top.nbytes + B2.bottom.nbytes + self.auxiliary_key.matrix.nbytes,
        }


def _context(relation, pp, auxiliary: bytes) -> bytes:
    if not isinstance(auxiliary, bytes):
        raise TypeError("auxiliary context must be bytes")
    from .relations import CompiledRelation
    if type(relation) is CompiledRelation:
        state = relation._yang_context_prefix.copy()
        for data in (pp.digest, auxiliary):
            state.update(len(data).to_bytes(8, 'little'))
            state.update(data)
        return state.digest()
    return framed_hash(b"Yang P1 FDGE statement context v1", bytes.fromhex(relation.statement_digest),
                       relation.y, relation.triples, pp.digest, auxiliary)


def _auxiliary_bits(context: bytes, vectors, q: int) -> np.ndarray:
    digest = auxiliary_digest(vectors, context, q)
    return np.unpackbits(np.frombuffer(digest, dtype=np.uint8), bitorder='little')


def _scale(vector, scalar: int, q: int):
    return scale_mod(vector, scalar, q)


def _product(a, b, q):
    if q <= 3_037_000_499:
        result = np.multiply(a, b)
        np.remainder(result, q, out=result)
        return result
    return np.asarray((a.astype(object) * b.astype(object)) % q, dtype=np.int64)


def _diagonal_constraint_block(triples, start, stop, width):
    if stop > width:
        return False
    block = triples[start:stop]
    dtype = triples.dtype if stop - 1 <= np.iinfo(triples.dtype).max else np.int64
    return (np.array_equal(block[:, 0], np.arange(start, stop, dtype=dtype))
            and np.array_equal(block[:, 1], block[:, 0])
            and np.array_equal(block[:, 2], block[:, 0]))


class _QuadraticLayoutPlan(tuple):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError('Validate witness before quadratic layout')


_NATIVE_CONSTRAINT_LAYOUTS = {}


def _constraint_layout_key(binding, width):
    return (id(binding[1]), *binding[2:], width)


def _remember_native_constraint_layout(triples, width):
    bound = _quadratic_layout_binding(triples)
    if bound is None or not 0 < width <= len(triples):
        raise ValueError('Immutable constraints and binary prefix required')
    binding = bound[1]
    key = _constraint_layout_key(binding, width)
    flags = (bytes([1]) * ((width + 65535)//65536)
             + bytes((len(triples) - width + 65535)//65536))
    _NATIVE_CONSTRAINT_LAYOUTS[key] = ({}, binding[1], flags)
    _attach_native_constraint_layout(triples, width)
    while len(_NATIVE_CONSTRAINT_LAYOUTS) > 32:
        _NATIVE_CONSTRAINT_LAYOUTS.pop(next(iter(_NATIVE_CONSTRAINT_LAYOUTS)))


def _attach_native_constraint_layout(triples, width):
    bound = _quadratic_layout_binding(triples)
    if bound is None:
        return
    key = _constraint_layout_key(bound[1], width)
    saved = _NATIVE_CONSTRAINT_LAYOUTS.get(key)
    if saved is None or saved[1] is not bound[1][1]:
        return
    identity = id(triples)
    def discard(reference):
        current = _NATIVE_CONSTRAINT_LAYOUTS.get(key)
        if current is not None and current[0].get(identity) is reference:
            current[0].pop(identity, None)
            if not current[0]:
                _NATIVE_CONSTRAINT_LAYOUTS.pop(key, None)
    saved[0][identity] = ref(triples, discard)


def _native_constraint_blocks(triples, width):
    if _diagonal_constraint_block is not _QUADRATIC_LAYOUT_FUNCTIONS[0]:
        return None
    bound = _quadratic_layout_binding(triples)
    if bound is None:
        return None
    key = _constraint_layout_key(bound[1], width)
    saved = _NATIVE_CONSTRAINT_LAYOUTS.get(key)
    return saved[2] if saved is not None and saved[1] is bound[1][1] else None


def _quadratic_layout_binding(triples):
    if (type(triples) is not np.ndarray or triples.ndim != 2 or triples.shape[1] != 3
            or triples.dtype.kind not in 'iu' or triples.flags.writeable
            or not triples.flags.c_contiguous):
        return None
    view = np.frombuffer(triples, dtype=triples.dtype).reshape(triples.shape)
    owner = view
    while isinstance(owner, (np.ndarray, memoryview)):
        if isinstance(owner, np.ndarray):
            owner = owner.base
        else:
            if not owner.readonly:
                return None
            owner = owner.obj
    if type(owner) is not bytes:
        return None
    return view, (triples, owner, view.__array_interface__['data'][0], view.dtype.str,
                  view.shape, view.strides)


def _quadratic_witness_layout(triples, x, q):
    bound = _quadratic_layout_binding(triples)
    functions = (_diagonal_constraint_block, _quadratic_witness_valid, _quadratic_masks)
    if (bound is None or type(x) is not np.ndarray or x.ndim != 1 or type(q) is not int
            or functions != _QUADRATIC_LAYOUT_FUNCTIONS):
        return _quadratic_witness_valid(triples, x, q), None
    view, binding = bound
    blocks = bytearray()
    if not _quadratic_witness_valid(view, x, q, _layout_blocks=blocks):
        return False, None
    return True, tuple.__new__(_QuadraticLayoutPlan,
                              (binding, len(x), q, bytes(blocks), functions))


def _quadratic_witness_valid(triples, x, q, *, _layout_blocks=None):
    binary_diagonal = (x.dtype == np.dtype(np.int64) and type(q) is int
                       and 1 < q <= np.iinfo(np.int64).max and _is_prime_64(q))
    start = 0
    flags = _native_constraint_blocks(triples, len(x))
    block_index = 0
    while start < len(triples):
        stop = min(start + 65536, len(triples), len(x) if start < len(x) else len(triples))
        diagonal = (flags[block_index] if flags is not None else
                    _diagonal_constraint_block(triples, start, stop, len(x)))
        block_index += 1
        if _layout_blocks is not None:
            _layout_blocks.append(diagonal)
        if diagonal:
            value = x[start:stop]
            if binary_diagonal:
                valid = int(value.view(np.uint64).max(initial=0)) <= 1
            else:
                valid = np.array_equal(value, _product(value, value, q))
            if not valid:
                return False
        else:
            out, left, right = triples[start:stop].T
            if not np.array_equal(x[out], _product(x[left], x[right], q)):
                return False
        start = stop
    return True


def _quadratic_masks(triples, r, x, q, *, _layout_plan=None):
    if 2 * (q - 1)**2 > np.iinfo(np.int64).max:
        out, left, right = triples.T
        a = _sub_canonical_mod(r[out], _product(r[left], x[right], q), q)
        a = _sub_canonical_mod(a, _product(r[right], x[left], q), q)
        return a, _product(r[left], r[right], q)
    blocks = _native_constraint_blocks(triples, len(r))
    if (type(_layout_plan) is _QuadraticLayoutPlan
            and _layout_plan[1:3] == (len(r), q) and len(x) == len(r)
            and _layout_plan[4] == (_diagonal_constraint_block, _quadratic_witness_valid,
                                   _quadratic_masks)):
        bound = _quadratic_layout_binding(triples)
        if bound is not None:
            view, binding = bound
            saved = _layout_plan[0]
            if (binding[0] is saved[0] and binding[1] is saved[1]
                    and binding[2:] == saved[2:]):
                triples, blocks = view, _layout_plan[3]
    a = np.empty(len(triples), dtype=np.int64)
    b = np.empty(len(triples), dtype=np.int64)
    start = 0
    block_index = 0
    while start < len(triples):
        stop = min(start + 65536, len(triples), len(r) if start < len(r) else len(triples))
        diagonal = (blocks[block_index] if blocks is not None
                    else _diagonal_constraint_block(triples, start, stop, len(r)))
        block_index += 1
        if diagonal:
            rvalue = r[start:stop]
            avalue = np.multiply(x[start:stop], -2, dtype=np.int64)
            avalue += 1
            avalue *= rvalue
            np.remainder(avalue, q, out=avalue)
            bvalue = rvalue * rvalue
            np.remainder(bvalue, q, out=bvalue)
            a[start:stop], b[start:stop] = avalue, bvalue
            start = stop
            continue
        out, left, right = triples[start:stop].T
        rleft, rright = r[left], r[right]
        avalue = r[out]
        avalue -= rleft * x[right]
        avalue -= rright * x[left]
        np.remainder(avalue, q, out=avalue)
        bvalue = rleft * rright
        np.remainder(bvalue, q, out=bvalue)
        a[start:stop], b[start:stop] = avalue, bvalue
        start = stop
    return a, b


_QUADRATIC_LAYOUT_FUNCTIONS = (_diagonal_constraint_block, _quadratic_witness_valid,
                             _quadratic_masks)


def _quadratic_mask_part(triples, r, q, x=None):
    result = np.empty(len(triples), dtype=np.int64)
    flags = _native_constraint_blocks(triples, len(r))
    start = block_index = 0
    while start < len(triples):
        stop = min(start + 65536, len(triples), len(r) if start < len(r) else len(triples))
        diagonal = (flags[block_index] if flags is not None
                    else _diagonal_constraint_block(triples, start, stop, len(r)))
        block_index += 1
        if diagonal and x is None:
            value = _product(r[start:stop], r[start:stop], q)
        elif x is not None and 2 * (q - 1)**2 <= np.iinfo(np.int64).max:
            if diagonal:
                value = np.multiply(x[start:stop], -2, dtype=np.int64)
                value += 1
                value *= r[start:stop]
            else:
                out, left, right = triples[start:stop].T
                value = r[out]
                value -= r[left] * x[right]
                value -= r[right] * x[left]
            np.remainder(value, q, out=value)
        elif diagonal:
            value = _sub_canonical_mod(r[start:stop],
                         _product(r[start:stop], _scale(x[start:stop], 2, q), q), q)
        else:
            out, left, right = triples[start:stop].T
            value = (_product(r[left], r[right], q) if x is None else
                     _sub_canonical_mod(_sub_canonical_mod(r[out], _product(r[left], x[right], q), q),
                                        _product(r[right], x[left], q), q))
        result[start:stop] = value
        start = stop
    return result


def _commitment_add_message(base, message, l1, q):
    tail = base[l1:]
    for start in range(0, len(message), 65536):
        stop = min(start + 65536, len(message))
        tail[start:stop] = _add_canonical_mod(tail[start:stop], message[start:stop], q)
    return base


def _quadratic_response(triples, z, alpha, q):
    centered = int(alpha) % q
    if centered > q // 2:
        centered -= q
    edge = centered * (q - 1)
    if (min(0, edge) - (q - 1)**2 < np.iinfo(np.int64).min
            or max(0, edge) > np.iinfo(np.int64).max):
        out, left, right = triples.T
        return _affine_canonical_mod(z[out], alpha, _product(z[left], z[right], q), q, subtract=True)
    result = np.empty(len(triples), dtype=np.int64)
    flags = _native_constraint_blocks(triples, len(z))
    start = 0
    block_index = 0
    while start < len(triples):
        stop = min(start + 65536, len(triples), len(z) if start < len(z) else len(triples))
        diagonal = (flags[block_index] if flags is not None else
                    _diagonal_constraint_block(triples, start, stop, len(z)))
        block_index += 1
        if diagonal:
            zvalue = z[start:stop]
            value = centered - zvalue
            value *= zvalue
            np.remainder(value, q, out=value)
            result[start:stop] = value
            start = stop
            continue
        out, left, right = triples[start:stop].T
        value = z[out] * centered
        value -= z[left] * z[right]
        np.remainder(value, q, out=value)
        result[start:stop] = value
        start = stop
    return result


@dataclass(frozen=True)
class YangCommitment:
    value: np.ndarray
    context: bytes


@dataclass(frozen=True)
class YangChallenge:
    alpha: int
    commitment: YangCommitment


@dataclass(frozen=True)
class YangResponse:
    t: np.ndarray
    c1: np.ndarray
    c2: np.ndarray
    c3: np.ndarray
    c4: np.ndarray
    rho: np.ndarray
    z0: np.ndarray
    z1: np.ndarray
    z2: np.ndarray


@dataclass(frozen=True)
class YangTranscript:
    commitment: YangCommitment
    challenge: YangChallenge
    response: YangResponse


def _immutable_vector(value):
    array = np.asarray(value, dtype=np.int64)
    if array.ndim == 1 and array.flags.c_contiguous and not array.flags.writeable:
        owner = array
        while isinstance(owner, (np.ndarray, memoryview)):
            if isinstance(owner, np.ndarray):
                owner = owner.base
            else:
                if not owner.readonly:
                    break
                owner = owner.obj
        if isinstance(owner, bytes):
            return np.frombuffer(array, dtype=np.int64)
    return np.frombuffer(array.tobytes(order='C'), dtype=np.int64)


def _copy_challenge(challenge):
    return YangChallenge(challenge.alpha, YangCommitment(
        _immutable_vector(challenge.commitment.value), challenge.commitment.context))


def _prover_first_phase(B1, triples, x, r, s1, s2, rng, quadratic_randomness_size,
                        sigma1, sigma2, gaussian_tail, *, executor=None, _layout_plan=None):
    first = (executor.submit(B1.commit_pair, (x, s1), (r, s2))
             if executor is not None else None)
    s3 = sample_standard_discrete_gaussian(rng, quadratic_randomness_size, sigma1, gaussian_tail)
    s4 = sample_standard_discrete_gaussian(rng, quadratic_randomness_size, sigma2, gaussian_tail)
    a, b = (_quadratic_masks(triples, r, x, B1.q, _layout_plan=_layout_plan)
            if type(_layout_plan) is _QuadraticLayoutPlan
            and _quadratic_masks is _layout_plan[4][2]
            else _quadratic_masks(triples, r, x, B1.q))
    c1, c2 = first.result() if first is not None else B1.commit_pair((x, s1), (r, s2))
    return c1, c2, s3, s4, a, b


def _prover_second_phase(B2, M, r, a, b, s3, s4, *, executor=None):
    if executor is None:
        c3, c4 = B2.commit_pair((a, s3), (b, s4))
        return M @ r, c3, c4
    second = executor.submit(B2.commit_pair, (a, s3), (b, s4))
    t = M @ r
    c3, c4 = second.result()
    return t, c3, c4


def _prover_second_payload_phase(B2, M, r, a, b, s3, s4, c1, c2, context, *, executor):
    from .commitments import _auxiliary_digest_state, _extend_auxiliary_digest
    pending = executor.submit(B2.commit_pair, (a, s3), (b, s4))
    try:
        t = M @ r
        digest = _auxiliary_digest_state(context, B2.q, 5)
        _extend_auxiliary_digest(digest, (t, c1, c2), B2.q)
        c3, c4 = pending.result()
        _extend_auxiliary_digest(digest, (c3, c4), B2.q)
        return (t, c1, c2, c3, c4), digest.digest()
    finally:
        from concurrent.futures import wait
        wait((pending,))


def _response_opening(s, mask, alpha, *, subtract=False):
    z = np.multiply(s, alpha)
    vf = z.astype(np.float64)
    if subtract:
        np.subtract(z, mask, out=z)
    else:
        np.add(z, mask, out=z)
    zf = z.astype(np.float64)
    return z, float(np.dot(vf, vf) - 2 * np.dot(zf, vf))


def _verifier_first_phase(B1, triples, z0, z1, alpha, *, executor=None):
    if executor is None:
        return B1.commit(z0, z1), _quadratic_response(triples, z0, alpha, B1.q)
    first = executor.submit(B1.commit, z0, z1)
    d = _quadratic_response(triples, z0, alpha, B1.q)
    return first.result(), d


def _verifier_equation_phase(B1, M, triples, z0, z1, alpha, y, t, c1, c2, *, executor=None):
    first = executor.submit(B1.commit, z0, z1) if executor is not None else None
    try:
        if not _affine_canonical_equal(M @ z0, y, alpha, t, B1.q):
            return None
        if first is None:
            if not _affine_canonical_equal(B1.commit(z0, z1), c1, alpha, c2, B1.q):
                return None
            return _quadratic_response(triples, z0, alpha, B1.q)
        d = _quadratic_response(triples, z0, alpha, B1.q)
        if not _affine_canonical_equal(first.result(), c1, alpha, c2, B1.q):
            return None
        return d
    finally:
        if first is not None:
            from concurrent.futures import wait
            wait((first,))


_VERIFIER_COMMIT_METHOD = LatticeCommitmentKey.commit


@lru_cache(maxsize=1)
def _verifier_native_block_codes():
    from types import CodeType
    from .relations import public_statement
    pending, codes = [public_statement.__code__], set()
    while pending:
        code = pending.pop()
        codes.add(code)
        pending.extend(value for value in code.co_consts if isinstance(value, CodeType))
    return frozenset(codes), public_statement.__globals__


def _verifier_parallel_relation(relation, B1, *vectors):
    from types import FunctionType
    from . import metal_matrix
    from .relations import (CompiledRelation, ExactLinearOperator, LinearBlock,
                            _immutable_snapshot_binding)
    if (type(relation) is not CompiledRelation or type(relation.M) is not ExactLinearOperator
            or type(B1) is not LatticeCommitmentKey or B1.width < 131072
            or getattr(B1.commit, '__func__', None) is not _VERIFIER_COMMIT_METHOD
            or not metal_matrix.is_prepared(B1.top) or not metal_matrix.is_prepared(B1.bottom)):
        return False
    relation._check_context_binding()
    if any(_immutable_snapshot_binding(vector) is None for vector in vectors):
        return False
    codes, namespace = _verifier_native_block_codes()
    return bool(relation.M.blocks) and all(
        type(block) is LinearBlock and type(block.apply) is FunctionType
        and block.apply.__code__ in codes and block.apply.__globals__ is namespace
        for block in relation.M.blocks)


def _prover_context_relation(relation, pp, auxiliary):
    from .relations import CompiledRelation, ExactLinearOperator
    if type(relation) is CompiledRelation:
        relation._check_context_binding()
    if (type(relation) is not CompiledRelation or type(relation.M) is not ExactLinearOperator
            or type(pp) is not YangPublicParameters or type(pp.digest) is not bytes
            or type(auxiliary) is not bytes or type(relation.statement_digest) is not str):
        return None
    for array in (relation.y, relation.triples):
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
    if relation.triples.ndim != 2 or len(relation.triples) < 65536:
        return None
    return relation


def _prover_witness_context_phase(public_relation, x, pp, auxiliary, rng, *, executor=None,
                                  _collect_layout=False):
    def context_digest():
        return _context(public_relation, pp, auxiliary)
    pending = (executor.submit(context_digest)
               if executor is not None else None)
    if not np.array_equal(public_relation.M @ x, public_relation.y):
        raise ValueError('Witness relation failed')
    valid, layout_plan = (_quadratic_witness_layout(public_relation.triples, x, pp.witness_key.q)
                          if _collect_layout else
                          (_quadratic_witness_valid(public_relation.triples, x, pp.witness_key.q), None))
    if not valid:
        raise ValueError('Witness relation failed')
    source = np.random.default_rng() if rng is None else rng
    context = pending.result() if pending is not None else context_digest()
    return (context, source, layout_plan) if _collect_layout else (context, source)


class YangPrecomputation:
    __slots__ = ('_lock', '_state')

    def __init__(self, payload):
        from threading import Lock
        self._lock, self._state = Lock(), [payload]

    @classmethod
    def prepare(cls, public_relation, pp, *, rng=None, auxiliary=b''):
        from concurrent.futures import ThreadPoolExecutor
        from . import metal_matrix
        from .fast_uniform import uniform_source, uniform_field
        q, rows, width, count = _check_parameters(public_relation, pp)
        plan = resource_plan(q, rows, width, count, pp.settings)
        require_memory(plan['conservative_working_bytes'], pp.settings.max_memory_bytes,
                       'Yang offline workspace')
        context = _context(public_relation, pp, auxiliary)
        source = np.random.default_rng() if rng is None else rng
        B1, B2, settings = pp.witness_key, pp.quadratic_key, pp.settings
        with uniform_source(source) as stream:
            r = uniform_field(stream, (width,), q)
        s1 = sample_standard_discrete_gaussian(source, B1.randomness_size, settings.sigma1, settings.gaussian_tail)
        s2 = sample_standard_discrete_gaussian(source, B1.randomness_size, pp.sigma2, settings.gaussian_tail)
        zero = np.broadcast_to(np.zeros(1, dtype=np.int64), (width,))
        parallel = metal_matrix.is_prepared(B1.top) and width >= 65536
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(B1.commit_pair, (zero, s1), (r, s2)) if parallel else None
            s3 = sample_standard_discrete_gaussian(source, B2.randomness_size, settings.sigma1, settings.gaussian_tail)
            s4 = sample_standard_discrete_gaussian(source, B2.randomness_size, pp.sigma2, settings.gaussian_tail)
            b = _quadratic_mask_part(public_relation.triples, r, q)
            c1, c2 = first.result() if first is not None else B1.commit_pair((zero, s1), (r, s2))
            zero = np.broadcast_to(np.zeros(1, dtype=np.int64), (count,))
            second = pool.submit(B2.commit_pair, (zero, s3), (b, s4)) if parallel else None
            t = public_relation.M @ r
            c3, c4 = second.result() if second is not None else B2.commit_pair((zero, s3), (b, s4))
        rho = source.integers(0, 2, size=pp.auxiliary_key.randomness_bits, dtype=np.int64)
        return cls((public_relation, pp, context, r, s1, s2, s3, s4, t, c1, c2, c3, c4, rho))

    def discard(self):
        with self._lock:
            self._state[0] = None

    def _take(self, public_relation, pp, auxiliary):
        with self._lock:
            payload, self._state[0] = self._state[0], None
        if payload is None:
            raise RuntimeError('Offline randomness already spent')
        original, saved_pp, context, *vectors = payload
        from .relations import CompiledRelation
        if type(original) is CompiledRelation:
            original._check_context_binding()
        if saved_pp is not pp or context != _context(public_relation, pp, auxiliary):
            raise ValueError('Offline statement/parameter mismatch')
        return context, vectors


class YangProverSession:
    def _initialize_witness(self, public_relation, witness, pp: YangPublicParameters, *, rng=None,
                 auxiliary: bytes = b""):
        bound_relation = _prover_context_relation(public_relation, pp, auxiliary)
        if bound_relation is not None:
            public_relation = bound_relation
        q, _, width, _ = _check_parameters(public_relation, pp)
        x = _snapshot_integer_vector(witness, width, "witness", q)
        layout_plan = None
        collect_layout = (type(self) is YangProverSession
                          and _prover_witness_context_phase is _PROVER_LAYOUT_PHASES[0]
                          and _prover_first_phase is _PROVER_LAYOUT_PHASES[1]
                          and getattr(self._commitment_phase, '__func__', None) is _YANG_COMMITMENT_PHASE)
        if bound_relation is not None:
            from concurrent.futures import ThreadPoolExecutor
            if public_relation._context_prepared:
                if collect_layout:
                    context, source, layout_plan = _prover_witness_context_phase(
                        public_relation, x, pp, auxiliary, rng, _collect_layout=True)
                else:
                    context, source = _prover_witness_context_phase(
                        public_relation, x, pp, auxiliary, rng)
            else:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    if collect_layout:
                        context, source, layout_plan = _prover_witness_context_phase(
                            public_relation, x, pp, auxiliary, rng, executor=pool, _collect_layout=True)
                    else:
                        context, source = _prover_witness_context_phase(public_relation, x, pp, auxiliary,
                                                                      rng, executor=pool)
        else:
            if (not np.array_equal(public_relation.M @ x, public_relation.y)
                    or not _quadratic_witness_valid(public_relation.triples, x, q)):
                raise ValueError('Witness relation failed')
            source = np.random.default_rng() if rng is None else rng
            context = _context(public_relation, pp, auxiliary)
        self.relation, self.pp, self.x = public_relation, pp, x
        self.rng, self.context = source, context
        self.spent = False
        return layout_plan

    def __init__(self, public_relation, witness, pp: YangPublicParameters, *, rng=None,
                 auxiliary: bytes = b''):
        layout_plan = self._initialize_witness(public_relation, witness, pp, rng=rng, auxiliary=auxiliary)
        B1, settings = pp.witness_key, pp.settings
        q, width = B1.q, B1.width
        from .fast_uniform import uniform_source, uniform_field
        with uniform_source(self.rng) as source:
            self.r = uniform_field(source, (width,), q)
        self.s1 = sample_standard_discrete_gaussian(self.rng, B1.randomness_size, settings.sigma1, settings.gaussian_tail)
        self.s2 = sample_standard_discrete_gaussian(self.rng, B1.randomness_size, pp.sigma2, settings.gaussian_tail)
        self._payload_digest = None
        self.vectors = (self._commitment_phase(_layout_plan=layout_plan)
                        if layout_plan is not None and
                        getattr(self._commitment_phase, '__func__', None) is _YANG_COMMITMENT_PHASE
                        else self._commitment_phase())
        self.rho = self.rng.integers(0, 2, size=pp.auxiliary_key.randomness_bits, dtype=np.int64)
        bits = (_auxiliary_bits(self.context, self.vectors, q) if self._payload_digest is None
                else np.unpackbits(np.frombuffer(self._payload_digest, dtype=np.uint8), bitorder='little'))
        value = pp.auxiliary_key.commit(bits, self.rho)
        self.commitment = YangCommitment(_immutable_vector(value), self.context)

    @classmethod
    def _from_precomputation(cls, offline, public_relation, witness, pp, *, rng=None, auxiliary=b''):
        context, vectors = offline._take(public_relation, pp, auxiliary)
        self = cls.__new__(cls)
        self._initialize_witness(public_relation, witness, pp, rng=rng, auxiliary=auxiliary)
        if self.context != context:
            raise ValueError('Offline context changed')
        (self.r, self.s1, self.s2, self.s3, self.s4, t, c1, c2, c3, c4, self.rho) = vectors
        B1, B2 = pp.witness_key, pp.quadratic_key
        a = _quadratic_mask_part(self.relation.triples, self.r, B1.q, self.x)
        c1 = _commitment_add_message(c1, self.x, B1.l1, B1.q)
        c3 = _commitment_add_message(c3, a, B2.l1, B2.q)
        self.vectors = t, c1, c2, c3, c4
        self._payload_digest = None
        value = pp.auxiliary_key.commit(_auxiliary_bits(self.context, self.vectors, B1.q), self.rho)
        self.commitment = YangCommitment(_immutable_vector(value), self.context)
        return self

    def _commitment_phase(self, *, _layout_plan=None):
        from concurrent.futures import ThreadPoolExecutor
        from . import metal_matrix
        B1, B2, settings = self.pp.witness_key, self.pp.quadratic_key, self.pp.settings
        args = (B1, self.relation.triples, self.x, self.r, self.s1, self.s2, self.rng,
                B2.randomness_size, settings.sigma1, self.pp.sigma2, settings.gaussian_tail)
        layout = ({'_layout_plan': _layout_plan}
                  if _layout_plan is not None and _prover_first_phase is _PROVER_LAYOUT_PHASES[1]
                  else {})
        if metal_matrix.is_prepared(B1.top) and B1.width >= 65536:
            with ThreadPoolExecutor(max_workers=1) as pool:
                c1, c2, self.s3, self.s4, a, b = _prover_first_phase(*args, executor=pool, **layout)
                vectors, self._payload_digest = _prover_second_payload_phase(
                    B2, self.relation.M, self.r, a, b, self.s3, self.s4,
                    c1, c2, self.context, executor=pool)
                return vectors
        else:
            c1, c2, self.s3, self.s4, a, b = _prover_first_phase(*args, **layout)
            t, c3, c4 = _prover_second_phase(B2, self.relation.M, self.r, a, b,
                                            self.s3, self.s4)
        return t, c1, c2, c3, c4

    def respond(self, challenge: YangChallenge) -> YangResponse | None:
        if self.spent:
            raise RuntimeError('Prover session already spent')
        self.spent = True
        alpha = challenge.alpha
        if (not isinstance(alpha, int) or isinstance(alpha, bool) or abs(alpha) > self.pp.settings.p
                or not _same_commitment(challenge.commitment, self.commitment)):
            raise ValueError('Challenge/commitment mismatch')
        z0 = _affine_canonical_mod(self.x, alpha, self.r, self.pp.witness_key.q)
        z1, first_exponent = _response_opening(self.s1, self.s2, alpha)
        z2, second_exponent = _response_opening(self.s3, self.s4, alpha, subtract=True)
        exponent = 0.0
        exponent += first_exponent
        exponent += second_exponent
        log_accept = min(0.0, exponent / (2 * self.pp.sigma2 ** 2) - math.log(self.pp.rejection_M))
        if math.log(max(float(self.rng.random()), np.finfo(float).tiny)) >= log_accept:
            return None
        return YangResponse(*self.vectors, self.rho.copy(), z0, z1, z2)


_YANG_COMMITMENT_PHASE = YangProverSession._commitment_phase
_PROVER_LAYOUT_PHASES = (_prover_witness_context_phase, _prover_first_phase)


def _same_commitment(first, second):
    return (isinstance(first, YangCommitment) and isinstance(second, YangCommitment)
            and first.context == second.context and np.array_equal(first.value, second.value))


def _check_parameters(relation, pp):
    q, rows, width, count = _public_dimensions(relation)
    pp.settings.validate(q)
    if (pp.witness_key.q != q or pp.quadratic_key.q != q or pp.auxiliary_key.q != q
            or pp.witness_key.width != width or pp.quadratic_key.width != count
            or pp.relation_rows != rows):
        raise ValueError('Yang/relation dimension mismatch')
    return q, rows, width, count


def _block_squared_norm(block, remaining, minimum=None, maximum=None):
    minimum = int(block.min(initial=0)) if minimum is None else minimum
    maximum = int(block.max(initial=0)) if maximum is None else maximum
    largest_square = max(abs(minimum), abs(maximum)) ** 2
    if largest_square > remaining:
        return None
    if not largest_square:
        return 0
    int64_max = np.iinfo(np.int64).max
    count = min(len(block), int64_max // largest_square)
    if count and block.dtype == np.dtype(np.int64):
        if count == len(block):
            squared = int(block @ block)
        else:
            stop = len(block) - len(block) % count
            parts = block[:stop].reshape(-1, count)
            products = np.einsum('ij,ij->i', parts, parts, optimize=False)
            tail = block[stop:]
            squared = sum(products.tolist()) + int(tail @ tail)
    else:
        squared = sum(int(value) ** 2 for value in block)
    return None if squared > remaining else squared


def _within_norm(vector, bound: float):
    limit = math.floor(bound ** 2)
    squared = 0
    for start in range(0, len(vector), 65_536):
        block = vector[start:start + 65_536]
        block_squared = _block_squared_norm(block, limit - squared)
        if block_squared is None:
            return False
        squared += block_squared
    return True


def _snapshot_integer_vector(value, size, name, q=None, *, bound=None, digest=None):
    source = np.asarray(value)
    if source.shape != (size,) or source.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be an integer vector of length {size}')
    vector = _immutable_vector(source)
    unsigned = source.dtype.kind == 'u'
    if digest is not None:
        if q is None:
            raise ValueError('auxiliary snapshot requires a modulus')
        digest.update(size.to_bytes(8, 'little'))
    if q is None and bound is None and not unsigned:
        return vector
    limit = None if bound is None else math.floor(bound ** 2)
    squared = 0
    chunk_entries = 65_536 if bound is not None else 262_144
    encoding_dtype = (np.dtype('<u4' if q <= 2**32 else '<u8')
                      if digest is not None else None)
    encoding_workspace = None
    canonical_only = q is not None and bound is None and 0 < q <= np.iinfo(np.int64).max
    for start in range(0, size, chunk_entries):
        piece = vector[start:start + chunk_entries]
        if canonical_only:
            maximum = int(piece.view(np.uint64).max(initial=0))
            if unsigned and maximum > np.iinfo(np.int64).max:
                raise ValueError(f'{name} exceeds signed int64')
            if maximum >= q:
                raise ValueError(f'{name} must use canonical representatives in [0,q)')
        else:
            minimum, maximum = int(piece.min(initial=0)), int(piece.max(initial=0))
            if unsigned and minimum < 0:
                raise ValueError(f'{name} exceeds signed int64')
            if q is not None and (minimum < 0 or maximum >= q):
                raise ValueError(f'{name} must use canonical representatives in [0,q)')
            if bound is not None:
                block_squared = _block_squared_norm(piece, limit - squared, minimum, maximum)
                if block_squared is None:
                    raise ValueError(f'{name} exceeds the response norm bound')
                squared += block_squared
        if digest is not None:
            encoding_workspace = _update_auxiliary_payload(digest, piece, encoding_dtype,
                                                            encoding_workspace)
    if q is not None:
        from .relations import _remember_canonical_vector
        _remember_canonical_vector(vector, q)
    return vector


class YangVerifierSession:
    def __init__(self, public_relation, pp: YangPublicParameters, *, rng=None,
                 auxiliary: bytes = b""):
        _check_parameters(public_relation, pp)
        self.relation, self.pp, self.rng = public_relation, pp, rng
        self.auxiliary = auxiliary
        self.context = _context(public_relation, pp, auxiliary)
        self.issued = False
        self.spent = False
        self._expected_challenge = None

    @property
    def challenge_message(self):
        return None if self._expected_challenge is None else _copy_challenge(self._expected_challenge)

    def challenge(self, commitment: YangCommitment) -> YangChallenge:
        if self.issued:
            raise RuntimeError("verifier already issued a challenge")
        if not isinstance(commitment, YangCommitment) or commitment.context != self.context:
            raise ValueError('Commitment context mismatch')
        value = _snapshot_integer_vector(commitment.value, self.pp.auxiliary_key.rows,
                                         "auxiliary commitment", self.pp.witness_key.q)
        saved = YangCommitment(value, commitment.context)
        p = self.pp.settings.p

        alpha = secrets.randbelow(2 * p + 1) - p if self.rng is None else int(self.rng.integers(-p, p + 1))
        self._expected_challenge = YangChallenge(alpha, saved)
        self.issued = True
        return self.challenge_message

    def verify(self, response: YangResponse | None) -> bool:
        if not self.issued or self.spent:
            return False
        self.spent = True
        return verify_response(self.relation, self.pp, response,
                               expected_challenge=self._expected_challenge, auxiliary_context=self.context,
                               auxiliary=self.auxiliary)


def _verified_response_payload(pp, response, expected_challenge, context, rho):
    from .commitments import _auxiliary_digest_state
    B1, B2 = pp.witness_key, pp.quadratic_key
    digest = _auxiliary_digest_state(context, B1.q, 5)
    t = _snapshot_integer_vector(response.t, pp.relation_rows, 't', B1.q, digest=digest)
    c1, c2 = [_snapshot_integer_vector(c, B1.commitment_size, 'witness commitment', B1.q, digest=digest)
              for c in (response.c1, response.c2)]
    c3, c4 = [_snapshot_integer_vector(c, B2.commitment_size, 'quadratic commitment', B1.q, digest=digest)
              for c in (response.c3, response.c4)]
    bits = np.unpackbits(np.frombuffer(digest.digest(), dtype=np.uint8), bitorder='little')
    if not np.array_equal(pp.auxiliary_key.commit(bits, rho), expected_challenge.commitment.value):
        return None
    return t, c1, c2, c3, c4


def _verifier_payload_equations(public, pp, response, challenge, context, rho, z0, z1, z2, *, executor):
    B1, B2 = pp.witness_key, pp.quadratic_key
    first = executor.submit(B1.commit, z0, z1)
    quadratic = executor.submit(_quadratic_response, public.triples, z0, challenge.alpha, B1.q)
    try:
        payload = _verified_response_payload(pp, response, challenge, context, rho)
        if payload is None:
            return False
        t, c1, c2, c3, c4 = payload
        alpha, q = challenge.alpha, B1.q
        if not _affine_canonical_equal(public.M @ z0, public.y, alpha, t, q):
            return False
        d = quadratic.result()
        if not _affine_canonical_equal(first.result(), c1, alpha, c2, q):
            return False
        return _affine_canonical_equal(B2.commit(d, z2), c3, alpha, c4, q, subtract=True)
    finally:
        from concurrent.futures import wait
        wait((first, quadratic))


def verify_response(public_relation, pp: YangPublicParameters, response: YangResponse | None, *,
                    expected_challenge: YangChallenge, auxiliary_context: bytes,
                    auxiliary: bytes = b"") -> bool:
    try:
        _check_parameters(public_relation, pp)
        if not isinstance(response, YangResponse) or not isinstance(expected_challenge, YangChallenge):
            return False
        if auxiliary_context != _context(public_relation, pp, auxiliary):
            return False
        q, B1, B2 = pp.witness_key.q, pp.witness_key, pp.quadratic_key
        alpha = expected_challenge.alpha
        if not isinstance(alpha, int) or isinstance(alpha, bool) or abs(alpha) > pp.settings.p:
            return False
        if expected_challenge.commitment.context != auxiliary_context:
            return False

        z0 = _snapshot_integer_vector(response.z0, B1.width, "masked witness", q)
        scale = pp.sigma2 + pp.settings.p * pp.settings.sigma1
        z1 = _snapshot_integer_vector(response.z1, B1.randomness_size, "witness opening response",
                                      bound=2 * math.sqrt(B1.randomness_size) * scale)
        z2 = _snapshot_integer_vector(response.z2, B2.randomness_size, "quadratic opening response",
                                      bound=2 * math.sqrt(B2.randomness_size) * scale)
        rho = _snapshot_integer_vector(response.rho, pp.auxiliary_key.randomness_bits, "auxiliary randomness")
        from . import metal_matrix
        if (type(pp) is YangPublicParameters
                and _verifier_parallel_relation(public_relation, B1, z0, z1)):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                return _verifier_payload_equations(public_relation, pp, response, expected_challenge,
                                                    auxiliary_context, rho, z0, z1, z2, executor=pool)
        else:
            payload = _verified_response_payload(pp, response, expected_challenge, auxiliary_context, rho)
            if payload is None:
                return False
            t, c1, c2, c3, c4 = payload
            if not _affine_canonical_equal(public_relation.M @ z0, public_relation.y, alpha, t, q):
                return False
            if metal_matrix.is_prepared(B1.top) and B1.width >= 131072:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=1) as pool:
                    first_value, d = _verifier_first_phase(B1, public_relation.triples,
                                                           z0, z1, alpha, executor=pool)
            else:
                first_value = B1.commit(z0, z1)
                d = None
            if not _affine_canonical_equal(first_value, c1, alpha, c2, q):
                return False
            if d is None:
                d = _quadratic_response(public_relation.triples, z0, alpha, q)
        return _affine_canonical_equal(B2.commit(d, z2), c3, alpha, c4, q, subtract=True)
    except (ValueError, TypeError, IndexError, AttributeError, OverflowError):
        return False
