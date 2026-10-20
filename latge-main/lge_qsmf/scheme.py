from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Tuple
import os

from .structured_crs import StructuredCRS
from .decomposition import rdec, rvdec
from .hash_to_ring import H1_mat, H2_vec
from .linalg import (as_column, concat_cols, first_column, mat_add, mat_vec_mul,
                     mat_transpose_vec_mul, matrix_equal, random_matrix, random_vec,
                     small_poly, small_vec, ternary_vec, vec_add, vec_equal,
                     vec_inf_norm)
from .ots import OTSParams, ots_gen, ots_sign, ots_verify
from .params import Params, table_iii_experiment_params
from .randomness import uniform_mod_coeffs
from .ring import Poly, Ring
from . import trapdoor as td


class MissingPaperParameterError(RuntimeError):
    pass


class PaperParameterInconsistencyError(RuntimeError):
    pass


@dataclass
class FullEnrollmentPublicParams:
    A_tags: List[List[List[Poly]]]
    B0: List[List[Poly]]
    B1: List[List[Poly]]
    B2: List[List[Poly]]
    u: List[Poly]


@dataclass
class FullPublicParams:
    params: Params
    ring_q: Ring
    s_full: int
    q1: int
    q2: int
    A_tags: List[List[List[Poly]]]
    B0: List[List[Poly]]
    B1: List[List[Poly]]
    B2: List[List[Poly]]
    u: List[Poly]
    parcom: StructuredCRS
    ots_hash_seed: bytes
    trapdoor_distribution: td.TrapdoorDistribution | None = None

    @property
    def pp_prime(self) -> FullEnrollmentPublicParams:
        return FullEnrollmentPublicParams(self.A_tags, self.B0, self.B1, self.B2, self.u)


@dataclass
class FullPublicKey:
    A: List[List[Poly]]


@dataclass
class FullGMPublicKey(FullPublicKey):
    pp_prime: FullEnrollmentPublicParams


@dataclass
class FullSecretKey:

    trapdoor: td.TrapdoorKey
    public_context: FullPublicParams | None = field(default=None, repr=False, compare=False)
    opening_context: FullGroupPublicKey | None = field(default=None, repr=False, compare=False)
    issuance_context: FullIssueContext | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class FullUserPublicKeyProof:
    pi_pki: object


@dataclass
class FullGMKeypair:
    pk: FullGMPublicKey
    sk: FullSecretKey


@dataclass
class FullOAKeypair:
    pk: FullPublicKey
    sk: FullSecretKey


@dataclass
class FullCertificate:
    tag: Tuple[int, ...]
    r_i: List[Poly]
    v_i: List[Poly]


@dataclass(frozen=True)
class FullJoinRequest:
    pk: FullPublicKey
    proof: FullUserPublicKeyProof


@dataclass(frozen=True)
class FullJoinAcceptance:

    pk: FullPublicKey
    cert: FullCertificate


@dataclass
class FullRegistrationEntry:
    pk: FullPublicKey
    cert: FullCertificate


def _public_key_equal(A, B) -> bool:

    try:
        a, b = A[0][0], B[0][0]
        if isinstance(a, Poly) and isinstance(b, Poly):
            if a.ring != b.ring:
                return False
            if int(a.coeffs[0]) % a.ring.q != int(b.coeffs[0]) % b.ring.q:
                return False
    except (TypeError, IndexError, AttributeError):
        pass
    return matrix_equal(A, B)


@dataclass
class FullRegistrationTable:
    entries: List[FullRegistrationEntry] = field(default_factory=list)

    def _contains_key(self, pk: FullPublicKey) -> bool:
        return any(_public_key_equal(entry.pk.A, pk.A) for entry in self.entries)

    def add(self, pk: FullPublicKey, cert: FullCertificate) -> None:
        if self._contains_key(pk):
            raise ValueError('user public key already registered')
        stored_pk = FullPublicKey([[a.copy() for a in row] for row in pk.A])
        stored_cert = FullCertificate(tuple(cert.tag), [a.copy() for a in cert.r_i],
                                      [a.copy() for a in cert.v_i])
        self.entries.append(FullRegistrationEntry(stored_pk, stored_cert))


@dataclass
class FullGroupPublicKey:
    pp: FullPublicParams
    pk_gm: FullGMPublicKey
    pk_oa: FullPublicKey

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:

        if (not isinstance(self.pk_gm, FullGMPublicKey)
                or not isinstance(self.pk_gm.pp_prime, FullEnrollmentPublicParams)
                or self.pk_gm.pp_prime != self.pp.pp_prime):
            raise ValueError("GM public key must include the same pp' as public parameters")


@dataclass(frozen=True)
class FullIssueContext:
    gpk: FullGroupPublicKey
    reg: FullRegistrationTable
    request: FullJoinRequest


def bind_key_context(pp: FullPublicParams, secret: FullSecretKey) -> FullSecretKey:

    if not isinstance(pp, FullPublicParams) or not isinstance(secret, FullSecretKey):
        raise ValueError('a working key and explicit public parameters are required')
    if secret.public_context is not None and secret.public_context != pp:
        raise ValueError('working key is bound to different public parameters')
    return FullSecretKey(secret.trapdoor, pp)


def bind_open_context(gpk: FullGroupPublicKey, secret: FullSecretKey) -> FullSecretKey:
    gpk.validate()
    return replace(bind_key_context(gpk.pp, secret), opening_context=gpk)


def bind_issue_context(gpk: FullGroupPublicKey, reg: FullRegistrationTable,
                       secret: FullSecretKey, request: FullJoinRequest) -> FullSecretKey:

    gpk.validate()
    p = gpk.pp.params
    if (not isinstance(reg, FullRegistrationTable) or not isinstance(request, FullJoinRequest)
            or not isinstance(request.pk, FullPublicKey)
            or not isinstance(request.proof, FullUserPublicKeyProof)
            or not _matrix_shape(request.pk.A, p.alpha1, 7*p.alpha1, gpk.pp.ring_q)):
        raise ValueError('issuer binding requires a registry and a valid JOIN message shape')
    received = FullJoinRequest(FullPublicKey([[a.copy() for a in row] for row in request.pk.A]),
                               request.proof)
    return replace(bind_key_context(gpk.pp, secret),
                   issuance_context=FullIssueContext(gpk, reg, received))


@dataclass
class FullCiphertext:
    vk_ots: bytes
    c_rec1: List[Poly]
    c_rec2: List[Poly]
    c_oa1: List[Poly]
    c_oa2: Poly
    sig: bytes
    label: bytes = b''
    relation: RISISStatement | None = None


@dataclass(frozen=True)
class RISISStatement:
    AR: Tuple[Poly, ...]
    uR: Poly


@dataclass(frozen=True)
class EncryptionCoins:
    x_rec: Tuple[Poly, ...]
    y_rec: Tuple[Poly, ...]
    z_rec: Tuple[Poly, ...]
    x_oa: Poly
    y_oa: Tuple[Poly, ...]
    z_oa: Tuple[Poly, ...]


def _root_floor(q: int, degree: int) -> int:
    lo, hi = 0, 1 << ((q.bit_length() + degree - 1) // degree)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if mid ** degree <= q:
            lo = mid
        else:
            hi = mid
    return lo if hi ** degree > q else hi


def _root_ceil(q: int, degree: int) -> int:
    v = _root_floor(q, degree)
    return v if v ** degree == q else v + 1


def _root_nearest(q: int, degree: int) -> int:
    v = _root_floor(q, degree)
    return v + int((2 * v + 1) ** degree <= (2 ** degree) * q)


def setup_init(security_parameter: int = 80) -> FullPublicParams:

    if type(security_parameter) is not int or security_parameter <= 0:
        raise ValueError('security_parameter must be a positive integer')
    if security_parameter != 80:
        raise NotImplementedError('Main provides only the Table III 80-bit target profile')
    return setup_init_from_params(table_iii_experiment_params())


def setup_init_from_params(p: Params, *,
                          trapdoor_distribution: td.TrapdoorDistribution | None = None
                          ) -> FullPublicParams:
    if not isinstance(p, Params):
        raise TypeError('an explicit Params profile is required')
    if trapdoor_distribution is not None:
        if not isinstance(trapdoor_distribution, td.TrapdoorDistribution):
            raise TypeError('trapdoor_distribution must be an explicit D sampler')
        if p.trapdoor_sigma is not None:
            raise ValueError('choose D or the Gaussian trapdoor_sigma profile, not both')
    ring = Ring(p.n, p.q)
    return FullPublicParams(
        params=p, ring_q=ring, s_full=p.s,
        q1=_root_nearest(p.q, 3), q2=_root_ceil(p.q, 5),
        A_tags=[random_matrix(ring, p.alpha, 3 * p.alpha) for _ in range(p.d + 1)],
        B0=random_matrix(ring, p.alpha, p.k * p.alpha),
        B1=random_matrix(ring, p.alpha, p.s),
        B2=random_matrix(ring, p.alpha, p.s * p.alpha),
        u=random_vec(ring, p.alpha),
        parcom=StructuredCRS.from_params(p),
        ots_hash_seed=os.urandom(32),
        trapdoor_distribution=trapdoor_distribution,
    )


def _required_sigma(pp: FullPublicParams, name: str) -> float:
    sigma = getattr(pp.params, name, None)
    if sigma is None or sigma <= 0:
        raise MissingPaperParameterError(f'{name} is required by the selected Gaussian reference profile; its source and sampling conditions must be specified')
    return float(sigma)


def _required_trapdoor_bound(pp: FullPublicParams) -> int:
    bound = pp.params.trapdoor_bound
    if bound is None:
        raise MissingPaperParameterError('trapdoor_bound must be specified independently of encryption-noise B')
    return bound


def setup_gm(pp: FullPublicParams) -> FullGMKeypair:
    if (pp.params.tag_boundaries_source != 'paper_formula'
            and not pp.params.zk_research_reference):
        raise MissingPaperParameterError(
            'SETUPGM requires explicit c and alpha0 for Main II-C tag sets; '
            'boundary-only parameters are a research configuration')
    t = _generate_group_trapdoor(pp, pp.params.alpha, 3, pp.q1)
    return FullGMKeypair(FullGMPublicKey(t.A, pp.pp_prime), FullSecretKey(t, pp))


def setup_oa(pp: FullPublicParams) -> FullOAKeypair:
    t = _generate_group_trapdoor(pp, pp.params.alpha1, 5, pp.q2)
    return FullOAKeypair(FullPublicKey(t.A), FullSecretKey(t, pp))


def _generate_group_trapdoor(pp: FullPublicParams, rows: int, digits: int,
                            base: int) -> td.TrapdoorKey:
    distribution = pp.trapdoor_distribution
    configuration = ({'trapdoor_distribution': distribution} if distribution is not None
                     else {'trapdoor_sigma': _required_sigma(pp, 'trapdoor_sigma')})
    bound = _required_trapdoor_bound(pp)
    if not pp.params.zk_research_reference:
        from .trapdoor_security import gentrap_support_bound
        effective_bound = bound
        if distribution is None:
            effective_bound = min(bound, td.gaussian_trapdoor_support_bound(
                pp.ring_q, rows, 2*rows, digits, configuration['trapdoor_sigma']))
        obstruction = gentrap_support_bound(pp.ring_q.q, pp.ring_q.n, rows,
                                            2*rows, rows*digits, effective_bound)
        if obstruction.excludes_statistical_distance(pp.params.security_bits):
            raise PaperParameterInconsistencyError(
                'GenTrap support contradicts Main\'s statistical-uniformity premise: '
                f'with {2*rows} prefix columns and R coefficients in [-{effective_bound},{effective_bound}], '
                f'the joint public A cannot be within 2^-{pp.params.security_bits} '
                'of uniform. This is a necessary-condition failure, not a missing '
                'parameter or evidence about computational indistinguishability. '
                'A larger declared bound alone does not establish security.')
    A_left = random_matrix(pp.ring_q, rows, 2 * rows)
    G = td.bind_gadget_context(td.gadget_matrix(pp.ring_q, rows, digits, base),
                              trapdoor_bound=bound, **configuration)
    return td._gen_trap_working_key(A_left, G)


def ukgen(pp: FullPublicParams) -> Tuple[FullPublicKey, FullSecretKey]:
    pair = setup_oa(pp)
    return pair.pk, pair.sk


def prove_user_public_key(pp: FullPublicParams, sk_i: FullSecretKey) -> FullUserPublicKeyProof:
    from .trapdoor_proof import prove as prove_join
    return FullUserPublicKeyProof(prove_join(pp, sk_i.trapdoor))


def join(gpk: FullGroupPublicKey, pk_i: FullPublicKey,
         sk_i: FullSecretKey) -> FullJoinRequest:
    gpk.validate()
    if (not isinstance(pk_i, FullPublicKey) or not isinstance(sk_i, FullSecretKey)
            or not isinstance(sk_i.trapdoor, td.TrapdoorKey)
            or not _matrix_shape(pk_i.A, gpk.pp.params.alpha1,
                                 7 * gpk.pp.params.alpha1, gpk.pp.ring_q)
            or not matrix_equal(pk_i.A, sk_i.trapdoor.A)):
        raise ValueError('JOIN public and secret keys do not match')
    return FullJoinRequest(pk_i, prove_user_public_key(gpk.pp, sk_i))


def verify_user_public_key(pp: FullPublicParams, pk_i: FullPublicKey, proof: FullUserPublicKeyProof) -> bool:
    from .trapdoor_proof import verify as verify_join
    if not isinstance(proof, FullUserPublicKeyProof):
        return False
    return verify_join(pp, pk_i.A, proof.pi_pki)


def _tag_boundaries(pp: FullPublicParams) -> tuple[int, ...]:
    boundaries = getattr(pp.params, 'tag_boundaries', None)
    if boundaries is None:
        raise MissingPaperParameterError('select explicit tag_boundaries (c_0,...,c_d) satisfying Main II-C and the chosen tag-domain conditions')
    return tuple(boundaries)


def _tag_polynomials(pp: FullPublicParams, tag: Tuple[int, ...]) -> List[Poly]:
    boundaries = _tag_boundaries(pp)
    if len(tag) != boundaries[-1] or any(type(bit) is not int or bit not in (0, 1) for bit in tag):
        raise ValueError('invalid DM polynomial tag')
    result = []
    for lo, hi in zip(boundaries, boundaries[1:]):
        result.append(Poly.from_ints(pp.ring_q, [tag[j] if lo <= j < hi else 0 for j in range(pp.params.n)]))
    return result


def _tag_matrix(pp: FullPublicParams, tag: Tuple[int, ...]) -> List[List[Poly]]:
    result = [[a.copy() for a in row] for row in pp.A_tags[0]]
    for part, A in zip(_tag_polynomials(pp, tag), pp.A_tags[1:]):
        result = mat_add(result, [[part * a for a in row] for row in A])
    return result


def _certificate_target(pp: FullPublicParams, h_i: List[Poly], r_i: List[Poly]):
    w = rvdec(vec_add(mat_vec_mul(pp.B0, r_i), mat_vec_mul(pp.B1, h_i)), (pp.params.q - 1) // 2)
    return w, vec_add(mat_vec_mul(pp.B2, w), pp.u)


def issue(gm_sk: FullSecretKey, pk_i: FullPublicKey) -> FullCertificate:

    context = getattr(gm_sk, 'issuance_context', None)
    if not isinstance(context, FullIssueContext):
        raise ValueError('ISSUE requires an explicitly bound JOIN interaction')
    gpk, reg, pk_proof = context.gpk, context.reg, context.request.proof
    gpk.validate()
    if (not isinstance(pk_i, FullPublicKey)
            or not _matrix_shape(pk_i.A, gpk.pp.params.alpha1,
                                 7*gpk.pp.params.alpha1, gpk.pp.ring_q)
            or not matrix_equal(pk_i.A, context.request.pk.A)):
        raise ValueError('ISSUE public key disagrees with the bound JOIN message')
    pp = gpk.pp
    if _bound_public_context(gm_sk, 'ISSUE') != pp:
        raise ValueError('ISSUE working key and interaction have different public contexts')
    boundaries = _tag_boundaries(pp)
    sigma = _required_sigma(pp, 'certificate_sigma')
    if not verify_user_public_key(pp, pk_i, pk_proof):
        raise ValueError('invalid user public-key proof')
    if reg._contains_key(pk_i):
        raise ValueError('user public key already registered')
    key = _secret_key_context(pp, gm_sk, pp.params.alpha, 3, 'ISSUE', gpk.pk_gm.A)
    tag = tuple(int(x) for x in uniform_mod_coeffs(boundaries[-1], 2))
    h = rdec(pk_i.A[0][-1], (pp.params.q - 1) // 2)
    r = ternary_vec(pp.ring_q, pp.params.k * pp.params.alpha)
    _, target = _certificate_target(pp, h, r)
    v = first_column(td.sample_d_extended(key, _tag_matrix(pp, tag), as_column(target), sigma))
    return FullCertificate(tag, r, v)


join_issue = issue


def join_accept(gpk: FullGroupPublicKey, pk_i: FullPublicKey,
                cert: FullCertificate) -> FullJoinAcceptance | None:
    if not verify_join_certificate(gpk, pk_i, cert):
        return None
    return FullJoinAcceptance(pk_i, cert)


def register_user(gpk: FullGroupPublicKey, reg: FullRegistrationTable,
                  acceptance: FullJoinAcceptance) -> None:
    if (not isinstance(acceptance, FullJoinAcceptance) or
            not verify_join_certificate(gpk, acceptance.pk, acceptance.cert)):
        raise ValueError('registration requires a valid user-accepted certificate')
    reg.add(acceptance.pk, acceptance.cert)


def _vector_shape(v, size: int, ring: Ring) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == size and all(isinstance(a, Poly) and a.ring == ring for a in v)


def _matrix_shape(A, rows: int, cols: int, ring: Ring) -> bool:
    return isinstance(A, (list, tuple)) and len(A) == rows and all(_vector_shape(row, cols, ring) for row in A)


_NO_EXPECTED_PUBLIC_KEY = object()


def _secret_key_context(pp: FullPublicParams, secret: FullSecretKey,
                        rows: int, digits: int, role: str,
                        expected_A=_NO_EXPECTED_PUBLIC_KEY):

    key = secret.trapdoor if isinstance(secret, FullSecretKey) else None
    A, R = getattr(key, 'A', None), getattr(key, 'R', None)
    if (not _matrix_shape(A, rows, (2+digits)*rows, pp.ring_q)
            or not _matrix_shape(R, 2*rows, digits*rows, pp.ring_q)):
        raise ValueError(f'{role} secret-key context has the wrong ring or dimensions')
    if expected_A is not _NO_EXPECTED_PUBLIC_KEY and (
            not _matrix_shape(expected_A, rows, (2+digits)*rows, pp.ring_q)
            or not matrix_equal(A, expected_A)):
        raise ValueError(f'{role} secret-key context does not match its group public key')
    return key


def _bound_public_context(secret: FullSecretKey, role: str) -> FullPublicParams:
    if not isinstance(secret, FullSecretKey) or not isinstance(secret.public_context, FullPublicParams):
        raise ValueError(f'{role} requires an explicitly bound public context')
    return secret.public_context


def _certificate_equation_ok(gpk: FullGroupPublicKey, pk_i: FullPublicKey,
                             cert: FullCertificate) -> bool:
    pp, p = gpk.pp, gpk.pp.params
    try:
        gpk.validate()
        if not isinstance(cert, FullCertificate) or not _matrix_shape(pk_i.A, p.alpha1, 7 * p.alpha1, pp.ring_q):
            return False
        if not _vector_shape(cert.r_i, p.k * p.alpha, pp.ring_q) or not _vector_shape(cert.v_i, 8 * p.alpha, pp.ring_q):
            return False
        if vec_inf_norm(cert.r_i) > 1:
            return False
        h = rdec(pk_i.A[0][-1], (p.q - 1) // 2)
        _, target = _certificate_target(pp, h, cert.r_i)
        return vec_equal(mat_vec_mul(concat_cols(gpk.pk_gm.A, _tag_matrix(pp, cert.tag)), cert.v_i), target)
    except (ValueError, TypeError, IndexError, AttributeError):
        return False


def verify_join_certificate(gpk: FullGroupPublicKey, pk_i: FullPublicKey,
                            cert: FullCertificate) -> bool:
    if not _certificate_equation_ok(gpk, pk_i, cert):
        return False
    p = gpk.pp.params
    squared_norm = sum(int(value) ** 2 for poly in cert.v_i for value in poly.centered())
    return squared_norm <= p.beta ** 2 * 8 * p.alpha * p.n


def verify_certificate(gpk: FullGroupPublicKey, pk_i: FullPublicKey,
                       cert: FullCertificate) -> bool:

    return (_certificate_equation_ok(gpk, pk_i, cert)
            and vec_inf_norm(cert.v_i) <= gpk.pp.params.beta)


def _opening_hint(pp: FullPublicParams, pk: FullPublicKey) -> Poly:
    value = pk.A[0][-1]
    if not isinstance(value, Poly) or value.ring != pp.ring_q:
        raise ValueError('opening hint has the wrong ring')
    beta = (pp.params.q - 1) // 2
    if beta < 1:
        raise ValueError('opening hint requires a nonempty decomposition')
    centered = value.centered()
    if bool((abs(centered) > beta).any()):
        raise ValueError('opening hint lies outside the decomposition bound')
    first_weight = (beta + 1) // 2
    digit = ((centered >= first_weight).astype('int64')
             - (centered <= -first_weight).astype('int64'))
    return Poly(pp.ring_q, digit)


def _matching_registration_keys(gpk: FullGroupPublicKey,
                                reg: FullRegistrationTable, hint: Poly):

    matches = []
    missing_tags = getattr(gpk.pp.params, 'tag_boundaries', None) is None
    for entry in reg.entries:
        pk, cert = entry.pk, entry.cert
        try:
            same_hint = _opening_hint(gpk.pp, pk) == hint
        except (ValueError, TypeError, IndexError, AttributeError):
            if verify_certificate(gpk, pk, cert):
                if rdec(pk.A[0][-1], (gpk.pp.params.q-1)//2)[0] == hint:
                    matches.append(pk)
            continue
        if same_hint:
            if verify_certificate(gpk, pk, cert):
                matches.append(pk)
        elif missing_tags:
            verify_certificate(gpk, pk, cert)
    return matches


def gr(pp: FullPublicParams) -> tuple[List[Poly], None]:
    return random_vec(pp.ring_q, pp.params.k), None


def _inner_public(a: List[Poly], b: List[Poly]) -> Poly:
    if not a or len(a) != len(b):
        raise ValueError('inner product dimension mismatch')
    result = a[0].ring.zero()
    for x, y in zip(a, b):
        result = result + x * y
    return result


def sample_risis(pk_R: List[Poly], sk_R: None = None
                 ) -> tuple[RISISStatement, List[Poly]]:
    if (sk_R is not None or not isinstance(pk_R, (list, tuple)) or not pk_R
            or not isinstance(pk_R[0], Poly)
            or not _vector_shape(pk_R, len(pk_R), pk_R[0].ring)):
        raise ValueError('RISIS requires A_R in R_q^(1 x k) and an empty secret key')
    message = ternary_vec(pk_R[0].ring, len(pk_R))
    statement = RISISStatement(tuple(a.copy() for a in pk_R), _inner_public(pk_R, message))
    return statement, message


def bind_public_relation(ct: FullCiphertext, statement: RISISStatement) -> FullCiphertext:

    if not isinstance(ct, FullCiphertext) or not isinstance(statement, RISISStatement):
        raise ValueError('ciphertext and explicit public RISIS statement are required')
    ring = ct.c_oa2.ring
    if (not _vector_shape(statement.AR, len(ct.c_rec2), ring)
            or not isinstance(statement.uR, Poly) or statement.uR.ring != ring):
        raise ValueError('public RISIS statement has the wrong shape or ring')
    if ct.relation is not None and (not vec_equal(ct.relation.AR, statement.AR)
                                   or ct.relation.uR != statement.uR):
        raise ValueError('ciphertext is already bound to a different public statement')
    return replace(ct, relation=RISISStatement(tuple(a.copy() for a in statement.AR),
                                              statement.uR.copy()))


def _encode_msg_scale(poly: Poly, q: int) -> Poly:
    return poly.scalar_mul(q // 4)


def _decode_scaled(poly: Poly, q: int, ternary: bool = True) -> Poly:
    scale = q // 4
    values = [int(x) for x in poly.centered()]
    rounded = [(1 if x >= 0 else -1) * ((2 * abs(x) + scale) // (2 * scale)) for x in values]
    alphabet = (-1, 0, 1) if ternary else (0, 1)
    if any(x not in alphabet for x in rounded):
        raise ValueError('decoded coefficient outside plaintext alphabet')
    return Poly.from_ints(poly.ring, rounded)


def _pack_vec(v: List[Poly]) -> bytes:
    return len(v).to_bytes(4, 'little') + b''.join(a.to_bytes() for a in v)


def _pack_matrix(A: List[List[Poly]]) -> bytes:
    return len(A).to_bytes(4, 'little') + b''.join(_pack_vec(row) for row in A)


def _ots_message(ct: FullCiphertext, label: bytes) -> bytes:
    return (b'LatGE-ENC-v2' + len(label).to_bytes(8, 'little') + label +
            _pack_vec(ct.c_rec1) + _pack_vec(ct.c_rec2) + _pack_vec(ct.c_oa1) + ct.c_oa2.to_bytes())


def verify_sender_sig(pp: FullPublicParams, ct: FullCiphertext, label: bytes) -> bool:
    try:
        return label == ct.label and ots_verify(ct.vk_ots, ct.sig, _ots_message(ct, label),
            params=OTSParams(n=pp.params.n, p=pp.params.p), public_seed=pp.ots_hash_seed)
    except (ValueError, TypeError, AttributeError, OverflowError):
        return False


def ciphertext_shape_ok(pp: FullPublicParams, ct: FullCiphertext) -> bool:
    p, ring = pp.params, pp.ring_q
    return (isinstance(ct, FullCiphertext) and isinstance(ct.vk_ots, bytes) and isinstance(ct.sig, bytes)
            and isinstance(ct.label, bytes) and _vector_shape(ct.c_rec1, 7 * p.alpha1, ring)
            and _vector_shape(ct.c_rec2, p.k, ring) and _vector_shape(ct.c_oa1, 7 * p.alpha1, ring)
            and isinstance(ct.c_oa2, Poly) and ct.c_oa2.ring == ring)


def _public_matrix_work(key):
    from .linalg import _PreparedMatrix
    work = getattr(key, '_matrix_work', None)
    if work is None or work.matrix is not key.A:
        work = _PreparedMatrix(key.A)
        key._matrix_work = work
    return work


def enc(gpk: FullGroupPublicKey, pk_i: FullPublicKey, cert: FullCertificate,
        msg: List[Poly], label: bytes):
    pp, p = gpk.pp, gpk.pp.params
    if not verify_certificate(gpk, pk_i, cert):
        return None
    if not isinstance(label, bytes) or not _vector_shape(msg, p.k, pp.ring_q) or vec_inf_norm(msg) > 1:
        raise ValueError('invalid message or label')
    ots = ots_gen(OTSParams(n=p.n, p=p.p), public_seed=pp.ots_hash_seed)
    H, h = H1_mat(pp.ring_q, ots.vk_bytes, p.alpha1, p.k), H2_vec(pp.ring_q, ots.vk_bytes, p.alpha1)
    xr, yr, zr = small_vec(pp.ring_q, p.k, p.B), small_vec(pp.ring_q, 7*p.alpha1, p.B), small_vec(pp.ring_q, p.alpha1, p.B)
    xo, yo, zo = small_poly(pp.ring_q, p.B), small_vec(pp.ring_q, 7*p.alpha1, p.B), small_vec(pp.ring_q, p.alpha1, p.B)
    hi0 = _opening_hint(pp, pk_i)
    ct = FullCiphertext(ots.vk_bytes,
        vec_add(_public_matrix_work(pk_i).transpose_vec(zr), yr),
        vec_add(vec_add(mat_transpose_vec_mul(H, zr), xr), [_encode_msg_scale(m, p.q) for m in msg]),
        vec_add(_public_matrix_work(gpk.pk_oa).transpose_vec(zo), yo),
        _inner_public(h, zo) + xo + _encode_msg_scale(hi0, p.q), b'', label)
    ct.sig = ots_sign(ots.sk_bytes, _ots_message(ct, label))
    return ct, EncryptionCoins(tuple(xr), tuple(yr), tuple(zr), xo, tuple(yo), tuple(zo))


def dec(sk_i: FullSecretKey, ct: FullCiphertext,
        label: bytes) -> List[Poly] | None:
    pp = _bound_public_context(sk_i, 'DEC')
    if not ciphertext_shape_ok(pp, ct) or not verify_sender_sig(pp, ct, label):
        return None
    key = _secret_key_context(pp, sk_i, pp.params.alpha1, 5, 'DEC')
    H = H1_mat(pp.ring_q, ct.vk_ots, pp.params.alpha1, pp.params.k)
    S = td.sample_pre(key.A, key.R, H, _required_sigma(pp, 'decryption_sigma'))
    masked = mat_transpose_vec_mul(S, ct.c_rec1)
    try:
        return [_decode_scaled(c - a, pp.params.q) for c, a in zip(ct.c_rec2, masked)]
    except ValueError:
        return None


def open_receiver(sk_oa: FullSecretKey, reg: FullRegistrationTable,
                  ct: FullCiphertext, label: bytes) -> FullPublicKey | None:
    gpk = getattr(sk_oa, 'opening_context', None)
    if not isinstance(gpk, FullGroupPublicKey):
        raise ValueError('OPEN requires an explicitly bound public group context')
    gpk.validate()
    pp = gpk.pp
    if _bound_public_context(sk_oa, 'OPEN') != pp:
        raise ValueError('OPEN working key and group have different public contexts')
    if not ciphertext_shape_ok(pp, ct) or not verify_sender_sig(pp, ct, label):
        return None
    key = _secret_key_context(pp, sk_oa, pp.params.alpha1, 5, 'OPEN', gpk.pk_oa.A)
    h = H2_vec(pp.ring_q, ct.vk_ots, pp.params.alpha1)
    s_oa = td.sample_pre(key.A, key.R, h, _required_sigma(pp, 'decryption_sigma'))
    try:
        hi0 = _decode_scaled(ct.c_oa2 - _inner_public(s_oa, ct.c_oa1), pp.params.q)
    except ValueError:
        return None
    matches = _matching_registration_keys(gpk, reg, hi0)
    if len(matches) != 1:
        return None
    return FullPublicKey([[a.copy() for a in row] for row in matches[0].A])


def prove(gpk: FullGroupPublicKey, pk_i: FullPublicKey, cert: FullCertificate,
          x: RISISStatement, msg: List[Poly], ct: FullCiphertext, coins: EncryptionCoins):
    from .zkp_test2 import prove as prove_paper
    return prove_paper(gpk, pk_i, cert, x, msg, ct, coins)


def verify(gpk: FullGroupPublicKey, ct: FullCiphertext, proof) -> bool:
    from .zkp_test2 import verify as verify_paper
    return verify_paper(gpk, ct, proof)

