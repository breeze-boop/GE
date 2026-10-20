from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from weakref import WeakKeyDictionary
import shutil
import numpy as np

from .accumulator import AccumulatorWitness, LNWXAccumulator
from .distributions import (BoundedErrorDistribution, require_error_distribution)
from .fdge import FDGECiphertext
from .gpv import (EncryptionCoins, GPVCipherPair, OACiphertext, OAPublicKey, OASecretKey,
                  RecipientCiphertext, UserPublicKey, UserSecretKey,
                  decrypt_oa, decrypt_recipient, encrypt_oa, encrypt_recipient,
                  keygen_oa)
from .linear_hash import LinearPublicKeyHash, linear_hash_storage_plan
from .opening import OPEN_CONTEXT, VERIFY_OPEN
from .mathutils import (bits_to_int_le,
                        digest_public_statement, int_to_bits_le, matmul_mod,
                        rand_bits, rand_mod_matrix, require_memory)
from .params import FDGEParams, load_selected_profile
from .policy import prohibited_ok, sample_prohibitive_instance
from .randomness import CryptographicRNG
from .fast_uniform import uniform_field, uniform_source
from .zk import ProofBackendUnavailable


@dataclass(frozen=True)
class ImplicitGadget:
    n: int
    k: int
    q: int

    @property
    def shape(self):
        return (self.n, self.n * self.k)

    def apply(self, vector):
        v = np.asarray(vector)
        if v.shape != (self.n * self.k,) or v.dtype.kind not in 'iub':
            raise ValueError('G expects an integer vector of length nk')

        return np.asarray([sum(int(x) * (1 << i) for i, x in enumerate(row)) % self.q
                           for row in v.reshape(self.n, self.k)], dtype=np.int64)


@dataclass(frozen=True)
class PaperPublicParameters:
    lambda_: int
    params: FDGEParams
    F: object
    Arec: np.ndarray
    Aoa: np.ndarray
    pp_COM: object | None = None
    chi: BoundedErrorDistribution | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if type(self.lambda_) is not int or self.lambda_ < 1:
            raise ValueError('lambda_ must be a positive integer')

        chi = self.params.chi if self.chi is None else self.chi
        if chi is not None:
            require_error_distribution(chi, self.params.error_bound)
            if self.params.chi is not None and self.params.chi != chi:
                raise ValueError('pp.chi must agree with the explicit params.chi')
            object.__setattr__(self, 'chi', chi)
            if self.params.chi is None:
                object.__setattr__(self, 'params', replace(self.params, chi=chi))

    @property
    def G(self):
        return ImplicitGadget(self.params.n, self.params.k, self.params.q)

    @property
    def N(self):
        return self.params.N

    @property
    def ell(self):
        return self.params.depth

    @property
    def n(self):
        return self.params.n

    @property
    def q(self):
        return self.params.q

    @property
    def k(self):
        return self.params.k

    @property
    def m(self):
        return self.params.m

    @property
    def B(self):
        return self.params.error_bound

    @property
    def sigma(self):
        return self.params.sigma

    @property
    def beta(self):
        return self.params.beta

    def as_paper(self):
        return {'N': self.N, 'ell': self.ell, 'lambda': self.lambda_,
                'n': self.n, 'q': self.q, 'k': self.k, 'm': self.m,
                'B': self.B, 'chi': self.chi, 'sigma': self.sigma, 'beta': self.beta,
                'pp_COM': self.pp_COM, 'F': self.F, 'G': self.G,
                'Arec': self.Arec, 'Aoa': self.Aoa}


@dataclass(frozen=True, eq=False)
class PaperGMPublicKey:
    pk: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'pk', _frozen_protocol_array(self.pk))

    def __reduce__(self):
        return type(self), (self.pk,)

    def as_paper(self):
        return self.pk


@dataclass(frozen=True)
class PaperGMSecretKey:
    sk: np.ndarray

    def as_paper(self):
        return self.sk


@dataclass(frozen=True)
class PaperGroupPublicKey:
    pp: PaperPublicParameters
    pkGM: PaperGMPublicKey
    pkOA: OAPublicKey

    def as_paper(self):
        return self.pp, self.pkGM.as_paper(), self.pkOA.as_paper()


@dataclass
class PaperRegistrationEntry:
    p: np.ndarray
    epoch: int

    def as_paper(self):
        return self.p, self.epoch


@dataclass(frozen=True)
class PaperGroupInfo:
    epoch: int
    root: np.ndarray
    witnesses: Mapping[int, AccumulatorWitness]

    def as_paper(self):
        return self.root, {j: _paper_path(w) for j, w in self.witnesses.items()}


@dataclass(frozen=True)
class PaperMembershipWitness:
    root: np.ndarray
    path: AccumulatorWitness

    def as_paper(self):
        return self.root, _paper_path(self.path)


@dataclass
class PaperGMState:
    accumulator: LNWXAccumulator
    c: int
    epoch: int
    free_indices: set[int]
    info: PaperGroupInfo | None = None


@dataclass(frozen=True)
class PaperCiphertext:
    c_rec: RecipientCiphertext
    c_oa: OACiphertext

    def as_paper(self):
        return self.c_rec.as_paper(), self.c_oa.as_paper()


def _label(L):
    if not isinstance(L, bytes):
        raise ValueError('L must be bytes')


def _field(value, shape, q, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'iub' or np.any(a < 0) or np.any(a >= q):
        raise ValueError(f'{name} must have shape {shape} with canonical Z_q entries')
    return a


@dataclass(frozen=True)
class SetupStorage:
    F_path: str | Path | None = None
    F_dtype: str = 'float64'
    minimum_free_disk_bytes: int = 32 * 1024**3


@dataclass
class _Runtime:
    params: FDGEParams
    rng: object
    proof_settings: object | None = None
    storage: SetupStorage | None = None
    pp: PaperPublicParameters | None = None
    patterns: np.ndarray | None = None

    gm_public_matrices: WeakKeyDictionary = field(default_factory=WeakKeyDictionary)
    groups: dict = field(default_factory=dict)
    user_leaves: dict = field(default_factory=dict)
    prepared_user_key: tuple | None = None
    opening_gpk: PaperGroupPublicKey | None = None


_active_runtime = ContextVar('fdge_paper_runtime', default=None)
_active_join_inputs = ContextVar('fdge_join_common_inputs', default=None)
_active_proof_channel = ContextVar('fdge_proof_channel', default=None)


def _runtime():
    runtime = _active_runtime.get()
    if runtime is None:

        params, settings, _ = load_selected_profile()
        runtime = _Runtime(params, CryptographicRNG(), settings)
        _active_runtime.set(runtime)
    return runtime


@contextmanager
def _experiment(params, *, rng=None, proof_settings=None, storage=None):
    runtime = _Runtime(params, CryptographicRNG() if rng is None else rng,
                       proof_settings, storage)
    token = _active_runtime.set(runtime)
    try:
        yield runtime
    finally:
        _active_runtime.reset(token)


@contextmanager
def _public_inputs(pp, patterns):
    _patterns(pp, patterns)
    policy = patterns.copy()
    policy.flags.writeable = False
    token = _active_runtime.set(replace(_runtime(), pp=pp, patterns=policy))
    try:
        yield
    finally:
        _active_runtime.reset(token)


def _public_parameters():
    pp = _runtime().pp
    if pp is None:
        raise RuntimeError('Public parameters not bound')
    return pp


def _remember_user_key(pp, pk, p_j):
    if not isinstance(pk, UserPublicKey):
        raise TypeError('UKGEN cache requires an immutable UserPublicKey')
    _check_key_storage(pk)
    leaf = _frozen_protocol_array(p_j)
    layout = tuple((a.shape, a.dtype.str, a.strides) for a in pk.as_paper())
    _runtime().user_leaves[id(pk)] = (pk, leaf, pp, layout)


def _check_key_storage(pk):
    for a in pk.as_paper():
        backing = a
        while isinstance(backing, np.ndarray):
            backing = backing.base
        if a.flags.writeable or not isinstance(backing, bytes):
            raise ValueError('Rebuild imported keys as UserPublicKey(U1,U2)')


def _prepare_user_key(pp, pk):
    if type(pk) is not UserPublicKey:
        raise TypeError('Immutable UserPublicKey required')
    _check_key_storage(pk)
    if type(pp.F) is not LinearPublicKeyHash or not pp.F.is_read_only or pp.F.storage != 'memory':
        return None
    cached = _runtime().prepared_user_key
    if cached is not None and cached[0] is pp.F and cached[1] is pk:
        cached[2]._check_binding()
        return cached[2]
    from .relations import relation_resource_plan
    p = pp.params
    plan = relation_resource_plan(p, F=pp.F)
    retained = 16*p.n*p.m + 2*p.n*p.m*p.k + 8*p.n + p.nk + 131072
    require_memory(plan['conservative_working_bytes'] + retained, plan['budget_bytes'],
                   'relation/key preparation')
    prepared = pp.F._prepare_key(pk.U1, pk.U2)
    _runtime().prepared_user_key = (pp.F, pk, prepared)
    return prepared


def _gm_matrix(pp, pkGM):
    entry = _runtime().gm_public_matrices.get(pkGM)
    if entry is None:
        raise RuntimeError('GM public matrix not bound')
    owner_pp, matrix = entry
    if owner_pp is not pp:
        raise ValueError('GM key/matrix parameters mismatch')
    return matrix


def EXPORT_GM_PUBLIC_CONTEXT(pp, pkGM):
    return _gm_matrix(pp, pkGM)


def BIND_GM_PUBLIC_CONTEXT(pp, pkGM, A):
    p = pp.params.validate(require_distributions=False)
    _field(pkGM.pk, (p.n,), p.q, 'pkGM')
    matrix = _field(A, (p.n, p.m), p.q, 'A')
    bindings = _runtime().gm_public_matrices
    old = bindings.get(pkGM)
    if old is not None:
        if old[0] is not pp or not np.array_equal(old[1], matrix):
            raise ValueError('GM setup already bound')
        return
    bindings[pkGM] = (pp, _frozen_protocol_array(matrix))


def _group_state(gpk):
    entry = _runtime().groups.get(id(gpk))
    if entry is None or entry[0] is not gpk:
        raise RuntimeError('GM state must be initialized in this experiment')
    return entry[1], entry[2]


def setup_execution_plan(params, proof_settings, storage=None):
    params.validate()
    params.resolved_chi()
    if proof_settings is None:
        raise ProofBackendUnavailable('SETUPinit requires explicit proof_settings')
    storage = SetupStorage() if storage is None else storage
    if not isinstance(storage, SetupStorage):
        raise TypeError('SetupStorage required')
    reserve = storage.minimum_free_disk_bytes
    if not isinstance(reserve, int) or isinstance(reserve, bool) or reserve < 0:
        raise ValueError('minimum_free_disk_bytes must be >= 0')
    from .relations import relation_resource_plan
    from .yang import resource_plan as proof_resource_plan
    dimensions = relation_resource_plan(params)
    proof_plan = proof_resource_plan(params.q, dimensions['rows'], dimensions['witness_coordinates'],
                                    dimensions['triples'], proof_settings)
    disk_backed = storage.F_path is not None
    f_plan = linear_hash_storage_plan(params, storage_dtype=storage.F_dtype,
                                      disk_backed=disk_backed, read_only=True)
    resident_f = 0 if disk_backed else f_plan['matrix_bytes']
    public_A_bytes = 16 * params.n * params.m

    required = max(f_plan['conservative_explicit_buffer_bytes'],
                   resident_f + public_A_bytes + proof_plan['conservative_working_bytes'])
    require_memory(proof_plan['conservative_working_bytes'], proof_plan['budget_bytes'],
                   'SETUPinit pp_COM')
    require_memory(required, params.max_memory_bytes, 'SETUPinit output/workspace')
    path = None if not disk_backed else Path(storage.F_path).absolute()
    disk_free = None
    if path is not None:
        if not path.parent.is_dir():
            raise FileNotFoundError('SETUP F storage parent must already exist')
        if path.exists() or path.is_symlink():
            raise FileExistsError('SETUP never overwrites an existing F file')
        disk_free = shutil.disk_usage(path.parent).free
        if disk_free < f_plan['matrix_bytes'] + reserve:
            raise OSError('Insufficient F disk space/reserve')
    return {'F': f_plan, 'F_path': None if path is None else str(path),
            'minimum_free_disk_bytes': reserve, 'disk_free_bytes': disk_free,
            'public_A_bytes': public_A_bytes, 'resident_F_bytes': resident_f,
            'pp_COM': proof_plan, 'relation_dimensions': dimensions,
            'conservative_working_bytes': required, 'budget_bytes': params.max_memory_bytes,
            'storage_changes_algorithm_parameters': False, 'hard_RSS_limit_enforced': False}


def SETUPinit(lambda_: int):
    if not isinstance(lambda_, int) or isinstance(lambda_, bool) or lambda_ < 1:
        raise ValueError('lambda_ must be > 0')
    runtime = _runtime()
    params, proof_settings, storage = runtime.params, runtime.proof_settings, runtime.storage
    params.validate()
    chi = params.resolved_chi()
    plan = setup_execution_plan(params, proof_settings, storage)
    p, rng = params, runtime.rng
    from .yang import YangPublicParameters
    dims = plan['relation_dimensions']
    F = LinearPublicKeyHash(p, rng, matrix_path=plan['F_path'],
                          storage_dtype=plan['F']['storage_dtype'], read_only=True,
                          min_disk_free_bytes=plan['minimum_free_disk_bytes'])
    try:
        Arec = rand_mod_matrix(rng, p.n, p.m, p.q)
        Aoa = rand_mod_matrix(rng, p.n, p.m, p.q)
        commitment_parameters = YangPublicParameters.generate_dimensions(
            q=p.q, rows=dims['rows'], width=dims['witness_coordinates'],
            count=dims['triples'], settings=proof_settings, rng=rng)
        pp = PaperPublicParameters(lambda_, p, F, Arec, Aoa, commitment_parameters, chi=chi)
        runtime.pp = pp
        return pp
    except BaseException:
        F.discard_owned_file()
        raise


def SETUPGM(pp: PaperPublicParameters):
    p, rng = pp.params, _runtime().rng
    p.validate(require_distributions=False)
    with uniform_source(rng) as source:
        A = uniform_field(source, (p.n, p.m), p.q, dtype=np.uint32)
    sk = rand_bits(rng, p.m).astype(np.int64)
    pkGM = PaperGMPublicKey(matmul_mod(A, sk, p.q))
    _runtime().gm_public_matrices[pkGM] = (pp, _frozen_protocol_array(A))
    return pkGM, PaperGMSecretKey(sk)


def SETUPOA(pp: PaperPublicParameters):
    return keygen_oa(pp.params, pp.Aoa, _runtime().rng)


def TSetup(A):
    return _new_tree(A, _public_parameters().params)


def _new_tree(A, params):
    p = params.validate(require_distributions=False)
    A = _field(A, (p.n, p.m), p.q, 'A')
    require_memory((2 * p.N - 1) * p.nk + 8 * p.n * p.m,
                   p.max_memory_bytes, 'accumulator workspace')

    fixed_A = np.frombuffer(A.astype(np.int64, copy=False).tobytes(order='C'),
                            dtype=np.int64).reshape(p.n, p.m)
    acc = object.__new__(LNWXAccumulator)
    acc.params, acc.A, acc.N = p, fixed_A, p.N
    acc.A0, acc.A1 = fixed_A[:, :p.nk], fixed_A[:, p.nk:]
    acc.zero_leaf = np.zeros(p.nk, dtype=np.uint8)
    acc.nodes = [None] + [acc.zero_leaf.copy() for _ in range(2 * p.N - 1)]
    return acc


def _leaves(acc, R):
    if len(R) != acc.N:
        raise ValueError('R must contain N ordered leaves')
    for leaf in R:
        acc._check_bits(leaf)
    return R


def TAcc(acc, R):
    R = _leaves(acc, R)

    for index, leaf in enumerate(R):
        acc.nodes[acc._leaf_position(index)] = leaf.astype(np.uint8, copy=True)
    for node in range(acc.N - 1, 0, -1):
        acc.nodes[node] = acc.hash(acc.nodes[2 * node], acc.nodes[2 * node + 1])
    return acc.root()


def TWitness(acc, R, d):
    R = _leaves(acc, R)
    acc._check_bits(d)
    index = next((i for i, leaf in enumerate(R) if np.array_equal(leaf, d)), None)

    if index is None:
        return None
    if all(np.array_equal(leaf, acc.nodes[acc._leaf_position(i)]) for i, leaf in enumerate(R)):
        return _paper_path(acc.witness(index))
    temporary = _new_tree(acc.A, acc.params)
    TAcc(temporary, R)
    return _paper_path(temporary.witness(index))


def _paper_path(w):
    return int_to_bits_le(w.index, len(w.siblings_bottom_up)), tuple(w.siblings_bottom_up)


def TVerify(acc, u, d, w):
    if isinstance(w, tuple) and len(w) == 2:
        binj, siblings = w
        try:
            bits = np.asarray(binj)
        except (TypeError, ValueError):
            return False
        if (bits.shape != (acc.params.depth,) or bits.dtype.kind not in 'iub'
                or not np.all((bits == 0) | (bits == 1))
                or not isinstance(siblings, (list, tuple))):
            return False
        w = AccumulatorWitness(bits_to_int_le(bits), list(siblings))
    return acc.verify(u, d, w)


def TUpdate(acc, binj, p):
    bits = np.asarray(binj)
    if (bits.shape != (acc.params.depth,) or bits.dtype.kind not in 'iub'
            or not np.all((bits == 0) | (bits == 1))):
        raise ValueError('binj must be an ell-bit little-endian identity')
    acc.update(bits_to_int_le(bits), p)


def INITIALIZE(pp, pkGM, pkOA):
    p = pp.params.validate(require_distributions=False)
    _field(pkGM.pk, (p.n,), p.q, 'pkGM')
    _field(pkOA.U1, (p.n, p.depth), p.q, 'pkOA.U1')
    _field(pkOA.U2, (p.n, p.depth), p.q, 'pkOA.U2')
    acc = _new_tree(_gm_matrix(pp, pkGM), p)
    reg = [PaperRegistrationEntry(np.zeros(p.nk, dtype=np.uint8), 0) for _ in range(p.N)]
    gm = PaperGMState(acc, 0, 0, set(range(p.N)), None)
    gpk = PaperGroupPublicKey(pp, pkGM, pkOA)
    runtime = _runtime()
    runtime.pp = pp
    runtime.opening_gpk = gpk
    runtime.groups[id(gpk)] = (gpk, reg, gm)
    return gpk, reg, gm


def PUBLISH_GROUP_INFO(gm_state, reg):
    acc = gm_state.accumulator
    if len(reg) != acc.N:
        raise ValueError('registration table must have N rows')

    for i, entry in enumerate(reg):
        if not np.array_equal(entry.p, acc.nodes[acc._leaf_position(i)]):
            raise ValueError('registration table does not match the GM tree')
    paths = {i: acc.witness(i) for i, entry in enumerate(reg) if np.any(entry.p)}
    if gm_state.info is not None:
        gm_state.epoch = max(gm_state.epoch, gm_state.info.epoch + 1)
    info = PaperGroupInfo(gm_state.epoch, acc.root(), paths)

    gm_state.info = info
    return info


def _check_gm_keypair(pp, pkGM, skGM):
    p = pp.params
    if (not isinstance(skGM, PaperGMSecretKey) or not isinstance(skGM.sk, np.ndarray)
            or skGM.sk.shape != (p.m,) or skGM.sk.dtype.kind not in 'iub'
            or not np.all((skGM.sk == 0) | (skGM.sk == 1))):
        raise ValueError('GM secret must contain m bits')
    pk = _field(pkGM.pk, (p.n,), p.q, 'pkGM')
    if not np.array_equal(matmul_mod(_gm_matrix(pp, pkGM), skGM.sk, p.q), pk):
        raise ValueError('GM secret does not match the public key')


def _manager_inputs(gpk, skGM, info, reg, gm_state):
    p = gpk.pp.params.validate(require_distributions=False)
    _check_gm_keypair(gpk.pp, gpk.pkGM, skGM)

    if len(reg) != p.N or not np.array_equal(gm_state.accumulator.A, _gm_matrix(gpk.pp, gpk.pkGM)):
        raise ValueError('GM state does not match gpk')
    actual_free = set()
    for index, entry in enumerate(reg):
        gm_state.accumulator._check_bits(entry.p)
        if not np.array_equal(entry.p, gm_state.accumulator.nodes[
                gm_state.accumulator._leaf_position(index)]):
            raise ValueError('registration table does not match the GM tree')
        if not np.any(entry.p):
            actual_free.add(index)
    if actual_free != gm_state.free_indices:
        raise ValueError('S0/vacant indices mismatch')
    if info is None:
        if gm_state.info is not None:
            raise ValueError('current group information is required')
    elif (gm_state.info is None or info.epoch != gm_state.epoch
          or not np.array_equal(info.root, gm_state.accumulator.root())):
        raise ValueError('Group info/GM state mismatch')


def UKGEN(pp):
    from .paper_measured import blocked_recipient_keygen
    pk, sk = blocked_recipient_keygen(pp.params, pp.Arec, _runtime().rng)
    p_j = pp.F.apply(pk.U1, pk.U2)
    if not np.any(p_j):
        raise RuntimeError('UKGEN produced zero leaf; retry')
    _remember_user_key(pp, pk, p_j)
    return pk, sk


@contextmanager
def JOIN_CONTEXT(gpk, pk, info):
    token = _active_join_inputs.set((gpk, pk, info))
    try:
        yield
    finally:
        _active_join_inputs.reset(token)


def _join_inputs():
    common = _active_join_inputs.get()
    if common is None:
        raise RuntimeError('JOIN/ISSUE requires JOIN_CONTEXT(gpk, pk, info)')
    return common


def _check_join_key_inputs(gpk, pk, sk):
    p = gpk.pp.params
    if not isinstance(pk, UserPublicKey):
        raise TypeError('JOIN requires a UserPublicKey')
    _check_key_storage(pk)
    _field(pk.U1, (p.n, p.m), p.q, 'pk.U1')
    _field(pk.U2, (p.n, p.m), p.q, 'pk.U2')
    if (not isinstance(sk, UserSecretKey) or not isinstance(sk.E1, np.ndarray)
            or sk.E1.shape != (p.m, p.m) or sk.E1.dtype.kind not in 'iub'):
        raise ValueError('JOIN requires integer E1 of shape (m,m)')


class PaperJoinSession:
    def __init__(self, gpk, pk, sk):
        _check_join_key_inputs(gpk, pk, sk)
        p = gpk.pp.params
        entry = _runtime().user_leaves.get(id(pk))
        if entry is None or entry[0] is not pk:

            leaf = gpk.pp.F.apply(pk.U1, pk.U2)
            if not _witness_bits(leaf, p.nk) or not np.any(leaf):
                raise ValueError('JOIN requires a nonzero binary F binding')
            _remember_user_key(gpk.pp, pk, leaf)
            entry = _runtime().user_leaves[id(pk)]
        if entry[2] is not gpk.pp:
            raise ValueError('JOIN key/group parameters mismatch')
        layout = tuple((a.shape, a.dtype.str, a.strides) for a in pk.as_paper())
        if entry[3] != layout:
            raise ValueError('JOIN key metadata changed')
        self._gpk, self._leaf = gpk, entry[1]
        self._spent = False

    @property
    def request(self):
        return self._leaf.copy()

    def finish(self, certificate):
        if self._spent:
            raise RuntimeError('JOIN session already completed')
        self._spent = True
        if not isinstance(certificate, PaperMembershipWitness):
            return None

        pp = self._gpk.pp
        acc = object.__new__(LNWXAccumulator)
        acc.params, acc.N = pp.params, pp.params.N
        acc.A = _gm_matrix(pp, self._gpk.pkGM)
        acc.A0, acc.A1 = acc.A[:, :pp.params.nk], acc.A[:, pp.params.nk:]
        return certificate if TVerify(acc, certificate.root, self._leaf, certificate.path) else None


class PaperIssueSession:
    def __init__(self, gpk, skGM, info):
        self._gpk, self._skGM, self._info = gpk, skGM, info
        self._messages = []
        self._spent = False

    def receive(self, p_j):
        if self._spent:
            raise RuntimeError('ISSUE session already completed')
        if not isinstance(p_j, np.ndarray):
            raise ValueError('the JOIN message must be the binary leaf p_j')
        self._messages.append(p_j.copy())

    def finish(self):
        if self._spent:
            raise RuntimeError('ISSUE session already completed')
        self._spent = True
        return _issue_batch(self._gpk, self._messages, self._skGM, self._info)


def JOIN(sk):
    gpk, pk, _ = _join_inputs()
    return PaperJoinSession(gpk, pk, sk)


def ISSUE(skGM):
    gpk, _, info = _join_inputs()
    return PaperIssueSession(gpk, skGM, info)


def _issue_batch(gpk, leaves, skGM, info):
    reg, gm_state = _group_state(gpk)
    _manager_inputs(gpk, skGM, info, reg, gm_state)
    p, acc, rng = gpk.pp.params, gm_state.accumulator, _runtime().rng
    leaves = list(leaves)
    if not leaves:
        raise ValueError('Empty enrollment batch')
    if len(leaves) > len(gm_state.free_indices):
        raise ValueError('Insufficient vacant slots')
    existing = {row.p.astype(np.uint8, copy=False).tobytes() for row in reg if np.any(row.p)}
    submitted = set()
    for leaf in leaves:
        acc._check_bits(leaf)

        key = leaf.astype(np.uint8, copy=False).tobytes()
        if not leaf.any() or key in existing or key in submitted:
            raise ValueError('JOIN requires distinct nonzero submitted leaves')
        submitted.add(key)

    if gm_state.info is not None:
        gm_state.epoch = max(gm_state.epoch, gm_state.info.epoch + 1)
    indices = []
    for leaf in leaves:
        free = sorted(gm_state.free_indices)
        index = free[int(rng.integers(len(free)))]
        TUpdate(acc, int_to_bits_le(index, p.depth), leaf)
        reg[index] = PaperRegistrationEntry(leaf.astype(np.uint8, copy=True), gm_state.epoch)
        gm_state.free_indices.remove(index)
        gm_state.c += 1
        indices.append(index)
    updated = PUBLISH_GROUP_INFO(gm_state, reg)
    witnesses = []
    for index in indices:
        path = updated.witnesses[index]

        member_path = AccumulatorWitness(path.index,
            [sibling.copy() for sibling in path.siblings_bottom_up])
        witnesses.append(PaperMembershipWitness(updated.root.copy(), member_path))
    return witnesses


def GUPDATE(gpk, skGM, info, S, reg):
    _, gm_state = _group_state(gpk)
    _manager_inputs(gpk, skGM, info, reg, gm_state)
    acc, p = gm_state.accumulator, gpk.pp.params
    indices = set()
    for leaf in S:
        acc._check_bits(leaf)
        matches = [i for i, row in enumerate(reg) if np.array_equal(row.p, leaf)] if leaf.any() else []
        if len(matches) != 1:
            raise ValueError('S requires distinct registered nonzero leaves')
        indices.add(matches[0])
    zero = np.zeros(p.nk, dtype=np.uint8)
    for index in sorted(indices):
        TUpdate(acc, int_to_bits_le(index, p.depth), zero)
        reg[index] = PaperRegistrationEntry(zero.copy(), reg[index].epoch)
        gm_state.free_indices.add(index)
    gm_state.epoch += 1
    updated = PUBLISH_GROUP_INFO(gm_state, reg)
    _runtime().groups[id(gpk)] = (gpk, reg, gm_state)
    return updated


def Gr(pp):
    pp.params.validate(require_distributions=False)
    return ((pp.params.t, pp.params.e), b'')


def sampleR(pkR):
    pp = _public_parameters()
    pp.params.validate(require_distributions=False)
    if pkR != (pp.params.t, pp.params.e):
        raise ValueError('pkR does not match public policy parameters')
    instance = sample_prohibitive_instance(pp.params, _runtime().rng)
    return instance.patterns, instance.message


def _same_path(a, b):
    return (a.index == b.index and len(a.siblings_bottom_up) == len(b.siblings_bottom_up)
            and all(np.array_equal(x, y) for x, y in zip(a.siblings_bottom_up, b.siblings_bottom_up)))


def _witness_bits(value, length):
    try:
        bits = np.asarray(value)
        return (bits.shape == (length,) and bits.dtype.kind in 'iub'
                and bool(np.all((bits == 0) | (bits == 1))))
    except (TypeError, ValueError):
        return False


def _witness_path(params, path):
    index = getattr(path, 'index', None)
    siblings = getattr(path, 'siblings_bottom_up', None)
    if (not isinstance(index, (int, np.integer)) or isinstance(index, (bool, np.bool_))
            or not 0 <= index < params.N or not isinstance(siblings, (list, tuple, np.ndarray))):
        return False
    if isinstance(siblings, np.ndarray) and siblings.ndim == 0:
        return False
    return (len(siblings) == params.depth
            and all(_witness_bits(node, params.nk) for node in siblings))


def _published_witness(params, info, witj):
    path = getattr(witj, 'path', None)
    root = getattr(witj, 'root', None)
    published_root = getattr(info, 'root', None)
    published = getattr(info, 'witnesses', None)
    if (not _witness_path(params, path) or not _witness_bits(root, params.nk)
            or not _witness_bits(published_root, params.nk) or not isinstance(published, Mapping)):
        return False
    stored = published.get(path.index)
    return (_witness_path(params, stored) and np.array_equal(published_root, root)
            and _same_path(stored, path))


def _patterns(pp, patterns):
    p = pp.params
    if (patterns.shape != (p.e, p.t) or patterns.dtype.kind not in 'iub'
            or not np.all((patterns == 0) | (patterns == 1))):
        raise ValueError('patterns must be an (e,t) binary array')


def ENC(pkGM, pkOA, pkj, witj, info, patterns, message, L):
    pp = _public_parameters()
    pp.params.validate()
    chi = require_error_distribution(pp.chi, pp.params.error_bound)
    _label(L)
    _patterns(pp, patterns)
    p, rng = pp.params, _runtime().rng
    _field(pkGM.pk, (p.n,), p.q, 'pkGM')
    if (message.shape != (p.m,) or message.dtype.kind not in 'iub'
            or not np.all((message == 0) | (message == 1))):
        raise ValueError('message must contain m bits')
    if not _published_witness(p, info, witj) or not prohibited_ok(patterns, message, p.d):
        return None

    c_rec, rec = encrypt_recipient(p, pp.Arec, pkj, message, rng, chi=chi)
    c_oa, oa = encrypt_oa(p, pp.Aoa, pkOA, int_to_bits_le(int(witj.path.index), p.depth), rng, chi=chi)
    coins = EncryptionCoins(rec['s'], rec['x'], rec['y'], oa['s'], oa['x'], oa['y'])
    return PaperCiphertext(c_rec, c_oa), coins


def DEC(sk, Psi, L):
    pp, patterns = _public_parameters(), _runtime().patterns
    if patterns is None:
        raise RuntimeError('DEC requires a bound public filtering policy')
    _label(L)
    _patterns(pp, patterns)
    p = pp.params
    msg = decrypt_recipient(p, sk, Psi.c_rec)
    return msg if prohibited_ok(patterns, msg, p.d) else None


def OPEN_IDENTITY(skOA, info, reg, Psi, L):
    pp = _public_parameters()
    _label(L)
    pp.params.validate()
    from .opening import _checked_opening_secret
    secret = _checked_opening_secret(pp, skOA, getattr(Psi, 'c_oa', None))
    return _opening_identity_with_checked_key(pp, OASecretKey(secret), info, reg, Psi, L)


def _opening_identity_with_checked_key(pp, skOA, info, reg, Psi, L):
    p = pp.params
    if len(reg) != p.N:
        raise ValueError('registration table must have N rows')
    index = bits_to_int_le(decrypt_oa(p, skOA, getattr(Psi, 'c_oa', None)))
    if info is None:
        return None
    if not isinstance(info, PaperGroupInfo) or not isinstance(info.witnesses, Mapping):
        raise ValueError('Missing published root or witnesses')
    if not _witness_bits(info.root, p.nk):
        raise ValueError('info root must contain nk bits')
    path = info.witnesses.get(index)
    if not _witness_path(p, path) or path.index != index:
        return None
    leaf = reg[index].p
    if not _witness_bits(leaf, p.nk):
        raise ValueError('registration leaves must contain nk bits')
    if not np.any(leaf):
        return None
    return index


def OPEN(skOA, info, reg, Psi, L) -> tuple[int | None, object]:
    from .opening import execute_open
    return execute_open(skOA, info, reg, Psi, L)


def _compiler_inputs(gpk, info, patterns, Psi, L):
    gpk.pp.params.validate()
    _label(L)
    _patterns(gpk.pp, patterns)
    if info is None:
        raise ValueError('proof requires published group information')
    pp = gpk.pp

    context = digest_public_statement(info.epoch, info.root, gpk.pkGM.pk)
    adapted = FDGECiphertext(Psi.c_rec, Psi.c_oa, info.epoch, L, context)
    return dict(params=pp.params, Aacc=_gm_matrix(pp, gpk.pkGM), Arec=pp.Arec, Aoa=pp.Aoa,
                F=pp.F, oa_pk=gpk.pkOA, root=info.root, patterns=patterns, ciphertext=adapted)


def PUBLIC_STATEMENT(gpk, info, patterns, Psi, L):
    from .relations import public_statement
    return public_statement(**_compiler_inputs(gpk, info, patterns, Psi, L))


def BUILD_RELATION(gpk, info, patterns, Psi, L, pkj, witj, message, coins):
    from .relations import _build_relation_with_key_hash
    gpk.pp.params.validate()
    if not _published_witness(gpk.pp.params, info, witj):
        raise ValueError('Witness not published')

    channel = _active_proof_channel.get()
    public = None
    if channel is not None and not channel._closed:
        c = channel._common
        if (gpk is c.gpk and info is c.info and patterns is c.patterns
                and Psi is c.Psi and L == c.L):
            channel.preprocess()
            public = channel._public
    prepared = _prepare_user_key(gpk.pp, pkj)
    return _build_relation_with_key_hash(**_compiler_inputs(gpk, info, patterns, Psi, L),
                                         user_pk=pkj, witness=witj.path,
                                         message=message, coins=coins,
                                         _prepared_key=prepared, _public_relation=public)


def _proof_pp(gpk):
    pp_COM = gpk.pp.pp_COM
    if pp_COM is None:
        raise ProofBackendUnavailable('pp_COM required')
    return pp_COM


@dataclass(frozen=True)
class _ProofCommonInputs:
    gpk: PaperGroupPublicKey
    info: PaperGroupInfo
    patterns: np.ndarray
    Psi: PaperCiphertext
    L: bytes


def _frozen_protocol_array(value):
    a = np.asarray(value)
    if a.dtype.kind not in 'iub':
        raise ValueError('public protocol arrays must contain integers')
    return np.frombuffer(a.tobytes(order='C'), dtype=a.dtype).reshape(a.shape)


def _proof_common_snapshot(gpk, info, patterns, Psi, L):
    _label(L)
    _patterns(gpk.pp, patterns)
    if info is None:
        raise ValueError('proof requires published group information')
    freeze = _frozen_protocol_array
    fixed_pp = replace(gpk.pp, Arec=freeze(gpk.pp.Arec), Aoa=freeze(gpk.pp.Aoa))
    pkGM = PaperGMPublicKey(gpk.pkGM.pk)
    pkOA = OAPublicKey(gpk.pkOA.U1, gpk.pkOA.U2)
    fixed_gpk = PaperGroupPublicKey(fixed_pp, pkGM, pkOA)

    _runtime().gm_public_matrices[pkGM] = (fixed_pp, _gm_matrix(gpk.pp, gpk.pkGM))
    paths = {index: AccumulatorWitness(path.index,
                tuple(freeze(sibling) for sibling in path.siblings_bottom_up))
             for index, path in info.witnesses.items()}
    fixed_info = PaperGroupInfo(info.epoch, freeze(info.root), MappingProxyType(paths))

    def pair(c):
        return GPVCipherPair(freeze(c.c1), freeze(c.c2))

    fixed_psi = PaperCiphertext(RecipientCiphertext(pair(Psi.c_rec.c1), pair(Psi.c_rec.c2)),
                               OACiphertext(pair(Psi.c_oa.c1), pair(Psi.c_oa.c2)))
    return _ProofCommonInputs(fixed_gpk, fixed_info, freeze(patterns), fixed_psi, L)


class PaperProofChannel:
    def __init__(self, common):
        from threading import Lock
        self._common = common
        self._public = None
        self._verifier = None
        self._closed = False
        self._offline = None
        self._offline_mode = False
        self._offline_lock = Lock()

    def preprocess(self, pkj=None):
        if self._closed:
            raise RuntimeError('proof context has ended')
        c = self._common
        if self._public is None:
            from .yang import _check_parameters, _context
            public = PUBLIC_STATEMENT(c.gpk, c.info, c.patterns, c.Psi, c.L)
            proof_pp = _proof_pp(c.gpk)
            _check_parameters(public, proof_pp)
            _context(public, proof_pp, c.L)
            self._public = public
        self._public._check_context_binding()
        if type(c.gpk.pp.F) is LinearPublicKeyHash:
            c.gpk.pp.F._ensure_open()
        if pkj is not None:
            _prepare_user_key(c.gpk.pp, pkj)
        return self

    def precompute(self):
        from .yang import YangPrecomputation
        with self._offline_lock:
            self.preprocess()
            if self._offline is not None:
                raise RuntimeError('an unused offline proof is already prepared')
            self._offline = YangPrecomputation.prepare(self._public, _proof_pp(self._common.gpk),
                                                       rng=_runtime().rng, auxiliary=self._common.L)
            self._offline_mode = True
        return self

    def _take_offline(self):
        with self._offline_lock:
            if not self._offline_mode:
                return None
            if self._offline is None:
                raise RuntimeError('Prepare fresh offline proof before P')
            offline, self._offline = self._offline, None
            return offline

    def challenge(self, commitment):
        from .yang import YangVerifierSession
        if self._closed:
            raise RuntimeError('proof context has ended')
        if self._verifier is None:
            c = self._common
            pp_COM = _proof_pp(c.gpk)
            self.preprocess()
            self._verifier = YangVerifierSession(self._public, pp_COM, auxiliary=c.L)
        return self._verifier.challenge(commitment)

    def _verify(self, piPsi):
        if self._closed or self._verifier is None:
            return False
        from .yang import YangChallenge, YangCommitment, YangTranscript
        expected = self._verifier.challenge_message
        matched = False
        if isinstance(piPsi, YangTranscript) and expected is not None:
            try:
                def same_commitment(message):
                    if not isinstance(message, YangCommitment):
                        return False
                    value = _field(message.value, expected.commitment.value.shape,
                                   self._common.gpk.pp.q, 'proof commitment')
                    return (message.context == expected.commitment.context
                            and np.array_equal(value, expected.commitment.value))

                challenge = piPsi.challenge
                matched = (same_commitment(piPsi.commitment)
                           and isinstance(challenge, YangChallenge)
                           and type(challenge.alpha) is int
                           and challenge.alpha == expected.alpha
                           and same_commitment(challenge.commitment))
            except (AttributeError, TypeError, ValueError):
                matched = False

        return self._verifier.verify(piPsi.response if matched else None)


@contextmanager
def PROOF_CONTEXT(gpk, info, patterns, Psi, L):
    common = _proof_common_snapshot(gpk, info, patterns, Psi, L)
    channel = PaperProofChannel(common)
    channel.preprocess()
    token = _active_proof_channel.set(channel)
    try:
        yield channel
    finally:
        with channel._offline_lock:
            channel._closed = True
            if channel._offline is not None:
                channel._offline.discard()
                channel._offline = None
        channel._public = None
        channel._verifier = None
        channel._common = None
        _active_proof_channel.reset(token)


def _proof_channel():
    channel = _active_proof_channel.get()
    if channel is None or channel._closed:
        raise RuntimeError('P/V requires PROOF_CONTEXT')
    return channel


class PaperProverSession:
    __slots__ = ('__session',)

    def __init__(self, session):
        self.__session = session

    @property
    def commitment(self):
        from .yang import YangCommitment
        message = self.__session.commitment
        return YangCommitment(message.value.copy(), message.context)

    def respond(self, challenge):
        return self.__session.respond(challenge)

    def finish(self, challenge):
        from .yang import YangTranscript, _copy_challenge
        commitment = self.commitment
        response = self.respond(challenge)
        if response is None:
            return None
        return YangTranscript(commitment, _copy_challenge(challenge), response)


def P(pkj, witj, message, coins):
    from .yang import YangProverSession
    channel = _proof_channel()
    c = channel._common
    pp_COM = _proof_pp(c.gpk)
    relation = BUILD_RELATION(c.gpk, c.info, c.patterns, c.Psi, c.L, pkj, witj, message, coins)

    public = relation._with_witness(None)
    offline = channel._take_offline()
    session = (YangProverSession(public, relation.x, pp_COM, rng=_runtime().rng, auxiliary=c.L)
               if offline is None else
               YangProverSession._from_precomputation(offline, public, relation.x, pp_COM,
                                                       rng=_runtime().rng, auxiliary=c.L))
    return PaperProverSession(session)


def V(piPsi):
    return _proof_channel()._verify(piPsi)
