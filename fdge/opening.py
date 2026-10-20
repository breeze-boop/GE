from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import json
import hashlib

import numpy as np

from .gpv import (GPVCipherPair, OACiphertext, OASecretKey, RecipientCiphertext,
                  validate_decryption_inputs)
from .zk import ProofBackendUnavailable
from .stern_opening import SternOpeningBackend


@dataclass(frozen=True)
class OpeningArray:
    shape: tuple[int, ...]
    data: bytes

    def array(self) -> np.ndarray:
        return np.frombuffer(self.data, dtype='<i8').reshape(self.shape)


def _array(value, shape, name, *, bound=None, q=None):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind not in 'iub':
        raise ValueError(f'{name} must contain integers of shape {shape}')
    if q is not None and (np.any(a < 0) or np.any(a >= q)):
        raise ValueError(f'{name} must contain canonical field elements')
    if bound is not None and (np.any(a > bound) or
                             (a.dtype.kind == 'i' and np.any(a < -bound))):
        raise ValueError(f'{name} exceeds its bound')
    if a.dtype.kind == 'u' and a.size and int(a.max()) > np.iinfo(np.int64).max:
        raise ValueError(f'{name} exceeds signed int64')
    return OpeningArray(tuple(shape), a.astype('<i8', copy=False).tobytes(order='C'))


def _bits(value, size, name):
    return _array(value, (size,), name, q=2)


def _nonnegative_integer(value, name):
    if (not isinstance(value, (int, np.integer)) or
            isinstance(value, (bool, np.bool_)) or value < 0):
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


@dataclass(frozen=True)
class OpeningGroupKey:
    mathematical_parameters: bytes
    Aoa: OpeningArray
    pkGM: OpeningArray
    Uoa1: OpeningArray
    Uoa2: OpeningArray


@dataclass(frozen=True)
class OpeningPath:
    index: int
    siblings_bottom_up: tuple[OpeningArray, ...]


@dataclass(frozen=True)
class OpeningGroupInfo:
    epoch: int
    root: OpeningArray
    witnesses: tuple[OpeningPath, ...]


@dataclass(frozen=True)
class OpeningCipherPair:
    c1: OpeningArray
    c2: OpeningArray


@dataclass(frozen=True)
class OpeningCiphertext:
    c_rec: tuple[OpeningCipherPair, OpeningCipherPair]
    c_oa: tuple[OpeningCipherPair, OpeningCipherPair]


@dataclass(frozen=True)
class OpeningStatement:
    group: OpeningGroupKey
    info: OpeningGroupInfo | None
    registration_nonzero: tuple[bool, ...]
    ciphertext: OpeningCiphertext
    label: bytes
    identity: int | None
    decoder: str = 'nearest_modular_0_floor_q_over_2_ties_to_zero_v1'
    registration_digest: bytes = b''


@dataclass(frozen=True)
class _OpeningContext:
    pp: object
    group: OpeningGroupKey
    backend: SternOpeningBackend


_active_opening = ContextVar('fdge_opening_context', default=None)


def _checked_opening_secret(pp, skOA, ciphertext):
    from . import paper_api as api

    p = pp.params
    secret, _, _ = validate_decryption_inputs(p, skOA, ciphertext, plaintext_length=p.depth)
    beta = p.resolved_beta()

    if np.any(secret > beta) or np.any(secret < -beta):
        raise ValueError('OA secret exceeds beta')
    E1 = _array(secret.astype(np.int64), (p.m, p.depth), 'OA secret', bound=beta).array()
    context = _active_opening.get()
    if context is not None:
        if context.pp is not pp:
            raise ValueError('OPEN context/parameter mismatch')
        A, U1 = context.group.Aoa.array(), context.group.Uoa1.array()
    else:
        gpk = api._runtime().opening_gpk
        if gpk is None or gpk.pp is not pp:
            raise RuntimeError('OPEN requires initialized group public inputs')

        A = _array(pp.Aoa, (p.n, p.m), 'Aoa', q=p.q).array()
        U1 = _array(gpk.pkOA.U1, (p.n, p.depth), 'pkOA.U1', q=p.q).array()
    if p.m * (p.q - 1) * beta >= 2**63:
        raise ValueError('OA key-check integer bound exceeds int64')

    if not np.array_equal((A @ np.asfortranarray(E1)) % p.q, U1):
        raise ValueError('OA secret/public key mismatch')
    return E1


@contextmanager
def OPEN_CONTEXT(gpk):
    from . import paper_api as api

    pp = gpk.pp
    p = pp.params.validate()
    backend=SternOpeningBackend(pp.lambda_,p.max_memory_bytes)
    parameters = {name: getattr(p, name) for name in
                  ('n', 'q', 'm', 'k', 'N', 'depth', 't', 'e', 'd',
                   'error_bound', 'sigma', 'beta')}
    parameters['lambda'] = pp.lambda_
    group = OpeningGroupKey(
        json.dumps(parameters, sort_keys=True, separators=(',', ':'),
                   allow_nan=False).encode('ascii'),
        _array(pp.Aoa, (p.n, p.m), 'Aoa', q=p.q),
        _array(gpk.pkGM.pk, (p.n,), 'pkGM', q=p.q),
        _array(gpk.pkOA.U1, (p.n, p.depth), 'pkOA.U1', q=p.q),
        _array(gpk.pkOA.U2, (p.n, p.depth), 'pkOA.U2', q=p.q))
    token = _active_opening.set(_OpeningContext(pp, group, backend))
    runtime_token = api._active_runtime.set(replace(api._runtime(), pp=pp))
    try:
        yield
    finally:
        api._active_runtime.reset(runtime_token)
        _active_opening.reset(token)


def _public_snapshot(p, info, reg, Psi):
    from . import paper_api as api
    from .accumulator import AccumulatorWitness

    if len(reg) != p.N:
        raise ValueError('registration table must have N rows')
    registry = tuple(api.PaperRegistrationEntry(
        _bits(entry.p, p.nk, 'registered leaf').array(),
        _nonnegative_integer(entry.epoch, 'registration epoch')) for entry in reg)
    active = tuple(bool(np.any(entry.p)) for entry in registry)
    fixed_info, paper_info = None, None
    if info is not None:
        epoch = _nonnegative_integer(info.epoch, 'info epoch')
        root = _bits(info.root, p.nk, 'info root')
        if not isinstance(info.witnesses, Mapping):
            raise ValueError('Witnesses must map identities to paths')
        paths = []
        for index, path in info.witnesses.items():
            index = _nonnegative_integer(index, 'published identity')
            if (index >= p.N or not api._witness_path(p, path) or path.index != index):
                raise ValueError('published identity/path is malformed')
            paths.append(OpeningPath(index, tuple(
                _bits(node, p.nk, 'path node') for node in path.siblings_bottom_up)))
        fixed_info = OpeningGroupInfo(epoch, root, tuple(sorted(paths, key=lambda x: x.index)))
        paper_info = api.PaperGroupInfo(epoch, root.array(), {
            path.index: AccumulatorWitness(path.index, tuple(
                node.array() for node in path.siblings_bottom_up)) for path in fixed_info.witnesses})

    def pair(value, length, name):
        return OpeningCipherPair(_array(value.c1, (p.m,), name + '.c1', q=p.q),
                                 _array(value.c2, (length,), name + '.c2', q=p.q))

    fixed_cipher = OpeningCiphertext(
        (pair(Psi.c_rec.c1, p.m, 'c_rec1'), pair(Psi.c_rec.c2, p.m, 'c_rec2')),
        (pair(Psi.c_oa.c1, p.depth, 'c_oa1'), pair(Psi.c_oa.c2, p.depth, 'c_oa2')))

    def paper_pair(value):
        return GPVCipherPair(value.c1.array(), value.c2.array())

    paper_cipher = api.PaperCiphertext(
        RecipientCiphertext(*(paper_pair(pair) for pair in fixed_cipher.c_rec)),
        OACiphertext(*(paper_pair(pair) for pair in fixed_cipher.c_oa)))
    return fixed_info, active, fixed_cipher, paper_info, registry, paper_cipher


def _registry_digest(registry):
    digest=hashlib.sha256(b'FDGE-opening-registration-v1\0'+len(registry).to_bytes(8,'little'))
    for entry in registry:
        epoch=entry.epoch.to_bytes(max(1,(entry.epoch.bit_length()+7)//8),'little')
        digest.update(len(epoch).to_bytes(8,'little'))
        digest.update(epoch)
        digest.update(entry.p.astype(np.uint8,copy=False).tobytes())
    return digest.digest()


def _prepare_with_context(context,skOA,info,reg,Psi,L):
    from . import paper_api as api
    if api._public_parameters() is not context.pp:
        raise ValueError('OPEN context/parameter mismatch')
    api._label(L)
    p = context.pp.params.validate()
    fixed_info, active, fixed_cipher, paper_info, registry, paper_cipher = _public_snapshot(
        p, info, reg, Psi)
    E1 = _checked_opening_secret(context.pp, skOA, paper_cipher.c_oa)
    identity = api._opening_identity_with_checked_key(
        context.pp, OASecretKey(E1), paper_info, registry, paper_cipher, L)
    statement = OpeningStatement(context.group, fixed_info, active, fixed_cipher, L, identity,
                                 registration_digest=_registry_digest(registry))
    return statement,E1,identity


def execute_open(skOA, info, reg, Psi, L) -> tuple[int | None, object]:
    from . import paper_api as api
    context = _active_opening.get()
    if context is None:
        gpk=api._runtime().opening_gpk
        if gpk is None:
            raise RuntimeError('OPEN requires group context')
        with OPEN_CONTEXT(gpk):
            return execute_open(skOA,info,reg,Psi,L)
    statement,E1,identity=_prepare_with_context(context,skOA,info,reg,Psi,L)
    pi = context.backend.prove(statement, E1)
    if pi is None:
        raise ProofBackendUnavailable('opening backend did not produce a proof')
    return identity, pi


def VERIFY_OPEN(gpk,info,reg,Psi,L,opened):
    from . import paper_api as api
    try:
        if not isinstance(opened,tuple) or len(opened)!=2:
            return False
        identity,pi=opened
        api._label(L)
        with OPEN_CONTEXT(gpk):
            context=_active_opening.get()
            fixed_info,active,fixed_cipher,_,registry,_=_public_snapshot(gpk.pp.params,info,reg,Psi)
            statement=OpeningStatement(context.group,fixed_info,active,fixed_cipher,L,identity,
                                       registration_digest=_registry_digest(registry))
            return context.backend.verify(statement,pi)
    except (ValueError,TypeError,AttributeError,OverflowError):
        return False
