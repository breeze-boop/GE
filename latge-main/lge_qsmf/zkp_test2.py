from __future__ import annotations
from dataclasses import dataclass, replace
from contextlib import ExitStack
from .proof_timing import excluded_from_prover_time, prover_timer
import hmac
from threading import Lock

from .paper_relation import build_public_statement, build_private_witness, check_private_relation
from .paper_zkp_backend import PaperZKPBackend


@dataclass(frozen=True)
class PiProof:
    statement_digest: bytes
    transcript: bytes


class _PublicProofPreparation:

    def __init__(self, gpk, public, strict, relation):
        self.encoding = public.encoding
        self.strict, self.relation = strict, relation
        self.closed = False
        self.gpk = replace(gpk)
        self.gpk._public_proof_preparation = self

    def bind(self, public, backend):
        if self.closed or not hmac.compare_digest(public.encoding, self.encoding):
            return None
        backend._prepared_public_relation = self.relation
        return self.strict

    def __copy__(self):raise TypeError('public preparation is a local execution plan')
    def __deepcopy__(self, memo):raise TypeError('public preparation is a local execution plan')
    def __reduce_ex__(self, protocol):raise TypeError('public preparation is not a protocol encoding')
    def close(self):
        self.closed = True
        self.gpk.__dict__.pop('_public_proof_preparation', None)
        from .public_crs_preparation import _prepared_owner
        owner = _prepared_owner(self.gpk.pp.parcom)
        if owner is not None:
            owner.close()
    def __enter__(self):
        if self.closed:raise ValueError('public preparation is closed')
        return self.gpk
    def __exit__(self, *exc):self.close()


def _prepare_public_proof(gpk, ciphertext):
    from .strict_relation import _compile_strict_relation_from_public
    from .strict_streaming import PreparedLinearOperators
    gpk = replace(gpk, pp=replace(gpk.pp, parcom=replace(gpk.pp.parcom)))
    public = build_public_statement(gpk, ciphertext)
    strict = _compile_strict_relation_from_public(gpk, ciphertext, public)
    relation = PaperZKPBackend()._compile_streaming(public, strict)
    relation._prepared_linear = PreparedLinearOperators(strict).prepare()
    return _PublicProofPreparation(gpk, public, strict, relation)


def _compiled_public_relation(gpk, ciphertext, public, backend):
    from .strict_relation import _compile_strict_relation_from_public
    preparation = getattr(gpk, '_public_proof_preparation', None)
    if type(preparation) is _PublicProofPreparation:
        strict = preparation.bind(public, backend)
        if strict is not None:
            return strict
    return _compile_strict_relation_from_public(gpk, ciphertext, public)


class _ProofPreparation:

    def __init__(self, coins, backend, package):
        self.backend, self.package = backend, package
        self.used = self.closed = False
        self._lock = Lock()
        self.coins = replace(coins)
        object.__setattr__(self.coins, '_proof_preparation', self)

    def claim(self):
        with self._lock:
            if self.used or self.closed:
                raise ValueError('proof preparation has already been consumed')
            self.used = True
            return self.backend

    def close(self):
        with self._lock:
            if not self.closed:
                self.closed = True
                self.package.close()

    def __enter__(self):
        if self.closed:raise ValueError('proof preparation is closed')
        return self.coins

    def __exit__(self, *exc):
        self.close()


def _prepare_proof_randomness(gpk, ciphertext, coins):
    with excluded_from_prover_time():
        from .scheme import EncryptionCoins
        if not isinstance(coins, EncryptionCoins):
            raise TypeError('encryption coins required')
        public = build_public_statement(gpk, ciphertext)
        backend = PaperZKPBackend()
        compiled = _compiled_public_relation(gpk, ciphertext, public, backend)
        relation = backend._compile_streaming(public, compiled)
        core = backend._streaming_core()
        from .public_crs_preparation import _prepared_owner, _prepare_public_crs
        crs = relation.parameters.commitment_crs
        if _prepared_owner(crs) is None:
            _prepare_public_crs(crs, core.workers)
        core._p_timer = prover_timer()
    package = core.prepare_randomness(relation)
    try:
        core.bind_prepared_randomness(package)
        backend._prepared_prover_core = core
        return _ProofPreparation(coins, backend, package)
    except BaseException:
        package.close()
        raise


def prove(gpk, pk_i, cert, x, msg, ciphertext, coins) -> PiProof:

    with ExitStack() as resources:
        preparation = getattr(coins, '_proof_preparation', None)
        if preparation is not None and type(preparation) is not _ProofPreparation:
            raise TypeError('invalid private proof preparation')
        if preparation is None:
            if getattr(gpk, '_public_proof_preparation', None) is None:
                with excluded_from_prover_time():
                    gpk = resources.enter_context(_prepare_public_proof(gpk, ciphertext))
            prepared = _prepare_proof_randomness(gpk, ciphertext, coins)
            coins = resources.enter_context(prepared)
            preparation = prepared
        backend = preparation.claim()
        from .scheme import (EncryptionCoins, FullCertificate, FullPublicKey,
                             RISISStatement, _matrix_shape, _vector_shape,
                             ciphertext_shape_ok, verify_sender_sig)
        from .linalg import vec_equal
        from .ring import Poly
        with excluded_from_prover_time():
            pp, p = gpk.pp, gpk.pp.params
            if (not ciphertext_shape_ok(pp, ciphertext) or
                    not verify_sender_sig(pp, ciphertext, ciphertext.label)):
                raise ValueError('invalid ciphertext signature')
            bound = ciphertext.relation
            for statement in (x, bound):
                if (not isinstance(statement, RISISStatement) or
                        not _vector_shape(statement.AR, p.k, pp.ring_q) or
                        not isinstance(statement.uR, Poly) or statement.uR.ring != pp.ring_q):
                    raise ValueError('explicit and bound public RISIS statements are required')
            if not vec_equal(x.AR, bound.AR) or x.uR != bound.uR:
                raise ValueError('explicit public relation disagrees with ciphertext context')
            if not isinstance(coins, EncryptionCoins):
                raise ValueError('coins must contain the six explicit encryption randomness fields')
            if (not isinstance(pk_i, FullPublicKey) or
                    not _matrix_shape(pk_i.A, p.alpha1, 7*p.alpha1, pp.ring_q)):
                raise ValueError('invalid private recipient public key')
            if not isinstance(cert, FullCertificate) or not _vector_shape(msg, p.k, pp.ring_q):
                raise ValueError('invalid private certificate or message')
        witness = {'pk_user': pk_i, 'cert': cert, 'm': msg,
                   'x_rec': coins.x_rec, 'y_rec': coins.y_rec, 'z_rec': coins.z_rec,
                   'x_oa': coins.x_oa, 'y_oa': coins.y_oa, 'z_oa': coins.z_oa}
        with excluded_from_prover_time():
            public = build_public_statement(gpk, ciphertext)
        private = build_private_witness(gpk, witness)
        with excluded_from_prover_time():
            check = check_private_relation(public, private)
            if not check.valid:
                raise ValueError(f'invalid private relation: {check.failed_constraints}')
            compiled = _compiled_public_relation(gpk, ciphertext, public, backend)
        transcript = backend.prove(public, private, strict_relation=compiled)
        if not isinstance(transcript, bytes) or not transcript:
            raise RuntimeError('proof backend returned an invalid transcript')
        return PiProof(public.digest(), transcript)


def verify(gpk, ciphertext, proof: PiProof) -> bool:
    from .scheme import ciphertext_shape_ok, verify_sender_sig
    if type(proof) is not PiProof or type(proof.statement_digest) is not bytes or len(proof.statement_digest) != 32:
        return False
    if type(proof.transcript) is not bytes or not proof.transcript:
        return False
    if (not ciphertext_shape_ok(gpk.pp, ciphertext) or
            not verify_sender_sig(gpk.pp, ciphertext, ciphertext.label)):
        return False
    try:
        public = build_public_statement(gpk, ciphertext)
    except (TypeError, ValueError, AttributeError, IndexError):
        return False
    if not hmac.compare_digest(proof.statement_digest, public.digest()):
        return False
    backend = PaperZKPBackend()
    compiled = _compiled_public_relation(gpk, ciphertext, public, backend)
    return backend.verify(public, proof.transcript, strict_relation=compiled)
