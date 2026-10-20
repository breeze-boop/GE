from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import os
from typing import Sequence

from .ring import Poly, Ring, is_prime_u64
from .structured_crs import StructuredCRS


class PaperProofUnavailable(NotImplementedError):
    pass


class PaperProofResourceLimit(PaperProofUnavailable):
    pass


@dataclass(frozen=True)
class LNS20Parameters:
    modulus: int
    degree: int = 16
    kappa1: int = 45
    kappa2: int = 52
    automorphisms: int = 4
    rejection_delta: int | None = None
    crs_seed: bytes = b""
    research_reference: bool = False
    commitment_crs: StructuredCRS | None = None

    def validate(self):
        q, d, k = self.modulus, self.degree, self.automorphisms
        if self.research_reference is not True:
            raise PaperProofUnavailable(
                'Only an explicit research_reference profile is implemented; '
                'its degree-16 CBD challenge has P(c=0)=2^-16. '
                'This is not a verified vulnerability claim about Main.pdf, '
                'nor a strict reproduction or an 80-bit security claim.')
        if not is_prime_u64(q) or d < 4 or d & (d - 1) or q % (4*d) != 2*d + 1:
            raise ValueError('LNS20 requires prime q with q - 1 == 2d (mod 4d)')
        if (k < 2 or k & (k - 1) or d % k or
                pow(2*d//k+1, k, 2*d) != 1 or
                pow(2*d//k+1, k//2, 2*d) == 1):
            raise ValueError('invalid automorphism subgroup')
        if self.kappa1 <= 0 or self.kappa2 <= 0:
            raise ValueError('commitment module ranks must be positive')
        delta = self.rejection_delta
        if delta is None:
            raise PaperProofUnavailable('LNS20 uniform rejection_delta must be an explicit public parameter; Main.pdf does not specify its conversion from sigma')
        if not isinstance(delta, int) or not d < delta or delta + d >= q//2:
            raise ValueError('rejection_delta must cover the challenge shift without modular wraparound')
        if not isinstance(self.crs_seed, bytes) or len(self.crs_seed) < 32:
            raise ValueError('a public CRS seed of at least 32 bytes is required')
        crs = self.commitment_crs
        if crs is not None and (not isinstance(crs, StructuredCRS) or
                crs.ring != Ring(d, q) or crs.kappa1 != self.kappa1 or
                crs.kappa2 != self.kappa2 or crs.auxiliary_count != 3 or
                crs.profile not in ('research_generic', 'paper_iii_d')):
            raise ValueError('commitment CRS does not match this research protocol')

    def encode(self) -> bytes:
        self.validate()
        return b''.join(int(x).to_bytes(8, 'little') for x in
                        (self.modulus,self.degree,self.kappa1,self.kappa2,
                         self.automorphisms,self.rejection_delta,int(self.research_reference))) + _frame(self.crs_seed) + _frame(b'' if self.commitment_crs is None else self.commitment_crs.encode())


@dataclass(frozen=True)
class PackedPolynomialConstraint:
    terms: tuple[tuple[Poly, tuple[int, ...]], ...]


def _frame(data):
    return len(data).to_bytes(8,'little') + data


@lru_cache(maxsize=32)
def _roots(degree: int, modulus: int):
    if not is_prime_u64(modulus) or (modulus-1) % (2*degree):
        raise ValueError('fully splitting prime ring required for NTT packing')
    for base in range(2,modulus):
        zeta=pow(base,(modulus-1)//(2*degree),modulus)
        if pow(zeta,degree,modulus)==modulus-1:
            return tuple(pow(zeta,2*i+1,modulus) for i in range(degree))
    raise ValueError('no primitive cyclotomic root')


def ntt_slots(poly: Poly) -> tuple[int,...]:
    q=poly.ring.q
    result=[]
    for root in _roots(poly.ring.n,q):
        acc=0
        for coefficient in reversed(poly.coeffs):acc=(acc*root+int(coefficient))%q
        result.append(acc)
    return tuple(result)


def pack_slots(ring: Ring, slots: Sequence[int]) -> Poly:
    if len(slots)!=ring.n:raise ValueError('slot vector has wrong dimension')
    q,d=ring.q,ring.n
    roots=_roots(d,q); inv_d=pow(d,-1,q)
    return Poly.from_ints(ring,[(sum(int(v)*pow(r,-j,q) for v,r in zip(slots,roots))*inv_d)%q for j in range(d)])


def automorphism(poly: Poly, power: int, count: int=4) -> Poly:
    d,q=poly.ring.n,poly.ring.q
    exponent=pow(2*d//count+1,power,2*d)
    result=[0]*d
    for i,a in enumerate(poly.coeffs):
        j=(i*exponent)%(2*d)
        result[j%d]=(result[j%d]+(-1 if j>=d else 1)*int(a))%q
    return Poly.from_ints(poly.ring,result)


def _sum_polys(ring, values):
    result=ring.zero()
    for v in values:result=result+v
    return result


def _monomial(ring, exponent):
    values=[0]*ring.n; exponent%=2*ring.n
    values[exponent%ring.n]=-1 if exponent>=ring.n else 1
    return Poly.from_ints(ring,values)


class _XOF:
    def __init__(self, seed):self.seed=seed;self.counter=0;self.pool=b'';self.position=0
    def bytes(self,n):
        return self.read_view(n).tobytes()
    def read_view(self,n):
        if type(n) is not int or n<0:raise ValueError('nonnegative byte count required')
        available=len(self.pool)-self.position
        if available>=n:
            result=memoryview(self.pool)[self.position:self.position+n];self.position+=n
            return result
        chunks=[memoryview(self.pool)[self.position:]] if available else []
        remaining=n-available
        self.pool=b'';self.position=0
        if remaining>=8192:
            from .bulk_crs import _counter_xof_blocks
            blocks=max(256,(remaining+4095)//4096)
            raw=_counter_xof_blocks(self.seed,self.counter,blocks)
            self.counter+=blocks;self.pool=raw;self.position=remaining
            if not chunks:return raw[:remaining]
            chunks.append(raw[:remaining])
            return memoryview(b''.join(chunks))
        while remaining:
            block=hashlib.shake_256(self.seed+self.counter.to_bytes(8,'little')).digest(4096)
            self.counter+=1
            take=min(remaining,len(block));chunks.append(block[:take])
            self.pool=block;self.position=take;remaining-=take
        return memoryview(b''.join(chunks))
    def uniform(self,bound):
        width=(bound.bit_length()+7)//8
        limit=(1<<(8*width))-((1<<(8*width))%bound)
        while True:
            candidate=int.from_bytes(self.bytes(width),'little')
            if candidate<limit:return candidate%bound
    def poly(self,ring):return Poly.from_ints(ring,[self.uniform(ring.q) for _ in range(ring.n)])
    def uniform_many(self,bound,count):

        import numpy as np
        if type(bound) is not int or not 0<bound<1<<64:
            raise ValueError('uniform bound must be an integer in [1,2^64)')
        if type(count) is not int or not 0<=count<=1<<20:
            raise ValueError('uniform batch must contain at most 2^20 values')
        width=(bound.bit_length()+7)//8
        limit=(1<<(8*width))-((1<<(8*width))%bound)
        result=np.empty(count,dtype=np.uint64);used=0
        while used<count:
            raw=self.bytes((count-used)*width)
            if width==5:
                words=np.frombuffer(raw,dtype=[('low','<u4'),('high','u1')])
                candidates=words['low'].astype(np.uint64)
                candidates|=words['high'].astype(np.uint64)<<32
            elif width in (1,2,4,8):
                candidates=np.frombuffer(raw,dtype=f'<u{width}')
            else:
                octets=np.frombuffer(raw,dtype=np.uint8).reshape(-1,width)
                candidates=np.zeros(len(octets),dtype=np.uint64)
                for j in range(width):candidates|=octets[:,j].astype(np.uint64)<<(8*j)
            if limit<1<<64:candidates=candidates[candidates<np.uint64(limit)]
            accepted=candidates%np.uint64(bound)
            result[used:used+len(accepted)]=accepted;used+=len(accepted)
        return result


def _encode_polys(polys):return b''.join(p.to_bytes() for p in polys)


def cubic_auxiliary_commitments(binding1,binding2,mask1,mask2,psi):

    psi0,psi1,psi2=psi
    return binding1-psi2,binding2-psi1-mask1,psi0+mask2


def _trace_polys(values,k):
    ring=values[0].ring
    return _sum_polys(ring,(_monomial(ring,mu)*_sum_polys(ring,(automorphism(value,nu,k) for nu in range(k))) for mu,value in enumerate(values))).scalar_mul(pow(k,-1,ring.q))


class PaperZKPBackend:


    def __init__(self,max_attempts=128,*,workspace_parent=None,
                 block_rows=1024,workers=None):
        if workers is None:
            configured=os.environ.get('LATGE_PROOF_WORKERS')
            if configured is None:
                workers=min(8,os.cpu_count() or 1)
            else:
                if not configured or not configured.isascii() or not configured.isdecimal():
                    raise ValueError('LATGE_PROOF_WORKERS must be a decimal integer in [1,32]')
                workers=int(configured,10)
                if not 1<=workers<=32:
                    raise ValueError('LATGE_PROOF_WORKERS must be a decimal integer in [1,32]')
        if type(workers) is not int or not 1<=workers<=32:
            raise ValueError('workers must be an integer in [1,32] or None')
        self._streaming_options=dict(workspace_parent=workspace_parent,
                                     block_rows=block_rows,workers=workers,
                                     max_attempts=max_attempts)
    @staticmethod
    def _parameters(statement):
        values=dict(getattr(statement,'zk_parameters',()))
        params=LNS20Parameters(statement.ring.q,degree=values.get('ell',16),
                              kappa1=values.get('kappa1',45),kappa2=values.get('kappa2',52),
                              automorphisms=values.get('conjugates',4),
                              rejection_delta=values.get('rejection_delta'),
                              crs_seed=getattr(statement,'zk_crs_seed',b''),
                              research_reference=values.get('research_reference',False))
        params.validate()
        return params
    @staticmethod
    def _strict_parameters(statement,strict_relation,params):
        from .strict_relation import StrictRelation
        if (not isinstance(strict_relation, StrictRelation) or
                strict_relation.public_statement_digest != statement.digest() or
                strict_relation.ring != statement.ring):
            raise ValueError('III-D relation does not match the public statement')
        parent_crs = getattr(statement, 'zk_crs', None)
        if (parent_crs is None or parent_crs.profile != 'paper_iii_d' or
                parent_crs.message_count != strict_relation.layout.packed_message_count):
            raise ValueError('III-D relation requires its original public CRS layout')
        params = replace(params, commitment_crs=parent_crs)
        params.validate()
        return params
    def _compile_streaming(self,statement,strict_relation):
        from .strict_streaming import StreamingStrictRelation
        params=self._strict_parameters(statement,strict_relation,self._parameters(statement))
        prepared=getattr(self,'_prepared_public_relation',None)
        if (isinstance(prepared,StreamingStrictRelation) and prepared.strict is strict_relation
                and prepared.parameters==params):
            return prepared
        return StreamingStrictRelation(strict_relation,params)
    def _streaming_core(self):
        from .streaming_zkp_backend import StreamingLNS20Core
        return StreamingLNS20Core(**self._streaming_options)
    def prove(self,statement,private_witness,*,strict_relation)->bytes:
        from .strict_relation import lower_strict_witness
        from .streaming_zkp_backend import StrictWitnessSlots
        from .proof_timing import excluded_from_prover_time
        with excluded_from_prover_time():
            relation=self._compile_streaming(statement,strict_relation)
        assignment=lower_strict_witness(strict_relation,private_witness)
        core=self.__dict__.pop('_prepared_prover_core',None)
        if core is None:core=self._streaming_core()
        return core.prove(relation,StrictWitnessSlots(strict_relation,assignment))
    def verify(self,statement,transcript:bytes,*,strict_relation)->bool:
        relation=self._compile_streaming(statement,strict_relation)
        return self._streaming_core().verify(relation,transcript)
