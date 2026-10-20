from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import os
import time

import numpy as np

from .decomposition import vdec, weights
from .mathutils import (bits_to_int_le, decrypt_bits_from_noisy_mod, int_to_bits_le,
                        require_memory)
from .randomness import CryptographicRNG
from .fast_uniform import uniform_field, uniform_source


_DOMAIN = b'FDGE-GPV-Stern-opening-binary-SHA256-FS-v1\0'


def _hash_public(value, digest):
    if value is None:
        digest.update(b'N')
    elif isinstance(value, bool):
        digest.update(b'T' if value else b'F')
    elif isinstance(value, int):
        data = str(value).encode('ascii')
        digest.update(b'I'+len(data).to_bytes(8,'little')+data)
    elif isinstance(value, (bytes, str)):
        data = value if isinstance(value,bytes) else value.encode('utf8')
        digest.update((b'B' if isinstance(value,bytes) else b'S')+len(data).to_bytes(8,'little'))
        digest.update(data)
    elif isinstance(value, tuple):
        digest.update(b'L'+len(value).to_bytes(8,'little'))
        for item in value:
            _hash_public(item,digest)
    elif is_dataclass(value):
        _hash_public(type(value).__name__,digest)
        for field in fields(value):
            _hash_public(field.name,digest)
            _hash_public(getattr(value,field.name),digest)
    else:
        raise ValueError('unsupported opening public statement field')


def statement_digest(statement):
    digest=hashlib.sha256(_DOMAIN+b'public-statement\0')
    _hash_public(statement,digest)
    return digest.digest()


def _rounds(bits):
    count=math.ceil(bits/math.log2(1.5))
    while (1 << bits)*(1 << count) > 3**count:
        count+=1
    return count


def _wire(vector):
    return np.asarray(vector,dtype='<u4').tobytes(order='C')


def _commit(context, round_index, role, salt, *payload):
    digest=hashlib.sha256(_DOMAIN+b'commitment\0'+context+round_index.to_bytes(4,'little')
                          +bytes([role])+salt)
    for part in payload:
        digest.update(len(part).to_bytes(8,'little'))
        digest.update(part)
    return digest.digest()


def _challenges(context, commitments, rounds):
    digest=hashlib.sha256(_DOMAIN+b'Fiat-Shamir\0'+context+rounds.to_bytes(4,'little'))
    for commitment in commitments:
        for part in commitment:
            digest.update(part)
    stream=hashlib.shake_256(digest.digest())
    length=max(256,2*rounds)
    while True:
        raw=stream.digest(length)
        result=tuple(value%3+1 for value in raw if value<252)
        if len(result)>=rounds:
            return result[:rounds]
        length*=2


@dataclass(frozen=True)
class SternOpeningResponse:
    challenge: int
    first: bytes
    second: bytes
    salt_a: bytes
    salt_b: bytes


@dataclass(frozen=True)
class SternOpeningProof:
    context: bytes
    commitments: tuple[tuple[bytes,bytes,bytes], ...]
    responses: tuple[SternOpeningResponse, ...]
    version: bytes = _DOMAIN

    def as_paper(self):
        return self.commitments,tuple(r.challenge for r in self.responses),self.responses


class OpeningRelation:
    def __init__(self, statement):
        import json
        p=json.loads(statement.group.mathematical_parameters)
        self.q,self.n,self.m,self.ell,self.N,self.beta=(p[name] for name in ('q','n','m','depth','N','beta'))
        if (self.q,self.n,self.m)!=(524309,222,8880) or self.ell not in range(4,11):
            raise ValueError('Original FDGE dimensions required')
        if statement.decoder!='nearest_modular_0_floor_q_over_2_ties_to_zero_v1':
            raise ValueError('unsupported decoder convention')
        self.half=self.q//2
        self.lower=self.half//2
        self.bound=self.half
        self.Eweights=weights(2*self.beta)
        self.Yweights=weights(self.bound)
        self.Esize=self.m*self.ell*len(self.Eweights)
        self.Ysize=self.ell*len(self.Yweights)
        published=set() if statement.info is None else {path.index for path in statement.info.witnesses}
        active={i for i,nonzero in enumerate(statement.registration_nonzero) if nonzero and i in published}
        if statement.identity is None:
            self.allowed=np.array(sorted(set(range(self.N))-active),dtype=np.int64)
        else:
            identity=statement.identity
            if type(identity) is not int or identity not in active:
                raise ValueError('Inactive opening identity')
            self.allowed=np.array([identity],dtype=np.int64)
        if not len(self.allowed):
            raise ValueError('No matching opening identity')
        self.allowed_bits=np.array([int_to_bits_le(int(i),self.ell) for i in self.allowed],dtype=np.int64)
        self.width=self.Esize+2*self.Ysize+self.ell+len(self.allowed)
        self.extended=2*self.width
        self.rows=self.n*self.ell+3*self.ell+1
        self.A=statement.group.Aoa.array()
        self.Afloat=self.A.astype(np.float64)
        self.U=statement.group.Uoa1.array()
        self.c1=statement.ciphertext.c_oa[0].c1.array()
        self.c2=statement.ciphertext.c_oa[0].c2.array()
        self.cfloat=self.c1.astype(np.float64)

        if self.m*(self.q-1)**2>2**53-1:
            raise ValueError('prepared exact BLAS bound fails')
        key=(self.U+self.beta*self.A.sum(axis=1)[:,None])%self.q
        decryption=(self.c2+self.beta*self.c1.sum()+self.lower)%self.q
        self.target=np.concatenate((key.ravel(),decryption,
                                    np.full(self.ell,self.bound,dtype=np.int64),
                                    np.zeros(self.ell,dtype=np.int64),np.ones(1,dtype=np.int64)))

    def split(self, vector):
        base=np.asarray(vector)[:self.width]
        offset=0
        E=base[:self.Esize].reshape(self.m,self.ell,len(self.Eweights))@self.Eweights
        offset+=self.Esize
        v=base[offset:offset+self.Ysize].reshape(self.ell,len(self.Yweights))@self.Yweights
        offset+=self.Ysize
        s=base[offset:offset+self.Ysize].reshape(self.ell,len(self.Yweights))@self.Yweights
        offset+=self.Ysize
        j=base[offset:offset+self.ell].astype(np.int64)
        selector=base[offset+self.ell:].astype(np.int64)
        return E%self.q,v%self.q,s%self.q,j,selector

    def apply(self, vector):
        E,v,s,j,selector=self.split(vector)
        Ef=E.astype(np.float64)
        key=(self.Afloat@Ef).astype(np.int64)%self.q
        decrypt=(self.cfloat@Ef).astype(np.int64)%self.q
        decrypt=(decrypt+v+(self.half+1)*j)%self.q
        ranges=(v+s+j)%self.q
        binding=(j-self.allowed_bits.T@selector)%self.q
        return np.concatenate((key.ravel(),decrypt,ranges,binding,np.array([selector.sum()%self.q],dtype=np.int64)))

    def witness(self,E):
        E=np.asarray(E)
        if (E.dtype.kind not in 'iub' or E.shape!=(self.m,self.ell)
                or np.any(E < -self.beta) or np.any(E > self.beta)):
            raise ValueError('Opening key shape/beta mismatch')
        E=E.astype(np.int64,copy=False)
        if not np.array_equal((self.A@E)%self.q,self.U):
            raise ValueError('opening key does not match public U1')
        raw=(self.c2-E.T@self.c1)%self.q
        j=decrypt_bits_from_noisy_mod(raw,self.q).astype(np.int64)
        identity=bits_to_int_le(j)
        positions=np.flatnonzero(self.allowed==identity)
        if len(positions)!=1:
            raise ValueError('Opening identity/decision mismatch')
        residual=(raw-self.half*j)%self.q
        residual=np.where(residual>self.q//2,residual-self.q,residual)
        v=residual+self.lower-j
        s=self.bound-j-v
        if np.any(v<0) or np.any(s<0) or np.any(v>self.bound) or np.any(s>self.bound):
            raise ValueError('Residual outside decision interval')
        selector=np.zeros(len(self.allowed),dtype=np.uint8)
        selector[positions[0]]=1
        base=np.concatenate((vdec((E+self.beta).ravel(),2*self.beta),vdec(v,self.bound),
                             vdec(s,self.bound),j.astype(np.uint8),selector))
        witness=np.concatenate((base,1-base)).astype(np.uint8)
        if not np.array_equal(self.apply(witness),self.target):
            raise AssertionError('compiled correct-decryption relation failed')
        return witness


def _permutation(rng,size):
    while True:
        priorities=np.frombuffer(rng._random_bytes(8*size),dtype='<u8')
        permutation=np.argsort(priorities,kind='quicksort')
        ordered=priorities[permutation]
        if not np.any(ordered[1:]==ordered[:-1]):
            return permutation.astype(np.uint32)


def _material(seed,relation):
    rng=CryptographicRNG(seed)
    with uniform_source(rng) as source:
        mask=uniform_field(source,(relation.extended,),relation.q,dtype=np.uint32)
        permutation=_permutation(source,relation.extended)
        salts=tuple(source._random_bytes(32) for _ in range(3))
    return mask,permutation,salts


def _add_binary(mask,bits,q):
    result=np.add(mask,bits,dtype=np.uint32)
    np.subtract(result,q,out=result,where=result>=q)
    return result


def _commit_material(material,relation,witness,context,index):
    mask,permutation,salts=material
    masked=_add_binary(mask,witness,relation.q)
    return (_commit(context,index,1,salts[0],_wire(permutation),_wire(relation.apply(mask))),
            _commit(context,index,2,salts[1],_wire(mask[permutation])),
            _commit(context,index,3,salts[2],_wire(masked[permutation])))


def _respond_material(material,relation,witness,challenge):
    mask,permutation,salts=material
    if challenge==1:
        return SternOpeningResponse(1,np.packbits(witness[permutation],bitorder='little').tobytes(),
                                    _wire(mask[permutation]),salts[1],salts[2])
    if challenge==2:
        return SternOpeningResponse(2,_wire(permutation),_wire(_add_binary(mask,witness,relation.q)),salts[0],salts[2])
    if challenge==3:
        return SternOpeningResponse(3,_wire(permutation),_wire(mask),salts[0],salts[1])
    raise ValueError('invalid Stern challenge')


def _verify_round(relation,context,index,commitment,response):
    if (not isinstance(response,SternOpeningResponse) or len(response.salt_a)!=32 or len(response.salt_b)!=32
            or len(response.second)!=4*relation.extended):
        return False
    vector=np.frombuffer(response.second,dtype='<u4')
    if np.any(vector>=relation.q):
        return False
    challenge=response.challenge
    if challenge==1:
        if len(response.first)!=(relation.extended+7)//8:
            return False
        bits=np.unpackbits(np.frombuffer(response.first,dtype=np.uint8),bitorder='little')
        if np.any(bits[relation.extended:]) or int(bits[:relation.extended].sum())!=relation.width:
            return False
        permuted=bits[:relation.extended]
        return (_commit(context,index,2,response.salt_a,response.second)==commitment[1]
                and _commit(context,index,3,response.salt_b,_wire(_add_binary(vector,permuted,relation.q)))==commitment[2])
    if challenge not in (2,3) or len(response.first)!=4*relation.extended:
        return False
    permutation=np.frombuffer(response.first,dtype='<u4')
    if np.any(permutation>=relation.extended) or not np.all(np.bincount(permutation,minlength=relation.extended)==1):
        return False
    product=relation.apply(vector)
    if challenge==2:
        product=(product-relation.target)%relation.q
        other=3
    else:
        other=2
    return (_commit(context,index,1,response.salt_a,response.first,_wire(product))==commitment[0]
            and _commit(context,index,other,response.salt_b,_wire(vector[permutation]))==commitment[other-1])


class SternOpeningBackend:
    def __init__(self,security_bits,max_memory_bytes):
        if type(security_bits) is not int or security_bits<1:
            raise ValueError('Opening security bits must be > 0')
        self.rounds=_rounds(security_bits)
        self.max_memory_bytes=max_memory_bytes

    def _preflight(self,relation):
        required=self.rounds*(8*relation.extended+256)+160*relation.extended+32*1024**2
        require_memory(required,self.max_memory_bytes,'Stern proof workspace')

    def prove(self,statement,E1):
        relation=OpeningRelation(statement)
        self._preflight(relation)
        witness=relation.witness(E1)
        context=statement_digest(statement)
        materials=[]
        commitments=[]
        for index in range(self.rounds):
            material=_material(os.urandom(32),relation)
            materials.append(material)
            commitments.append(_commit_material(material,relation,witness,context,index))
        commitments=tuple(commitments)
        challenges=_challenges(context,commitments,self.rounds)
        responses=[]
        for index,challenge in enumerate(challenges):
            material=materials[index]
            materials[index]=None
            responses.append(_respond_material(material,relation,witness,challenge))
            del material
        return SternOpeningProof(context,commitments,tuple(responses))

    def verify(self,statement,proof):
        try:
            if (not isinstance(proof,SternOpeningProof) or proof.version!=_DOMAIN
                    or len(proof.commitments)!=self.rounds or len(proof.responses)!=self.rounds
                    or proof.context!=statement_digest(statement)):
                return False
            if any(len(c)!=3 or any(type(x) is not bytes or len(x)!=32 for x in c) for c in proof.commitments):
                return False
            relation=OpeningRelation(statement)
            self._preflight(relation)
            challenges=_challenges(proof.context,proof.commitments,self.rounds)
            return all(response.challenge==challenge and
                       _verify_round(relation,proof.context,index,commitment,response)
                       for index,(commitment,response,challenge) in enumerate(zip(proof.commitments,proof.responses,challenges)))
        except (ValueError,TypeError,AttributeError,OverflowError):
            return False


def calibrate_opening(statement,E1,*,repeats=3):
    relation=OpeningRelation(statement)
    witness=relation.witness(E1)
    context=statement_digest(statement)
    samples={name:[] for name in ('material','commit','response1','response2','response3','verify1','verify2','verify3')}
    for index in range(repeats):
        seed=os.urandom(32)
        start=time.perf_counter_ns()
        material=_material(seed,relation)
        samples['material'].append((time.perf_counter_ns()-start)/1e6)
        start=time.perf_counter_ns()
        commitment=_commit_material(material,relation,witness,context,index)
        samples['commit'].append((time.perf_counter_ns()-start)/1e6)
        for challenge in (1,2,3):
            start=time.perf_counter_ns()
            response=_respond_material(material,relation,witness,challenge)
            samples[f'response{challenge}'].append((time.perf_counter_ns()-start)/1e6)
            start=time.perf_counter_ns()
            valid=_verify_round(relation,context,index,commitment,response)
            samples[f'verify{challenge}'].append((time.perf_counter_ns()-start)/1e6)
            if not valid:
                raise AssertionError('Stern opening calibration branch rejected')
    mean={key:sum(values)/len(values) for key,values in samples.items()}
    rounds=_rounds(json.loads(statement.group.mathematical_parameters)['lambda'])
    return {'rounds':rounds,'witness_bits':relation.width,'extended_coordinates':relation.extended,
            'rows':relation.rows,'samples_ms':samples,'mean_ms':mean,
            'estimated_prove_ms':rounds*(mean['material']+mean['commit']+sum(mean[f'response{i}'] for i in (1,2,3))/3),
            'estimated_verify_ms':rounds*sum(mean[f'verify{i}'] for i in (1,2,3))/3,
            'all_three_branches_checked':True,'complete_proof_generated':False,
            'scope':'Stern_rounds; excludes_retention/dispatch/setup'}
