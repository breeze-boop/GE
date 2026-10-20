from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
from itertools import islice
import numpy as np

from .decomposition import g_beta
from .paper_zkp_backend import LNS20Parameters, PackedPolynomialConstraint, _XOF, _frame, pack_slots, ntt_slots
from .ring import Ring
from .strict_packing import PACKING_PROFILE, _dimensions
from .strict_relation import StrictRelation, _TERNARY_BLOCKS


STREAMING_DOMAIN = b'LatGE-LNS20-Main-streaming-context-v2\x00'
COMPILER_VERSION = b'main-iii-d-streaming-operators-v1\x00'


class RepeatableStream:
    def __init__(self, count, factory):
        self.count, self.factory = count, factory
    def __len__(self):
        return self.count
    def __iter__(self):
        return self.factory()


@dataclass(frozen=True)
class StreamingLinearConstraint:
    scalar_row: object
    degree: int
    modulus: int
    @property
    def target(self):
        return self.scalar_row.target % self.modulus
    @property
    def terms(self):
        for a, index in self.scalar_row.terms:
            yield a % self.modulus, index//self.degree, index%self.degree


class StreamingStrictRelation:

    def __init__(self, relation: StrictRelation, parameters: LNS20Parameters):
        ring, messages = _dimensions(relation, parameters, relation.scalar_count)
        self.strict, self.parameters, self.ring = relation, parameters, ring
        self.message_count = messages
        self.statement_digest = hashlib.sha256(
            PACKING_PROFILE.encode('ascii')+b'\x00'+relation.digest()).digest()
        stats = relation.statistics()
        count = (stats['product_constraints']+stats['ternary_constraints']+
                 stats['tag_constraints'])//ring.n
        self.linear = RepeatableStream(stats['linear_constraints'],
            lambda: (StreamingLinearConstraint(row,ring.n,ring.q)
                     for row in relation.iter_linear_constraints()))
        self.polynomial = RepeatableStream(count,self._polynomials)
        self._context = hashlib.sha256(STREAMING_DOMAIN +
            _frame(COMPILER_VERSION) + _frame(parameters.encode()) +
            _frame(self.statement_digest) +
            b''.join(x.to_bytes(8,'little') for x in
                (messages,len(self.linear),len(self.polynomial)))).digest()

    @classmethod
    def from_public(cls,gpk,ciphertext,parameters):
        from .strict_relation import compile_strict_relation
        return cls(compile_strict_relation(gpk,ciphertext),parameters)

    def context(self):
        return self._context

    def _polynomials(self):
        r, ring = self.strict, self.ring
        p,n,e = r.params,r.n,ring.n
        k0,k1 = r.layout.k0,r.layout.k1
        one,negative = ring.one(),ring.one().scalar_mul(-1)
        off = {b.name:b.offset for b in r.layout.witness}
        for tag in range(p.d):
            for h in range(3*p.alpha*k0):
                for i in range(n):
                    for z in range(0,n,e):
                        local = (h*n+i)*n+z
                        factors = ((off['v2_coefficient_copies']+local)//e,
                                   (off['tag']+tag*n+z)//e)
                        out = (off[f'tag_product[{tag+1}]']+local)//e
                        yield PackedPolynomialConstraint(((one,factors),(negative,(out,))))
        for column in range(7*p.alpha1):
            for row in range(p.alpha1):
                for digit in range(k1):
                    for i in range(n):
                        for z in range(0,n,e):
                            copy = ((row*k1+digit)*n+i)*n+z
                            product = (((column*p.alpha1+row)*k1+digit)*n+i)*n+z
                            factors = ((off['z_rec_coefficient_copies']+copy)//e,
                                (off['A_user_column_major']+(column*p.alpha1+row)*n+z)//e)
                            out = (off['recipient_products']+product)//e
                            yield PackedPolynomialConstraint(((one,factors),(negative,(out,))))
        for name in _TERNARY_BLOCKS:
            block = r.block(name)
            for index in range(block.offset,block.offset+block.scalar_length,e):
                j=index//e
                yield PackedPolynomialConstraint(((one,(j,j,j)),(negative,(j,))))
        boundaries=p.tag_boundaries
        for tag in range(p.d):
            for z in range(0,n,e):
                j=(off['tag']+tag*n+z)//e
                indicator=[-int(boundaries[tag]<=z+s<boundaries[tag+1]) for s in range(e)]
                terms=((one,(j,j)),)
                if any(indicator):
                    terms += ((pack_slots(ring,indicator),(j,)),)
                yield PackedPolynomialConstraint(terms)

    def _row_term_count(self,row,position):
        r=self.strict;p,n=r.params,r.n;k0,k1=r.layout.k0,r.layout.k1
        component=position//n
        C=r.coefficient
        nz=lambda vector:sum(int(a)%p.q != 0 for poly in vector for a in poly)
        if row==1:
            tags=C('A_tags')
            return k0*(nz(C('A')[component])+nz(tags[0][component])+
                n*sum(nz(matrix[component]) for matrix in tags[1:]))+nz(C('B2')[component])
        if row in (2,8,5):return 2
        if row==4:return 1+p.s
        if row==6:return nz(C('B0')[component])+nz(C('B1')[component])+p.s
        if row==7:return p.alpha1*k1*n+k1
        if row in (9,11,12):
            matrix=C('H1') if row==9 else C('A_oa') if row==11 else tuple((a,) for a in C('H2'))
            return k1*nz([matrix[i][component] for i in range(p.alpha1)])+k1+int(row in (9,12))
        if row==10:return nz(C('AR'))
        raise ValueError('unknown normalized row')

    def resource_report(self):
        r=self.strict;p,n=r.params,r.n
        total=0;counts={}
        for row in r.layout.rows:
            if row.index==3:continue
            if row.index in (2,5,8):
                count=2*row.scalar_length
            else:
                count=sum(n*self._row_term_count(row.index,start)
                          for start in range(0,row.scalar_length,n))
            counts[str(row.index)]=count;total+=count
        k=self.parameters.automorphisms
        crs=self.parameters.commitment_crs
        columns=p.kappa1+self.message_count+3+p.kappa2
        polynomial_terms=2*len(self.polynomial)
        for tag in range(p.d):
            for z in range(0,n,p.ell):
                if not any(p.tag_boundaries[tag]<=j<p.tag_boundaries[tag+1]
                           for j in range(z,z+p.ell)):
                    polynomial_terms-=1
        return {'context_profile':'streaming-v2', 'message_count':self.message_count,
            'scalar_count':r.scalar_count,'linear_count':len(self.linear),
            'packed_polynomial_count':len(self.polynomial),'linear_terms_by_paper_row':counts,
            'linear_term_count':total, 'legacy_linear_encoding_bytes':16*len(self.linear)+24*total,
            'polynomial_term_count':polynomial_terms,
            'one_gamma_uint64_bytes':len(self.linear)*8,
            'one_message_slots_uint64_bytes':r.scalar_count*8,
            'materialized_all_p_vectors_uint64_bytes':k*r.scalar_count*8,
            'crs_random_polynomials_per_binding':p.kappa1*(columns-p.kappa1),
            'crs_random_polynomials_per_messages':(self.message_count+3)*p.kappa2,
            'research_full_transcript_bytes':8+len(self.ring.zero().to_bytes())*(
                p.kappa1+self.message_count+1+k*p.kappa1+4+k+k*columns),
            'crs_is_supplied':crs is not None,
            'source_conflicts':len(r.layout.conflicts),
            'full_proof_executed':False}


class _UniformChallengeVector:
    def __init__(self,stream,bound,count,bulk=None):
        self.stream,self.bound,self.remaining=stream,bound,count
        self.bulk=bulk
    def __iter__(self):return self
    def __next__(self):
        if not self.remaining:raise StopIteration
        value=self.stream.uniform(self.bound);self.remaining-=1
        return value
    def read(self,count):
        if type(count) is not int or not 0<=count<=self.remaining:
            raise ValueError('gamma read exceeds remaining challenge vector')
        out=np.empty(count,dtype=np.uint64)
        for start in range(0,count,1<<18):
            part=out[start:start+(1<<18)]
            if self.bulk is None:
                part[:]=self.stream.uniform_many(self.bound,len(part))
            else:
                _uniform_challenge_slots(self.stream,self.bulk,part)
        self.remaining-=count
        return out


class _PolynomialChallengeVector:
    def __init__(self,parent,count,ring):
        self.parent,self.remaining,self.ring=parent,count,ring
        if not count:parent.phase='phi'
    def __iter__(self):return self
    def _consumed(self,count):
        self.remaining-=count
        if not self.remaining:self.parent.phase='phi'
    def __next__(self):
        if not self.remaining:raise StopIteration
        value=self.parent.stream.poly(self.ring);self._consumed(1)
        return value
    def read_slots(self,count,bulk):
        return self._read(count,bulk,transform=True)
    def read_coefficients(self,count,bulk):
        return self._read(count,bulk,transform=False)
    def _read(self,count,bulk,*,transform):
        if type(count) is not int or not 0<=count<=self.remaining or count>4096:
            raise ValueError('alpha read exceeds remaining vector or batch limit')
        if bulk.ring!=self.ring:raise ValueError('wrong alpha ring')
        result=np.empty((count,self.ring.n),dtype=np.uint64)
        _uniform_challenge_slots(self.parent.stream,bulk,result,transform=transform)
        self._consumed(count)
        return result


def _uniform_challenge_slots(stream,bulk,output,*,transform=False):

    width=(bulk.ring.q.bit_length()+7)//8;produced=0;count=output.size
    while produced<count:
        raw=stream.read_view((count-produced)*width)
        produced=bulk._uniform_challenge_into(raw,produced,output,transform=transform)
    return output


def _challenge_array(iterator,count,modulus,dtype=np.int64):
    reader=getattr(iterator,'read',None)
    if callable(reader):
        values=np.asarray(reader(count))
        if values.dtype==np.dtype(np.uint64) and np.dtype(dtype)==np.dtype(np.int64):
            return values.view(np.int64)
        return np.asarray(values,dtype=dtype)
    return np.fromiter((int(x)%modulus for x in islice(iterator,count)),dtype=dtype,count=count)


class FirstChallengeStream:

    def __init__(self,relation,first):
        self.relation=relation;self.phase='gamma';self.gamma_index=0
        self.stream=_XOF(hashlib.sha256(b'LNS20-round-1-v1'+relation.context()+_frame(first)).digest())
    def gamma(self,*,bulk=None):
        if self.phase!='gamma':raise RuntimeError('gamma phase already consumed')
        if bulk is not None and bulk.ring.q!=self.relation.parameters.modulus:
            raise ValueError('wrong gamma modulus')
        self.phase='gamma_active'
        p=self.relation.parameters
        for mu in range(p.automorphisms):
            self.gamma_index=mu
            values=_UniformChallengeVector(self.stream,p.modulus,len(self.relation.linear),bulk)
            yield mu,values
            if values.remaining:
                raise RuntimeError('each gamma vector must be exhausted before advancing')
        self.phase='alpha'
    def alpha(self):
        if self.phase!='alpha':raise RuntimeError('gamma phase must be exhausted first')
        self.phase='alpha_active'
        ring=Ring(self.relation.parameters.degree,self.relation.parameters.modulus)
        return _PolynomialChallengeVector(self,len(self.relation.polynomial),ring)
    def phi(self):
        if self.phase!='phi':raise RuntimeError('alpha phase must be exhausted first')
        p=self.relation.parameters;ring=Ring(p.degree,p.modulus)
        result=tuple(self.stream.poly(ring) for _ in range(p.automorphisms))
        self.phase='done'
        return result


def _transpose_rot_matrix(matrix,gamma,ring):

    from .ring_native import product
    coefficients=np.asarray(matrix,dtype=np.int64)
    if coefficients.ndim!=3 or coefficients.shape[2]!=ring.n:
        raise ValueError('adjoint public matrix shape mismatch')
    adjoint=coefficients.copy()
    adjoint[:,:,1:]=(-coefficients[:,:,1:][:,:,::-1])%ring.q
    values=np.asarray(gamma,dtype=np.int64)
    if values.shape!=(coefficients.shape[0],ring.n):
        raise ValueError('adjoint gamma shape mismatch')
    return product(adjoint,values[:,None,:],ring.q,transpose_left=True)[:,0,:]


def _slot_product_mod(left,right,modulus):
    if not 1 < modulus < 1<<33:
        raise ValueError('slot multiplication supports moduli below 2^33')
    a,b=np.asarray(left,dtype=np.uint64),np.asarray(right,dtype=np.uint64)
    if modulus <= 1<<32:
        return (a*b)%modulus
    low=(a&65535)*b
    high=((a>>16)*b)%modulus
    return (low+high*65536)%modulus


class PreparedLinearOperators:

    def __init__(self,relation):
        self.relation=relation;self.matrices={};self._transformed={}
        n=relation.n;ij=np.arange(n)[:,None]+np.arange(n)[None,:]
        self.indices=ij%n;self.sign=np.where(ij<n,1,-1)
        self.indices.flags.writeable=False;self.sign.flags.writeable=False
    def _prepare_matrix(self,key,matrix):
        if key not in self.matrices:
            ring=self.relation.ring
            coefficients=np.asarray(matrix,dtype=np.int64)
            if coefficients.ndim!=3 or coefficients.shape[2]!=ring.n:
                raise ValueError('adjoint public matrix shape mismatch')
            adjoint=coefficients.copy()
            adjoint[:,:,1:]=(-coefficients[:,:,1:][:,:,::-1])%ring.q
            if ring.n==128 and ring.q==4294967969:
                from .ring_native import _PreparedLeft
                prepared=_PreparedLeft(adjoint)
                self.matrices[key]=prepared.values
                self._transformed[key]=prepared
            else:
                self.matrices[key]=np.frombuffer(adjoint.tobytes(),dtype=np.int64).reshape(adjoint.shape)
    def prepare(self):

        C=self.relation.coefficient
        matrices=[('A',C('A'))]
        matrices.extend((('A_tags',i),matrix) for i,matrix in enumerate(C('A_tags')))
        matrices.extend((name,C(name)) for name in ('B0','B1','B2'))
        matrices.extend((('H1_transposed',tuple(zip(*C('H1')))),
                         ('A_oa_transposed',tuple(zip(*C('A_oa')))),
                         ('H2',(C('H2'),)),('AR',(C('AR'),))))
        for key,matrix in matrices:self._prepare_matrix(key,matrix)
        return self
    def transpose(self,key,matrix,gamma,ring):
        from .ring_native import product
        self._prepare_matrix(key,matrix)
        adjoint=self.matrices[key];values=np.asarray(gamma,dtype=np.int64)
        if values.shape!=(adjoint.shape[0],ring.n):raise ValueError('adjoint gamma shape mismatch')
        prepared=self._transformed.get(key)
        if prepared is not None:
            return prepared.multiply(values[:,None,:],transpose_left=True)[:,0,:]
        return product(adjoint,values[:,None,:],ring.q,transpose_left=True)[:,0,:]


class StructuredLinearAdjoint:

    def __init__(self,relation:StrictRelation,gamma,*,prepared=None):
        self.relation=relation;p,n=relation.params,relation.n;q=p.q
        if prepared is not None and prepared.relation is not relation:
            raise ValueError('linear preparation belongs to another relation')
        self.prepared=prepared
        max_weight=max(*g_beta(p.beta),*g_beta(p.B),*g_beta((q-1)//2),q//4)
        max_rows=max(n,p.alpha,p.alpha1,p.k,7*p.alpha1)
        if q*max(max_weight,max_rows) >= 1<<63:
            raise ValueError('structured adjoint integer intermediates exceed signed int64')
        self.ring=Ring(n,q);self.rows={};it=iter(gamma)
        for block in relation.layout.rows:
            if block.index==3:continue
            values=_challenge_array(it,block.scalar_length,q)
            if len(values)!=block.scalar_length:raise ValueError('too few gamma coefficients')
            self.rows[block.index]=values.reshape(-1,n)
        if next(it,None) is not None:raise ValueError('too many gamma coefficients')
        self.base=self._base_blocks()
        self._b_sources={}
        C=relation.coefficient
        for tag in range(p.d):
            self._b_sources[f'tag_product[{tag+1}]']=self._transpose(
                ('A_tags',tag+1),C('A_tags')[tag+1],self.rows[1])
        from .slot_native import dot_sums
        targets={row:np.asarray(C(name),dtype=np.uint64).reshape(-1,p.ell)
                 for row,name in ((1,'u'),(7,'c_rec1'),(9,'c_rec2'),
                                  (10,'uR'),(11,'c_oa1'),(12,'c_oa2'))}
        self.target=sum(int(dot_sums(self.rows[row].reshape(-1,p.ell),
                          (target,),q).sum(dtype=np.uint64))
                        for row,target in targets.items())%q

    def _transpose(self,key,matrix,gamma):
        if self.prepared is None:return _transpose_rot_matrix(matrix,gamma,self.ring)
        return self.prepared.transpose(key,matrix,gamma,self.ring)

    def _base_blocks(self):
        r=self.relation;p,n,q=r.params,r.n,r.modulus;C=r.coefficient;G=self.rows
        gb,ge,gh=g_beta(p.beta),g_beta(p.B),g_beta((q-1)//2)
        base={name:np.zeros(r.block(name).scalar_length,dtype=np.int64).reshape(-1,n)
              for name in _TERNARY_BLOCKS+('tag','A_user_column_major')}
        def add(name,poly,value):base[name][poly]=(base[name][poly]+value)%q
        def rotated(key,matrix,gamma,name,weights=(1,),sign=1):
            columns=self._transpose(key,matrix,gamma)
            weights=np.asarray(weights,dtype=np.int64)*sign
            values=(columns[:,None,:]*weights[None,:,None]%q).reshape(-1,n)
            base[name][:len(values)]=(base[name][:len(values)]+values)%q
        rotated('A',C('A'),G[1],'v1_digits',gb)
        rotated(('A_tags',0),C('A_tags')[0],G[1],'v2_digits',gb)
        copies=G[2].reshape(3*p.alpha*len(gb),n,n)
        base['v2_digits']=(base['v2_digits']+copies.sum(axis=2))%q
        rotated('B2',C('B2'),G[1],'w',sign=-1)
        add('A_user_column_major',(7*p.alpha1-1)*p.alpha1,G[4][0])
        base['h'][:len(gh)]=(base['h'][:len(gh)]+(-np.asarray(gh)[:,None]*G[4][0])%q)%q
        add('h',0,G[5][0]);add('h_first',0,-G[5][0]%q)
        rotated('B0',C('B0'),G[6],'r');rotated('B1',C('B1'),G[6],'h')
        gh=np.asarray(gh,dtype=np.int64);ge=np.asarray(ge,dtype=np.int64)
        w=base['w'].reshape(p.alpha,p.s,n)
        w[:,:len(gh)]=(w[:,:len(gh)]+(-G[6][:,None,:]*gh[None,:,None])%q)%q
        for name,row in (('y_rec_digits',7),('y_oa_digits',11)):
            base[name]=(G[row][:,None,:]*ge[None,:,None]%q).reshape(-1,n)
        copies=G[8].reshape(p.alpha1*len(ge),n,n)
        base['z_rec_digits']=(base['z_rec_digits']+copies.sum(axis=2))%q
        rotated('H1_transposed',tuple(zip(*C('H1'))),G[9],'z_rec_digits',ge)
        rotated('A_oa_transposed',tuple(zip(*C('A_oa'))),G[11],'z_oa_digits',ge)
        rotated('H2',(C('H2'),),G[12],'z_oa_digits',ge)
        base['x_rec_digits']=(G[9][:,None,:]*ge[None,:,None]%q).reshape(-1,n)
        base['message']=((q//4)*G[9])%q
        rotated('AR',(C('AR'),),G[10],'message')
        base['x_oa_digits']=(ge[:,None]*G[12][0])%q
        add('h_first',0,(q//4)*G[12][0]%q)
        return base

    def chunks(self,*,chunk_scalars=16384):
        r=self.relation;p,n,q=r.params,r.n,r.modulus;e=p.ell
        if type(chunk_scalars) is not int or chunk_scalars<e or chunk_scalars%e:
            raise ValueError('chunk size must be a positive multiple of ell')
        if self.prepared is None:
            ij=np.arange(n)[:,None]+np.arange(n)[None,:]
            indices=ij%n;sign=np.where(ij<n,1,-1)
        else:indices,sign=self.prepared.indices,self.prepared.sign
        for block in r.layout.witness:
            if block.name in self.base:
                source=self.base[block.name].reshape(-1)
                for start in range(0,len(source),chunk_scalars):
                    yield block.offset+start,source[start:start+chunk_scalars].copy()
            elif block.name in ('v2_coefficient_copies','z_rec_coefficient_copies'):
                source=self.rows[2 if block.name.startswith('v2') else 8].reshape(-1)
                for start in range(0,len(source),chunk_scalars):
                    yield block.offset+start,(-source[start:start+chunk_scalars])%q
            else:
                if block.name.startswith('tag_product['):
                    source=self._b_sources[block.name];weights=g_beta(p.beta)
                elif block.name=='recipient_products':
                    source=np.repeat(self.rows[7],p.alpha1,axis=0);weights=g_beta(p.B)
                else:raise ValueError('unknown strict block')
                for poly,values in enumerate(source):
                    for digit,weight in enumerate(weights):
                        data=(values[indices]*sign*weight%q).reshape(-1)
                        start=(poly*len(weights)+digit)*n*n
                        for local in range(0,len(data),chunk_scalars):
                            yield block.offset+start+local,data[local:local+chunk_scalars].copy()

    def packed(self):
        ring=Ring(self.relation.params.ell,self.relation.modulus)
        for _,data in self.chunks():
            for row in data.reshape(-1,ring.n):yield pack_slots(ring,row)

    def dot(self,messages):
        e,q=self.relation.params.ell,self.relation.modulus
        if len(messages)!=self.relation.scalar_count//e:raise ValueError('wrong message count')
        result=[0]*e
        for start,data in self.chunks():
            for index,row in enumerate(data.reshape(-1,e)):
                values=ntt_slots(messages[start//e+index])
                result=[(a+int(b)*int(c))%q for a,b,c in zip(result,row,values)]
        return pack_slots(Ring(e,q),result)


    def dot_slot_readers(self,readers):

        e,q=self.relation.params.ell,self.relation.modulus
        if q>=1<<33:raise ValueError('slot dot currently supports q below 2^33')
        readers=tuple(readers)
        from .slot_native import dot_sums
        totals=np.zeros((len(readers),e),dtype=np.uint64)
        for start,data in self.chunks():
            left=data.reshape(-1,e)
            rights=[]
            for reader in readers:
                right=np.asarray(reader(start,len(data)))
                if right.size!=len(data):
                    raise ValueError('slot reader must return the requested integer coordinates')
                right=right.reshape(-1,e)
                rights.append(right)
            totals=(totals+dot_sums(left,rights,q))%q
        return tuple(pack_slots(Ring(e,q),total) for total in totals)

    def dot_slot_reader(self,reader):
        return self.dot_slot_readers((reader,))[0]


def _ordered_jobs(jobs,operation,workers):

    if type(workers) is not int or not 1<=workers<=32:
        raise ValueError('aggregation worker count must be in 1..32')
    if workers==1:
        for job in jobs:yield operation(job)
        return
    jobs=iter(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=deque(pool.submit(operation,job) for job in islice(jobs,2*workers))
        while pending:
            yield pending.popleft().result()
            job=next(jobs,None)
            if job is not None:pending.append(pool.submit(operation,job))


def joint_adjoint_dots(adjoints,readers,*,workers=1):

    adjoints=tuple(adjoints);readers=tuple(readers)
    if not adjoints:raise ValueError('at least one adjoint required')
    r=adjoints[0].relation;e,q=r.params.ell,r.modulus
    if any(a.relation is not r for a in adjoints):raise ValueError('adjoint relation mismatch')
    if r.n<=1024:
        return _joint_adjoint_dots_compact(adjoints,readers,workers=workers)
    from .slot_native import multi_dot_sums
    totals=np.zeros((len(adjoints),len(readers),e),dtype=np.uint64)
    for chunks in zip(*(a.chunks() for a in adjoints),strict=True):
        start,data=chunks[0]
        if any(s!=start or len(values)!=len(data) for s,values in chunks[1:]):
            raise ValueError('adjoint chunk alignment mismatch')
        rights=[]
        for reader in readers:
            right=np.asarray(reader(start,len(data)))
            if right.size!=len(data):raise ValueError('slot reader size mismatch')
            rights.append(right.reshape(-1,e))
        totals=(totals+multi_dot_sums([values.reshape(-1,e) for _,values in chunks],rights,q))%q
    ring=Ring(e,q)
    return [tuple(pack_slots(ring,value) for value in row) for row in totals]


def _joint_adjoint_dots_compact(adjoints,readers,*,workers=1):

    from .slot_native import multi_dot_sums,adjoint_product_dot,adjoint_digit_dot
    from .streaming_zkp_backend import _ScratchPolynomialVector
    trusted=all(type(getattr(reader,'__self__',None)) is _ScratchPolynomialVector for reader in readers)
    r=adjoints[0].relation;p,n,q=r.params,r.n,r.modulus;e=p.ell
    totals=np.zeros((len(adjoints),len(readers),e),dtype=np.uint64)
    def rights(start,count):
        result=[]
        for reader in readers:
            value=np.asarray(reader(start,count))
            if value.size!=count:raise ValueError('slot reader size mismatch')
            if workers>1:
                owner_array=getattr(getattr(reader,'__self__',None),'array',None)
                if not isinstance(owner_array,np.ndarray) or not np.shares_memory(value,owner_array):
                    value=value.copy()
            result.append(value.reshape(-1,e))
        return result
    def jobs():
        for block in r.layout.witness:
            if block.name in adjoints[0].base:
                values=[a.base[block.name].reshape(-1) for a in adjoints]
                sign=1
            elif block.name in ('v2_coefficient_copies','z_rec_coefficient_copies'):
                row=2 if block.name.startswith('v2') else 8
                values=[a.rows[row].reshape(-1) for a in adjoints]
                sign=-1
            else:
                if block.name.startswith('tag_product['):
                    sources=[a._b_sources[block.name] for a in adjoints];weights=g_beta(p.beta)
                    count=len(sources[0]);source_span=1
                elif block.name=='recipient_products':
                    sources=[a.rows[7] for a in adjoints];weights=g_beta(p.B)
                    count=len(sources[0])*p.alpha1;source_span=p.alpha1
                else:raise ValueError('unknown strict block')
                group=(source_span if len(weights)*source_span<=64 and
                       len(weights)*source_span*n*n<=1<<20 else 1)
                packed_weights=np.tile(np.asarray(weights,dtype=np.uint64)%q,group)
                for poly in range(0,count,group):
                    source=[values[poly//source_span] for values in sources]
                    if len(packed_weights)<=64 and len(packed_weights)*n*n<=1<<20:
                        start=block.offset+poly*len(weights)*n*n
                        yield ('digits',source,rights(start,len(packed_weights)*n*n),packed_weights)
                    else:
                        for digit,weight in enumerate(weights):
                            start=block.offset+(poly*len(weights)+digit)*n*n
                            yield ('product',source,rights(start,n*n),int(weight)%q)
                continue
            for start in range(0,block.scalar_length,16384):
                count=min(16384,block.scalar_length-start)
                yield (sign,[v[start:start+count].reshape(-1,e) for v in values],
                       rights(block.offset+start,count),None)
    def calculate(job):
        kind,left,right,weight=job
        if kind=='digits':return 1,adjoint_digit_dot(left,right,e,q,weight,_trusted=trusted)
        if kind=='product':return 1,adjoint_product_dot(left,right,e,q,weight)
        return kind,multi_dot_sums(left,right,q,_trusted=trusted)
    for sign,sums in _ordered_jobs(jobs(),calculate,workers):
        totals=(totals+sums)%q if sign==1 else (totals+q-sums)%q
    ring=Ring(e,q)
    return [tuple(pack_slots(ring,value) for value in row) for row in totals]

