from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import numpy as np

from .decomposition import g_beta
from .mp12_structured import negacyclic_product
from .paper_zkp_backend import automorphism,pack_slots,ntt_slots
from .strict_relation import _TERNARY_BLOCKS
from .strict_streaming import StreamingStrictRelation,_slot_product_mod,_ordered_jobs
from .slot_native import (multi_expansion_sums,multi_homogeneous_sums,
                         collapse_product_rows,check_polynomial_rows)


@dataclass(frozen=True)
class PolynomialBatch:
    kind: str
    left: range | np.ndarray
    right: range | np.ndarray | None = None
    output: range | np.ndarray | None = None
    indicator: np.ndarray | None = None
    def __len__(self):return len(self.left)


def _block_guard(size):
    if type(size) is not int or not 1<=size<=4096:
        raise ValueError('batch size must be an integer in 1..4096')


def _period_rows(base,start,count,period):
    count=min(count,period);offset=start%period
    if offset+count<=period:return range(base+offset,base+offset+count)
    return base+(np.arange(count,dtype=np.int64)+offset)%period


def polynomial_batches(relation,*,batch_size=1024):
    if not isinstance(relation,StreamingStrictRelation):raise TypeError('StreamingStrictRelation required')
    _block_guard(batch_size)
    r=relation.strict;p,n,e=r.params,r.n,relation.ring.n
    off={b.name:b.offset//e for b in r.layout.witness}
    nz=n//e
    count=3*p.alpha*r.layout.k0*n*n//e
    for tag in range(p.d):
        for start in range(0,count,batch_size):
            take=min(batch_size,count-start)
            yield PolynomialBatch('product',range(off['v2_coefficient_copies']+start,off['v2_coefficient_copies']+start+take),
                _period_rows(off['tag']+tag*nz,start,take,nz),
                range(off[f'tag_product[{tag+1}]']+start,off[f'tag_product[{tag+1}]']+start+take))
    count=7*p.alpha1*p.alpha1*r.layout.k1*n*n//e
    column_span=p.alpha1*r.layout.k1*n*n//e
    key_span=r.layout.k1*n*n//e
    start=0
    while start<count:
        take=min(batch_size,count-start,key_span-start%key_span)
        left=off['z_rec_coefficient_copies']+start%column_span
        output=off['recipient_products']+start
        yield PolynomialBatch('product',range(left,left+take),
            _period_rows(off['A_user_column_major']+(start//key_span)*nz,start,take,nz),
            range(output,output+take))
        start+=take
    for name in _TERNARY_BLOCKS:
        b=r.block(name);count=b.scalar_length//e
        for start in range(0,count,batch_size):
            yield PolynomialBatch('ternary',range(off[name]+start,off[name]+min(start+batch_size,count)))
    for tag in range(p.d):
        for start in range(0,nz,batch_size):
            t=np.arange(start,min(start+batch_size,nz),dtype=np.int64)
            position=t[:,None]*e+np.arange(e)[None,:]
            indicator=((position>=p.tag_boundaries[tag])&(position<p.tag_boundaries[tag+1])).astype(np.uint64)
            yield PolynomialBatch('tag',off['tag']+tag*nz+t,indicator=indicator)


def _validate_vector(relation,vector):
    if vector.ring!=relation.ring or len(vector)<relation.message_count:
        raise ValueError('slot vector ring or length does not match relation')
    if getattr(vector,'domain','slots')!='slots':raise ValueError('NTT slot vector required')


def _canonical(values,rows,ring):
    a=np.asarray(values)
    if a.shape!=(rows,ring.n) or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>=ring.q):
        raise ValueError('noncanonical slot block')
    return np.asarray(a,dtype=np.uint64)


def _index_plan(indices):
    if isinstance(indices,range) and indices.step==1:return len(indices),indices.start
    index_array=np.asarray(indices);count=index_array.size;start=None
    if (index_array.ndim==1 and index_array.dtype.kind in 'iu' and count
            and 0<=int(index_array[0])
            and int(index_array[-1])==int(index_array[0])+count-1
            and np.all(np.diff(index_array)==1)):
        start=int(index_array[0])
    return count,start


def _gather_values(vector,indices,ring,*,plan=None):
    if isinstance(indices,range) and indices.step==1 and hasattr(vector,'read'):
        return vector.read(indices.start,len(indices))
    if hasattr(vector,'read_indices'):
        values=vector.read_indices(indices)
    elif hasattr(vector,'array'):
        vector.read(0,0)
        count,start=_index_plan(indices) if plan is None else plan
        if start is not None and start+count<=len(vector.array):
            values=vector.array[start:start+count].view()
            values.flags.writeable=False
        else:
            values=vector.array[indices]
    else:
        values=np.empty((len(indices),ring.n),dtype=np.uint64)
        starts=np.r_[0,np.flatnonzero(np.diff(indices)!=1)+1,len(indices)]
        for a,b in zip(starts[:-1],starts[1:]):
            piece=_canonical(vector.read(int(indices[a]),int(b-a)),int(b-a),ring)
            values[a:b]=piece
    return values


def _gather_group(vectors,indices,ring,*,retain=False):
    plan=_index_plan(indices);result=[]
    for vector in vectors:
        values=_gather_values(vector,indices,ring,plan=plan)
        if retain and (hasattr(vector,'read_indices') or not hasattr(vector,'array')):
            values=np.array(values,copy=True)
        result.append(values)
    return result


def _gather(vector,indices,ring):
    return _canonical(_gather_values(vector,indices,ring),len(indices),ring)


def _product_right_indices(batch,period):

    right=batch.right
    if len(right)>period and np.array_equal(right[period:],right[:-period]):
        return right[:period]
    return right


def _alpha_slots(alpha,count,bulk):
    reader=getattr(alpha,'read_slots',None)
    if callable(reader):return reader(count,bulk)
    values=list(islice(alpha,count))
    if len(values)!=count:raise ValueError('too few alpha challenges')
    if any(getattr(a,'ring',None)!=bulk.ring for a in values):raise ValueError('wrong alpha ring')
    return bulk.transform(np.asarray([a.coeffs for a in values],dtype=np.uint64))


def _finish(locals_,phi,ring,k):
    phi=phi() if callable(phi) else tuple(phi)
    if len(phi)!=k or any(x.ring!=ring for x in phi):raise ValueError('wrong phi challenges')
    return tuple(sum((phi[i]*automorphism(pack_slots(ring,locals_[i,j]),-i,k)
                      for i in range(k)),ring.zero()) for j in range(locals_.shape[1]))


def aggregate_expansions_slots(relation,masks,messages,alpha,phi,bulk,*,batch_size=1024,workers=1):
    k=relation.parameters.automorphisms;ring=relation.ring;q=ring.q
    if bulk.ring!=ring or len(masks)!=k:raise ValueError('aggregation ring or mask count mismatch')
    for v in (messages,*masks):_validate_vector(relation,v)
    locals_=np.zeros((k,3,ring.n),dtype=np.uint64);it=iter(alpha)
    coefficients=getattr(it,'read_coefficients',None)
    from .streaming_zkp_backend import _ScratchPolynomialVector
    from .strict_streaming import _PolynomialChallengeVector
    trusted=(type(it) is _PolynomialChallengeVector and
             all(type(v) is _ScratchPolynomialVector for v in (messages,*masks)))
    def jobs():
        for batch in polynomial_batches(relation,batch_size=batch_size):
            a=(coefficients(len(batch),bulk) if callable(coefficients)
               else _alpha_slots(it,len(batch),bulk))
            left=_gather_group((messages,*masks),batch.left,ring,retain=workers>1)
            right=_product_right_indices(batch,relation.strict.n//ring.n) if batch.kind=='product' else None
            others=(_gather_group((messages,*masks),right,ring,retain=workers>1)
                    if right is not None else None)
            outputs=(_gather_group(masks,batch.output,ring,retain=workers>1)
                     if batch.kind=='product' else None)
            yield batch,a,left,others,outputs
    def calculate(job):
        batch,a,left,others,outputs=job
        if callable(coefficients):bulk._transform_canonical(a,out=a)
        return multi_expansion_sums(batch.kind,a,left[0],left[1:],q,
            other_message=others[0] if others is not None else None,
            other_masks=others[1:] if others is not None else None,
            output_masks=outputs,indicator=batch.indicator,_trusted=trusted)
    for sums in _ordered_jobs(jobs(),calculate,workers):
        locals_=(locals_+sums)%q
    if next(it,None) is not None:raise ValueError('too many alpha challenges')
    return _finish(locals_,phi,ring,k)


def aggregate_homogeneous_slots(relation,values,challenges,alpha,phi,bulk,*,batch_size=1024,workers=1,output_target=None):

    k=relation.parameters.automorphisms;ring=relation.ring;q=ring.q
    if bulk.ring!=ring or len(values)!=k or len(challenges)!=k:raise ValueError('aggregation public shape mismatch')
    for v in values:_validate_vector(relation,v)
    if output_target is not None:_validate_vector(relation,output_target)
    if any(c.ring!=ring for c in challenges):raise ValueError('wrong challenge ring')
    c=np.asarray([ntt_slots(x) for x in challenges],dtype=np.uint64)
    locals_=np.zeros((k,1,ring.n),dtype=np.uint64);it=iter(alpha)
    coefficients=getattr(it,'read_coefficients',None)
    from .streaming_zkp_backend import _ScratchPolynomialVector
    from .strict_streaming import _PolynomialChallengeVector
    trusted=(type(it) is _PolynomialChallengeVector and
             all(type(v) is _ScratchPolynomialVector for v in values) and
             (output_target is None or type(output_target) is _ScratchPolynomialVector))
    def jobs():
        for batch in polynomial_batches(relation,batch_size=batch_size):
            a=(coefficients(len(batch),bulk) if callable(coefficients)
               else _alpha_slots(it,len(batch),bulk))
            xs=_gather_group(values,batch.left,ring,retain=workers>1)
            right=_product_right_indices(batch,relation.strict.n//ring.n) if batch.kind=='product' else None
            ys=_gather_group(values,right,ring,retain=workers>1) if right is not None else None
            zs=(_gather_group(values,batch.output,ring,retain=workers>1)
                if batch.kind=='product' else None)
            target=(_gather_group((output_target,),batch.output,ring,retain=workers>1)[0]
                    if output_target is not None and batch.kind=='product' else None)
            yield batch,a,xs,ys,zs,target
    def calculate(job):
        batch,a,xs,ys,zs,target=job
        if callable(coefficients):bulk._transform_canonical(a,out=a)
        return multi_homogeneous_sums(batch.kind,a,xs,c,q,other_values=ys,
            output_values=zs,indicator=batch.indicator,output_target=target,_trusted=trusted)
    for sums in _ordered_jobs(jobs(),calculate,workers):
        locals_[:,0]=(locals_[:,0]+sums)%q
    if next(it,None) is not None:raise ValueError('too many alpha challenges')
    return _finish(locals_,phi,ring,k)[0]


class _Forward:
    def __init__(self,relation,vector):
        self.r=relation.strict;self.p=self.r.params;self.n=self.r.n
        self.e=relation.ring.n;self.q=relation.ring.q;self.vector=vector;self.ring=relation.ring
        _validate_vector(relation,vector)
        max_sum=max(self.n,self.p.alpha1,*(len(g_beta(bound)) for bound in
                    (self.p.beta,self.p.B,(self.q-1)//2)))
        if self.q>=1<<33 or max_sum*self.q>=1<<63:
            raise ValueError('forward operator exceeds exact integer guard')
        self.direct={}
    def read(self,name,start,count):
        b=self.r.block(name)
        if start<0 or count<0 or start+count>b.scalar_length or start%self.e or count%self.e:
            raise ValueError('unaligned forward block')
        return _canonical(self.vector.read((b.offset+start)//self.e,count//self.e),count//self.e,self.ring).reshape(-1)
    def block(self,name):
        if name not in self.direct:
            self.direct[name]=self.read(name,0,self.r.block(name).scalar_length).reshape(-1,self.n)
        return self.direct[name]
    def compose(self,name,weights):
        weights=tuple(int(weight)%self.q for weight in weights)
        if not weights or len(weights)*self.q>=1<<63:
            raise ValueError('composition weights exceed exact integer guard')
        value=self.block(name).reshape(-1,len(weights),self.n)
        return _slot_product_mod(value,np.asarray(weights,dtype=np.uint64)[None,:,None],self.q).sum(axis=1,dtype=np.uint64)%self.q
    def matvec(self,matrix,vector):
        out=np.zeros((len(matrix),self.n),dtype=np.uint64)
        for i,row in enumerate(matrix):
            if len(row)!=len(vector):raise ValueError('forward matrix dimension mismatch')
            for coefficient,value in zip(row,vector):
                product=negacyclic_product(tuple(int(x) for x in coefficient),tuple(int(x) for x in value))
                out[i]=(out[i]+np.asarray([int(x)%self.q for x in product],dtype=np.uint64))%self.q
        return out
    def collapse(self,name,count,weights):
        weights=tuple(int(weight)%self.q for weight in weights)
        out=collapse_product_rows(self.block(name),weights,self.n,self.q)
        if out.shape!=(count,self.n):raise ValueError('product block size disagrees')
        return out


def forward_linear_residuals(relation,vector,*,batch_size=1024):

    if not isinstance(relation,StreamingStrictRelation):raise TypeError('StreamingStrictRelation required')
    _block_guard(batch_size);f=_Forward(relation,vector);r,p,n,q=f.r,f.p,f.n,f.q
    C=r.coefficient;gb,ge,gh=g_beta(p.beta),g_beta(p.B),g_beta((q-1)//2)
    def target(row):
        name={1:'u',7:'c_rec1',9:'c_rec2',10:'uR',11:'c_oa1',12:'c_oa2'}.get(row)
        return np.asarray(C(name),dtype=np.uint64).reshape(-1) if name else 0
    for row in (1,2,4,5,6,7,8,9,10,11,12):
        if row in (2,8):
            original,copy=(('v2_digits','v2_coefficient_copies') if row==2 else ('z_rec_digits','z_rec_coefficient_copies'))
            base=f.block(original).reshape(-1)
            count=r.block(copy).scalar_length
            step=batch_size*f.e
            for start in range(0,count,step):
                size=min(step,count-start);positions=np.arange(start,start+size,dtype=np.int64)//n
                yield row,start,(base[positions]+q-f.read(copy,start,size))%q
            continue
        if row==1:
            value=(f.matvec(C('A'),f.compose('v1_digits',gb))+
                   f.matvec(C('A_tags')[0],f.compose('v2_digits',gb)))%q
            for tag in range(p.d):
                value=(value+f.matvec(C('A_tags')[tag+1],f.collapse(f'tag_product[{tag+1}]',3*p.alpha,gb)))%q
            value=(value+q-f.matvec(C('B2'),f.block('w')))%q
        elif row==4:
            value=(f.block('A_user_column_major')[(7*p.alpha1-1)*p.alpha1]+q-f.compose('h',gh)[0])%q
        elif row==5:value=(f.block('h')[0]+q-f.block('h_first')[0])%q
        elif row==6:
            value=(f.matvec(C('B0'),f.block('r'))+f.matvec(C('B1'),f.block('h'))+q-f.compose('w',gh))%q
        elif row==7:
            product=f.collapse('recipient_products',7*p.alpha1*p.alpha1,ge)
            value=(product.reshape(7*p.alpha1,p.alpha1,n).sum(axis=1,dtype=np.uint64)+f.compose('y_rec_digits',ge))%q
        elif row==9:
            value=(f.matvec(tuple(zip(*C('H1'))),f.compose('z_rec_digits',ge))+
                f.compose('x_rec_digits',ge)+_slot_product_mod(f.block('message'),np.uint64(q//4),q))%q
        elif row==10:value=f.matvec((C('AR'),),f.block('message'))
        elif row==11:
            value=(f.matvec(tuple(zip(*C('A_oa'))),f.compose('z_oa_digits',ge))+f.compose('y_oa_digits',ge))%q
        else:
            value=(f.matvec((C('H2'),),f.compose('z_oa_digits',ge))+f.compose('x_oa_digits',ge)+
                   _slot_product_mod(f.block('h_first'),np.uint64(q//4),q))%q
        residual=(value.reshape(-1)+q-target(row))%q
        for start in range(0,len(residual),batch_size*f.e):
            yield row,start,residual[start:start+batch_size*f.e]


def check_witness_slots(relation,vector,*,batch_size=1024):
    for _,_,residual in forward_linear_residuals(relation,vector,batch_size=batch_size):
        if np.any(residual):return False
    ring=relation.ring
    for batch in polynomial_batches(relation,batch_size=batch_size):
        left=_gather_values(vector,batch.left,ring)
        right=output=None
        if batch.kind=='product':
            indices=_product_right_indices(batch,relation.strict.n//ring.n)
            right=_gather_values(vector,indices,ring)
            output=_gather_values(vector,batch.output,ring)
        if not check_polynomial_rows(batch.kind,left,ring.q,right=right,output=output,indicator=batch.indicator):return False
    return True
