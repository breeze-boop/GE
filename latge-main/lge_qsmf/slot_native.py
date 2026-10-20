from __future__ import annotations
import ctypes
import numpy as np
from .bulk_crs import _load_native, _pointer


def _shape(q,n):
    if type(q) is not int or not 2<q<1<<33 or not 4<=n<=256 or n&(n-1):
        raise ValueError('native slots require power-of-two degree4..256 and 2<q<2^33')


def _array(values,q,shape=None,*,trusted=False):
    a=np.asarray(values)
    if shape is not None and a.shape!=shape:raise ValueError('slot array shape mismatch')
    if a.dtype.kind not in 'iu':
        raise ValueError('native slots require canonical residues')
    if a.dtype==np.dtype(np.int64):a=a.view(np.uint64)
    a=np.ascontiguousarray(a,dtype=np.uint64)
    if not trusted and np.any(a>=q):raise ValueError('native slots require canonical residues')
    return a


def _check(code):
    if code:raise ValueError('native slot shape/parameter guard failed')


def collapse_product_rows(values,weights,n,q):
    _shape(q,n);values=_array(values,q);weights=_array(weights,q)
    if (values.ndim!=2 or values.shape[1]!=n or weights.ndim!=1
            or not 1<=len(weights)<=64 or len(values)%(len(weights)*n)):
        raise ValueError('product collapse dimensions disagree')
    polys=len(values)//(len(weights)*n);out=np.empty((polys,n),dtype=np.uint64)
    _check(_load_native()[0].latge_collapse_products(_pointer(values),_pointer(weights),
        polys,len(weights),n,q,_pointer(out)))
    return out


def check_polynomial_rows(kind,left,q,*,right=None,output=None,indicator=None):
    left=_array(left,q)
    if left.ndim!=2:raise ValueError('polynomial check requires packed rows')
    _shape(q,left.shape[1]);period=0
    if kind=='product':
        right=_array(right,q);output=_array(output,q,left.shape)
        if right.ndim!=2 or right.shape[1]!=left.shape[1] or not 0<len(right)<=len(left):
            raise ValueError('product right period disagrees')
        period=right.size
    elif kind=='tag':indicator=_array(indicator,2,left.shape)
    elif kind!='ternary':raise ValueError('unknown polynomial family')
    ptr=lambda a:None if a is None else _pointer(a)
    code=_load_native()[0].latge_check_witness_polynomial(('product','ternary','tag').index(kind),
        _pointer(left),ptr(right),ptr(output),ptr(indicator),left.size,period,q)
    if code<0:_check(code)
    return code==0


def dot_sums(left,rights,q):
    left=_array(left,q)
    if left.ndim!=2:raise ValueError('slot dot requires 2D rows')
    rows,n=left.shape;_shape(q,n)
    rights=tuple(rights)
    if rows>1<<20 or not 1<=len(rights)<=16:raise ValueError('native dot batch limit')
    arrays=tuple(_array(a,q,left.shape) for a in rights)
    pointer=ctypes.POINTER(ctypes.c_uint64)
    pointers=(pointer*len(arrays))(*(_pointer(a) for a in arrays))
    out=np.empty((len(arrays),n),dtype=np.uint64)
    _check(_load_native()[0].latge_slot_dot(_pointer(left),pointers,rows,n,q,len(arrays),_pointer(out)))
    return out


def multi_dot_sums(lefts,rights,q,*,_trusted=False):
    lefts=tuple(lefts);rights=tuple(rights)
    if not 1<=len(lefts)<=16 or not 1<=len(rights)<=16:raise ValueError('native dot batch limit')
    first=_array(lefts[0],q,trusted=_trusted)
    if first.ndim!=2:raise ValueError('slot dot requires 2D rows')
    rows,n=first.shape;_shape(q,n)
    if rows>1<<20:raise ValueError('native dot row limit')
    lefts=(first,)+tuple(_array(a,q,first.shape,trusted=_trusted) for a in lefts[1:])
    rights=tuple(_array(a,q,first.shape,trusted=_trusted) for a in rights)
    pointer=ctypes.POINTER(ctypes.c_uint64)
    lp=(pointer*len(lefts))(*(_pointer(a) for a in lefts))
    rp=(pointer*len(rights))(*(_pointer(a) for a in rights))
    out=np.empty((len(lefts),len(rights),n),dtype=np.uint64)
    code=_load_native()[0].latge_slot_multi_dot(lp,rp,rows,n,q,len(lefts),len(rights),_pointer(out))
    if code==-2:raise MemoryError('native joint dot allocation failed')
    _check(code);return out


def adjoint_product_dot(sources,rights,degree,q,weight):
    sources=tuple(sources);rights=tuple(rights)
    if not 1<=len(sources)<=16 or not 1<=len(rights)<=16:raise ValueError('native dot batch limit')
    first=_array(sources[0],q)
    if first.ndim!=1:raise ValueError('adjoint source must be one coefficient vector')
    n=len(first);_shape(q,degree)
    if not degree<=n<=1024 or n%degree or type(weight) is not int or not 0<=weight<q:
        raise ValueError('native adjoint shape/weight mismatch')
    sources=(first,)+tuple(_array(a,q,(n,)) for a in sources[1:])
    rights=tuple(_array(a,q,(n*n//degree,degree)) for a in rights)
    pointer=ctypes.POINTER(ctypes.c_uint64)
    sp=(pointer*len(sources))(*(_pointer(a) for a in sources))
    rp=(pointer*len(rights))(*(_pointer(a) for a in rights))
    out=np.empty((len(sources),len(rights),degree),dtype=np.uint64)
    code=_load_native()[0].latge_slot_adjoint_product_dot(sp,rp,n,degree,q,weight,
        len(sources),len(rights),_pointer(out))
    if code==-2:raise MemoryError('native adjoint dot allocation failed')
    _check(code);return out


def adjoint_digit_dot(sources,rights,degree,q,weights,*,_trusted=False):
    from .structured_native import native_library
    sources=tuple(sources);rights=tuple(rights)
    if not 1<=len(sources)<=16 or not 1<=len(rights)<=16:raise ValueError('native dot batch limit')
    first=_array(sources[0],q,trusted=_trusted)
    if first.ndim!=1:raise ValueError('adjoint source must be one coefficient vector')
    n=len(first);_shape(q,degree)
    weights=_array(weights,q)
    if (weights.ndim!=1 or not 1<=len(weights)<=64 or not degree<=n<=1024
            or n%degree or len(weights)*n*n>1<<20):
        raise ValueError('native adjoint digit batch limit')
    sources=(first,)+tuple(_array(a,q,(n,),trusted=_trusted) for a in sources[1:])
    rights=tuple(_array(a,q,(len(weights)*n*n//degree,degree),trusted=_trusted) for a in rights)
    pointer=ctypes.POINTER(ctypes.c_uint64)
    sp=(pointer*len(sources))(*(_pointer(a) for a in sources))
    rp=(pointer*len(rights))(*(_pointer(a) for a in rights))
    out=np.empty((len(sources),len(rights),degree),dtype=np.uint64)
    code=native_library().latge_adjoint_digit_dot(sp,rp,_pointer(weights),len(weights),
        n,degree,q,len(sources),len(rights),_pointer(out))
    if code==-2:raise MemoryError('native adjoint digit allocation failed')
    _check(code);return out


def _inputs(kind,alpha,q,values,indicator,*,trusted=False):
    if kind not in ('product','ternary','tag'):raise ValueError('unknown polynomial family')
    alpha=_array(alpha,q,trusted=trusted)
    if alpha.ndim!=2:raise ValueError('slot aggregation requires 2D rows')
    rows,n=alpha.shape;_shape(q,n)
    if rows>4096:raise ValueError('native polynomial batch limit')
    arrays=tuple(_array(a,q,alpha.shape,trusted=trusted) if a is not None else None for a in values)
    indicator=_array(indicator,2,alpha.shape) if indicator is not None else None
    if kind=='tag' and indicator is None:raise ValueError('tag indicator required')
    return ('product','ternary','tag').index(kind),alpha,arrays,indicator


def _ptr(a):return None if a is None else _pointer(a)


def expansion_sums(kind,alpha,message,mask,q,*,other_message=None,other_mask=None,output_mask=None,indicator=None):
    code,alpha,arrays,indicator=_inputs(kind,alpha,q,(message,mask,other_message,other_mask,output_mask),indicator)
    if arrays[0] is None or arrays[1] is None or (kind=='product' and any(x is None for x in arrays[2:])):
        raise ValueError('polynomial expansion operands required')
    rows,n=alpha.shape;out=np.empty((3,n),dtype=np.uint64)
    _check(_load_native()[0].latge_expansion_sums(code,_pointer(alpha),*(_ptr(a) for a in arrays),
        _ptr(indicator),rows,n,q,_pointer(out)))
    return out


def homogeneous_sums(kind,alpha,x,challenge,q,*,y=None,z=None,indicator=None):
    code,alpha,arrays,indicator=_inputs(kind,alpha,q,(x,y,z),indicator)
    if arrays[0] is None or (kind=='product' and any(a is None for a in arrays[1:])):
        raise ValueError('homogeneous operands required')
    rows,n=alpha.shape;challenge=_array(challenge,q,(n,));out=np.empty(n,dtype=np.uint64)
    _check(_load_native()[0].latge_homogeneous_sums(code,_pointer(alpha),*(_ptr(a) for a in arrays),
        _ptr(indicator),_pointer(challenge),rows,n,q,_pointer(out)))
    return out


def _lane_arrays(values,q,shape,lanes=None,*,trusted=False):
    values=tuple(values)
    if not 1<=len(values)<=16 or (lanes is not None and len(values)!=lanes):
        raise ValueError('native automorphism batch limit/mismatch')
    arrays=tuple(_array(a,q,shape,trusted=trusted) for a in values)
    pointer=ctypes.POINTER(ctypes.c_uint64)
    return arrays,(pointer*len(arrays))(*(_pointer(a) for a in arrays))


def multi_expansion_sums(kind,alpha,message,masks,q,*,other_message=None,other_masks=None,output_masks=None,indicator=None,_trusted=False):
    code,alpha,shared,indicator=_inputs(kind,alpha,q,(message,),indicator,trusted=_trusted)
    if shared[0] is None:raise ValueError('expansion messages required')
    rows,n=alpha.shape;arrays,mp=_lane_arrays(masks,q,alpha.shape,trusted=_trusted);lanes=len(arrays)
    other=output=other_message_array=None;op=zp=None;other_rows=rows
    if kind=='product':
        if other_message is None or other_masks is None or output_masks is None:raise ValueError('product operands required')
        other_message_array=_array(other_message,q,trusted=_trusted)
        if (other_message_array.ndim!=2 or other_message_array.shape[1]!=n
                or not (0<len(other_message_array)<=rows or rows==len(other_message_array)==0)):
            raise ValueError('product right-factor shape mismatch')
        other_rows=len(other_message_array)
        other,op=_lane_arrays(other_masks,q,(other_rows,n),lanes,trusted=_trusted)
        output,zp=_lane_arrays(output_masks,q,alpha.shape,lanes,trusted=_trusted)
    out=np.empty((lanes,3,n),dtype=np.uint64)
    _check(_load_native()[0].latge_multi_expansion_sums(code,_pointer(alpha),_pointer(shared[0]),
        _ptr(other_message_array),mp,op,zp,_ptr(indicator),rows,other_rows,n,q,lanes,_pointer(out)))
    return out


def multi_homogeneous_sums(kind,alpha,values,challenges,q,*,other_values=None,output_values=None,indicator=None,output_target=None,_trusted=False):
    code,alpha,_,indicator=_inputs(kind,alpha,q,(),indicator,trusted=_trusted)
    rows,n=alpha.shape;arrays,xp=_lane_arrays(values,q,alpha.shape,trusted=_trusted);lanes=len(arrays)
    challenges=_array(challenges,q,(lanes,n));other=output=None;yp=zp=None;other_rows=rows
    if kind=='product':
        if other_values is None or output_values is None:raise ValueError('homogeneous product operands required')
        other_values=tuple(other_values)
        if len(other_values)!=lanes:raise ValueError('native automorphism batch limit/mismatch')
        right_shape=np.asarray(other_values[0]).shape
        if (len(right_shape)!=2 or right_shape[1]!=n
                or not (0<right_shape[0]<=rows or rows==right_shape[0]==0)):
            raise ValueError('product right-factor shape mismatch')
        other_rows=right_shape[0]
        other,yp=_lane_arrays(other_values,q,right_shape,lanes,trusted=_trusted)
        output,zp=_lane_arrays(output_values,q,alpha.shape,lanes,trusted=_trusted)
    out=np.empty((lanes,n),dtype=np.uint64)
    arguments=(code,_pointer(alpha),xp,yp,zp,_ptr(indicator),_pointer(challenges),
               rows,other_rows,n,q,lanes,_pointer(out))
    if output_target is None:
        _check(_load_native()[0].latge_multi_homogeneous_sums(*arguments))
    else:
        if kind!='product':raise ValueError('affine output target requires product constraints')
        target=_array(output_target,q,alpha.shape,trusted=_trusted)
        _check(_load_native()[0].latge_multi_homogeneous_affine_outputs(*arguments,_pointer(target)))
    return out
