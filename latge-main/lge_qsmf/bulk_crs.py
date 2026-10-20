from __future__ import annotations
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import ctypes
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import weakref
import numpy as np

from .structured_crs import StructuredCRS,_ENTRY_DOMAIN,_frame
from .paper_zkp_backend import _roots


class NativeCRSUnavailable(RuntimeError):pass


class _ProjectionBlock(ctypes.Structure):
    _fields_=[('a1',ctypes.c_void_p),('h1',ctypes.c_void_p),
              ('a2',ctypes.c_void_p),('h2',ctypes.c_void_p),
              ('start',ctypes.c_uint64),('rows',ctypes.c_uint64)]


@lru_cache(maxsize=1)
def _load_native():
    source=Path(__file__).with_name('bulk_crs_native.c')
    compiler=shutil.which('clang') or shutil.which('cc')
    if compiler is None:raise NativeCRSUnavailable('a local C compiler is required for bulk CRS')
    private_build=tempfile.TemporaryDirectory(prefix='latge-bulk-crs-private-')
    output=Path(private_build.name)/('kernel.dylib' if sys.platform=='darwin' else 'kernel.so')
    command=[compiler,'-O3','-std=c11','-fPIC',
        '-dynamiclib' if sys.platform=='darwin' else '-shared',str(source),'-o',str(output)]
    result=subprocess.run(command,capture_output=True,text=True)
    if result.returncode:
        private_build.cleanup()
        raise NativeCRSUnavailable('bulk CRS compiler failed: '+result.stderr[-2000:])
    library=ctypes.CDLL(str(output))
    library._private_build=private_build
    u64=ctypes.c_uint64;size=ctypes.c_size_t;integer=ctypes.c_int
    pointer=ctypes.POINTER(u64);bytepointer=ctypes.c_void_p
    library.latge_shake256.argtypes=[bytepointer,size,bytepointer,size]
    library.latge_counter_xof.argtypes=[bytepointer,size,u64,u64,bytepointer]
    library.latge_expand_witness.argtypes=[integer,u64,u64,u64,u64,u64,u64,pointer,u64,pointer,u64,pointer]
    library.latge_collapse_products.argtypes=[pointer,pointer,u64,u64,integer,u64,pointer]
    library.latge_check_witness_polynomial.argtypes=[integer,pointer,pointer,pointer,pointer,u64,u64,u64]
    library.latge_transform.argtypes=[pointer,u64,integer,u64,u64,integer,pointer]
    library.latge_entries.argtypes=[bytepointer,size,u64,u64,u64,u64,integer,u64,pointer]
    library.latge_crs_dot.argtypes=[bytepointer,size,u64,u64,u64,u64,integer,u64,u64,pointer,u64,pointer]
    library.latge_crs_dot_identity.argtypes=library.latge_crs_dot.argtypes[:-1]+[pointer,pointer]
    library.latge_slot_dot.argtypes=[pointer,ctypes.POINTER(pointer),u64,integer,u64,u64,pointer]
    library.latge_slot_multi_dot.argtypes=[ctypes.POINTER(pointer),ctypes.POINTER(pointer),u64,integer,u64,u64,u64,pointer]
    library.latge_slot_adjoint_product_dot.argtypes=[ctypes.POINTER(pointer),ctypes.POINTER(pointer),integer,integer,u64,u64,u64,u64,pointer]
    library.latge_expansion_sums.argtypes=[integer]+[pointer]*7+[u64,integer,u64,pointer]
    library.latge_homogeneous_sums.argtypes=[integer]+[pointer]*6+[u64,integer,u64,pointer]
    pointers=ctypes.POINTER(pointer)
    library.latge_multi_expansion_sums.argtypes=[integer,pointer,pointer,pointer,pointers,pointers,pointers,pointer,u64,u64,integer,u64,u64,pointer]
    library.latge_multi_homogeneous_sums.argtypes=[integer,pointer,pointers,pointers,pointers,pointer,pointer,u64,u64,integer,u64,u64,pointer]
    library.latge_multi_homogeneous_affine_outputs.argtypes=library.latge_multi_homogeneous_sums.argtypes+[pointer]
    library.latge_encode_coefficients.argtypes=[pointer,u64,integer,u64,bytepointer]
    library.latge_decode_coefficients.argtypes=[bytepointer,u64,integer,u64,pointer]
    library.latge_pack_proof.argtypes=[pointer,u64,integer,u64,u64,bytepointer]
    library.latge_unpack_proof.argtypes=[bytepointer,u64,integer,u64,u64,pointer]
    library.latge_pack_proof.restype=integer
    library.latge_unpack_proof.restype=integer
    library.latge_original_transform.argtypes=[pointer,u64,integer,pointer]
    library.latge_original_transform.restype=integer
    library.latge_original_mul.argtypes=[pointer,pointer,u64,integer,integer,integer,pointer]
    library.latge_original_mul.restype=integer
    library.latge_original_dot.argtypes=[pointer,pointer,u64,u64,u64,pointer]
    library.latge_original_dot.restype=integer
    library.latge_context_create.argtypes=[integer,u64,u64,bytepointer,size,bytepointer,size]
    library.latge_context_create.restype=bytepointer
    library.latge_context_destroy.argtypes=[bytepointer]
    library.latge_context_destroy.restype=None
    library.latge_context_prepare_matrix.argtypes=[bytepointer,integer,u64,u64,u64,u64,bytepointer,bytepointer]
    library.latge_context_cached_a2_dot.argtypes=[bytepointer,bytepointer,bytepointer,u64,u64,pointer,u64,pointer,pointer]
    library.latge_context_transform.argtypes=[bytepointer,pointer,u64,integer,pointer]
    library.latge_unpack_transform.argtypes=[bytepointer,bytepointer,u64,integer,u64,pointer,bytepointer]
    library.latge_commitment_encode.argtypes=[bytepointer,pointer,pointer,u64,integer,bytepointer]
    library.latge_context_signed_transform.argtypes=[bytepointer,ctypes.POINTER(ctypes.c_int64),u64,pointer]
    library.latge_trinary_bytes_transform.argtypes=[bytepointer,bytepointer,u64,u64,pointer]
    library.latge_bounded_bytes_transform.argtypes=[bytepointer,bytepointer,u64,u64,u64,pointer,pointer]
    library.latge_uniform_challenge.argtypes=[bytepointer,bytepointer,u64,u64,integer,pointer,pointer]
    library.latge_context_dot.argtypes=[bytepointer,integer,u64,u64,u64,u64,pointer,u64,pointer,pointer]
    library.latge_context_dot_split.argtypes=[bytepointer,integer,u64,u64,u64,u64,pointers,u64,pointers,pointers]
    library.latge_context_cached_a2_dot_split.argtypes=[bytepointer,bytepointer,bytepointer,u64,u64,pointers,u64,pointers,pointers]
    library.latge_context_cached_project.argtypes=[bytepointer]*5+[u64]*3+[pointers,pointers,u64,pointer,pointers]
    library.latge_projection_pack_tail.argtypes=[pointers,u64,u64,bytepointer]
    library.latge_context_project_reuse.argtypes=[bytepointer]*5+[u64]*3+[
        pointers,pointers,u64,bytepointer,bytepointer,pointer,pointers,bytepointer,u64,integer,u64,u64,u64]
    library.latge_projection_group.argtypes=[bytepointer,ctypes.POINTER(_ProjectionBlock)]+[u64]*6+[
        pointers,pointers,pointers,bytepointer,bytepointer,pointer,pointer,bytepointer,u64,integer,u64]
    library.latge_unpack_transform_lanes.argtypes=[bytepointer,bytepointer,u64,u64,integer,u64,u64,pointer]
    library.latge_response.argtypes=[bytepointer,pointer,pointer,pointer,u64,u64,pointer]
    library.latge_responses.argtypes=[bytepointer,pointer,pointers,pointer,u64,u64,u64]
    library.latge_decode_response.argtypes=[bytepointer,bytepointer,u64,integer,u64,pointer]
    library.latge_decode_transform.argtypes=[bytepointer,bytepointer,u64,integer,pointer]
    library.latge_subtract_challenge_products.argtypes=[bytepointer,pointers,pointer,pointer,u64,u64]
    for name in ('latge_shake256','latge_counter_xof','latge_commitment_encode',
                 'latge_multi_homogeneous_affine_outputs','latge_expand_witness','latge_collapse_products','latge_check_witness_polynomial',
                 'latge_transform','latge_entries','latge_crs_dot','latge_crs_dot_identity',
                 'latge_slot_dot','latge_slot_multi_dot','latge_slot_adjoint_product_dot','latge_expansion_sums','latge_homogeneous_sums',
                 'latge_multi_expansion_sums','latge_multi_homogeneous_sums',
                 'latge_encode_coefficients','latge_decode_coefficients','latge_context_transform',
                 'latge_context_signed_transform','latge_context_dot','latge_response','latge_responses','latge_decode_response',
                 'latge_decode_transform','latge_unpack_transform',
                 'latge_context_prepare_matrix','latge_context_cached_a2_dot',
                 'latge_context_dot_split','latge_context_cached_a2_dot_split',
                 'latge_context_cached_project','latge_unpack_transform_lanes',
                 'latge_projection_pack_tail','latge_context_project_reuse',
                 'latge_projection_group',
                 'latge_trinary_bytes_transform','latge_bounded_bytes_transform','latge_uniform_challenge',
                 'latge_subtract_challenge_products'):
        getattr(library,name).restype=integer
    return library,str(output),hashlib.sha256(source.read_bytes()).hexdigest()


def _pointer(array):return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


@lru_cache(maxsize=1)
def _challenge_pool():
    return ThreadPoolExecutor(max_workers=4,thread_name_prefix='latge-challenge')


def _counter_xof_blocks(seed,counter,blocks):
    if not 0<=counter<1<<64 or blocks<1 or counter+blocks>1<<64:
        raise OverflowError('challenge block counter outside uint64')
    output=ctypes.create_string_buffer(blocks*4096)
    native=_load_native()[0].latge_counter_xof
    def generate(task):
        first,count=task
        if native(seed,len(seed),counter+first,count,ctypes.byref(output,first*4096)):
            raise ValueError('challenge block range mismatch')
    if blocks>=64:
        chunk=2*((blocks+7)//8)
        tasks=((first,min(chunk,blocks-first)) for first in range(0,blocks,chunk))
        for _ in _challenge_pool().map(generate,tasks):pass
    else:generate((0,blocks))
    return memoryview(output).cast('B').toreadonly()


def _expand_witness_into(kind,n,alpha1,digits,start,q,left,right,out):
    code=_load_native()[0].latge_expand_witness(kind,n,alpha1,digits,start,out.size,q,
        _pointer(left),left.size,None if right is None else _pointer(right),
        0 if right is None else right.size,_pointer(out))
    if code:raise ValueError('private witness expansion dimensions disagree')


def encode_coefficients(values,ring):
    values=np.asarray(values)
    if (values.ndim!=2 or values.shape[1]!=ring.n or values.dtype.kind not in 'iu'
            or (values.dtype.kind=='i' and np.any(values<0))):
        raise ValueError('canonical integer coefficient rows required')
    if not 2<=ring.q<1<<64:raise ValueError('unsupported coefficient modulus')
    data=np.ascontiguousarray(values,dtype=np.uint64)
    width=(ring.q.bit_length()+7)//8
    out=np.empty(data.size*width,dtype=np.uint8)
    code=_load_native()[0].latge_encode_coefficients(
        _pointer(data),data.size,width,ring.q,ctypes.c_void_p(out.ctypes.data))
    if code:raise ValueError('noncanonical coefficient vector')
    return memoryview(out)


def decode_coefficients(raw,count,ring):
    if type(count) is not int or count<0 or not 2<=ring.q<1<<64:
        raise ValueError('invalid coefficient count/modulus')
    width=(ring.q.bit_length()+7)//8
    octets=np.frombuffer(raw,dtype=np.uint8)
    if octets.size!=count*ring.n*width:raise ValueError('truncated polynomial vector')
    out=np.empty((count,ring.n),dtype=np.uint64)
    code=_load_native()[0].latge_decode_coefficients(
        ctypes.c_void_p(octets.ctypes.data),out.size,width,ring.q,_pointer(out))
    if code:raise ValueError('noncanonical proof coefficient')
    return out


def proof_row_bytes(ring, bound=0):
    bits=(2*bound-2).bit_length() if bound else (ring.q-1).bit_length()
    if ring.n*bits%8:
        raise ValueError('proof polynomial must occupy a whole number of bytes')
    return ring.n*bits//8


def pack_proof_coefficients(values,ring,bound=0):
    data=np.ascontiguousarray(values,dtype=np.uint64)
    if data.ndim!=2 or data.shape[1]!=ring.n:raise ValueError('wrong proof coefficient shape')
    bits=(2*bound-2).bit_length() if bound else (ring.q-1).bit_length()
    out=np.empty(len(data)*proof_row_bytes(ring,bound),dtype=np.uint8)
    code=_load_native()[0].latge_pack_proof(_pointer(data),data.size,bits,
        ring.q,bound,ctypes.c_void_p(out.ctypes.data))
    if code:raise ValueError('noncanonical or out-of-bound proof coefficient')
    return memoryview(out)


def unpack_proof_coefficients(raw,count,ring,bound=0):
    octets=np.frombuffer(raw,dtype=np.uint8)
    if count<0 or octets.size!=count*proof_row_bytes(ring,bound):
        raise ValueError('truncated packed proof coefficients')
    bits=(2*bound-2).bit_length() if bound else (ring.q-1).bit_length()
    out=np.empty((count,ring.n),dtype=np.uint64)
    code=_load_native()[0].latge_unpack_proof(ctypes.c_void_p(octets.ctypes.data),
        out.size,bits,ring.q,bound,_pointer(out))
    if code:raise ValueError('noncanonical or out-of-bound proof coefficient')
    return out


class BulkCRS:
    def __init__(self,crs:StructuredCRS):
        if not isinstance(crs,StructuredCRS):raise TypeError('StructuredCRS required')
        if not 4<=crs.ring.n<=256 or not crs.ring.q<1<<33:
            raise ValueError('native CRS supports degree4..256 and q<2^33')
        self.crs=crs;self.ring=crs.ring
        self.psi=_roots(self.ring.n,self.ring.q)[0]
        self.library,self.native_path,self.source_sha256=_load_native()
        self.prefix={kind:_ENTRY_DOMAIN+_frame(crs.encode())+_frame(kind.encode('ascii'))
                     for kind in ('A1','A2')}
        self._context=self.library.latge_context_create(self.ring.n,self.ring.q,self.psi,
            self.prefix['A1'],len(self.prefix['A1']),self.prefix['A2'],len(self.prefix['A2']))
        if not self._context:raise MemoryError('native CRS context allocation failed')
        self._release_context=weakref.finalize(self,self.library.latge_context_destroy,self._context)
        from .public_crs_preparation import _prepared_owner
        self._public_preparation=_prepared_owner(crs)
    def __copy__(self):
        return type(self)(self.crs)
    def _check(self,code):
        if code==-2:raise MemoryError('native CRS scratch allocation failed')
        if code:raise ValueError('native CRS shape/parameter guard failed')
    def shake256(self,data,count):
        out=ctypes.create_string_buffer(count)
        self._check(self.library.latge_shake256(data,len(data),out,count))
        return out.raw
    def transform(self,values,*,inverse=False):
        values=np.asarray(values)
        if values.ndim<2 or values.shape[-1]!=self.ring.n or values.dtype.kind not in 'iu':
            raise ValueError('transform needs canonical coefficient/slot rows')
        if (values.dtype.kind=='i' and np.any(values<0)) or np.any(values>=self.ring.q):
            raise ValueError('noncanonical transform input')
        data=np.ascontiguousarray(values,dtype=np.uint64)
        return self._transform_canonical(data,inverse=inverse)
    def _transform_canonical(self,data,*,inverse=False,out=None):
        if out is None:out=np.empty_like(data)
        self._check(self.library.latge_context_transform(self._context,_pointer(data),
            data.size//self.ring.n,int(inverse),_pointer(out)))
        return out
    def _unpack_transform_into(self,raw,out,bound=0,*,hash_encoding=False):
        data=np.frombuffer(raw,dtype=np.uint8)
        if data.size!=len(out)*proof_row_bytes(self.ring,bound):
            raise ValueError('truncated packed proof coefficients')
        bits=(2*bound-2).bit_length() if bound else (self.ring.q-1).bit_length()
        encoded=np.empty(out.size*((self.ring.q.bit_length()+7)//8),dtype=np.uint8) if hash_encoding else None
        self._check(self.library.latge_unpack_transform(self._context,
            ctypes.c_void_p(data.ctypes.data),len(out),bits,bound,_pointer(out),
            None if encoded is None else ctypes.c_void_p(encoded.ctypes.data)))
        return None if encoded is None else memoryview(encoded)
    def _unpack_response_lanes_into(self,raw,stride,out,bound):
        if (out.ndim!=3 or out.shape[2]!=16 or out.dtype!=np.dtype(np.uint64)
                or not out.flags.c_contiguous or not out.flags.writeable):
            raise ValueError('response lane destination shape/dtype mismatch')
        lanes,rows,_=out.shape
        size=(lanes-1)*stride+rows*proof_row_bytes(self.ring,bound)
        data=np.frombuffer(raw,dtype=np.uint8)
        if data.size!=size:raise ValueError('truncated packed response lanes')
        bits=(2*bound-2).bit_length()
        self._check(self.library.latge_unpack_transform_lanes(self._context,
            ctypes.c_void_p(data.ctypes.data),stride,rows,bits,bound,lanes,_pointer(out)))
    def _encode_commitment_sum(self,left,right):
        if (left.ndim!=2 or left.shape[1]!=self.ring.n or right.shape!=left.shape
                or any(a.dtype!=np.dtype(np.uint64) or not a.flags.c_contiguous for a in (left,right))):
            raise ValueError('commitment slot tile shape/dtype mismatch')
        width=(self.ring.q.bit_length()+7)//8
        output=np.empty(left.size*width,dtype=np.uint8)
        self._check(self.library.latge_commitment_encode(self._context,_pointer(left),
            _pointer(right),len(left),width,ctypes.c_void_p(output.ctypes.data)))
        return memoryview(output)
    def _signed_transform_into(self,data,out):

        data=np.asarray(data)
        if (data.dtype!=np.dtype(np.int64) or not data.flags.c_contiguous
                or data.ndim!=2 or data.shape[1]!=self.ring.n
                or out.shape!=data.shape or out.dtype!=np.dtype(np.uint64)
                or not out.flags.c_contiguous or not out.flags.writeable):
            raise ValueError('signed transform buffer shape/dtype mismatch')
        self._check(self.library.latge_context_signed_transform(self._context,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),len(data),_pointer(out)))
    def transform_vector(self,source,destination,*,block_rows=4096):
        self._validate_vectors((source,),len(source))
        self._validate_vectors((destination,),len(source))
        if source.domain==destination.domain:raise ValueError('transform needs opposite domains')
        self._block_guard(block_rows,65536)
        for start in range(0,len(source),block_rows):
            count=min(block_rows,len(source)-start)
            destination.write(start,self.transform(source.read(start,count),inverse=source.domain=='slots'))
    def _entropy_output(self,raw,output):
        if (not (isinstance(raw,bytes) or
                 isinstance(raw,ctypes.Array) and raw._type_ is ctypes.c_char)
                or not isinstance(output,np.ndarray)
                or output.ndim!=2 or output.shape[1]!=self.ring.n
                or output.dtype!=np.dtype(np.uint64) or not output.flags.c_contiguous
                or not output.flags.writeable):
            raise ValueError('entropy transform buffer shape/dtype mismatch')
    def _trinary_bytes_transform_into(self,raw,output):
        self._entropy_output(raw,output)
        self._check(self.library.latge_trinary_bytes_transform(self._context,raw,len(raw),len(output),_pointer(output)))
    def _bounded_bytes_transform_into(self,raw,bound,produced,output):
        self._entropy_output(raw,output)
        if (type(bound) is not int or not 0<=bound<=self.ring.q//2
                or type(produced) is not int or not 0<=produced<=output.size):
            raise ValueError('bounded entropy transform range mismatch')
        count=ctypes.c_uint64(produced)
        self._check(self.library.latge_bounded_bytes_transform(self._context,raw,len(raw),bound,
            len(output),ctypes.byref(count),_pointer(output)))
        return int(count.value)
    def _uniform_challenge_into(self,raw,produced,output,*,transform=False):
        output=np.asarray(output)
        data=np.frombuffer(raw,dtype=np.uint8)
        if (output.dtype!=np.dtype(np.uint64)
                or output.ndim not in (1,2) or not output.flags.c_contiguous
                or not output.flags.writeable or type(produced) is not int
                or not 0<=produced<=output.size
                or (transform and output.size%self.ring.n)):
            raise ValueError('uniform challenge destination/range mismatch')
        count=ctypes.c_uint64(produced)
        self._check(self.library.latge_uniform_challenge(self._context,ctypes.c_void_p(data.ctypes.data),data.size,
            output.size,int(transform),ctypes.byref(count),_pointer(output)))
        return int(count.value)
    def _response_into(self,r,y,challenge,bound,out):
        if (r.ndim!=2 or r.shape[1]!=self.ring.n or y.shape!=r.shape or out.shape!=r.shape
                or challenge.shape!=(self.ring.n,) or not out.flags.writeable
                or any(a.dtype!=np.dtype(np.uint64) or not a.flags.c_contiguous
                       for a in (r,y,challenge,out)) or type(bound) is not int or bound<=0):
            raise ValueError('response buffer shape/dtype/bound mismatch')
        code=self.library.latge_response(self._context,_pointer(r),_pointer(y),_pointer(challenge),
                                        len(r),bound,_pointer(out))
        if code==1:return False
        self._check(code);return True
    def _responses_into(self,r,masks,challenges,bound):
        masks=tuple(masks);lanes=len(masks)
        if (not 1<=lanes<=16 or type(bound) is not int or bound<=0
                or r.ndim!=2 or r.shape[1]!=self.ring.n
                or challenges.shape!=(lanes,self.ring.n)
                or any(a.shape!=r.shape or not a.flags.writeable for a in masks)
                or any(a.dtype!=np.dtype(np.uint64) or not a.flags.c_contiguous
                       for a in (r,*masks,challenges))):
            raise ValueError('response lane shape/dtype mismatch')
        pointers=(ctypes.POINTER(ctypes.c_uint64)*lanes)(*map(_pointer,masks))
        code=self.library.latge_responses(self._context,_pointer(r),pointers,
            _pointer(challenges),len(r),bound,lanes)
        if code<0:self._check(code)
        return code==0
    def _decode_response_into(self,raw,count,bound,out):
        width=(self.ring.q.bit_length()+7)//8
        octets=np.frombuffer(raw,dtype=np.uint8)
        if (type(count) is not int or count<0 or type(bound) is not int or bound<=0
                or octets.size!=count*self.ring.n*width or out.shape!=(count,self.ring.n)
                or out.dtype!=np.dtype(np.uint64) or not out.flags.c_contiguous or not out.flags.writeable):
            raise ValueError('response encoding/buffer mismatch')
        code=self.library.latge_decode_response(self._context,ctypes.c_void_p(octets.ctypes.data),
                                               count,width,bound,_pointer(out))
        if code==1:raise ValueError('response exceeds rejection bound')
        self._check(code)

    def _decode_slots_into(self,raw,count,out):
        octets=np.frombuffer(raw,dtype=np.uint8)
        width=(self.ring.q.bit_length()+7)//8
        if (type(count) is not int or count<0
                or octets.size!=count*self.ring.n*width or out.shape!=(count,self.ring.n)
                or out.dtype!=np.dtype(np.uint64) or not out.flags.c_contiguous or not out.flags.writeable):
            raise ValueError('coefficient encoding/buffer mismatch')
        self._check(self.library.latge_decode_transform(self._context,
            ctypes.c_void_p(octets.ctypes.data),count,width,_pointer(out)))

    def _subtract_challenge_products_into(self,values,challenges,target):

        values=tuple(values);batch=len(values)
        if (not 1<=batch<=16 or target.ndim!=2 or target.shape[1]!=self.ring.n
                or challenges.shape!=(batch,self.ring.n)
                or any(a.shape!=target.shape or not a.flags.writeable for a in values)
                or any(a.dtype!=np.dtype(np.uint64) or not a.flags.c_contiguous
                       for a in (*values,challenges,target))):
            raise ValueError('challenge subtraction buffer shape/dtype mismatch')
        pointers=(ctypes.POINTER(ctypes.c_uint64)*batch)(*map(_pointer,values))
        self._check(self.library.latge_subtract_challenge_products(self._context,
            pointers,_pointer(challenges),_pointer(target),len(target),batch))

    def _project_vectors(self,vectors,outputs,*,block_rows=1024,workers=1):

        vectors,outputs=tuple(vectors),tuple(outputs)
        self._validate_vectors(vectors,self.crs.columns)
        self._validate_vectors(outputs,self.crs.total_messages)
        if len(vectors)!=len(outputs):raise ValueError('one output per input vector required')
        def overlaps(a,b):
            if a is b:return True
            if hasattr(a,'path') and hasattr(b,'path'):
                if Path(a.path).samefile(Path(b.path)):return True
            return hasattr(a,'array') and hasattr(b,'array') and np.shares_memory(a.array,b.array)
        for i,output in enumerate(outputs):
            if any(overlaps(output,x) for x in outputs[:i]+vectors):
                raise ValueError('CRS outputs must not alias each other or an input')
        from .streaming_zkp_backend import _ScratchPolynomialVector
        native_inputs=all(type(v) is _ScratchPolynomialVector and v.domain=='slots' for v in vectors)
        if native_inputs:
            read_slots=lambda start,count:tuple(v._native_slot_view(start,count) for v in vectors)
        else:read_slots=lambda start,count:self._slot_block(vectors,start,count)
        return self._project_responses(read_slots,outputs,block_rows=block_rows,workers=workers,
                                       slot_inputs=vectors if native_inputs else None)

    def _project_responses(self,read_slots,outputs,*,block_rows=1024,workers=1,
                           compact_responses=None,slot_inputs=None):

        outputs=tuple(outputs);batch=len(outputs)
        self._validate_vectors(outputs,self.crs.total_messages)
        if any(v.domain!='slots' for v in outputs):raise ValueError('projection outputs need slots')
        if len({id(v) for v in outputs})!=batch:raise ValueError('projection outputs alias')
        self._block_guard(block_rows,4096);self._block_guard(workers,32)
        head=self.crs.kappa1;tail_start=head+self.crs.total_messages
        from .streaming_zkp_backend import _ScratchPolynomialVector
        direct=all(type(v) is _ScratchPolynomialVector for v in outputs)
        tail=tuple(read_slots(tail_start,self.crs.kappa2))
        owner=self._public_preparation
        if owner is not None:
            owner.prepare(self,[('A1',0,head,tail_start,self.crs.kappa2)])
        out=self._dot_lanes('A1',0,head,tail_start,tail)
        for lane,identity in enumerate(read_slots(0,head)):out[lane]+=identity
        out%=self.ring.q
        pointer_table=ctypes.POINTER(ctypes.c_uint64)*batch
        tail_pointers=pointer_table(*map(_pointer,tail))
        if owner is not None and direct:
            destinations=pointer_table(*(_pointer(v._native_slot_view(0,len(v),writable=True)) for v in outputs))
            source_pointers=None if slot_inputs is None else pointer_table(*(
                _pointer(v._native_slot_view(0,len(v))) for v in slot_inputs))
            packed_tail=np.empty(8+batch*self.crs.kappa2*130,dtype=np.uint8)
            packed_tail_pointer=ctypes.c_void_p(packed_tail.ctypes.data)
            self._check(self.library.latge_projection_pack_tail(
                tail_pointers,batch,self.crs.kappa2,packed_tail_pointer))
            local=threading.local()
            if compact_responses is not None:
                raw,stride,bound=compact_responses
                raw=np.frombuffer(raw,dtype=np.uint8)
                row_bytes=proof_row_bytes(self.ring,bound)
                if raw.size!=batch*stride or stride!=self.crs.columns*row_bytes:
                    raise ValueError('truncated compact response vector')
                bits=(2*bound-2).bit_length()
                raw_address=raw.ctypes.data
            else:raw_address=stride=bits=bound=0
        def calculate(start):
            count=min(block_rows,self.crs.total_messages-start)
            if owner is not None and direct:
                state=getattr(local,'projection',None)
                if state is None:
                    workspace=np.empty(batch*block_rows*128,dtype=np.uint8)
                    state=(workspace,ctypes.c_void_p(workspace.ctypes.data),pointer_table())
                    local.projection=state
                workspace,workspace_pointer,inputs=state
                input_row=0
                if not raw_address and source_pointers is not None:
                    inputs=source_pointers;input_row=head+start
                elif not raw_address:
                    values=tuple(read_slots(head+start,count))
                    if len(values)!=batch or any(a.shape!=(count,16) or
                            a.dtype!=np.dtype(np.uint64) or not a.flags.c_contiguous for a in values):
                        raise ValueError('CRS projection input tile shape/dtype mismatch')
                    for i,value in enumerate(values):inputs[i]=_pointer(value)
                a1,h1=owner._block(self.crs,'A1',0,head,head+start,count)
                a2,h2=owner._block(self.crs,'A2',start,count,tail_start,self.crs.kappa2)
                binding=np.empty((batch,head,16),dtype=np.uint64)
                self._check(self.library.latge_context_project_reuse(self._context,
                    ctypes.c_void_p(a1.ctypes.data),ctypes.c_void_p(h1.ctypes.data),
                    ctypes.c_void_p(a2.ctypes.data),ctypes.c_void_p(h2.ctypes.data),
                    head,count,self.crs.kappa2,inputs,tail_pointers,batch,
                    packed_tail_pointer,workspace_pointer,_pointer(binding),destinations,
                    ctypes.c_void_p(raw_address+(head+start)*row_bytes) if raw_address else None,
                    stride,bits,bound,input_row,start))
                return binding
            values=tuple(read_slots(head+start,count))
            binding=self._dot_lanes('A1',0,head,head+start,values)
            if direct:
                targets=tuple(v._native_slot_view(start,count,writable=True) for v in outputs)
                self._dot_lanes_into('A2',start,count,tail_start,tail,values,targets)
            else:
                messages=self._dot_lanes('A2',start,count,tail_start,tail,values)
                for i,destination in enumerate(outputs):destination.write(start,messages[i])
            return binding
        starts=iter(range(0,self.crs.total_messages,block_rows))
        def prepare(starts,pool=None):
            requests=[]
            for start in starts:
                count=min(block_rows,self.crs.total_messages-start)
                requests.extend((('A1',0,head,head+start,count),
                                 ('A2',start,count,tail_start,self.crs.kappa2)))
            owner.prepare(self,requests,pool)
        if owner is not None and direct and raw_address:
            wave_tiles=max(workers,2*workers*1024//block_rows)
            blocks=(_ProjectionBlock*wave_tiles)()
            partials=np.empty((workers,batch,head,16),dtype=np.uint64)
            partial_pointers=tuple(_pointer(partial) for partial in partials)
            raw_pointer=ctypes.c_void_p(raw_address) if raw_address else None
            def calculate_group(first):
                state=getattr(local,'group',None)
                if state is None:
                    scratch=np.empty(batch*block_rows*128,dtype=np.uint8)
                    binding=np.empty((batch,head,16),dtype=np.uint64)
                    state=(scratch,binding,ctypes.c_void_p(scratch.ctypes.data),_pointer(binding))
                    local.group=state
                self._check(self.library.latge_projection_group(self._context,
                    blocks,len(wave),first,groups,head,self.crs.kappa2,batch,
                    source_pointers,tail_pointers,destinations,packed_tail_pointer,
                    state[2],state[3],partial_pointers[first],raw_pointer,stride,bits,bound))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                while wave:=tuple(islice_compat(starts,wave_tiles)):
                    prepare(wave,pool)
                    for i,start in enumerate(wave):
                        count=min(block_rows,self.crs.total_messages-start)
                        a1,h1=owner._block(self.crs,'A1',0,head,head+start,count)
                        a2,h2=owner._block(self.crs,'A2',start,count,tail_start,self.crs.kappa2)
                        block=blocks[i]
                        block.a1=a1.ctypes.data;block.h1=h1.ctypes.data
                        block.a2=a2.ctypes.data;block.h2=h2.ctypes.data
                        block.start=start;block.rows=count
                    groups=min(workers,len(wave))
                    for _ in pool.map(calculate_group,range(groups)):pass
                    out+=partials[:groups].sum(axis=0,dtype=np.uint64)
                    np.remainder(out,self.ring.q,out=out)
            return self._transform_canonical(out,inverse=True)
        if workers==1:
            for start in starts:
                if owner is not None:prepare((start,))
                out=(out+calculate(start))%self.ring.q
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                if owner is not None:
                    wave_tiles=max(workers,2*workers*1024//block_rows)
                    while wave:=tuple(islice_compat(starts,wave_tiles)):
                        prepare(wave,pool)
                        for result in pool.map(calculate,wave):out=(out+result)%self.ring.q
                else:
                    pending=deque(pool.submit(calculate,start) for start in islice_compat(starts,2*workers))
                    while pending:
                        out=(out+pending.popleft().result())%self.ring.q
                        start=next(starts,None)
                        if start is not None:pending.append(pool.submit(calculate,start))
        return self._transform_canonical(out,inverse=True)
    def _block_guard(self,value,limit):
        if type(value) is not int or not 1<=value<=limit:raise ValueError('block size outside supported range')
    def _range(self,kind,row,rows,col,cols):
        if kind not in self.prefix:raise ValueError('matrix kind must be A1/A2')
        height=self.crs.kappa1 if kind=='A1' else self.crs.total_messages
        if any(type(x) is not int for x in (row,rows,col,cols)) or min(row,rows,col,cols)<0 or row+rows>height or col+cols>self.crs.columns:
            raise ValueError('CRS block outside public matrix')
        if rows*cols*self.ring.n>1<<20:raise ValueError('entry block exceeds bounded allocation')
    def entries(self,kind,row,rows,col,cols):
        self._range(kind,row,rows,col,cols)
        out=np.zeros((rows,cols,self.ring.n),dtype=np.uint64)
        boundary=self.crs.kappa1 if kind=='A1' else self.crs.kappa1+self.crs.total_messages
        for r in range(row,row+rows):
            identity=r if kind=='A1' else self.crs.kappa1+r
            if col<=identity<col+cols:out[r-row,identity-col,0]=1
        start=max(col,boundary)
        if start<col+cols and rows:
            random=np.empty((rows,col+cols-start,self.ring.n),dtype=np.uint64)
            prefix=self.prefix[kind]
            self._check(self.library.latge_entries(prefix,len(prefix),row,rows,start,col+cols-start,
                self.ring.n,self.ring.q,_pointer(random)))
            out[:,start-col:]=random
        return out
    def _validate_vectors(self,vectors,length):
        if not 1<=len(vectors)<=16:raise ValueError('one to sixteen vectors required')
        for vector in vectors:
            if vector.ring!=self.ring or len(vector)!=length or vector.domain not in ('coefficients','slots'):
                raise ValueError('vector public shape/ring/domain mismatch')
    def _slot_block(self,vectors,start,count):
        result=np.empty((len(vectors),count,self.ring.n),dtype=np.uint64)
        for i,vector in enumerate(vectors):
            data=np.asarray(vector.read(start,count))
            if (data.shape!=(count,self.ring.n) or data.dtype.kind not in 'iu'
                    or (data.dtype.kind=='i' and np.any(data<0)) or np.any(data>=self.ring.q)):
                raise ValueError('noncanonical vector block')
            if vector.domain=='slots':result[i]=data
            else:
                self._transform_canonical(np.ascontiguousarray(data,dtype=np.uint64),out=result[i])
        return result
    def _dot_lanes(self,kind,row,rows,col,values,identity=None):
        out=np.empty((len(values),rows,self.ring.n),dtype=np.uint64)
        self._dot_lanes_into(kind,row,rows,col,values,identity,tuple(out))
        return out
    def _dot_lanes_into(self,kind,row,rows,col,values,identity,outputs):
        batch=len(values);cols=len(values[0]);n=self.ring.n
        if (not 1<=batch<=16 or len(outputs)!=batch or
                any(a.shape!=(cols,n) for a in values) or
                any(a.shape!=(rows,n) or not a.flags.writeable for a in outputs) or
                (identity is not None and (len(identity)!=batch or
                    any(a.shape!=(rows,n) for a in identity)))):
            raise ValueError('CRS lane shape mismatch')
        arrays=tuple(values)+tuple(outputs)+(tuple(identity) if identity is not None else ())
        if any(a.dtype!=np.dtype(np.uint64) or not a.flags.c_contiguous for a in arrays):
            raise ValueError('CRS lanes require contiguous uint64 slots')
        pointer_table=ctypes.POINTER(ctypes.c_uint64)*batch
        inputs=pointer_table(*map(_pointer,values));targets=pointer_table(*map(_pointer,outputs))
        identities=None if identity is None else pointer_table(*map(_pointer,identity))
        owner=self._public_preparation
        if owner is not None:
            cached=owner._block(self.crs,kind,row,rows,col,cols)
            if cached is not None:
                low,high=cached
                self._check(self.library.latge_context_cached_a2_dot_split(self._context,
                    ctypes.c_void_p(low.ctypes.data),ctypes.c_void_p(high.ctypes.data),
                    rows,cols,inputs,batch,identities,targets))
                return
        self._check(self.library.latge_context_dot_split(self._context,int(kind=='A2'),
            row,rows,col,cols,inputs,batch,identities,targets))
    def _dot(self,kind,row,rows,col,slot_vectors):
        prefix=self.prefix[kind];batch,cols,n=slot_vectors.shape
        data=np.ascontiguousarray(slot_vectors,dtype=np.uint64)
        out=np.empty((batch,rows,n),dtype=np.uint64)
        if self._cached_a2_dot(kind,row,rows,col,data,None,out):return out
        self._check(self.library.latge_context_dot(self._context,int(kind=='A2'),
            row,rows,col,cols,_pointer(data),batch,None,_pointer(out)))
        return out
    def _dot_identity(self,kind,row,rows,col,slot_vectors,identity):
        prefix=self.prefix[kind];batch,cols,n=slot_vectors.shape
        data=np.ascontiguousarray(slot_vectors,dtype=np.uint64)
        identity=np.ascontiguousarray(identity,dtype=np.uint64)
        if identity.shape!=(batch,rows,n):raise ValueError('identity block shape mismatch')
        out=np.empty_like(identity)
        if self._cached_a2_dot(kind,row,rows,col,data,identity,out):return out
        self._check(self.library.latge_context_dot(self._context,int(kind=='A2'),
            row,rows,col,cols,_pointer(data),batch,_pointer(identity),_pointer(out)))
        return out
    def _cached_a2_dot(self,kind,row,rows,col,data,identity,out):
        owner=self._public_preparation
        if owner is None:return False
        cached=owner._block(self.crs,kind,row,rows,col,data.shape[1])
        if cached is None:return False
        low,high=cached
        self._check(self.library.latge_context_cached_a2_dot(self._context,
            ctypes.c_void_p(low.ctypes.data),ctypes.c_void_p(high.ctypes.data),rows,data.shape[1],
            _pointer(data),len(data),None if identity is None else _pointer(identity),_pointer(out)))
        return True
    def binding(self,vectors,*,block_columns=4096,output_domain='coefficients',workers=1):
        vectors=tuple(vectors);self._validate_vectors(vectors,self.crs.columns)
        self._block_guard(block_columns,65536);self._block_guard(workers,32)
        if self.crs.kappa1>4096:raise ValueError('binding rank exceeds native row-block limit')
        if output_domain not in ('coefficients','slots'):raise ValueError('unknown output domain')
        out=self._slot_block(vectors,0,self.crs.kappa1)
        def calculate(start):
            count=min(block_columns,self.crs.columns-start)
            return self._dot('A1',0,self.crs.kappa1,start,self._slot_block(vectors,start,count))
        starts=iter(range(self.crs.kappa1,self.crs.columns,block_columns))
        if workers==1:
            for start in starts:out=(out+calculate(start))%self.ring.q
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending=deque(pool.submit(calculate,start) for start in islice_compat(starts,2*workers))
                while pending:
                    out=(out+pending.popleft().result())%self.ring.q
                    start=next(starts,None)
                    if start is not None:pending.append(pool.submit(calculate,start))
        return out if output_domain=='slots' else self.transform(out,inverse=True)
    def messages(self,vectors,outputs,*,block_rows=256,workers=1):
        vectors,outputs=tuple(vectors),tuple(outputs)
        self._validate_vectors(vectors,self.crs.columns)
        self._validate_vectors(outputs,self.crs.total_messages)
        if len(vectors)!=len(outputs):raise ValueError('one output per input vector required')
        def overlaps(a,b):
            if a is b:return True
            if hasattr(a,'path') and hasattr(b,'path'):
                if Path(a.path).samefile(Path(b.path)):return True
            return hasattr(a,'array') and hasattr(b,'array') and np.shares_memory(a.array,b.array)
        for i,output in enumerate(outputs):
            if any(overlaps(output,x) for x in outputs[:i]+vectors):
                raise ValueError('CRS outputs must not alias each other or an input')
        self._block_guard(block_rows,4096);self._block_guard(workers,32)
        if self.crs.kappa2>65536:raise ValueError('random tail exceeds native column-block limit')
        tail_start=self.crs.kappa1+self.crs.total_messages
        tail=self._slot_block(vectors,tail_start,self.crs.kappa2)
        def calculate(start):
            count=min(block_rows,self.crs.total_messages-start)
            identity=self._slot_block(vectors,self.crs.kappa1+start,count)
            out=self._dot_identity('A2',start,count,tail_start,tail,identity)
            coefficients=None
            for i,destination in enumerate(outputs):
                if destination.domain=='coefficients':
                    if coefficients is None:coefficients=self._transform_canonical(out,inverse=True)
                    destination.write(start,coefficients[i])
                else:destination.write(start,out[i])
            return count
        starts=iter(range(0,self.crs.total_messages,block_rows))
        if workers==1:
            for start in starts:calculate(start)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending=deque()
                for start in islice_compat(starts,2*workers):pending.append(pool.submit(calculate,start))
                while pending:
                    pending.popleft().result()
                    start=next(starts,None)
                    if start is not None:pending.append(pool.submit(calculate,start))


def islice_compat(iterator,count):
    for _ in range(count):
        item=next(iterator,None)
        if item is None:return
        yield item
