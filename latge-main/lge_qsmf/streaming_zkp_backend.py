from __future__ import annotations
from contextlib import contextmanager
from bisect import bisect_right
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import hashlib
import hmac
import io
import math
import os
import operator
from pathlib import Path
import shutil
import tempfile
import threading
import weakref
import numpy as np

from .bulk_crs import (BulkCRS, encode_coefficients, decode_coefficients,
    _expand_witness_into, pack_proof_coefficients, unpack_proof_coefficients,
    proof_row_bytes)
from .disk_vectors import DiskPolynomialVector
from .paper_zkp_backend import PaperProofResourceLimit, _XOF, _encode_polys, _monomial, _trace_polys, automorphism, ntt_slots, cubic_auxiliary_commitments
from .randomness import (_random_slots_into,_random_slot_entropy,
    _random_slots_from_entropy,uniform_mod_coeffs)
from .ring import Poly, Ring
from .proof_timing import excluded_from_prover_time, prover_timer
from .strict_relation import StrictRelation, StrictPrivateAssignment
from .strict_streaming import StreamingStrictRelation, FirstChallengeStream, StructuredLinearAdjoint, PreparedLinearOperators, joint_adjoint_dots, _ordered_jobs
from .strict_slot_ops import (check_witness_slots,aggregate_expansions_slots,
    aggregate_homogeneous_slots)


HEADER=b'LNS20C3\x01'


def _challenge_from_seed(ring,seed):
    stream=_XOF(seed)
    return Poly.from_ints(ring,[(b&1)-((b>>1)&1) for b in stream.bytes(ring.n)])


class StrictWitnessSlots:

    def __init__(self,relation,assignment):
        if not isinstance(relation,StrictRelation):raise TypeError('StrictRelation required')
        if len(assignment)!=relation.scalar_count:raise ValueError('wrong scalar assignment length')
        from .ring import Ring
        self.ring=Ring(relation.params.ell,relation.modulus)
        self.assignment=assignment;self.length=relation.scalar_count//self.ring.n
        self._batch_assignment=(type(assignment) is StrictPrivateAssignment
                                and assignment.relation is relation and self.ring.q<1<<33)
        self._direct_arrays={}
    def __len__(self):return self.length
    def read(self,start,count):
        if type(start) is not int or type(count) is not int or start<0 or count<0 or start+count>len(self):
            raise IndexError('private slot block out of range')
        e,q=self.ring.n,self.ring.q
        if getattr(self,'_batch_assignment',False):
            return self._read_assignment(start*e,count*e).reshape(count,e)
        return np.fromiter((operator.index(self.assignment[i])%q
            for i in range(start*e,(start+count)*e)),dtype=np.uint64,count=count*e).reshape(count,e)

    def _direct_array(self,name):
        source=self.assignment._direct[name]
        cached=self._direct_arrays.get(name)
        if cached is None or cached[0] is not source:
            array=np.fromiter((operator.index(x)%self.ring.q for x in source),
                              dtype=np.uint64,count=len(source))
            self._direct_arrays[name]=(source,array)
        return self._direct_arrays[name][1]

    def _read_assignment(self,start,count,out=None):
        assignment=self.assignment;relation=assignment.relation
        n,q,k1=relation.n,self.ring.q,relation.layout.k1
        result=np.empty(count,dtype=np.uint64) if out is None else out
        stop=start+count;written=0
        block_index=bisect_right(assignment._starts,start)-1
        while start<stop:
            block=relation.layout.witness[block_index]
            take=min(stop,block.offset+block.scalar_length)-start
            offset=start-block.offset
            target=result[written:written+take]
            if block.name in assignment._direct:
                target[:]=self._direct_array(block.name)[offset:offset+take]
            else:
                if block.name in ('v2_coefficient_copies','z_rec_coefficient_copies'):
                    name='v2_digits' if block.name.startswith('v2') else 'z_rec_digits'
                    kind,left,right=0,self._direct_array(name),None
                elif block.name.startswith('tag_product['):
                    tag=int(block.name.removeprefix('tag_product[').removesuffix(']'))-1
                    kind,left,right=1,self._direct_array('v2_digits'),self._direct_array('tag')[tag*n:(tag+1)*n]
                elif block.name=='recipient_products':
                    kind,left,right=2,self._direct_array('z_rec_digits'),self._direct_array('A_user_column_major')
                else:raise ValueError('unknown strict witness block')
                _expand_witness_into(kind,n,relation.params.alpha1,k1,offset,q,left,right,target)
            start+=take;written+=take;block_index+=1
        return result


class _BytesReader:

    def __init__(self,data):
        if not isinstance(data,bytes):raise TypeError('immutable bytes required')
        self.data=memoryview(data);self.position=0
    def read(self,size=-1):
        size=operator.index(size)
        stop=len(self.data) if size<0 else min(len(self.data),self.position+size)
        result=self.data[min(self.position,len(self.data)):stop]
        self.position=max(self.position,stop)
        return result
    def seek(self,offset,whence=0):
        offset=operator.index(offset);whence=operator.index(whence)
        if whence not in (0,1,2):raise ValueError('invalid whence')
        position=offset+(self.position if whence==1 else len(self.data) if whence==2 else 0)
        if position<0:raise ValueError('negative seek position')
        self.position=position
        return position
    def tell(self):return self.position


@contextmanager
def _open_transcript(source):
    if isinstance(source,bytes):
        reader=_BytesReader(source)
        try:yield reader
        finally:reader.data.release()
    else:
        with Path(source).open('rb') as reader:yield reader


def _update_file_frame(h,path,offset,length,chunk=1<<20):
    h.update(length.to_bytes(8,'little'))
    with _open_transcript(path) as source:
        source.seek(offset)
        while length:
            data=source.read(min(chunk,length))
            if not data:raise ValueError('truncated transcript frame')
            h.update(data);length-=len(data)


class _FileFirstChallenges(FirstChallengeStream):
    def __init__(self,relation,path,offset,length):
        self.relation=relation;self.phase='gamma';self.gamma_index=0
        h=hashlib.sha256(b'LNS20-round-1-v1'+relation.context())
        _update_file_frame(h,path,offset,length)
        self.stream=_XOF(h.digest())


class _FirstRoundHashes:

    def __init__(self,relation,length):
        self.relation=relation;self.length=length;self.seen=0
        frame=length.to_bytes(8,'little')
        self.first=hashlib.sha256(b'LNS20-round-1-v1'+relation.context()+frame)
        self.last=hashlib.sha256(b'LNS20-round-2-v1'+relation.context()+frame)
    def update(self,data):
        if self.seen+len(data)>self.length:raise ValueError('first-round frame overflow')
        self.first.update(data);self.last.update(data);self.seen+=len(data)
    def update_many(self,buffers,pool):

        buffers=tuple(buffers);size=sum(map(len,buffers))
        if self.seen+size>self.length:raise ValueError('first-round frame overflow')
        def update(state):
            for data in buffers:state.update(data)
        if size>=1<<20:
            for _ in pool.map(update,(self.first,self.last)):pass
        else:
            update(self.first);update(self.last)
        self.seen+=size
    def copy(self):
        result=object.__new__(type(self))
        result.relation=self.relation;result.length=self.length;result.seen=self.seen
        result.first=self.first.copy();result.last=self.last.copy()
        return result
    def _complete(self):
        if self.seen!=self.length:raise ValueError('incomplete first-round frame')
    def first_stream(self):
        self._complete()
        stream=object.__new__(_FileFirstChallenges)
        stream.relation=self.relation;stream.phase='gamma';stream.gamma_index=0
        stream.stream=_XOF(self.first.digest())
        return stream
    def challenge_seed(self,second):
        self._complete();h=self.last.copy()
        h.update(len(second).to_bytes(8,'little'));h.update(second)
        return h.digest()
    def challenge(self,second):
        return _challenge_from_seed(self.relation.ring,self.challenge_seed(second))


class _HashingIO:
    def __init__(self,handle,hashes):self.handle=handle;self.hashes=hashes
    def write(self,data):
        with excluded_from_prover_time():
            count=self.handle.write(data)
        self.hashes.update(memoryview(data)[:count]);return count
    def read(self,count=-1):
        data=self.handle.read(count);self.hashes.update(data);return data
    def write_many(self,buffers,pool):
        with excluded_from_prover_time():
            counts=[self.handle.write(data) for data in buffers]
        written=[memoryview(data)[:count] for data,count in zip(buffers,counts)]
        self.hashes.update_many(written,pool)


class _ScratchPolynomialVector(DiskPolynomialVector):

    @classmethod
    def create(cls,path,ring,length,*,domain='coefficients'):
        if type(length) is not int or length<=0 or domain not in ('coefficients','slots'):
            raise ValueError('positive length and known domain required')
        if not isinstance(ring,Ring) or not 2<ring.q<1<<63:
            raise ValueError('supported Ring required')
        vector=object.__new__(cls)
        vector.ring=ring;vector.domain=domain
        vector.array=np.empty((length,ring.n),dtype=np.uint64)
        vector.writable=True;vector._closed=False
        return vector
    def read(self,start,count):
        self._bounds(start,count)
        result=self.array[start:start+count].view()
        result.flags.writeable=False
        return result
    def _native_slot_view(self,start,count,*,writable=False):
        self._bounds(start,count)
        if self.domain!='slots':raise ValueError('native slot view requires slots')
        if writable:
            if not self.writable:raise PermissionError('scratch vector is read-only')
            return self.array[start:start+count]
        return self.read(start,count)
    def flush(self):
        self._ensure_open()
    def close(self):
        if self._closed:return
        self._closed=True
        self.array=None


class _Workspace:
    def __init__(self,path,ring):self.path=Path(path);self.ring=ring;self.vectors=[]
    def vector(self,name,length,domain='slots'):
        v=_ScratchPolynomialVector.create(self.path/(name+'.npy'),self.ring,length,domain=domain)
        self.vectors.append(v);return v
    def close(self):
        for v in reversed(self.vectors):v.close()


@contextmanager
def _workspace(parent,ring):
    with tempfile.TemporaryDirectory(prefix='latge-streaming-zkp-',dir=parent) as path:
        work=_Workspace(path,ring)
        try:yield work
        finally:work.close()


def _randomness_binding(relation):
    return hashlib.sha256(b'LatGE-single-use-prover-randomness-v1\x00'+
        relation.context()+relation.parameters.encode()+
        relation.parameters.commitment_crs.encode()).digest()


def _dispose_prepared_workspace(work,owner_pid):
    if os.getpid()!=owner_pid:return
    try:work.close()
    finally:shutil.rmtree(work.path,ignore_errors=True)


class _PreparedRandomness:

    def __init__(self,core,relation,bulk):
        self._lock=threading.Lock();self._state='preparing';self._owner=None
        self._pid=os.getpid();self._binding=_randomness_binding(relation)
        self.bulk=bulk;self.metrics={}
        self.work=_Workspace(tempfile.mkdtemp(prefix='latge-one-use-randomness-',
                                             dir=core.workspace_parent),relation.ring)
        self._dispose=weakref.finalize(self,_dispose_prepared_workspace,self.work,self._pid)
    def __copy__(self):raise TypeError('prepared randomness cannot be copied')
    def __deepcopy__(self,memo):raise TypeError('prepared randomness cannot be copied')
    def __reduce_ex__(self,protocol):raise TypeError('prepared randomness cannot be serialized')
    @property
    def closed(self):return self._state=='closed'
    def _bind(self,core):
        with self._lock:
            if os.getpid()!=self._pid or self._state!='ready':
                raise RuntimeError('prepared randomness is unavailable or already bound')
            self._owner=weakref.ref(core);self._state='bound'
    def _claim(self,core,relation):
        with self._lock:
            if os.getpid()!=self._pid or self._state!='bound' or self._owner() is not core:
                raise RuntimeError('prepared randomness is unavailable or already consumed')
            self._state='consumed'
        try:
            if self._binding!=_randomness_binding(relation):
                raise ValueError('prepared randomness belongs to another public proof context')
        except BaseException:
            self.close();raise
    def close(self):
        if os.getpid()!=self._pid:return
        with self._lock:self._state='closed'
        self._dispose()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()


@contextmanager
def _prover_workspace(parent,ring,prepared):
    if prepared is not None:yield prepared.work
    else:
        with _workspace(parent,ring) as work:yield work


@contextmanager
def _prover_file(path, mode):
    with excluded_from_prover_time():
        handle = path.open(mode)
    try:
        yield handle
    finally:
        with excluded_from_prover_time():
            handle.close()


def _write_coefficients(handle,array,ring):
    handle.write(encode_coefficients(array,ring))


def _read_coefficients(handle,count,ring):
    width=(ring.q.bit_length()+7)//8
    raw=handle.read(count*ring.n*width)
    if len(raw)!=count*ring.n*width:raise ValueError('truncated polynomial vector')
    return decode_coefficients(raw,count,ring)


def _write_commitment(handle,left,right,count,bulk,block,workers=1):
    if count>=65536:block=max(block,8192)
    def encode(start):
        rows=min(block,count-start)
        return bulk._encode_commitment_sum(left.read(start,rows),right.read(start,rows))
    if workers==1:
        for start in range(0,count,block):handle.write(encode(start))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for base in range(0,count,2*workers*block):
                encoded=list(pool.map(encode,range(base,min(count,base+2*workers*block),block)))
                handle.write_many(encoded,pool)


def _write_transcript(output,first_path,second,vectors,bulk,block,seed,bound):

    ring=bulk.ring;crs=bulk.crs;tile=max(block,8192)
    output.write(HEADER);output.write(seed)
    retained=crs.kappa1+crs.message_count+1
    with Path(first_path).open('rb') as first:
        for start in range(0,retained,tile):
            values=_read_coefficients(first,min(tile,retained-start),ring)
            output.write(pack_proof_coefficients(values,ring))
    step=len(ring.zero().to_bytes())
    values=decode_coefficients(second[:2*step]+second[-step:],3,ring)
    output.write(pack_proof_coefficients(values,ring))
    for vector in vectors:
        for start in range(0,len(vector),tile):
            values=vector.read(start,min(tile,len(vector)-start))
            if vector.domain=='slots':values=bulk.transform(values,inverse=True)
            output.write(pack_proof_coefficients(values,ring,bound))


def _publish_transcript(path,first_path,second,vectors,bulk,block,seed,bound):
    if path is None:
        with io.BytesIO() as output:
            _write_transcript(output,first_path,second,vectors,bulk,block,seed,bound)
            return output.getvalue()
    path=Path(path)
    with tempfile.TemporaryDirectory(prefix='.latge-proof-publish-',dir=path.parent) as directory:
        staged=Path(directory)/'proof.bin'
        with staged.open('xb') as output:
            _write_transcript(output,first_path,second,vectors,bulk,block,seed,bound)
        os.link(staged,path)
    return path


def _linear_aggregate(relation,stream,vectors,*,bulk=None,workers=1):
    operator=getattr(relation,'linear_aggregate',None)
    if callable(operator):return operator(stream,vectors)
    adjoints=[];targets=[];ring=relation.ring
    if not hasattr(relation,'_prepared_linear'):
        relation._prepared_linear=PreparedLinearOperators(relation.strict)
    for _,gamma in stream.gamma(bulk=bulk):
        adjoint=StructuredLinearAdjoint(relation.strict,gamma,prepared=relation._prepared_linear)
        adjoints.append(adjoint)
        targets.append(ring.one().scalar_mul(adjoint.target*pow(ring.n,-1,ring.q)%ring.q))
    return joint_adjoint_dots(adjoints,[v.slot_reader for v in vectors],workers=workers),targets


def _mixed_from_dots(dots,extra,k,ring,offset=1):
    result=[]
    for i in range(k):
        value=ring.zero()
        for mu in range(k):
            inner=sum((automorphism(dots[mu][offset+(i-nu)%k],nu,k) for nu in range(k)),ring.zero())
            value=value+_monomial(ring,mu)*inner
        result.append(value.scalar_mul(pow(k,-1,ring.q))+extra[i])
    return result


class StreamingLNS20Core:
    def __init__(self,*,workspace_parent=None,block_rows=1024,workers=1,max_attempts=128):
        if type(block_rows) is not int or not 1<=block_rows<=4096:raise ValueError('invalid block_rows')
        if type(workers) is not int or not 1<=workers<=32:raise ValueError('invalid workers')
        if type(max_attempts) is not int or max_attempts<1:raise ValueError('invalid max_attempts')
        self.workspace_parent=workspace_parent;self.block=block_rows
        self.workers=workers;self.max_attempts=max_attempts
        self._package_lock=threading.Lock();self._prepared_randomness=None
        self._outstanding_package=None;self._p_timer=None
    def prepare_randomness(self,relation):

        with excluded_from_prover_time():
            with self._package_lock:
                old=self._outstanding_package() if self._outstanding_package else None
                if self._prepared_randomness is not None or (old is not None and not old.closed):
                    raise RuntimeError('only one outstanding randomness package is permitted')
                bulk=self._prepare(relation)
                self._check_prover_cost(relation,bulk)
                package=_PreparedRandomness(self,relation,bulk)
                self._outstanding_package=weakref.ref(package)
        try:
            with excluded_from_prover_time():
                p=relation.parameters;k=p.automorphisms;work=package.work
                package.r=work.vector('r',bulk.crs.columns)
                package.y=[work.vector(f'y{i}',bulk.crs.columns) for i in range(k)]
                package.by=[work.vector(f'br{i}',bulk.crs.total_messages) for i in range(k+1)]
                package.binding=np.empty((k+1,bulk.crs.kappa1,bulk.ring.n),dtype=np.uint64)
            self._random_vector(package.r,bulk,'r',p.rejection_delta)
            package.metrics.update(random_vectors=1,
                storage_bytes=sum(v.nbytes for v in (package.r,*package.y,*package.by))+package.binding.nbytes,
                context_binding_sha256=package._binding.hex())
            with package._lock:package._state='ready'
            return package
        except BaseException:
            package.close();raise
    def bind_prepared_randomness(self,package):
        if type(package) is not _PreparedRandomness:raise TypeError('owned prepared randomness required')
        with self._package_lock:
            if self._prepared_randomness is not None:raise RuntimeError('a randomness package is already bound')
            package._bind(self);self._prepared_randomness=package
    def _take_prepared_randomness(self,relation):
        with self._package_lock:
            package=self._prepared_randomness;self._prepared_randomness=None
        if package is not None:package._claim(self,relation)
        return package
    def _prepare(self,relation):
        if not isinstance(relation,StreamingStrictRelation):
            raise TypeError('a locally compiled StreamingStrictRelation is required')
        p=relation.parameters;p.validate()
        crs=p.commitment_crs
        if (crs is None or crs.profile!=getattr(relation,'crs_profile','paper_iii_d')
                or crs.message_count!=relation.message_count):
            raise ValueError('streaming proof CRS does not match its locally compiled relation')
        return BulkCRS(crs)
    def _lengths(self,relation,bulk):
        p=relation.parameters;m=relation.message_count;k=p.automorphisms
        step=len(relation.ring.zero().to_bytes())
        first=step*(p.kappa1+m+1+k*p.kappa1)
        second=step*(4+k)
        compact=40+proof_row_bytes(relation.ring)*(p.kappa1+m+4)
        compact+=proof_row_bytes(relation.ring,p.rejection_delta-p.degree)*k*bulk.crs.columns
        return first,second,compact
    def _check_prover_cost(self,relation,bulk):
        p=relation.parameters;ring=relation.ring;delta=p.rejection_delta
        log_accept=p.automorphisms*ring.n*bulk.crs.columns*math.log(
            (2*(delta-ring.n)-1)/(2*delta-1))
        if log_accept < -20:
            raise PaperProofResourceLimit('configured rejection_delta makes reference acceptance probability below exp(-20)')
    def _ingest(self,source,destination,bulk):
        if len(source)!=len(destination):raise ValueError('wrong witness message count')
        if (type(source) is StrictWitnessSlots and source._batch_assignment
                and type(destination) is _ScratchPolynomialVector):
            if source.ring!=bulk.ring:raise ValueError('wrong witness ring')
            target=destination._native_slot_view(0,len(destination),writable=True)
            source._read_assignment(0,target.size,target.reshape(-1))
            return
        for start in range(0,len(destination),self.block):
            count=min(self.block,len(destination)-start)
            if isinstance(source,StrictWitnessSlots):
                if source.ring!=bulk.ring:raise ValueError('wrong witness ring')
                slots=source.read(start,count)
            elif isinstance(source,DiskPolynomialVector):
                if source.ring!=bulk.ring:raise ValueError('wrong witness ring')
                data=source.read(start,count)
                slots=data if source.domain=='slots' else bulk.transform(data)
            else:
                values=[source[start+i] for i in range(count)]
                if any(not isinstance(x,Poly) or x.ring!=bulk.ring for x in values):raise ValueError('wrong witness polynomial')
                slots=bulk.transform(np.array([x.coeffs for x in values],dtype=np.uint64))
            destination.write(start,slots)
    def _random_vector(self,vector,bulk,kind,delta):
        self._random_vectors(((kind,vector),),bulk,delta)
    def _random_vectors(self,vector_kinds,bulk,delta):

        vector_kinds=tuple(vector_kinds)
        for _,vector in vector_kinds:
            if not isinstance(vector,_ScratchPolynomialVector) or not vector.writable:
                raise TypeError('owned private scratch vector required')
            vector._ensure_open()
        parallel=self.workers>1 and any(len(v)>=65536 for _,v in vector_kinds)
        def tiles():
            for kind,vector in vector_kinds:
                block=max(self.block,8192) if self.workers>1 and len(vector)>=65536 else self.block
                for start in range(0,len(vector),block):
                    yield kind,vector.array[start:start+min(block,len(vector)-start)]
        if parallel:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                pending=deque()
                for kind,output in tiles():
                    if len(pending)>=2*self.workers:pending.popleft().result()
                    raw=_random_slot_entropy(kind,delta-1,output.size)
                    pending.append(pool.submit(_random_slots_from_entropy,
                        bulk,kind,delta-1,output,raw))
                for future in pending:future.result()
        else:
            for kind,output in tiles():
                _random_slots_into(bulk,kind,delta-1,output)
    def _responses(self,bulk,r,y,challenges,bound):

        parallel=self.workers>1 and len(r)>=65536
        block=max(self.block,4096) if parallel else self.block
        def calculate(start):
            count=min(block,len(r)-start)
            return bulk._responses_into(r.read(start,count),
                tuple(v.array[start:start+count] for v in y),challenges,bound)
        tiles=iter(range(0,len(r),block))
        if parallel:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                pending=deque()
                for _ in range(2*self.workers):
                    tile=next(tiles,None)
                    if tile is None:break
                    pending.append(pool.submit(calculate,tile))
                while pending:
                    if not pending.popleft().result():
                        for future in pending:future.cancel()
                        return False
                    tile=next(tiles,None)
                    if tile is not None:pending.append(pool.submit(calculate,tile))
        else:
            for tile in tiles:
                if not calculate(tile):return False
        for vector in y:vector.domain='coefficients'
        return True
    def prove(self,relation,messages):
        return self._run_prover(relation,messages,None)
    def prove_file(self,relation,messages,path):
        return self._run_prover(relation,messages,Path(path))
    def _run_prover(self,relation,messages,path):
        with excluded_from_prover_time():
            prepared=self._take_prepared_randomness(relation)
        try:return self._prove_file(relation,messages,path,prepared)
        finally:
            if prepared is not None:prepared.close()
    def _prove_file(self,relation,messages,path,prepared):
        with excluded_from_prover_time():
            bulk=prepared.bulk if prepared is not None else self._prepare(relation)
            p=relation.parameters;ring=relation.ring
            m,k,delta=relation.message_count,p.automorphisms,p.rejection_delta
            path=None if path is None else Path(path)
            if path is not None and path.exists():raise FileExistsError(str(path))
            self._check_prover_cost(relation,bulk)
        with _prover_workspace(self.workspace_parent,ring,prepared) as work:
            with excluded_from_prover_time():
                message=work.vector('witness',m)
            self._ingest(messages,message,bulk)
            with excluded_from_prover_time():
                check=getattr(relation,'check_witness_slots',None)
                valid=(check(message,bulk,batch_size=self.block) if callable(check)
                       else check_witness_slots(relation,message,batch_size=self.block))
                if not valid:
                    raise ValueError('witness does not satisfy public relation')
            if prepared is None:
                with excluded_from_prover_time():
                    r=work.vector('r',bulk.crs.columns)
                    y=[work.vector(f'y{i}',bulk.crs.columns) for i in range(k)]
                    by_all=[work.vector(f'br{i}',bulk.crs.total_messages) for i in range(k+1)]
                    binding=np.empty((k+1,bulk.crs.kappa1,ring.n),dtype=np.uint64)
                self._random_vector(r,bulk,'r',delta)
            else:
                r,y,by_all=prepared.r,prepared.y,prepared.by
                binding=prepared.binding
            br,*by=by_all
            first_path=work.path/'first.bin'
            first_len,_,_=self._lengths(relation,bulk)
            timer=self._p_timer or prover_timer()
            if timer is not None:timer.keep()
            for attempt in range(self.max_attempts):
                if attempt and timer is not None:timer.restart()
                for vector in y:vector.domain='slots'
                self._random_vectors((('y',v) for v in y),bulk,delta)
                binding[:]=bulk._project_vectors((r,*y),by_all,
                    block_rows=self.block,workers=self.workers)
                gcoeff=uniform_mod_coeffs(ring.n,ring.q);gcoeff[:k]=0
                g=Poly.from_ints(ring,gcoeff);tg=br[m]+g
                hashes=_FirstRoundHashes(relation,first_len)
                with _prover_file(first_path,'wb') as handle:
                    first=_HashingIO(handle,hashes)
                    _write_coefficients(first,binding[0],ring)
                    _write_commitment(first,br,message,m,bulk,self.block,self.workers)
                    first.write(tg.to_bytes())
                    _write_coefficients(first,binding[1:].reshape(-1,ring.n),ring)
                with excluded_from_prover_time():
                    if first_path.stat().st_size!=first_len:raise AssertionError('first-round encoding length')
                stream=hashes.first_stream()
                dots,targets=_linear_aggregate(relation,stream,(message,*by),bulk=bulk,workers=self.workers)
                expand=getattr(relation,'aggregate_expansions_slots',None)
                args=(by,message,stream.alpha(),stream.phi,bulk)
                psi0,psi1,psi2=(expand(*args,batch_size=self.block) if callable(expand)
                    else aggregate_expansions_slots(relation,*args,batch_size=4096,workers=self.workers))
                tp1,tp2,v=cubic_auxiliary_commitments(br[m+1],br[m+2],
                    by[0][m+1],by[0][m+2],(psi0,psi1,psi2))
                wprime=_mixed_from_dots(dots,[b[m] for b in by],k,ring)
                h=_trace_polys([row[0]-target for row,target in zip(dots,targets)],k)+g
                second=_encode_polys([tp1,tp2,v]+wprime+[h])
                c=hashes.challenge(second)
                z=y
                cs=np.asarray([ntt_slots(automorphism(c,i,k)) for i in range(k)],dtype=np.uint64)
                if not self._responses(bulk,r,y,cs,delta-ring.n):continue
                if timer is not None:timer.pause()
                for vector in (message,r,*by_all):vector.close()
                return _publish_transcript(path,first_path,second,z,bulk,self.block,
                    hashes.challenge_seed(second),delta-ring.n)
        raise RuntimeError('LNS20 rejection sampling exceeded its public attempt budget')
    def verify(self,relation,transcript):
        if not isinstance(transcript,bytes):return False
        bulk=self._prepare(relation)
        if len(transcript)!=self._lengths(relation,bulk)[2]:return False
        try:
            with _workspace(self.workspace_parent,relation.ring) as work:
                return self._verify_transcript(relation,transcript,bulk,work)
        except (ValueError,TypeError,IndexError,OverflowError):
            return False
    def verify_file(self,relation,path):
        bulk=self._prepare(relation);path=Path(path)
        expected=self._lengths(relation,bulk)[2]
        if path.stat().st_size!=expected:return False
        try:
            with _workspace(self.workspace_parent,relation.ring) as work:
                snapshot=work.path/'transcript.bin'
                with path.open('rb') as source,snapshot.open('xb') as destination:
                    remaining=expected
                    while remaining:
                        data=source.read(min(1<<20,remaining))
                        if not data:return False
                        destination.write(data);remaining-=len(data)
                    if source.read(1):return False
                return self._verify_transcript(relation,snapshot,bulk,work)
        except (ValueError,TypeError,IndexError,OverflowError):
            return False
    def _verify_transcript(self,relation,path,bulk,work):
        p=relation.parameters;ring=relation.ring;m,k=relation.message_count,p.automorphisms
        first_len,_,_=self._lengths(relation,bulk)
        t=work.vector('t',m)
        bz=[work.vector(f'bz{i}',bulk.crs.total_messages) for i in range(k)]
        hashes=_FirstRoundHashes(relation,first_len)
        with _open_transcript(path) as source:
            if source.read(8)!=HEADER:return False
            seed=bytes(source.read(32))
            if len(seed)!=32:return False
            c=_challenge_from_seed(ring,seed)
            challenges=[automorphism(c,i,k) for i in range(k)]
            public_row=proof_row_bytes(ring)
            def take(count):
                values=unpack_proof_coefficients(source.read(count*public_row),count,ring)
                return [Poly.from_ints(ring,row) for row in values]
            t0=take(p.kappa1);hashes.update(_encode_polys(t0))
            block=max(self.block,8192) if m>=65536 else self.block
            def commitment_tiles():
                for start in range(0,m,block):
                    count=min(block,m-start)
                    yield start,count,source.read(count*public_row)
            def decode_commitment(tile):
                start,count,raw=tile
                return bulk._unpack_transform_into(raw,t.array[start:start+count],hash_encoding=True)
            decoded=_ordered_jobs(commitment_tiles(),decode_commitment,self.workers)
            if self.workers>1 and m>=65536:
                with ThreadPoolExecutor(max_workers=2) as hash_pool:
                    while batch:=tuple(islice(decoded,2*self.workers)):
                        hashes.update_many(batch,hash_pool)
            else:
                for encoded in decoded:hashes.update(encoded)
            tg=take(1)[0];hashes.update(tg.to_bytes())
            tp1,tp2,h=take(3)
            if any(int(h.coeffs[j]) for j in range(k)):return False
            bound=p.rejection_delta-ring.n
            offset=source.tell();row_bytes=proof_row_bytes(ring,bound)
            stride=bulk.crs.columns*row_bytes
            memory=source.data if isinstance(source,_BytesReader) else None
            descriptor=None if memory is not None else source.fileno()
            def read_slots(start,count):
                result=np.empty((k,count,ring.n),dtype=np.uint64)
                if memory is not None:
                    begin=offset+start*row_bytes
                    size=(k-1)*stride+count*row_bytes
                    bulk._unpack_response_lanes_into(memory[begin:begin+size],stride,result,bound)
                    return result
                for i in range(k):
                    begin=offset+i*stride+start*row_bytes;size=count*row_bytes
                    raw=os.pread(descriptor,size,begin)
                    bulk._unpack_transform_into(raw,result[i],bound)
                return result
            compact=None if memory is None else (memory[offset:offset+k*stride],stride,bound)
            binding=bulk._project_responses(read_slots,bz,block_rows=self.block,
                                          workers=self.workers,compact_responses=compact)
        w=[[Poly.from_ints(ring,row)-challenges[i]*t0[j]
            for j,row in enumerate(binding[i])] for i in range(k)]
        hashes.update(_encode_polys([a for row in w for a in row]))
        fp1=bz[0][m+1]-c*tp1;fp2=bz[0][m+2]-c*tp2
        stream=hashes.first_stream()
        dots,targets=_linear_aggregate(relation,stream,(t,*bz),bulk=bulk,workers=self.workers)
        tau=_trace_polys([row[0]-u for row,u in zip(dots,targets)],k)
        actual=_mixed_from_dots(dots,[b[m] for b in bz],k,ring)
        f=bz
        homogeneous=getattr(relation,'aggregate_homogeneous_slots',None)
        cs=np.ascontiguousarray([ntt_slots(c) for c in challenges],dtype=np.uint64)
        block=max(self.block,8192) if m>=65536 else self.block
        if callable(homogeneous):spans=((0,m),)
        else:
            spans=((b.offset//ring.n,b.scalar_length//ring.n) for b in relation.strict.layout.witness
                   if not b.name.startswith('tag_product[') and b.name!='recipient_products')
        tiles=((base+start,min(block,count-start)) for base,count in spans for start in range(0,count,block))
        def subtract(tile):
            start,count=tile
            bulk._subtract_challenge_products_into(
                tuple(vector.array[start:start+count] for vector in f),cs,t.read(start,count))
        if self.workers>1 and m>=65536:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for _ in pool.map(subtract,tiles):pass
        else:
            for tile in tiles:subtract(tile)
        args=(f,challenges,stream.alpha(),stream.phi,bulk)
        lhs=(homogeneous(*args,batch_size=self.block) if callable(homogeneous)
             else aggregate_homogeneous_slots(relation,*args,batch_size=4096,workers=self.workers,output_target=t))
        t.close()
        v=fp2-c*fp1-lhs
        wprime=[actual[i]-challenges[i]*(tau+tg-h) for i in range(k)]
        second=_encode_polys([tp1,tp2,v]+wprime+[h])
        return hmac.compare_digest(hashes.challenge_seed(second),seed)
