from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .ring import Ring,Poly


class DiskPolynomialVector:
    FORMAT='LatGE-disk-polynomial-vector-v1'
    def __init__(self,path,mode='r'):
        self.path=Path(path)
        metadata=json.loads(self.path.with_suffix(self.path.suffix+'.json').read_text())
        if metadata.get('format')!=self.FORMAT:raise ValueError('unknown disk vector format')
        self.ring=Ring(metadata['degree'],metadata['modulus'])
        self.domain=metadata['domain']
        if self.domain not in ('coefficients','slots'):raise ValueError('invalid disk vector domain')
        self.array=np.load(self.path,mmap_mode=mode,allow_pickle=False)
        if self.array.dtype!=np.dtype('<u8') or self.array.shape!=(metadata['length'],self.ring.n):
            raise ValueError('disk vector metadata/array mismatch')
        self.writable=mode=='r+'
        self._closed=False
    @classmethod
    def create(cls,path,ring,length,*,domain='coefficients'):
        path=Path(path)
        if type(length) is not int or length<=0 or domain not in ('coefficients','slots'):
            raise ValueError('positive length and known domain required')
        if not isinstance(ring,Ring) or not 2<ring.q<1<<63:raise ValueError('supported Ring required')
        metadata=path.with_suffix(path.suffix+'.json')
        if metadata.exists():raise FileExistsError(str(metadata))
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb'):pass
        try:
            array=np.lib.format.open_memmap(path,mode='w+',dtype='<u8',shape=(length,ring.n))
            array.flush();del array
            with metadata.open('x') as f:json.dump({'format':cls.FORMAT,'degree':ring.n,
                'modulus':ring.q,'length':length,'domain':domain},f,sort_keys=True)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return cls(path,'r+')
    def _ensure_open(self):
        if self._closed:raise RuntimeError('disk vector is closed')
    def __len__(self):
        self._ensure_open()
        return self.array.shape[0]
    @property
    def nbytes(self):
        self._ensure_open()
        return self.array.nbytes
    def _bounds(self,start,count):
        self._ensure_open()
        if type(start) is not int or type(count) is not int or start<0 or count<0 or start+count>len(self):
            raise IndexError('disk vector slice outside range')
    def read(self,start,count):
        self._bounds(start,count)
        result=self.array[start:start+count].view()
        if np.any(result>=self.ring.q):raise ValueError('noncanonical on-disk vector value')
        result.flags.writeable=False
        return result
    def write(self,start,values):
        self._ensure_open()
        if not self.writable:raise PermissionError('disk vector is read-only')
        values=np.asarray(values)
        if values.ndim!=2 or values.shape[1]!=self.ring.n or values.dtype.kind not in 'iu':
            raise ValueError('integer coefficient/slot rows required')
        self._bounds(start,len(values))
        if np.any(values<0) or np.any(values>=self.ring.q):raise ValueError('noncanonical vector value')
        self.array[start:start+len(values)]=values
    def __getitem__(self,index):
        self._bounds(index,1)
        row=self.read(index,1)[0]
        if self.domain=='coefficients':return Poly.from_ints(self.ring,row)
        from .paper_zkp_backend import pack_slots
        return pack_slots(self.ring,row)
    def slot_reader(self,start_scalar,count):
        self._ensure_open()
        if self.domain!='slots':raise ValueError('slot reader needs an NTT-slot vector')
        n=self.ring.n
        if start_scalar%n or count%n:raise ValueError('unaligned slot range')
        return self.read(start_scalar//n,count//n).reshape(-1)
    def flush(self):
        self._ensure_open()
        self.array.flush()
    def close(self):
        if self._closed:return
        try:
            self.array.flush()
        finally:
            self._closed=True
            self.array=None
    def __enter__(self):
        self._ensure_open()
        return self
    def __exit__(self,*_):self.close()
