from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


class NativeStructuredUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def native_library():
    compiler = shutil.which('clang') or shutil.which('cc')
    if compiler is None:
        raise NativeStructuredUnavailable('a local C compiler is required for exact sparse integer operators')
    directory = tempfile.TemporaryDirectory(prefix='latge-structured-private-')
    source = Path(__file__).with_suffix('.c')
    binary = Path(directory.name)/('kernel.dylib' if sys.platform == 'darwin' else 'kernel.so')
    result = subprocess.run([compiler, '-O3', '-std=c11', '-fPIC',
        '-dynamiclib' if sys.platform == 'darwin' else '-shared', str(source), '-o', str(binary)],
        capture_output=True, text=True)
    if result.returncode:
        directory.cleanup()
        raise NativeStructuredUnavailable(result.stderr[-2000:])
    library = ctypes.CDLL(str(binary))
    library._private_build = directory
    up = ctypes.POINTER(ctypes.c_uint64)
    ip = ctypes.POINTER(ctypes.c_int64)
    library.latge_sparse_integer_product.argtypes = [up,up,up,ip,
        ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,
        ip,ctypes.c_size_t,ip]
    library.latge_sparse_integer_product.restype = ctypes.c_int
    library.latge_adjoint_digit_dot.argtypes = [ctypes.POINTER(up),ctypes.POINTER(up),up,
        ctypes.c_size_t,ctypes.c_size_t,ctypes.c_size_t,ctypes.c_uint64,
        ctypes.c_size_t,ctypes.c_size_t,up]
    library.latge_adjoint_digit_dot.restype = ctypes.c_int
    return library


@dataclass(frozen=True,slots=True)
class SparseIntegerMatrix:
    source: tuple
    n: int
    rows: int
    cols: int
    offsets: np.ndarray
    columns: np.ndarray
    shifts: np.ndarray
    weights: np.ndarray
    row_bound: int

    @classmethod
    def from_matrix(cls,matrix,n):
        if (type(n) is not int or n<1 or type(matrix) is not tuple or not matrix
                or any(type(row) is not tuple for row in matrix) or not matrix[0]):
            return None
        cols=len(matrix[0])
        if any(len(row)!=cols for row in matrix):
            return None
        if any(type(poly) is not tuple or len(poly)!=n
               or any(type(x) is not int or abs(x)>=(1<<63) for x in poly)
               for row in matrix for poly in row):
            return None
        coefficients=np.asarray(matrix,dtype=np.int64)
        row_indices,columns,shifts=np.nonzero(coefficients)
        weights=coefficients[row_indices,columns,shifts]
        offsets=np.empty(len(matrix)+1,dtype=np.uint64)
        offsets[0]=0
        np.cumsum(np.bincount(row_indices,minlength=len(matrix)),out=offsets[1:])
        bounds=[sum(map(abs,weights[int(offsets[i]):int(offsets[i+1])].tolist()))
                for i in range(len(matrix))]
        return cls(matrix,n,len(matrix),cols,
            np.frombuffer(np.asarray(offsets,dtype=np.uint64).tobytes(),dtype=np.uint64),
            np.frombuffer(np.asarray(columns,dtype=np.uint64).tobytes(),dtype=np.uint64),
            np.frombuffer(np.asarray(shifts,dtype=np.uint64).tobytes(),dtype=np.uint64),
            np.frombuffer(np.asarray(weights,dtype=np.int64).tobytes(),dtype=np.int64),max(bounds))

    def adjoint(self):
        from .mp12_structured import adjoint_poly
        source=tuple(tuple(adjoint_poly(self.source[i][j]) for i in range(self.rows))
                     for j in range(self.cols))
        rows=self.columns.astype(np.int64)
        columns=np.repeat(np.arange(self.rows,dtype=np.uint64),
                          np.diff(self.offsets).astype(np.int64))
        shifts=(self.n-self.shifts)%self.n
        weights=np.where(self.shifts==0,self.weights,-self.weights)
        order=np.lexsort((shifts,columns,rows))
        columns,shifts,weights=columns[order],shifts[order],weights[order]
        offsets=np.empty(self.cols+1,dtype=np.uint64)
        offsets[0]=0
        np.cumsum(np.bincount(rows,minlength=self.cols),out=offsets[1:])
        bounds=[sum(map(abs,weights[int(offsets[i]):int(offsets[i+1])].tolist()))
                for i in range(self.cols)]
        return type(self)(source,self.n,self.cols,self.rows,
            np.frombuffer(offsets.tobytes(),dtype=np.uint64),
            np.frombuffer(columns.tobytes(),dtype=np.uint64),
            np.frombuffer(shifts.tobytes(),dtype=np.uint64),
            np.frombuffer(weights.tobytes(),dtype=np.int64),max(bounds))

    def apply_many(self,vectors):
        try:
            data=np.asarray(vectors)
        except (ValueError,OverflowError):
            return None
        if data.ndim!=2 or data.shape[1]!=self.cols*self.n:
            raise ValueError('structured operator dimension mismatch')
        if not data.shape[0]:
            return []
        if data.dtype.kind not in 'iu':
            return None
        magnitude=max(abs(int(data.min())),abs(int(data.max())))
        if magnitude*self.row_bound>=(1<<63) or magnitude>=(1<<63):
            return None
        if not self.row_bound:
            return [[0]*(self.rows*self.n) for _ in range(data.shape[0])]
        library=native_library()
        data=np.ascontiguousarray(data,dtype=np.int64)
        output=np.empty((len(data),self.rows*self.n),dtype=np.int64)
        up=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        ip=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        code=library.latge_sparse_integer_product(up(self.offsets),up(self.columns),
            up(self.shifts),ip(self.weights),self.rows,self.cols,self.n,len(self.weights),
            ip(data),len(data),ip(output))
        if code:
            raise ArithmeticError('native sparse integer operator invariant failed')
        return output.tolist()
