from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


class NativeRingUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load():
    compiler = shutil.which('clang') or shutil.which('cc')
    if compiler is None:
        raise NativeRingUnavailable('a local C compiler is required for the exact ring kernel')
    directory = tempfile.TemporaryDirectory(prefix='latge-ring-private-')
    source = Path(__file__).with_suffix('.c')
    binary = Path(directory.name)/('kernel.dylib' if sys.platform == 'darwin' else 'kernel.so')
    result = subprocess.run([compiler, '-O3', '-std=c11', '-fPIC',
                             '-dynamiclib' if sys.platform == 'darwin' else '-shared',
                             str(source), '-o', str(binary)], capture_output=True, text=True)
    if result.returncode:
        directory.cleanup()
        raise NativeRingUnavailable(result.stderr[-2000:])
    library = ctypes.CDLL(str(binary))
    library._private_build = directory
    pointer = ctypes.POINTER(ctypes.c_int64)
    library.latge_ring_product.argtypes = [pointer, pointer, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_uint64,
        ctypes.c_int, pointer]
    library.latge_ring_product.restype = ctypes.c_int
    library.latge_table_prepare.argtypes = [pointer, ctypes.c_size_t, pointer]
    library.latge_table_prepare.restype = ctypes.c_int
    library.latge_ring_product_prepared.argtypes = library.latge_ring_product.argtypes[:-1]+[pointer, pointer]
    library.latge_ring_product_prepared.restype = ctypes.c_int
    return library


def supports(n, q, inner):
    return (type(n) is int and type(q) is int and type(inner) is int
            and 1 <= n <= 4096 and 2 <= q < 1 << 63 and inner >= 1
            and n * inner * (q//2)**2 < 1 << 126)


def _product(left, right, q, *, transpose_left=False, prepared_left=None):

    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 3 or right.ndim != 3 or left.shape[2] != right.shape[2]:
        raise ValueError('ring product requires compatible 3D coefficient arrays')
    rows, inner = (left.shape[1], left.shape[0]) if transpose_left else left.shape[:2]
    columns, n = right.shape[1], left.shape[2]
    if right.shape[0] != inner or not rows or not columns or not supports(n, q, inner):
        raise ValueError('unsupported ring product shape or accumulator bound')
    for array in (left, right):
        if array.dtype.kind not in 'iu' or np.any(array < 0) or np.any(array >= q):
            raise ValueError('native ring product requires canonical integer coefficients')
    a = np.ascontiguousarray(left, dtype=np.int64)
    b = np.ascontiguousarray(right, dtype=np.int64)
    out = np.empty((rows, columns, n), dtype=np.int64)
    ptr = lambda array: array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    arguments = (ptr(a), ptr(b), rows, inner, columns, n, q, int(transpose_left))
    if prepared_left is None:
        code = _load().latge_ring_product(*arguments, ptr(out))
    else:
        code = _load().latge_ring_product_prepared(*arguments, ptr(prepared_left), ptr(out))
    if code == -2:
        raise MemoryError('exact ring product scratch allocation failed')
    if code:
        raise ValueError('native ring product parameter guard failed')
    return out


def product(left, right, q, *, transpose_left=False):
    return _product(left, right, q, transpose_left=transpose_left)


class _PreparedLeft:
    def __init__(self, values):
        values = np.ascontiguousarray(values, dtype=np.int64)
        if values.ndim != 3 or values.shape[2] != 128 or not all(values.shape):
            raise ValueError('Table III matrix must have shape (rows, columns, 128)')
        self.shape = values.shape
        self.canonical = values.tobytes()
        self.values = np.frombuffer(self.canonical, dtype=np.int64).reshape(self.shape)
        transformed = np.empty(self.shape, dtype=np.int64)
        pointer = ctypes.POINTER(ctypes.c_int64)
        code = _load().latge_table_prepare(self.values.ctypes.data_as(pointer),
            self.shape[0]*self.shape[1], transformed.ctypes.data_as(pointer))
        if code:
            raise ValueError('invalid Table III matrix preparation')
        self.transformed = np.frombuffer(transformed.tobytes(), dtype=np.int64).reshape(self.shape)

    def multiply(self, right, *, transpose_left=False):
        return _product(self.values, right, 4294967969, transpose_left=transpose_left,
                        prepared_left=self.transformed)
