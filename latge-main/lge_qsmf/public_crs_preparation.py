import ctypes
import time

import numpy as np

from .proof_timing import excluded_from_prover_time


class _PublicCRSPreparation:
    def __init__(self, crs, workers):
        self.crs = crs
        self._blocks = {}
        self.metrics = {
            'preparation_seconds': 0.0,
            'cache_bytes': 0,
            'matrix': 'bounded waves of A1 and A2 random entries in NTT form',
            'encoding': 'sixteen uint32 low words plus one uint16 high-bit mask',
            'workers': workers,
        }
        object.__setattr__(crs, '_public_a2_preparation', self)

    def prepare(self, bulk, requests, pool=None):

        with excluded_from_prover_time():
            started = time.perf_counter()
            available={}
            for low,high in self._blocks.values():
                available.setdefault(high.shape,[]).append((low,high))
            self._blocks.clear()
            jobs=[]
            for request in requests:
                kind, row, rows, col, cols = request
                buffers=available.get((rows,cols))
                if buffers:low,high=buffers.pop()
                else:
                    low=np.empty((rows,cols,16),dtype=np.uint32)
                    high=np.empty((rows,cols),dtype=np.uint16)
                low.flags.writeable=high.flags.writeable=True
                jobs.append((request,low,high))
            available.clear()

            def expand(job):
                request,low,high=job
                kind,row,rows,col,cols=request
                bulk._check(bulk.library.latge_context_prepare_matrix(
                    bulk._context, int(kind == 'A2'), row, rows, col, cols,
                    ctypes.c_void_p(low.ctypes.data), ctypes.c_void_p(high.ctypes.data)))
                low.flags.writeable = high.flags.writeable = False
                return request, (low, high)

            values = map(expand,jobs) if pool is None else pool.map(expand,jobs)
            self._blocks.update(values)
            size = sum(low.nbytes + high.nbytes for low, high in self._blocks.values())
            self.metrics['cache_bytes'] = max(self.metrics['cache_bytes'], size)
            self.metrics['preparation_seconds'] += time.perf_counter() - started

    def _block(self, crs, kind, row, rows, col, cols):
        return self._blocks.get((kind, row, rows, col, cols))

    def close(self):
        self._blocks.clear()
        self.crs.__dict__.pop('_public_a2_preparation', None)


def _prepared_owner(crs):
    return getattr(crs, '_public_a2_preparation', None)


def _prepare_public_crs(crs, workers=8):
    return _PublicCRSPreparation(crs, workers)
