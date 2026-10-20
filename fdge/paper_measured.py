from __future__ import annotations

import numpy as np

from .gpv import UserPublicKey, UserSecretKey
from .mathutils import matmul_mod, rand_discrete_gaussian, require_memory
from .params import FDGEParams


_RECIPIENT_COLUMN_BLOCK = 512


def blocked_recipient_keygen(p: FDGEParams, Arec: np.ndarray, rng):
    p.validate()
    if Arec.shape != (p.n, p.m):
        raise ValueError("Arec must have shape (n,m)")
    width = min(_RECIPIENT_COLUMN_BLOCK, p.m)
    dtype = np.dtype(np.int64)

    required = dtype.itemsize * p.m * p.m + 32 * p.n * p.m + 24 * p.m * width + 4 * 1024 * 1024
    require_memory(required, p.max_memory_bytes, 'GPV keygen workspace')
    E1 = np.empty((p.m, p.m), dtype=dtype)
    public = []
    for branch in range(2):
        U = np.empty((p.n, p.m), dtype=np.int64)
        for offset in range(0, p.m, width):
            stop = min(offset + width, p.m)
            block = rand_discrete_gaussian(rng, (p.m, stop - offset), p.resolved_sigma(),
                                           p.resolved_beta(), p.max_memory_bytes)
            U[:, offset:stop] = matmul_mod(Arec, block, p.q)
            if branch == 0:
                E1[:, offset:stop] = block
        public.append(U)
    return UserPublicKey(*public), UserSecretKey(E1)
