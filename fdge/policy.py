from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .mathutils import rand_bits
from .params import FDGEParams


@dataclass
class ProhibitiveInstance:
    patterns: np.ndarray
    message: np.ndarray


def prohibited_ok(patterns: np.ndarray, message: np.ndarray, d: int) -> bool:
    if patterns.ndim != 2:
        raise ValueError("patterns must be e x t")
    e, t = patterns.shape
    if e < 1 or t < 1 or not 1 <= d <= t:
        raise ValueError('Require e >= 1 and 1 <= d <= t')
    if message.ndim != 1 or message.size < t:
        raise ValueError("message must be a bit vector with length >= t")
    if (patterns.dtype.kind not in 'iub' or message.dtype.kind not in 'iub'
            or not np.all((patterns == 0) | (patterns == 1))
            or not np.all((message == 0) | (message == 1))):
        raise ValueError("policy inputs must be binary before conversion")
    windows = sliding_window_view(message.astype(np.uint8), t)
    for s in patterns.astype(np.uint8):
        dist = np.count_nonzero(windows != s.reshape(1, -1), axis=1)
        if np.any(dist < d):
            return False
    return True


def sample_prohibitive_instance(params: FDGEParams, rng) -> ProhibitiveInstance:
    params.validate(require_distributions=False)
    while True:
        patterns = rand_bits(rng, params.e * params.t).reshape(params.e, params.t)
        msg = rand_bits(rng, params.m)
        if prohibited_ok(patterns, msg, params.d):
            return ProhibitiveInstance(patterns=patterns, message=msg)
