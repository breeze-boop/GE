from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .ring import Poly


def delta_beta(beta: int) -> int:
    if not isinstance(beta, (int, np.integer)) or beta < 0:
        raise ValueError("decomposition bound must be a non-negative integer")
    return int(beta).bit_length()


def g_beta(beta: int) -> List[int]:
    delta = delta_beta(beta)
    return [(beta + (1 << (j - 1))) // (1 << j) for j in range(1, delta + 1)]


def idec(value: int, beta: int) -> List[int]:
    if not isinstance(value, (int, np.integer)) or value < 0 or value > beta:
        raise ValueError("value outside decomposition bound")
    weights = g_beta(beta)
    rest = int(value)
    out = []
    for weight in weights:
        if rest >= weight:
            out.append(1)
            rest -= weight
        else:
            out.append(0)
    if rest != 0:
        raise ValueError("decomposition failed")
    return out


def rdec(poly: Poly, beta: int | None = None) -> List[Poly]:
    ring = poly.ring
    if beta is None:
        beta = (ring.q - 1) // 2
    centered = poly.centered().astype(np.int64, copy=False)
    delta_beta(beta)
    if bool(np.any(np.abs(centered) > beta)):
        raise ValueError("polynomial outside decomposition bound")
    rest = np.abs(centered).astype(np.int64, copy=True)
    signs = np.where(centered < 0, -1, 1).astype(np.int64)
    out = []
    for weight in g_beta(beta):
        take = rest >= weight
        out.append(Poly(ring, (signs * take.astype(np.int64)) % ring.q))
        rest[take] -= weight
    if bool(np.any(rest != 0)):
        raise ValueError("decomposition failed")
    return out


def rcompose(parts: Sequence[Poly], beta: int | None = None) -> Poly:
    if not parts:
        raise ValueError("empty decomposition")
    ring = parts[0].ring
    if beta is None:
        beta = (ring.q - 1) // 2
    weights = g_beta(beta)
    if len(weights) != len(parts):
        raise ValueError("wrong decomposition width")
    acc = ring.zero()
    for weight, part in zip(weights, parts):
        acc = acc + part.scalar_mul(weight)
    return acc


def rvdec(vec: Sequence[Poly], beta: int | None = None) -> List[Poly]:
    out = []
    for poly in vec:
        out.extend(rdec(poly, beta))
    return out


