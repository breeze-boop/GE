from __future__ import annotations

import numpy as np


def weights(beta: int) -> np.ndarray:
    if not isinstance(beta, (int, np.integer)) or beta < 0:
        raise ValueError("beta must be a nonnegative integer")
    beta = int(beta)
    if beta > np.iinfo(np.int64).max:
        raise ValueError("beta exceeds the supported int64 representation")
    return np.array([(beta + (1 << (j - 1))) // (1 << j)
                     for j in range(1, beta.bit_length() + 1)], dtype=np.int64)


def _integers(value, *, name: str) -> np.ndarray:
    out = np.asarray(value)
    if out.dtype.kind not in "iub":
        raise ValueError(f"{name} must contain integers")
    if out.size and (np.any(out < 0) or np.any(out > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} is outside the supported nonnegative range")
    return out.astype(np.int64, copy=False)


def idec(value: int, beta: int) -> np.ndarray:
    if not isinstance(value, (int, np.integer)):
        raise ValueError("value must be an integer")
    return vdec(np.array([value]), beta)


def vdec(vector: np.ndarray, beta: int) -> np.ndarray:
    w = weights(beta)
    values = _integers(vector, name="vector")
    if values.ndim != 1 or np.any(values > beta):
        raise ValueError('Expected vector in [0,beta]')
    remainder = values.copy()
    result = np.zeros((values.size, w.size), dtype=np.uint8)
    for j, weight in enumerate(w):
        selected = remainder >= weight
        result[:, j] = selected
        remainder -= selected.astype(np.int64) * weight
    if np.any(remainder):
        raise AssertionError("bounded greedy decomposition did not reconstruct")
    return result.reshape(-1)


def mdec(matrix: np.ndarray, q: int) -> np.ndarray:
    values = _integers(matrix, name="matrix")
    if values.ndim != 2 or q < 2 or np.any(values >= q):
        raise ValueError('Expected matrix in [0,q)')
    return vdec(values.reshape(-1, order="F"), q - 1)
