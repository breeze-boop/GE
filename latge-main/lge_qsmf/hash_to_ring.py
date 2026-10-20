from __future__ import annotations
import hashlib
from functools import lru_cache
from typing import List
from .ring import Poly, Ring


@lru_cache(maxsize=4096)
def _hash_coefficients(n: int, q: int, data: bytes, domain: bytes) -> tuple[int, ...]:
    if n <= 0 or q < 3:
        raise ValueError("invalid hash-to-ring parameters")
    width = (q.bit_length() + 7) // 8
    limit = (1 << (8 * width)) // q * q
    prefix = len(domain).to_bytes(4, 'little') + domain + len(data).to_bytes(8, 'little') + data
    xof = hashlib.shake_256(prefix)
    size, used = n * width, 0
    coefficients = []
    while len(coefficients) < n:
        stream = xof.digest(size)
        while used + width <= len(stream) and len(coefficients) < n:
            value = int.from_bytes(stream[used:used + width], 'little')
            used += width
            if value < limit:
                coefficients.append(value % q)
        size *= 2
    return tuple(coefficients)


def hash_to_poly(ring: Ring, data: bytes, domain: bytes) -> Poly:
    return Poly.from_ints(ring, _hash_coefficients(ring.n, ring.q, bytes(data), bytes(domain)))


def H2(ring: Ring, vk_bytes: bytes, alpha1: int) -> List[Poly]:
    if type(alpha1) is not int or alpha1 <= 0:
        raise ValueError('alpha1 must be a positive integer')
    return [hash_to_poly(ring, vk_bytes + i.to_bytes(4, 'little'), b'GE:H2V') for i in range(alpha1)]


def H2_vec(ring: Ring, vk_bytes: bytes, alpha1: int) -> List[Poly]:
    return H2(ring, vk_bytes, alpha1)


def H1(ring: Ring, vk_bytes: bytes, alpha1: int, k: int) -> List[List[Poly]]:
    if any(type(value) is not int or value <= 0 for value in (alpha1, k)):
        raise ValueError('matrix dimensions must be positive integers')
    return [[hash_to_poly(ring, vk_bytes + r.to_bytes(4, 'little') + c.to_bytes(4, 'little'),
                          b'GE:H1M') for c in range(k)] for r in range(alpha1)]


def H1_mat(ring: Ring, vk_bytes: bytes, alpha1: int, k: int) -> List[List[Poly]]:
    return H1(ring, vk_bytes, alpha1, k)
