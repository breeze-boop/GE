from __future__ import annotations

import math
import os
import sys
import ctypes

import numpy as np


if sys.platform == 'darwin':
    _system_random = ctypes.CDLL('/usr/lib/libSystem.B.dylib').CCRandomGenerateBytes
    _system_random.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _system_random.restype = ctypes.c_int


def _proof_entropy(count):
    if sys.platform != 'darwin':
        return os.urandom(count)
    raw = ctypes.create_string_buffer(count)
    if _system_random(raw, count):
        raise OSError('system cryptographic random generation failed')
    return raw


def _uniform_mod_u64(modulus: int, count: int) -> np.ndarray:
    if not isinstance(modulus, int) or isinstance(modulus, bool) or not 0 < modulus < (1 << 63):
        raise ValueError("modulus must be a positive integer below 2^63")
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if count == 0:
        return np.array([], dtype=np.int64)
    word_bits = 32 if modulus < (1 << 32) else 64
    word_dtype = np.uint32 if word_bits == 32 else np.uint64
    cardinality = 1 << word_bits
    limit = (cardinality // modulus) * modulus
    acceptance = limit / float(cardinality)
    out = np.empty(count, dtype=np.int64)
    need = count
    while need:
        draw_count = max(int(math.ceil(need / acceptance * 1.05)), need + 8, 16)
        draw = np.frombuffer(os.urandom((word_bits // 8) * draw_count), dtype=word_dtype)
        if limit != cardinality:
            draw = draw[draw < limit]
        take = min(need, draw.size)
        out[count - need:count - need + take] = draw[:take] % word_dtype(modulus)
        need -= take
    return out


def uniform_mod_coeffs(count: int, modulus: int) -> np.ndarray:
    return _uniform_mod_u64(modulus, count)


def uniform_bounded_coeffs(count: int, bound: int) -> np.ndarray:
    if not isinstance(bound, int) or isinstance(bound, bool) or bound < 0:
        raise ValueError("bound must be a non-negative integer")
    span = 2 * int(bound) + 1
    return _uniform_mod_u64(span, count) - int(bound)


def lns20_commitment_coeffs(count: int) -> np.ndarray:

    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if count == 0:
        return np.array([], dtype=np.int64)
    raw = np.frombuffer(os.urandom((count + 1) // 2), dtype=np.uint8)
    nibbles = np.empty(2 * len(raw), dtype=np.uint8)
    nibbles[::2] = raw & 15
    nibbles[1::2] = raw >> 4
    lookup = np.array([((0xA815 >> i) & 3) - 1 for i in range(16)], dtype=np.int64)
    return lookup[nibbles[:count]]


def _random_slots_into(bulk, kind, bound, output):

    if kind != 'r' and (type(bound) is not int or not 0 <= bound <= bulk.ring.q // 2):
        raise ValueError('bound must fit the centered coefficient range')
    raw = _random_slot_entropy(kind, bound, output.size)
    _random_slots_from_entropy(bulk, kind, bound, output, raw)


def _random_slot_entropy(kind, bound, count):
    if kind == 'r':
        return _proof_entropy((count + 1) // 2)
    span = 2 * bound + 1
    word_bits = 32 if span < (1 << 32) else 64
    cardinality = 1 << word_bits
    limit = (cardinality // span) * span
    acceptance = limit / float(cardinality)
    draw_count = max(int(math.ceil(count / acceptance * 1.05)), count + 8, 16)
    return _proof_entropy((word_bits // 8) * draw_count)


def _random_slots_from_entropy(bulk, kind, bound, output, raw):
    if kind == 'r':
        bulk._trinary_bytes_transform_into(raw, output)
        return
    if type(bound) is not int or not 0 <= bound <= bulk.ring.q // 2:
        raise ValueError('bound must fit the centered coefficient range')
    count = output.size
    produced = 0
    while produced < count:
        produced = bulk._bounded_bytes_transform_into(raw, bound, produced, output)
        if produced < count:
            raw = _random_slot_entropy(kind, bound, count - produced)
