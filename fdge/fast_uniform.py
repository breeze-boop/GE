from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import ctypes
import math
import sys

import numpy as np

from .randomness import CryptographicRNG
from .mathutils import SystemUniformRNG

_ZERO_BLOCK = bytes(1024 * 1024)


class _AES256CTRBytes(SystemUniformRNG):
    geometric = CryptographicRNG.geometric

    def __init__(self, key, iv, library):
        if len(key) != 32 or len(iv) != 16:
            raise ValueError('AES-256-CTR needs a 32-byte key and 16-byte counter')
        self._library, self._context = library, ctypes.c_void_p()
        self._remaining = 16 * ((1 << 64) - int.from_bytes(iv[8:], 'big'))
        status = library.CCCryptorCreateWithMode(
            0, 4, 0, 0, iv, key, len(key), None, 0, 0, 2,
            ctypes.byref(self._context))
        if status != 0:
            raise RuntimeError(f'CommonCrypto CTR initialization failed: {status}')

    def _validate_length(self, length):
        if type(length) is not int or length < 0 or length > self._remaining:
            raise ValueError('invalid CTR stream length or exhausted counter')
        if not self._context.value:
            raise ValueError('CTR stream is closed')

    def _fill_bytes(self, output):
        length, address = output.nbytes, output.ctypes.data
        offset = 0
        while offset < length:
            count = min(length - offset, len(_ZERO_BLOCK))
            written = ctypes.c_size_t()
            status = self._library.CCCryptorUpdate(
                self._context, _ZERO_BLOCK, count, ctypes.c_void_p(address + offset), count,
                ctypes.byref(written))
            if status != 0 or written.value != count:
                self.close()
                raise RuntimeError(f'CommonCrypto CTR update failed: {status}')
            offset += count
        self._remaining -= length

    def _random_bytes(self, length):
        self._validate_length(length)
        output = np.empty(length, dtype=np.uint8)
        self._fill_bytes(output)
        return output.tobytes()

    def _random_words(self, count, word_bytes):
        if type(count) is not int or count < 0 or word_bytes not in (4, 8):
            raise ValueError('invalid CTR word count or size')
        self._validate_length(count * word_bytes)
        output = np.empty(count, dtype='<u4' if word_bytes == 4 else '<u8')
        self._fill_bytes(output)
        return output

    def random(self, size=None):
        shape = () if size is None else (size,) if isinstance(size, int) else tuple(size)
        if any(type(value) is not int or value < 0 for value in shape):
            raise ValueError('Invalid sample shape')
        words = self._random_words(math.prod(shape), 8)
        np.right_shift(words, 11, out=words)
        values = words.astype(np.float64)
        values *= 2.0**-53
        values = values.reshape(shape)
        return values[()] if size is None else values

    def integers(self, low, high=None, size=None, dtype=np.int64):
        if high is None:
            low, high = 0, low
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               for value in (low, high)):
            raise ValueError('uniform integer endpoints must be integers')
        low, high = int(low), int(high)
        dtype = np.dtype(dtype)
        if dtype.kind not in 'iu':
            raise ValueError('uniform integer dtype must be an integer dtype')
        limits = np.iinfo(dtype)
        span = high - low
        if (not 0 < span <= np.iinfo(np.int64).max or low < limits.min or high - 1 > limits.max
                or low < np.iinfo(np.int64).min or high - 1 > np.iinfo(np.int64).max):
            raise ValueError('Integer interval exceeds dtype/range')
        shape = () if size is None else (size,) if isinstance(size, (int, np.integer)) else tuple(size)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0
               for value in shape):
            raise ValueError('Invalid uniform sample shape')
        count = math.prod(shape)
        result = np.empty(count, dtype=dtype)
        if span == 1:
            result.fill(low)
        else:
            bits = (span - 1).bit_length()
            word_bytes = 4 if bits <= 32 else 8
            mask = (1 << bits) - 1
            filled = 0
            while filled < count:
                batch = min(65536, max(64, count - filled if bits <= 32 else 2 * (count - filled)))
                words = self._random_words(batch, word_bytes)
                if bits <= 32:
                    product = words.astype(np.uint64)
                    product *= np.uint64(span)
                    accepted = (product >> 32)[(product & 0xffffffff) >= (1 << 32) % span]
                else:
                    words &= mask
                    accepted = words[words < span]
                take = min(len(accepted), count - filled)
                np.add(accepted[:take], np.int64(low), out=result[filled:filled + take],
                       dtype=np.int64, casting='unsafe')
                filled += take
        result = result.reshape(shape)
        return result[()] if size is None else result

    def close(self):
        if self._context.value:
            self._library.CCCryptorRelease(self._context)
            self._context = ctypes.c_void_p()


@lru_cache(maxsize=1)
def _native_library():
    if sys.platform != 'darwin':
        return None
    try:
        library = ctypes.CDLL('/usr/lib/system/libcommonCrypto.dylib')
    except OSError:
        return None
    pointer, size = ctypes.c_void_p, ctypes.c_size_t
    library.CCCryptorCreateWithMode.argtypes = [
        ctypes.c_uint32] * 4 + [pointer, pointer, size, pointer, size,
                              ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(pointer)]
    library.CCCryptorCreateWithMode.restype = ctypes.c_int32
    library.CCCryptorUpdate.argtypes = [pointer, pointer, size, pointer, size,
                                      ctypes.POINTER(size)]
    library.CCCryptorUpdate.restype = ctypes.c_int32
    library.CCCryptorRelease.argtypes = [pointer]
    library.CCCryptorRelease.restype = ctypes.c_int32

    key = bytes.fromhex('603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4')
    iv = bytes.fromhex('f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff')
    plain = bytes.fromhex('6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51')
    cipher = bytes.fromhex('601ec313775789a5b7a7f504bbf3d228f443e3ca4d62b59aca84e990cacaf5c5')
    source = _AES256CTRBytes(key, iv, library)
    try:
        stream = source._random_bytes(5) + source._random_bytes(27)
        if bytes(a ^ b for a, b in zip(stream, plain)) != cipher:
            raise RuntimeError('CommonCrypto failed the NIST AES-256-CTR vector')
    finally:
        source.close()
    return library


def uniform_backend():
    return 'aes256_ctr' if _native_library() is not None else 'shake256'


@contextmanager
def uniform_source(parent):
    library = _native_library() if isinstance(parent, CryptographicRNG) else None
    if library is None:
        yield parent
        return
    seed = parent._random_bytes(40)
    source = _AES256CTRBytes(seed[:32], seed[32:] + bytes(8), library)
    try:
        yield source
    finally:
        source.close()


def uniform_field(source, shape, q, *, dtype=np.int64):
    dtype, shape = np.dtype(dtype), tuple(shape)
    if (type(q) is not int or not 1 < q <= 1 << 32 or dtype.kind not in 'iu'
            or q - 1 > np.iinfo(dtype).max
            or any(type(value) is not int or value < 0 for value in shape)):
        raise ValueError('invalid unsigned-word field sampling domain')
    if not callable(getattr(source, '_random_bytes', None)):
        return source.integers(0, q, size=shape, dtype=dtype)
    output = np.empty(math.prod(shape), dtype=dtype)
    filled, threshold = 0, (1 << 32) % q
    while filled < output.size:
        count = min(262144, max(64, output.size - filled))
        words = (source._random_words(count, 4) if type(source) is _AES256CTRBytes
                 else np.frombuffer(source._random_bytes(4 * count), dtype='<u4'))
        product = words.astype(np.uint64)
        product *= np.uint64(q)
        accepted = (product >> 32)[(product & 0xffffffff) >= threshold]
        take = min(accepted.size, output.size - filled)
        output[filled:filled + take] = accepted[:take]
        filled += take
    return output.reshape(shape)
