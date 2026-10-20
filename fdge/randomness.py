from __future__ import annotations

from decimal import Context, Decimal, ROUND_FLOOR, ROUND_CEILING, ROUND_HALF_EVEN
from functools import lru_cache
import hashlib
import math
import os

import numpy as np

from .mathutils import SystemUniformRNG, require_memory


@lru_cache(maxsize=32)
def _pi_interval(precision):
    lo = Context(prec=precision + 12, rounding=ROUND_FLOOR)
    hi = Context(prec=precision + 12, rounding=ROUND_CEILING)
    tolerance = Decimal(1).scaleb(-precision - 8)

    def atan_inverse(a):
        lower = upper = Decimal(0)
        j = 0
        power = a
        while True:
            denominator = Decimal((2*j+1)*power)
            term_lo, term_hi = lo.divide(Decimal(1), denominator), hi.divide(Decimal(1), denominator)
            if j % 2:
                lower, upper = lo.subtract(lower, term_hi), hi.subtract(upper, term_lo)
            else:
                lower, upper = lo.add(lower, term_lo), hi.add(upper, term_hi)
            j += 1
            power *= a*a
            remainder = hi.divide(Decimal(1), Decimal((2*j+1)*power))
            if remainder < tolerance:
                if j % 2:
                    lower = lo.subtract(lower, remainder)
                else:
                    upper = hi.add(upper, remainder)
                return lower, upper

    a, b = atan_inverse(5), atan_inverse(239)
    return (lo.subtract(lo.multiply(Decimal(16), a[0]), hi.multiply(Decimal(4), b[1])),
            hi.subtract(hi.multiply(Decimal(16), a[1]), lo.multiply(Decimal(4), b[0])))


@lru_cache(maxsize=64)
def gaussian_cdf_intervals(sigma_text, window, precision=60):
    lo = Context(prec=precision, rounding=ROUND_FLOOR)
    hi = Context(prec=precision, rounding=ROUND_CEILING)
    nearest = Context(prec=precision, rounding=ROUND_HALF_EVEN)
    sigma = Decimal(sigma_text)
    if not sigma.is_finite() or sigma <= 0 or window < 1:
        raise ValueError('Gaussian sigma/window must be positive')
    pi_lo, pi_hi = _pi_interval(precision)
    square_lo, square_hi = lo.multiply(sigma, sigma), hi.multiply(sigma, sigma)

    def mass(factor):
        alpha_lo = lo.divide(lo.multiply(pi_lo, Decimal(factor)), square_hi)
        alpha_hi = hi.divide(hi.multiply(pi_hi, Decimal(factor)), square_lo)

        lower = nearest.exp(alpha_hi.copy_negate()).next_minus(nearest)
        upper = nearest.exp(alpha_lo.copy_negate()).next_plus(nearest)
        return max(Decimal(0), lower), upper

    masses = [mass(x*x) for x in range(-window, window+1)]
    total_lo = total_hi = Decimal(0)
    for lower, upper in masses:
        total_lo, total_hi = lo.add(total_lo, lower), hi.add(total_hi, upper)
    first_tail = mass((window+1)**2)[1]
    ratio = mass(2*window+3)[1]
    tail = hi.divide(first_tail, lo.subtract(Decimal(1), ratio))
    normalizer_hi = hi.add(total_hi, hi.multiply(Decimal(2), tail))
    result = [(Decimal(0), min(Decimal(1), hi.divide(tail, total_lo)))]
    cumulative_lo = cumulative_hi = Decimal(0)
    for lower, upper in masses:
        cumulative_lo = lo.add(cumulative_lo, lower)
        cumulative_hi = hi.add(cumulative_hi, upper)
        result.append((max(Decimal(0), lo.divide(cumulative_lo, normalizer_hi)),
                       min(Decimal(1), hi.divide(hi.add(cumulative_hi, tail), total_lo))))
    return tuple(result)


@lru_cache(maxsize=32)
def _prefix_table(sigma_text, window):
    intervals = gaussian_cdf_intervals(sigma_text, window)
    scale = Decimal(1 << 64)
    lo = Context(prec=100, rounding=ROUND_FLOOR)
    hi = Context(prec=100, rounding=ROUND_CEILING)
    cuts, ambiguous = [], set()
    for i, (lower, upper) in enumerate(intervals):
        lower_integer = int(lo.multiply(lower, scale).to_integral_value(rounding=ROUND_FLOOR))
        upper_integer = int(hi.multiply(upper, scale).to_integral_value(rounding=ROUND_CEILING))-1

        if upper_integer-lower_integer > 1:
            raise ValueError('Gaussian window exceeds 64-bit comparison')
        ambiguous.update(x for x in range(lower_integer, upper_integer+1) if 0 <= x < 1 << 64)
        if 0 < i < len(intervals)-1:
            cuts.append(lower_integer)
    return np.array(cuts, dtype=np.uint64), np.array(sorted(ambiguous), dtype=np.uint64)


class CryptographicRNG(SystemUniformRNG):
    def __init__(self, seed=None):
        self._seed = os.urandom(32) if seed is None else bytes(seed)
        if len(self._seed) != 32:
            raise ValueError('cryptographic stream seed must be 32 bytes')
        self._counter = 0

    def _random_bytes(self, size):
        if type(size) is not int or size < 0:
            raise ValueError('random byte length must be nonnegative')
        block = b'FDGE-CSPRNG-SHAKE256-v1\0'+self._seed+self._counter.to_bytes(16,'little')+size.to_bytes(8,'little')
        self._counter += 1
        return hashlib.shake_256(block).digest(size)

    def __deepcopy__(self, memo):
        copy = type(self)(self._seed)
        copy._counter = self._counter
        return copy

    def random(self, size=None):
        shape=() if size is None else (size,) if isinstance(size,int) else tuple(size)
        if any(type(s) is not int or s<0 for s in shape):
            raise ValueError('Invalid sample shape')
        words=np.frombuffer(self._random_bytes(8*math.prod(shape)),dtype='<u8')
        values=((words>>11).astype(np.float64)*(2.0**-53)).reshape(shape)
        return values[()] if size is None else values

    def geometric(self, probability, size=None):
        if not 0<probability<=1:
            raise ValueError('geometric probability must lie in (0,1]')
        if probability==1:
            return np.ones(() if size is None else size,dtype=np.int64)

        return (np.floor(np.log1p(-self.random(size))/math.log1p(-probability))+1).astype(np.int64)

    def _refine_gaussian(self, prefix, sigma_text, initial_window):
        bits, window, precision = 64, initial_window, 60
        while True:
            intervals = gaussian_cdf_intervals(sigma_text, window, precision)
            lo = Context(prec=precision, rounding=ROUND_FLOOR)
            hi = Context(prec=precision, rounding=ROUND_CEILING)
            denominator = Decimal(1 << bits)
            lower = lo.divide(Decimal(prefix), denominator)
            upper = hi.divide(Decimal(prefix+1), denominator)
            for i in range(1, len(intervals)):
                if lower >= intervals[i-1][1] and upper <= intervals[i][0]:
                    return i-window-1
            prefix = (prefix << 64) | int.from_bytes(self._random_bytes(8), 'little')
            bits += 64
            window += 2
            precision += 32

    def discrete_gaussian(self, shape, sigma, beta, max_memory_bytes):
        if (not math.isfinite(sigma) or sigma <= 0 or type(beta) is not int or beta < 1
                or any(type(s) is not int or s < 0 for s in shape)):
            raise ValueError('invalid discrete Gaussian parameters')
        count = math.prod(shape)
        require_memory(8*count+4*1024*1024, max_memory_bytes, 'lazy Gaussian output and bounded batches')
        sigma_text = str(sigma)

        window = max(beta, math.ceil(8*sigma))
        cuts, ambiguous = _prefix_table(sigma_text, window)
        output = np.empty(count, dtype=np.int64)
        for offset in range(0, count, 65536):
            length = min(65536,count-offset)
            prefixes = np.frombuffer(self._random_bytes(8*length), dtype='<u8')
            indices = np.searchsorted(cuts, prefixes, side='right')
            values = indices.astype(np.int64)-window
            positions = np.searchsorted(ambiguous, prefixes)
            possible = positions < ambiguous.size
            unresolved = np.zeros(length, dtype=bool)
            unresolved[possible] = ambiguous[positions[possible]] == prefixes[possible]
            for i in np.flatnonzero(unresolved):
                values[i] = self._refine_gaussian(int(prefixes[i]), sigma_text, window)
            output[offset:offset+length] = values
        return output.reshape(shape)
