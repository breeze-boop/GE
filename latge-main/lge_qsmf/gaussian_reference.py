from __future__ import annotations

import bisect
import math
import secrets
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
from functools import lru_cache
from dataclasses import dataclass

from .sampler_native import native_library


_PI = Decimal('3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408128')


def _decimal(value) -> Decimal:
    if isinstance(value, Fraction):
        return Decimal(value.numerator) / Decimal(value.denominator)
    return value if isinstance(value, Decimal) else Decimal(str(value))


def as_fraction(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ValueError("Boolean is not a Gaussian parameter")
    return Fraction(str(value))


@lru_cache(maxsize=32)
def pi_interval(bits: int) -> tuple[Fraction, Fraction]:
    def atan_inverse(denominator):
        total, i = Fraction(0), 0
        tolerance = Fraction(1, 1 << (bits + 8))
        while True:
            term = Fraction(1, (2 * i + 1) * denominator ** (2 * i + 1))
            total += term if i % 2 == 0 else -term
            following = Fraction(1, (2 * i + 3) * denominator ** (2 * i + 3))
            if following < tolerance:
                return (total - following, total) if i % 2 == 0 else (total, total + following)
            i += 1
    lo5, hi5 = atan_inverse(5)
    lo239, hi239 = atan_inverse(239)
    return 16 * lo5 - 4 * hi239, 16 * hi5 - 4 * lo239


def exp_interval(exponent: Fraction, bits: int) -> tuple[Fraction, Fraction]:

    if exponent > 0:
        raise ValueError("this enclosure accepts non-positive exponents")
    cutoff = bits + 64
    if exponent <= -cutoff:
        return Fraction(0), Fraction(1, 1 << cutoff)
    with localcontext() as context:
        context.prec = (bits * 30103) // 100000 + 50
        context.rounding = ROUND_FLOOR
        lo = Decimal(exponent.numerator) / Decimal(exponent.denominator)
        context.rounding = ROUND_CEILING
        hi = Decimal(exponent.numerator) / Decimal(exponent.denominator)
        lower = max(Decimal(0), lo.exp().next_minus())
        upper = hi.exp().next_plus()
        return Fraction(lower), Fraction(upper)


@lru_cache(maxsize=8)
def _pi_fixed(bits: int) -> tuple[int, int]:
    lower, upper = pi_interval(bits+32)
    scale = 1 << bits
    return (lower.numerator*scale//lower.denominator,
            -((-upper.numerator*scale)//upper.denominator))


def _rho_dyadic_interval(factor: Fraction) -> tuple[int, int, int]:

    if factor < 0:
        raise ValueError('Gaussian probability factor must be nonnegative')
    if factor == 0:
        return 1, 1, 0
    pi_lo, pi_hi = _pi_fixed(128)
    low_num, high_num = pi_lo*factor.numerator, pi_hi*factor.numerator
    denominator = factor.denominator << 128
    if low_num >= 128*denominator:
        return 0, 1, 128
    squares = max(0, high_num.bit_length()-denominator.bit_length()+4)
    while 8*high_num > denominator << squares:
        squares += 1
    bits = 80+squares
    scale = 1 << bits
    divisor = denominator << squares
    y_lo = low_num*scale//divisor
    y_hi = -((-high_num*scale)//divisor)
    term_lo = term_hi = scale
    lower = upper = scale
    degree = 0
    while True:
        degree += 1
        denominator_term = scale*degree
        term_lo = term_lo*y_lo//denominator_term
        term_hi = -((-term_hi*y_hi)//denominator_term)
        if degree & 1:
            lower -= term_hi; upper -= term_lo
        else:
            lower += term_lo; upper += term_hi
        if term_hi <= 1:
            lower -= term_hi; upper += term_hi
            break
    lower, upper = max(0, lower), min(scale, upper)
    for _ in range(squares):
        lower = lower*lower//scale
        upper = -((-upper*upper)//scale)
    return lower, upper, bits


def _rho_native_decision(numerator: int, denominator: int, draw: int):

    library = native_library()
    if library is None:
        return None
    pi_lo, pi_hi = _pi_fixed(128)
    low_num, high_num = pi_lo*numerator, pi_hi*numerator
    divisor = denominator << 128
    if low_num >= 128*divisor:
        return 0 if draw else -1
    squares = max(0, high_num.bit_length()-divisor.bit_length()+4)
    while 8*high_num > divisor << squares:
        squares += 1
    divisor <<= squares
    ylo = (low_num << 60)//divisor
    yhi = -((-high_num << 60)//divisor)
    status = library.latge_rho_decide(ylo, yhi, squares, draw)
    if status < -1:
        raise ArithmeticError('native rho decision invariant failed')
    return status


def _bernoulli_rho_ratio(numerator: int, denominator: int, randbelow) -> bool:

    if numerator < 0 or denominator <= 0:
        raise ValueError('Gaussian probability ratio must be nonnegative')
    if numerator == 0:
        return True
    bits = 48
    draw = randbelow(1 << bits)
    integral = numerator//denominator
    if draw and integral and (integral >= 16 or draw >= 1 << (bits-3*integral)):
        return False
    status = _rho_native_decision(numerator, denominator, draw)
    if status is not None and status != -1:
        return bool(status)
    factor = Fraction(numerator, denominator)
    if status is None:
        lower, upper, enclosure_bits = _rho_dyadic_interval(factor)
        shift = enclosure_bits-bits
        if (draw+1) << shift <= lower:
            return True
        if draw << shift >= upper:
            return False
    return _bernoulli_rho_refine(numerator, denominator, draw, bits, randbelow)


def _bernoulli_rho_refine(numerator, denominator, draw, bits, randbelow):
    factor = Fraction(numerator, denominator)
    while True:
        lo_pi, hi_pi = pi_interval(bits + 64)
        lower = exp_interval(-hi_pi * factor, bits + 16)[0]
        upper = exp_interval(-lo_pi * factor, bits + 16)[1]
        if Fraction(draw + 1, 1 << bits) <= lower:
            return True
        if Fraction(draw, 1 << bits) >= upper:
            return False
        draw = (draw << 32) + randbelow(1 << 32)
        bits += 32


@lru_cache(maxsize=256)
def _gaussian_support(variance: Fraction, precision_bits: int):

    with localcontext() as context:
        context.prec = (precision_bits * 30103) // 100000 + 50
        sigma = _decimal(variance).sqrt()
        radius = int((sigma * (Decimal(precision_bits+32)*Decimal(2).ln()/_PI).sqrt()).to_integral_value(rounding=ROUND_CEILING))+2
    pi_lo, _ = pi_interval(precision_bits+96)
    budget = Fraction(1, 1 << precision_bits)
    while True:
        distance = radius+1
        first = exp_interval(-pi_lo*(Fraction(distance*distance)-Fraction(1,4))/variance, precision_bits+24)[1]
        ratio = exp_interval(-pi_lo*(2*distance+1)/variance, precision_bits+24)[1]
        tail = 2*first/(1-ratio)
        if tail <= budget/4:
            return sigma, radius, tail
        radius += max(1, radius//4)


@lru_cache(maxsize=32)
def smoothing_integer(bits: int) -> tuple[int, Fraction]:

    if not isinstance(bits, int) or not 16 <= bits <= 512:
        raise ValueError("smoothing bits must be between 16 and 512")
    pi_lo, _ = pi_interval(bits + 64)
    width = 1
    while True:
        first = exp_interval(-pi_lo * width * width, bits + 16)[1]
        ratio = exp_interval(-3 * pi_lo * width * width, bits + 16)[1]
        bound = 2 * first / (1 - ratio)
        if bound <= Fraction(1, 1 << bits):
            return width, bound
        width += 1


@dataclass(frozen=True, slots=True, init=False)
class ShiftedGaussianTemplate:

    radius: int
    count: int
    variance_numerator: int
    variance_denominator: int
    tail_error_bound: Fraction

    def __init__(self, sigma_squared, precision_bits):
        variance = as_fraction(sigma_squared)
        if variance <= 0:
            raise ValueError('sigma squared must be positive')
        if not isinstance(precision_bits, int) or not 32 <= precision_bits <= 512:
            raise ValueError('precision_bits must be between 32 and 512')
        _, radius, tail_bound = _gaussian_support(variance, precision_bits)
        object.__setattr__(self, 'radius', radius)
        object.__setattr__(self, 'count', 2*radius+2)
        object.__setattr__(self, 'variance_numerator', variance.numerator)
        object.__setattr__(self, 'variance_denominator', variance.denominator)
        object.__setattr__(self, 'tail_error_bound', tail_bound)

    def sample_ratio(self, center_num, center_den, *, randbelow=secrets.randbelow):
        if type(center_num) is not int or type(center_den) is not int or center_den <= 0:
            raise ValueError('center must have an integer numerator and positive denominator')
        common = math.gcd(center_num, center_den)
        cn, cd = center_num//common, center_den//common
        shift, remainder = divmod(cn, cd)
        nearest = min(remainder, cd-remainder)**2
        denominator = cd*cd*self.variance_numerator
        start = shift-self.radius
        while True:
            candidate = start+randbelow(self.count)
            displacement = candidate*cd-cn
            numerator = (displacement*displacement-nearest)*self.variance_denominator
            if _bernoulli_rho_ratio(numerator, denominator, randbelow):
                return candidate


class DiscreteGaussian1D:


    def __init__(self, sigma=None, center=0, *, precision_bits: int = 128,
                 sigma_squared=None, table_max_support: int = 256):
        if not isinstance(precision_bits, int) or not 32 <= precision_bits <= 512:
            raise ValueError("precision_bits must be between 32 and 512")
        self.precision_bits = precision_bits
        if (not isinstance(table_max_support, int) or isinstance(table_max_support, bool)
                or table_max_support < 0):
            raise ValueError('table_max_support must be a nonnegative integer')
        variance = as_fraction(sigma_squared) if sigma_squared is not None else as_fraction(sigma) ** 2
        c = as_fraction(center)
        if variance <= 0 or (sigma is not None and as_fraction(sigma) <= 0):
            raise ValueError("sigma must be positive")
        self.sigma_squared, self.center = variance, c
        self.sigma, radius, tail_bound = _gaussian_support(variance, precision_bits)
        shift = c.numerator // c.denominator
        fraction = c - shift
        nearest_squared = min(fraction, 1 - fraction) ** 2
        nearest_numerator = min(c.numerator % c.denominator,
                                (-c.numerator) % c.denominator)
        self._rho_geometry = (c.numerator, c.denominator,
                              nearest_numerator*nearest_numerator,
                              variance.denominator,
                              c.denominator*c.denominator*variance.numerator)
        budget = Fraction(1, 1 << precision_bits)
        self.support = range(shift - radius, shift + radius + 2)
        self._nearest_squared = nearest_squared
        self.tail_error_bound = tail_bound
        if len(self.support) > table_max_support:
            self.integer_weights = None
            self.total_weight, self._cdf = None, None
            self.table_error_bound = Fraction(0)
            self.total_variation_bound = tail_bound
            return
        offsets = tuple(range(-radius, radius + 2))
        pi_lo, pi_hi = pi_interval(precision_bits + 96)
        scale = 1 << (precision_bits + len(offsets).bit_length() + 16)
        weights, gaps = [], 0
        for offset in offsets:
            squared_difference = (offset - fraction) ** 2 - nearest_squared
            factor = squared_difference / variance
            lower = exp_interval(-pi_hi * factor, precision_bits + 48)[0]
            upper = exp_interval(-pi_lo * factor, precision_bits + 48)[1]
            weight = (lower.numerator * scale) // lower.denominator
            upper_integer = -((-upper.numerator * scale) // upper.denominator)
            weights.append(weight)
            gaps += upper_integer - weight
        self.support = tuple(shift + offset for offset in offsets)
        self.integer_weights = tuple(weights)
        total, cdf = 0, []
        for weight in weights:
            total += weight
            cdf.append(total)
        if total <= 0:
            raise ArithmeticError("Gaussian table lost all probability mass")
        self.total_weight, self._cdf = total, tuple(cdf)
        self.tail_error_bound = tail_bound
        self.table_error_bound = Fraction(gaps, total)
        self.total_variation_bound = tail_bound + self.table_error_bound
        if self.total_variation_bound > budget:
            raise ArithmeticError("Gaussian finite-precision budget was exceeded")

    def sample(self, *, randbelow=secrets.randbelow) -> int:
        if self._cdf is None:
            cn, cd, nearest, multiplier, denominator = self._rho_geometry
            while True:
                candidate = self.support[randbelow(len(self.support))]
                displacement = candidate*cd-cn
                numerator = (displacement*displacement-nearest)*multiplier
                if _bernoulli_rho_ratio(numerator, denominator, randbelow):
                    return candidate
        value = randbelow(self.total_weight)
        if not isinstance(value, int) or not 0 <= value < self.total_weight:
            raise ValueError("randbelow returned an out-of-range integer")
        return self.support[bisect.bisect_right(self._cdf, value)]

    def sample_many(self, count, *, randbelow=secrets.randbelow):
        if type(count) is not int or count < 0:
            raise ValueError('sample count must be a nonnegative integer')
        from .sampler_native import (SystemRandomBuffer, integer_gaussian_vector,
                                     rational_gaussian_vector, table_gaussian_vector)
        rng = getattr(randbelow, '__self__', None)
        if (self._cdf is not None and type(rng) is SystemRandomBuffer
                and getattr(randbelow, '__func__', None) is SystemRandomBuffer.randbelow):
            result = table_gaussian_vector(self.support, self._cdf, count, rng)
            if result is not None:
                return result
        if (self._cdf is None and self.center == 0
                and -(1 << 31) <= self.support.start
                and self.support.stop <= 1 << 31
                and type(rng) is SystemRandomBuffer
                and getattr(randbelow, '__func__', None) is SystemRandomBuffer.randbelow):
            if self.sigma_squared.denominator == 1 and self.sigma_squared <= 1 << 56:
                result = integer_gaussian_vector(self.support, int(self.sigma_squared), count, rng)
            else:
                result = rational_gaussian_vector(self.support, self.sigma_squared, count, rng)
            if result is not None:
                return result
        return [self.sample(randbelow=randbelow) for _ in range(count)]


@lru_cache(maxsize=128)
def gadget_kernel_basis(q: int, base: int, digits: int) -> tuple[tuple[int, ...], ...]:

    if any(not isinstance(x, int) or isinstance(x, bool) for x in (q, base, digits)):
        raise ValueError('gadget parameters must be integers')
    if q < 2 or base < 2 or not 1 <= digits <= 64:
        raise ValueError('invalid gadget parameters')
    columns = []
    for i in range(digits - 1):
        column = [0] * digits
        column[i], column[i + 1] = -base, 1
        columns.append(tuple(column))
    remaining, last = q, []
    for _ in range(digits - 1):
        remaining, digit = divmod(remaining, base)
        last.append(digit)
    last.append(remaining)
    columns.append(tuple(last))
    return tuple(columns)


@lru_cache(maxsize=128)
def _gram_schmidt(columns: tuple[tuple[int, ...], ...]):
    orthogonal, squared_norms = [], []
    for column in columns:
        vector = [Fraction(x) for x in column]
        for previous, norm in zip(orthogonal, squared_norms):
            coefficient = sum(x * y for x, y in zip(vector, previous)) / norm
            vector = [x - coefficient * y for x, y in zip(vector, previous)]
        norm = sum(x * x for x in vector)
        if norm <= 0:
            raise ValueError('gadget basis is singular')
        orthogonal.append(tuple(vector))
        squared_norms.append(norm)
    return tuple(orthogonal), tuple(squared_norms)


class GadgetCosetSampler:


    def __init__(self, q: int, base: int, digits: int, sigma, *, precision_bits: int = 128):
        self.q, self.base, self.digits = q, base, digits
        self.columns = gadget_kernel_basis(q, base, digits)
        self.orthogonal, self.squared_norms = _gram_schmidt(self.columns)
        if not isinstance(precision_bits, int) or not 32 <= precision_bits <= 512:
            raise ValueError('precision_bits must be between 32 and 512')
        self.precision_bits = precision_bits
        self.sigma_squared = as_fraction(sigma) ** 2
        if as_fraction(sigma) <= 0:
            raise ValueError('sigma must be positive')
        omega, delta = smoothing_integer(precision_bits)
        self.minimum_sigma_squared = max(self.squared_norms) * omega * omega
        self.normalization_delta = delta
        if self.sigma_squared < self.minimum_sigma_squared:
            raise ValueError(f'gadget sigma below smoothing bound squared={self.minimum_sigma_squared}')
        with localcontext() as context:
            context.prec = (precision_bits * 30103) // 100000 + 50
            self.sigma = _decimal(self.sigma_squared).sqrt()
            self.minimum_sigma = _decimal(self.minimum_sigma_squared).sqrt()
        self.total_variation_bound = 8 * digits * delta + Fraction(digits, 1 << precision_bits)
        self.widths_squared = tuple(self.sigma_squared/norm for norm in self.squared_norms)
        projections = []
        for vector, norm in zip(self.orthogonal, self.squared_norms):
            weights = tuple(value/norm for value in vector)
            denominator = math.lcm(*(weight.denominator for weight in weights))
            projections.append((tuple((j, weight.numerator*(denominator//weight.denominator))
                                      for j, weight in enumerate(weights) if weight), denominator))
        self.projections = tuple(projections)
        self.sampling_templates = tuple(
            (_gaussian_support(width, precision_bits)[1], width.numerator, width.denominator)
            for width in self.widths_squared)

    def sample(self, target: int, *, randbelow=secrets.randbelow) -> tuple[int, ...]:
        if not isinstance(target, int) or isinstance(target, bool):
            raise ValueError('target must be an integer')
        particular = [target % self.q] + [0] * (self.digits - 1)
        center = [-x for x in particular]
        result = list(particular)
        for i in reversed(range(self.digits)):
            weights, denominator = self.projections[i]
            numerator = sum(center[j]*weight for j, weight in weights)
            radius, width_num, width_den = self.sampling_templates[i]
            shift, remainder = divmod(numerator, denominator)
            nearest = min(remainder, denominator-remainder)**2
            start, count = shift-radius, 2*radius+2
            rho_denominator = denominator*denominator*width_num
            while True:
                coefficient = start+randbelow(count)
                displacement = coefficient*denominator-numerator
                if _bernoulli_rho_ratio((displacement*displacement-nearest)*width_den,
                                        rho_denominator, randbelow):
                    break
            for j in range(self.digits):
                delta = coefficient * self.columns[i][j]
                result[j] += delta
                center[j] -= delta
        if sum((self.base ** i) * x for i, x in enumerate(result)) % self.q != target % self.q:
            raise ArithmeticError('gadget sampler returned an invalid coset element')
        return tuple(result)

    def sample_many(self, targets, *, randbelow=secrets.randbelow):

        targets = tuple(targets)
        if any(not isinstance(x, int) or isinstance(x, bool) for x in targets):
            raise ValueError('targets must be integers')
        from .sampler_native import SystemRandomBuffer, gadget_gaussian_vector
        rng = getattr(randbelow, '__self__', None)
        if (type(rng) is SystemRandomBuffer
                and getattr(randbelow, '__func__', None) is SystemRandomBuffer.randbelow):
            result = gadget_gaussian_vector(self, targets, rng)
            if result is not None:
                return result
        return [self.sample(target, randbelow=randbelow) for target in targets]
