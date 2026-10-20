from __future__ import annotations

import math
from dataclasses import dataclass, replace
from fractions import Fraction


def is_prime_u64(value: int) -> bool:
    if not isinstance(value, int) or isinstance(value, bool) or not 2 <= value < 1 << 64:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    odd, power = value - 1, 0
    while odd % 2 == 0:
        odd //= 2
        power += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        residue = pow(base, odd, value)
        if residue in (1, value - 1):
            continue
        for _ in range(power - 1):
            residue = residue * residue % value
            if residue == value - 1:
                break
        else:
            return False
    return True


def ceil_log_integer(value: int, base: int) -> int:
    if value < 1 or base < 2:
        raise ValueError('value must be positive and base at least two')
    exponent, power = 0, 1
    while power < value:
        exponent += 1
        power *= base
    return exponent


def cyclotomic_factor_count(n: int, q: int) -> int:
    if n < 1 or n & (n - 1) or not is_prime_u64(q) or q % 2 == 0:
        raise ValueError('requires power-of-two n and an odd prime q')
    modulus, residue, order = 2 * n, q % (2 * n), 1
    while residue != 1:
        residue = residue * q % modulus
        order += 1
        if order > n:
            raise ValueError('invalid cyclotomic multiplicative order')
    return n // order


def _tag_fraction(value: str | int, name: str) -> Fraction:
    if type(value) not in (str, int):
        raise ValueError(f'{name} must be an integer or exact decimal/fraction string, not a float')
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f'{name} must be a valid finite rational value') from exc


def derive_tag_boundaries(n: int, d: int, tag_c: str | int,
                          tag_alpha0: str | int) -> tuple[int, ...]:

    if type(n) is not int or n <= 0 or type(d) is not int or not 0 < d <= n:
        raise ValueError('tag dimensions must satisfy positive integers 1 <= d <= n')
    c = _tag_fraction(tag_c, 'tag_c')
    alpha0 = _tag_fraction(tag_alpha0, 'tag_alpha0')
    if c <= 1:
        raise ValueError('tag_c must be greater than one')
    if alpha0 < 1 / (c - 1):
        raise ValueError('tag_alpha0 must be at least 1/(tag_c-1)')
    boundaries, value = [0], alpha0
    for _ in range(d):
        value *= c
        boundary = value.numerator // value.denominator
        if boundary <= boundaries[-1]:
            raise ValueError('derived tag boundaries must increase strictly')
        if boundary > n:
            raise ValueError('derived tag boundaries must end at or below n')
        boundaries.append(boundary)
    return tuple(boundaries)


@dataclass(frozen=True)
class Params:
    n: int
    q: int
    p: int
    k: int
    B: int
    ell: int
    alpha: int
    alpha1: int
    d: int
    beta: int
    kappa1: int
    kappa2: int
    s: int = 32
    security_bits: int = 80
    tag_boundaries: tuple[int, ...] | None = None
    tag_c: str | int | None = None
    tag_alpha0: str | int | None = None
    trapdoor_sigma: float | None = None
    trapdoor_bound: int | None = None
    certificate_sigma: float | None = None
    decryption_sigma: float | None = None
    zk_rejection_delta: int | None = None
    zk_conjugates: int = 4
    zk_research_reference: bool = False

    def __post_init__(self) -> None:
        names = ('n', 'q', 'p', 'k', 'B', 'ell', 'alpha', 'alpha1', 'd',
                 'beta', 'kappa1', 'kappa2', 's', 'security_bits')
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.n & (self.n - 1):
            raise ValueError('n must be a power of two')
        if not is_prime_u64(self.q) or not is_prime_u64(self.p):
            raise ValueError('p and q must be prime')
        if max(self.p, self.q) >= 1 << 62:
            raise ValueError('moduli exceed the supported signed-int64 arithmetic range')
        if self.q % 64 != 33:
            raise ValueError('paper Section IV-A requires q == 33 (mod 64)')
        if cyclotomic_factor_count(self.n, self.q) != self.ell:
            raise ValueError('ell must equal the actual cyclotomic factor count')
        if self.k != ceil_log_integer(self.q, 3):
            raise ValueError('k must be ceil(log_3(q))')
        if self.s != ceil_log_integer(self.q, 2) - 1:
            raise ValueError('s must be ceil(log_2(q)) - 1')
        if (self.tag_c is None) != (self.tag_alpha0 is None):
            raise ValueError('tag_c and tag_alpha0 must be specified together')
        if self.tag_c is not None:
            derived = derive_tag_boundaries(self.n, self.d, self.tag_c, self.tag_alpha0)
            if self.tag_boundaries is not None and (not isinstance(self.tag_boundaries, tuple) or
                    any(type(value) is not int for value in self.tag_boundaries)):
                raise ValueError('explicit tag_boundaries must be a tuple of integers')
            if self.tag_boundaries is not None and self.tag_boundaries != derived:
                raise ValueError('tag_boundaries disagree with floor(tag_alpha0 * tag_c**i)')
            object.__setattr__(self, 'tag_boundaries', derived)
        if self.tag_boundaries is not None:
            boundaries = self.tag_boundaries
            if not isinstance(boundaries, tuple) or len(boundaries) != self.d + 1:
                raise ValueError('tag_boundaries must be the tuple (c_0,...,c_d)')
            if any(not isinstance(x, int) or isinstance(x, bool) for x in boundaries):
                raise ValueError('tag boundaries must be integers')
            if boundaries[0] != 0 or boundaries[-1] > self.n:
                raise ValueError('tag boundaries must start at zero and end at or below n')
            if any(a >= b for a, b in zip(boundaries, boundaries[1:])):
                raise ValueError('tag boundaries must increase strictly')
        for name in ('trapdoor_sigma', 'certificate_sigma', 'decryption_sigma'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be a positive finite Gaussian parameter or None')
        if self.trapdoor_bound is not None and (not isinstance(self.trapdoor_bound, int) or isinstance(self.trapdoor_bound, bool) or self.trapdoor_bound <= 0):
            raise ValueError('trapdoor_bound must be a positive integer or None')
        if not isinstance(self.zk_conjugates, int) or isinstance(self.zk_conjugates, bool) or self.zk_conjugates <= 0 or self.ell % self.zk_conjugates:
            raise ValueError('zk_conjugates must be a positive divisor of ell')
        if not isinstance(self.zk_research_reference, bool):
            raise ValueError('zk_research_reference must be a boolean')
        if self.zk_rejection_delta is not None:
            delta = self.zk_rejection_delta
            if not isinstance(delta, int) or isinstance(delta, bool) or delta <= self.ell or 2 * (delta + self.ell) >= self.q:
                raise ValueError('zk_rejection_delta must satisfy ell < delta and delta+ell < q/2')

    @property
    def tag_boundaries_source(self) -> str:

        if self.tag_c is not None:
            return 'paper_formula'
        if self.tag_boundaries is not None:
            return 'explicit_research_boundaries'
        return 'unspecified'

    def unspecified_protocol_parameters(self) -> tuple[str, ...]:

        return tuple(name for name in ('tag_boundaries', 'trapdoor_sigma',
                                      'trapdoor_bound', 'certificate_sigma', 'decryption_sigma', 'zk_rejection_delta')
                     if getattr(self, name) is None)


def table_iii_params(**overrides) -> Params:
    q = 4294967969
    params = Params(n=128, q=q, p=70368744177679,
                    k=ceil_log_integer(q, 3), B=8, ell=16,
                    alpha=4, alpha1=6, d=2, beta=95255,
                    kappa1=45, kappa2=52,
                    s=ceil_log_integer(q, 2) - 1,
                    decryption_sigma=5 * 22075)
    return replace(params, **overrides) if overrides else params


def table_ii_params(**overrides) -> Params:
    return table_iii_params(**overrides)


def table_iii_experiment_params() -> Params:

    return table_iii_params(tag_c='8', tag_alpha0='1',
        trapdoor_sigma=0.65, trapdoor_bound=12,
        certificate_sigma=40000., zk_rejection_delta=1 << 31,
        zk_research_reference=True)


paper_params = table_iii_params


