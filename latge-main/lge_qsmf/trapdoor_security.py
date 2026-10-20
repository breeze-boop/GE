from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@dataclass(frozen=True)
class GenTrapSupportBound:
    modulus: int
    ring_degree: int
    rows: int
    prefix_columns: int
    gadget_columns: int
    coefficient_bound: int
    precision_bits: int
    uniform_support_mass_upper: Fraction

    @property
    def statistical_distance_lower(self) -> Fraction:
        return 1 - self.uniform_support_mass_upper

    def excludes_statistical_distance(self, security_bits: int) -> bool:

        if type(security_bits) is not int or security_bits < 1:
            raise ValueError('security_bits must be a positive integer')
        width = min(self.modulus, 2*self.coefficient_bound+1)
        numerator = width**self.prefix_columns
        denominator = self.modulus**self.rows
        if numerator >= denominator:
            return False
        exponent = self.gadget_columns*self.ring_degree
        if exponent >= 1 and ((denominator-numerator) << security_bits) > denominator:
            return True
        numerator, denominator = numerator**exponent, denominator**exponent
        return ((denominator-numerator) << security_bits) > denominator

    def summary(self, security_bits: int | None = None) -> dict:
        result = {
            'modulus': self.modulus, 'ring_degree': self.ring_degree,
            'rows': self.rows, 'prefix_columns': self.prefix_columns,
            'gadget_columns': self.gadget_columns,
            'coefficient_bound': self.coefficient_bound,
            'bound_source': 'enforced support of every successfully returned R; not an observed maximum',
            'uniform_support_mass_upper': str(self.uniform_support_mass_upper),
            'statistical_distance_lower': str(self.statistical_distance_lower),
            'scope': 'joint public A; necessary statistical-uniformity condition only; abort and computational security separate',
        }
        if security_bits is not None:
            result['requested_statistical_bits'] = security_bits
            result['statistical_uniformity_excluded'] = self.excludes_statistical_distance(security_bits)
        return result


def gentrap_support_bound(modulus: int, ring_degree: int, rows: int,
                          prefix_columns: int, gadget_columns: int,
                          coefficient_bound: int, *, precision_bits: int = 192
                          ) -> GenTrapSupportBound:
    if (type(modulus) is not int or modulus < 2 or
            any(type(v) is not int or v <= 0 for v in
                (ring_degree, rows, prefix_columns, gadget_columns))):
        raise ValueError('invalid ring or matrix dimensions')
    if (type(coefficient_bound) is not int or
            not 0 <= coefficient_bound <= modulus//2):
        raise ValueError('coefficient_bound must be a centered support bound')
    if type(precision_bits) is not int or not 32 <= precision_bits <= 4096:
        raise ValueError('precision_bits must be between 32 and 4096')
    scale = 1 << precision_bits
    width = min(modulus, 2*coefficient_bound+1)
    numerator, denominator = width**prefix_columns, modulus**rows
    base = min(scale, _ceil_div(numerator*scale, denominator))
    value, exponent = scale, gadget_columns*ring_degree
    while exponent:
        if exponent & 1:
            value = _ceil_div(value*base, scale)
        exponent >>= 1
        if exponent:
            base = _ceil_div(base*base, scale)
    return GenTrapSupportBound(modulus, ring_degree, rows, prefix_columns,
        gadget_columns, coefficient_bound, precision_bits, Fraction(value, scale))
