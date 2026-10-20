from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import numpy as np

from .gaussian_reference import (DiscreteGaussian1D, as_fraction, exp_interval,
                                 pi_interval, smoothing_integer)


class StructuredAcceptanceCostError(ValueError):
    pass


def negacyclic_product(a, b):
    n = len(a)
    if n == 0 or len(b) != n:
        raise ValueError('nonempty equal-length integer polynomials required')
    bound = min(sum(abs(int(x)) for x in a) * max(abs(int(x)) for x in b),
                sum(abs(int(x)) for x in b) * max(abs(int(x)) for x in a))
    if not bound:
        return (0,) * n
    if bound < 1 << 62:
        convolution = np.convolve(np.asarray(a, dtype=np.int64),
                                  np.asarray(b, dtype=np.int64))
        result = convolution[:n].copy()
        if n > 1:
            result[:n-1] -= convolution[n:]
        return tuple(map(int, result))
    bits = (2 * bound).bit_length()
    radix, mask, half = 1 << bits, (1 << bits) - 1, 1 << (bits - 1)
    aa = sum(int(x) << (bits * i) for i, x in enumerate(a))
    bb = sum(int(x) << (bits * i) for i, x in enumerate(b))
    packed, result = aa * bb, [0] * n
    for i in range(2 * n - 1):
        coefficient = packed & mask
        if coefficient >= half:
            coefficient -= radix
        packed = (packed - coefficient) >> bits
        result[i % n] += coefficient if i < n else -coefficient
    if packed:
        raise ArithmeticError('signed polynomial packing bound was violated')
    return tuple(result)


def adjoint_poly(a):
    return (a[0],) + tuple(-x for x in reversed(a[1:]))


def _matrix_vector(matrix, vector, n):
    blocks = [vector[i:i+n] for i in range(0, len(vector), n)]
    if matrix and len(matrix[0]) != len(blocks):
        raise ValueError('structured operator dimension mismatch')
    output = []
    for row in matrix:
        total = [0] * n
        for polynomial, block in zip(row, blocks):
            if any(polynomial) and any(block):
                product = negacyclic_product(polynomial, block)
                total = [x + y for x, y in zip(total, product)]
        output.extend(total)
    return output


@dataclass
class StructuredTrapdoor:
    n: int
    R: tuple
    adjoint: tuple
    gram: tuple
    norm_squared_bound: int
    frobenius_squared: int
    _integer_operators: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def from_key(cls, key):
        n = key.ring.n
        R = tuple(tuple(tuple(map(int, poly.centered())) for poly in row) for row in key.R)
        adjoint, gram, norm, frobenius, prepared, prepared_adjoint = _structured_geometry(n, R)
        result = cls(n, prepared.source if prepared is not None else R,
                     adjoint, gram, norm, frobenius)
        result._integer_operators['R'] = prepared
        result._integer_operators['adjoint'] = prepared_adjoint
        return result

    def apply_R(self, x):
        return self._apply_many('R', [x])[0]

    def apply_adjoint(self, x):
        return self._apply_many('adjoint', [x])[0]

    def apply_gram(self, x):
        return self._apply_many('gram', [x])[0]

    def lift(self, z):
        return self.apply_R(z) + list(z)

    def _apply_many(self, name, vectors):
        from .structured_native import SparseIntegerMatrix
        matrix = getattr(self, name)
        prepared = self._integer_operators.get(name)
        if prepared is None or prepared.source is not matrix or prepared.n != self.n:
            prepared = SparseIntegerMatrix.from_matrix(matrix, self.n)
            self._integer_operators[name] = prepared
        if prepared is not None:
            result = prepared.apply_many(vectors)
            if result is not None:
                return result
        return [_matrix_vector(matrix, vector, self.n) for vector in vectors]

    def lift_many(self, vectors):
        products = self._apply_many('R', vectors)
        return [product + list(vector) for product, vector in zip(products, vectors)]


@lru_cache(maxsize=2)
def _structured_geometry(n, R):

    from .structured_native import SparseIntegerMatrix
    prepared = SparseIntegerMatrix.from_matrix(R, n)
    prepared_adjoint = None if prepared is None else prepared.adjoint()
    adjoint = (prepared_adjoint.source if prepared_adjoint is not None else
               tuple(tuple(adjoint_poly(R[i][j]) for i in range(len(R)))
                     for j in range(len(R[0]))))
    products = None if prepared is None else prepared.apply_many([
        [value for row in adjoint for value in row[j]] for j in range(len(R))])
    if products is not None:
        gram = [[tuple(products[j][i*n:(i+1)*n]) for j in range(len(R))]
                for i in range(len(R))]
    else:
        zero = (0,) * n
        gram = [[zero] * len(R) for _ in R]
        active = [i for i, row in enumerate(R) if any(any(poly) for poly in row)]
        for position, i in enumerate(active):
            for j in active[:position + 1]:
                entry = [0] * n
                for k in range(len(R[0])):
                    product = negacyclic_product(R[i][k], adjoint[k][j])
                    entry = [x + y for x, y in zip(entry, product)]
                gram[i][j] = tuple(entry)
                gram[j][i] = adjoint_poly(entry)
    coefficients = (prepared.weights.tolist() if prepared is not None else
                    (x for row in R for p in row for x in p))
    frobenius = n * sum(x*x for x in coefficients)
    row_sum = max(sum(abs(x) for p in row for x in p) for row in gram)
    return adjoint, tuple(tuple(row) for row in gram), min(frobenius, row_sum), frobenius, prepared, prepared_adjoint


@dataclass
class StructuredPerturbation:
    operator: StructuredTrapdoor
    target_squared: Fraction
    gadget_squared: Fraction
    proposal: DiscreteGaussian1D
    acceptance_lower_bound: Fraction
    proposal_distance_bound: Fraction
    accepted_distance_bound: Fraction
    max_proposals: int
    proposals_used: int = 0

    def quadratic_intervals(self, x):

        op = self.operator
        prefix = len(op.R) * op.n
        y = [a+b for a, b in zip(op.apply_adjoint(x[:prefix]), x[prefix:])]
        yield from self._quadratic_intervals_from_projection(y)

    def _quadratic_intervals_from_projection(self, y):
        op = self.operator
        z = op.apply_R(y)
        b = self.gadget_squared / (self.target_squared - self.gadget_squared)
        alpha = b / self.target_squared
        rho = b * op.norm_squared_bound
        if not 0 <= rho < 1:
            raise ValueError('Neumann contraction is not certified')
        total = alpha * sum(v*v for v in y)
        z_norm = sum(v*v for v in z)
        if z_norm == 0:
            while True:
                yield total, total
        vector, power, tail_power = z, Fraction(1), rho
        while True:
            term = sum(a*c for a, c in zip(z, vector))
            if term < 0:
                raise ArithmeticError('positive semidefinite Gram form became negative')
            total += alpha * b * power * term
            tail = alpha * b * z_norm * tail_power / (1 - rho)
            yield total, total + tail
            vector = op.apply_gram(vector)
            power *= b
            tail_power *= rho

    def _accept(self, x, randbelow):
        bits, draw = 64, randbelow(1 << 64)
        op = self.operator
        prefix = len(op.R) * op.n
        y = [a+b for a, b in zip(op.apply_adjoint(x[:prefix]), x[prefix:])]
        b = self.gadget_squared / (self.target_squared-self.gadget_squared)
        alpha, rho = b/self.target_squared, b*op.norm_squared_bound
        if not 0 <= rho < 1:
            raise ValueError('Neumann contraction is not certified')
        upper_form = alpha*sum(v*v for v in y)/(1-rho)
        hi_pi = pi_interval(bits+64)[1]
        lower_probability = exp_interval(-hi_pi*upper_form,bits+24)[0]
        if Fraction(draw+1,1 << bits) <= lower_probability:
            return True
        for step, (lower_form, upper_form) in enumerate(self._quadratic_intervals_from_projection(y)):
            lo_pi, hi_pi = pi_interval(bits + 64)
            lower_probability = exp_interval(-hi_pi * upper_form, bits + 24)[0]
            upper_probability = exp_interval(-lo_pi * lower_form, bits + 24)[1]
            if Fraction(draw + 1, 1 << bits) <= lower_probability:
                return True
            if Fraction(draw, 1 << bits) >= upper_probability:
                return False
            if step % 8 == 7 or lower_form == upper_form:
                draw = (draw << 32) + randbelow(1 << 32)
                bits += 32
        raise AssertionError('the exact interval generator cannot terminate')

    def sample(self, *, randbelow=secrets.randbelow):
        dimension = (len(self.operator.R) + len(self.operator.adjoint)) * self.operator.n
        for _ in range(self.max_proposals):
            self.proposals_used += 1
            candidate = self.proposal.sample_many(dimension, randbelow=randbelow)
            if self._accept(candidate, randbelow):
                return candidate
        from .mp12_reference import MP12ResourceLimitError
        raise MP12ResourceLimitError('structured MP12 rejection exhausted the explicit proposal cost budget')


def prepare_structured_perturbation(operator, target, gadget, *, dimension,
                                   gadget_dimension, omega, delta, component_bits,
                                   minimum_acceptance=Fraction(1, 64), max_proposals=4096):
    target, gadget, delta, minimum_acceptance = map(
        as_fraction, (target, gadget, delta, minimum_acceptance))
    certified_omega, certified_delta = smoothing_integer(component_bits)
    if omega < certified_omega or delta < certified_delta:
        raise ValueError('supplied smoothing bound is not certified at component_bits')
    expected_gadget = len(operator.adjoint)*operator.n
    expected_dimension = len(operator.R)*operator.n + expected_gadget
    if dimension != expected_dimension or gadget_dimension != expected_gadget:
        raise ValueError('perturbation dimensions do not match the structured operator')
    if target < omega or omega <= 0 or not 0 <= dimension*delta <= Fraction(1, 2):
        raise ValueError('proposal normalization/smoothing preconditions are not satisfied')
    if gadget <= 0:
        raise ValueError('gadget width must be positive')
    target_squared, gadget_squared = target*target, gadget*gadget
    rho = gadget_squared * (operator.norm_squared_bound + 1) / target_squared
    if not 0 <= rho < 1:
        raise ValueError('perturbation covariance is not certified positive definite')
    if target_squared - gadget_squared * (operator.norm_squared_bound + 1) < omega*omega:
        raise ValueError('perturbation covariance is below the smoothing bound')
    if not 0 < minimum_acceptance <= 1:
        raise ValueError('minimum_acceptance must be in (0,1]')
    if not isinstance(max_proposals, int) or isinstance(max_proposals, bool) or max_proposals <= 0:
        raise ValueError('max_proposals must be a positive integer')
    trace = gadget_squared * (operator.frobenius_squared + gadget_dimension) / target_squared
    exponent = -trace / (2 * (1 - rho))
    acceptance = exp_interval(exponent, component_bits + 32)[0] / (1 + 2*dimension*delta)
    if acceptance < minimum_acceptance:
        raise StructuredAcceptanceCostError('certified structured rejection acceptance lower bound '
                         f'{float(acceptance):.6g} is below cost budget {float(minimum_acceptance):.6g}; '
                         'the requested target sigma was not changed')
    extra_bits = 4
    while Fraction(1, 1 << (extra_bits - 4)) > acceptance:
        extra_bits += 1
    proposal = DiscreteGaussian1D(target, precision_bits=component_bits + extra_bits)
    proposal_error = dimension * proposal.total_variation_bound
    if proposal_error > acceptance / 2:
        raise ArithmeticError('proposal error exceeds its acceptance budget')
    accepted_error = 4 * proposal_error / acceptance
    if accepted_error > Fraction(dimension, 1 << component_bits):
        raise ArithmeticError('structured perturbation error exceeds the composed MP12 budget')
    return StructuredPerturbation(operator, target_squared, gadget_squared, proposal,
                                  acceptance, proposal_error, accepted_error, max_proposals)
