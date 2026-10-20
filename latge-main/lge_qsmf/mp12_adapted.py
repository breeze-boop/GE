from __future__ import annotations

import math
import operator
import secrets
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np

from .gaussian_reference import (DiscreteGaussian1D, ShiftedGaussianTemplate, as_fraction, exp_interval,
                                 pi_interval, smoothing_integer)
from .mp12_structured import StructuredPerturbation, StructuredTrapdoor


class AdaptedAcceptanceCostError(ValueError):
    pass


def _ceil_fraction(value):
    return -((-value.numerator)//value.denominator)


def _sqrt_upper(value, bits=128):
    if value < 0:
        raise ValueError('square root of negative bound')
    scaled = value * (1 << (2*bits))
    lower = math.isqrt(scaled.numerator//scaled.denominator)
    if lower*lower < scaled:
        lower += 1
    return Fraction(lower, 1 << bits)


def _nearest_integer(numerator, denominator):
    if denominator <= 0:
        raise ValueError('positive denominator required')
    quotient, remainder = divmod(numerator, denominator)
    return quotient + (2*remainder >= denominator)


def _rotation(coefficients):
    n = len(coefficients)
    row, column = np.indices((n, n))
    return np.asarray(coefficients, dtype=np.float64)[(row-column) % n] * np.where(row >= column, 1., -1.)


def exact_integer_covariance(operator, target_squared, gadget_squared):
    target_squared, gadget_squared = map(as_fraction, (target_squared, gadget_squared))
    if target_squared.denominator != 1 or gadget_squared.denominator != 1:
        raise ValueError('adapted prototype requires integral squared widths')
    s2, r2 = int(target_squared), int(gadget_squared)
    entry_bound = max([1] + [abs(x) for row in operator.gram for poly in row for x in poly]
                      + [abs(x) for row in operator.R for poly in row for x in poly])
    if max(s2, r2)*max(1, entry_bound)+s2 >= 1 << 53:
        raise ValueError('target covariance cannot be represented exactly by binary64 integers')
    n, prefix, gadget = operator.n, len(operator.R), len(operator.adjoint)
    dimension = (prefix+gadget)*n
    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    for i in range(prefix):
        for j in range(prefix):
            covariance[i*n:(i+1)*n, j*n:(j+1)*n] = -r2*_rotation(operator.gram[i][j])
        for j in range(gadget):
            block = -r2*_rotation(operator.R[i][j])
            covariance[i*n:(i+1)*n, (prefix+j)*n:(prefix+j+1)*n] = block
            covariance[(prefix+j)*n:(prefix+j+1)*n, i*n:(i+1)*n] = block.T
    covariance.flat[::dimension+1] += s2
    for i in range(prefix*n, dimension):
        covariance[i, i] -= r2
    return covariance


@dataclass
class AdaptedCertificate:
    dimension: int
    residual_norm_bound: Fraction
    target_min_eigenvalue: Fraction
    target_max_eigenvalue: Fraction
    inflation: Fraction
    proposal_min_eigenvalue: Fraction
    proposal_max_eigenvalue: Fraction
    covariance_difference_bound: Fraction
    factor_fraction_bits: int
    factorization_model: str = 'IEEE binary64 ordinary BLAS; deterministic gamma_(2d) residual enclosure'

    def summary(self):
        return {name: str(value) if isinstance(value,Fraction) else value
                for name,value in vars(self).items()}


def _certify_factor(covariance, factor, target_min, target_max, components=None,
                    diagonal_prefix=0):

    d = len(factor)
    if covariance.shape != (d, d) or factor.shape != (d, d):
        raise ValueError('factor/covariance dimensions disagree')
    if (factor.dtype != np.float64 or covariance.dtype != np.float64 or
            not np.isfinite(factor).all() or not np.isfinite(covariance).all() or
            np.any(np.diag(factor) <= 0) or np.any(np.triu(factor, 1) != 0)):
        raise ValueError('a finite positive-diagonal binary64 lower factor is required')
    unit = Fraction(1, 1 << 53)
    gamma = 2*d*unit/(1-2*d*unit)
    if gamma >= Fraction(1, 4):
        raise ValueError('dimension exceeds the binary64 certificate profile')
    max_row_sum, maximum_residual = 0., 0.
    blocks = (None,) if components is None else components
    for component in blocks:
        if component is None:
            block, target_block = factor, covariance
        else:
            index = np.ix_(component, component)
            block, target_block = factor[index], covariance[index]
        row_sums = np.sum(block*block, axis=1, dtype=np.float64)
        split = diagonal_prefix if component is None else int(np.searchsorted(component, diagonal_prefix))
        if split:
            diagonal = np.diag(block)[:split]
            residual = np.zeros_like(block)
            residual[np.arange(split), np.arange(split)] = diagonal*diagonal
            if split < len(block):
                cross, tail = block[split:, :split], block[split:, split:]
                residual[split:, :split] = cross*diagonal
                residual[:split, split:] = residual[split:, :split].T
                residual[split:, split:] = cross @ cross.T + tail @ tail.T
        else:
            residual = block @ block.T
        residual -= target_block
        if not np.isfinite(row_sums).all() or not np.isfinite(residual).all():
            raise ValueError('factor residual computation overflowed')
        max_row_sum = max(max_row_sum, float(np.max(row_sums)))
        maximum_residual = max(maximum_residual, float(np.max(np.abs(residual))))
    subnormal = Fraction(1, 1 << 1074)
    frobenius_upper = d*(Fraction(max_row_sum) + 2*d*subnormal)/(1-gamma)
    max_residual = Fraction(maximum_residual)
    eta = d*(max_residual+subnormal)/(1-unit) + gamma*frobenius_upper + 4*d*d*subnormal/(1-gamma)
    mu, nu = as_fraction(target_min), as_fraction(target_max)
    if not 0 < eta < mu or not 0 < mu <= nu:
        raise ValueError('factor residual does not certify positive target domination')
    inflation_bits = 48
    inflation = 1 + Fraction(_ceil_fraction(eta/(mu-eta)*(1 << inflation_bits)), 1 << inflation_bits)
    h2 = inflation*inflation
    proposal_min, proposal_max = h2*(mu-eta), h2*(nu+eta)
    difference = h2*eta+(h2-1)*nu
    if h2*(mu-eta) < mu:
        raise ArithmeticError('dyadic inflation did not establish covariance domination')
    minimum_exponent = 0
    for row in factor:
        nonzero = np.abs(row[row != 0])
        if nonzero.size:
            minimum_exponent = min(minimum_exponent, math.frexp(float(np.min(nonzero)))[1])
    fraction_bits = 53-minimum_exponent
    return AdaptedCertificate(d, eta, mu, nu, inflation, proposal_min, proposal_max,
                              difference, fraction_bits)


def _component_cholesky(covariance, diagonal_prefix=0):

    dimension = len(covariance)
    seen = np.zeros(dimension, dtype=bool)
    components = []
    for vertex in range(dimension):
        if seen[vertex]:
            continue
        seen[vertex] = True
        stack, component = [vertex], []
        while stack:
            current = stack.pop()
            component.append(current)
            adjacent = np.flatnonzero((covariance[current] != 0) & ~seen)
            seen[adjacent] = True
            stack.extend(adjacent.tolist())
        components.append(np.asarray(sorted(component), dtype=np.intp))
    factor = np.zeros_like(covariance)
    for component in components:
        index = np.ix_(component, component)
        block = covariance[index]
        split = int(np.searchsorted(component, diagonal_prefix))
        if not split:
            factor[index] = np.linalg.cholesky(block)
            continue
        lower = np.zeros_like(block)
        diagonal = np.sqrt(np.diag(block)[:split])
        lower[np.arange(split), np.arange(split)] = diagonal
        if split < len(component):
            cross = block[split:, :split] / diagonal
            lower[split:, :split] = cross
            schur = block[split:, split:] - cross @ cross.T
            lower[split:, split:] = np.linalg.cholesky(schur)
        factor[index] = lower
    return factor, tuple(components)


class DyadicTriangular:
    def __init__(self, factor, certificate, active_indices=None, independent_diagonal=None):
        self.factor, self.certificate = factor, certificate
        self.dimension = certificate.dimension
        self.active_indices = (tuple(range(self.dimension)) if active_indices is None
                               else active_indices)
        self._positions = {coordinate: i for i, coordinate in enumerate(self.active_indices)}
        self.order = self.active_indices + tuple(i for i in range(self.dimension)
                                                if i not in self._positions)
        self._zero_center = tuple(i not in self._positions or not np.any(
            factor[self._positions[i], :self._positions[i]]) for i in range(self.dimension))
        self.bits = certificate.factor_fraction_bits
        self.h = certificate.inflation
        self.denominator = self.h.denominator << self.bits
        maximum = max(abs(float(np.min(factor))), abs(float(np.max(factor))),
                      abs(float(independent_diagonal or 0)))
        self._float_scaling_safe = math.frexp(maximum)[1]+self.bits <= 1024
        self.diagonal_integers = [self._scaled_integer(float(
            factor[self._positions[i], self._positions[i]] if i in self._positions
            else independent_diagonal)) for i in range(self.dimension)]
        self.variances = [Fraction(self.h.numerator*v, self.denominator)**2
                          for v in self.diagonal_integers]

    def _scaled_integer(self, value):
        if self._float_scaling_safe:
            return int(math.ldexp(value, self.bits))
        numerator, denominator = value.as_integer_ratio()
        return numerator << (self.bits-(denominator.bit_length()-1))

    def _row_dot(self, i, integers):
        position = self._positions.get(i)
        if position is None:
            return 0
        indices = np.flatnonzero(self.factor[position, :position])
        row = self.factor[position, indices]
        if self._float_scaling_safe:
            weights = map(int, np.ldexp(row, self.bits).tolist())
        else:
            weights = map(self._scaled_integer, row.tolist())
        coordinates = map(self.active_indices.__getitem__, indices.tolist())
        return sum(map(operator.mul, weights, map(integers.__getitem__, coordinates)))

    def solve_fixed(self, x, bits):
        integers = [0] * self.dimension
        for i in self.order:
            numerator = self.h.numerator*self._row_dot(i, integers)
            integers[i] = _nearest_integer((int(x[i])*self.denominator << bits)-numerator,
                                           self.h.numerator*self.diagonal_integers[i])
        return integers

    def draw(self, bits, precision_bits, *, randbelow=secrets.randbelow):
        integers, values = [0]*self.dimension, [0]*self.dimension
        position = 0
        independent_templates = {}
        shifted_templates = {}
        while position < self.dimension:
            i = self.order[position]
            if self._zero_center[i]:
                stop = position+1
                variance = self.variances[i]
                while (stop < self.dimension and self._zero_center[self.order[stop]]
                       and self.variances[self.order[stop]] == variance):
                    stop += 1
                if variance not in independent_templates:
                    independent_templates[variance] = DiscreteGaussian1D(
                        sigma_squared=variance, center=0, precision_bits=precision_bits,
                        table_max_support=0)
                block = independent_templates[variance].sample_many(stop-position, randbelow=randbelow)
                for j, value in zip(self.order[position:stop], block):
                    integers[j] = _nearest_integer(value*self.denominator << bits,
                                                   self.h.numerator*self.diagonal_integers[j])
                    values[j] = value
                position = stop
                continue
            numerator = self.h.numerator*self._row_dot(i, integers)
            variance = self.variances[i]
            if variance not in shifted_templates:
                shifted_templates[variance] = ShiftedGaussianTemplate(variance, precision_bits)
            value = shifted_templates[variance].sample_ratio(
                numerator, self.denominator << bits, randbelow=randbelow)
            integers[i] = _nearest_integer((value*self.denominator << bits)-numerator,
                                           self.h.numerator*self.diagonal_integers[i])
            values[i] = value
            position += 1
        return values, integers

    def inverse_quadratic_interval(self, x, bits, integers=None):
        if integers is None:
            integers = self.solve_fixed(x, bits)
        qhat = Fraction(sum(v*v for v in integers), 1 << (2*bits))
        cert = self.certificate
        squared_error = cert.dimension*cert.proposal_max_eigenvalue / (
            cert.proposal_min_eigenvalue * (1 << (2*bits+2)))
        error = 2*_sqrt_upper(qhat*squared_error, bits+16)+squared_error
        return max(Fraction(0), qhat-error), qhat+error


class AdaptedPerturbation:
    def __init__(self, operator, target, gadget, triangular, *, component_bits,
                 minimum_acceptance=Fraction(1, 64), max_proposals=4096):
        self.operator, self.target_squared = operator, target*target
        self.gadget_squared, self.triangular = gadget*gadget, triangular
        self.certificate = triangular.certificate
        c, d = self.certificate, self.certificate.dimension
        theta_width, theta_delta = smoothing_integer(component_bits+32)
        if c.proposal_min_eigenvalue < max(25, theta_width*theta_width):
            raise ValueError('adapted proposal does not meet normalization/center-shift conditions')
        if d*theta_delta > Fraction(1, 4):
            raise ValueError('adapted proposal normalization budget is too large')
        sqrt_dimension = math.isqrt(d)+1
        condition = c.proposal_max_eigenvalue/c.proposal_min_eigenvalue
        bits = component_bits+16
        while condition*sqrt_dimension/Fraction(1 << (bits+1)) > Fraction(1, 1 << (component_bits+13)):
            bits += 1
        self.fixed_bits, self.precision_bits = bits, component_bits+10
        center_ratio = condition*sqrt_dimension/Fraction(1 << (bits+1))
        self.proposal_distance_bound = d*(8*center_ratio+Fraction(1, 1 << self.precision_bits))+8*d*theta_delta
        exponent = -d*c.covariance_difference_bound/(2*c.target_min_eigenvalue)
        self.acceptance_lower_bound = exp_interval(exponent, component_bits+32)[0]/(1+2*d*theta_delta)
        if self.acceptance_lower_bound < minimum_acceptance:
            raise AdaptedAcceptanceCostError('adapted proposal certified acceptance is below its cost budget')
        if self.proposal_distance_bound > self.acceptance_lower_bound/2:
            raise ValueError('adapted proposal error exceeds the accepted-law budget')
        self.accepted_distance_bound = 4*self.proposal_distance_bound/self.acceptance_lower_bound
        if self.accepted_distance_bound > Fraction(d, 1 << component_bits):
            raise ValueError('adapted perturbation exceeds the existing MP12 numerical budget')
        self.max_proposals, self.proposals_used, self.refined_decisions = max_proposals, 0, 0
        self._isotropic_form = StructuredPerturbation(operator, target*target, gadget*gadget,
                                                     None, Fraction(0), Fraction(0), Fraction(0), 0)

    def correction_interval(self, x):
        c = self.certificate
        return Fraction(0), c.covariance_difference_bound*sum(v*v for v in x)/(
            c.target_min_eigenvalue*c.proposal_min_eigenvalue)

    def _accept(self, x, integers, *, randbelow):
        bits, draw = 64, randbelow(1 << 64)
        lo, hi = self.correction_interval(x)
        fixed_bits = self.fixed_bits
        target_intervals = None
        step = 0
        while True:
            pi_lo, pi_hi = pi_interval(bits+64)
            lower = exp_interval(-pi_hi*hi, bits+24)[0]
            upper = exp_interval(-pi_lo*lo, bits+24)[1]
            if Fraction(draw+1, 1 << bits) <= lower:
                return True
            if Fraction(draw, 1 << bits) >= upper:
                return False
            if target_intervals is None:
                self.refined_decisions += 1
                target_intervals = self._isotropic_form.quadratic_intervals(x)
            target_lo, target_hi = next(target_intervals)
            target_base = Fraction(sum(v*v for v in x), self.target_squared)
            prop_lo, prop_hi = self.triangular.inverse_quadratic_interval(x, fixed_bits, integers)
            lo = max(lo, target_lo+target_base-prop_hi, Fraction(0))
            hi = min(hi, target_hi+target_base-prop_lo)
            if hi < lo:
                raise ArithmeticError('certified target/proposal inverse intervals disagree')
            if step % 16 == 15:
                draw = (draw << 32)+randbelow(1 << 32)
                bits += 32
                fixed_bits += 32
                integers = self.triangular.solve_fixed(x, fixed_bits)
            step += 1

    def sample(self, *, randbelow=secrets.randbelow):
        for _ in range(self.max_proposals):
            self.proposals_used += 1
            x, integers = self.triangular.draw(self.fixed_bits, self.precision_bits, randbelow=randbelow)
            if self._accept(x, integers, randbelow=randbelow):
                return x
        from .mp12_reference import MP12ResourceLimitError
        raise MP12ResourceLimitError('adapted MP12 proposal cost budget exhausted')


@lru_cache(maxsize=1)
def _certified_geometry(n, R, adjoint, gram, norm_bound, frobenius,
                        target_squared, gadget_squared):

    operator = StructuredTrapdoor(n, R, adjoint, gram, norm_bound, frobenius)
    mu = target_squared-gadget_squared*(norm_bound+1)
    full_dimension = (len(R)+len(adjoint))*n
    active_rows = tuple(i for i, row in enumerate(R) if any(any(poly) for poly in row))
    active_indices = tuple(i*n+j for i in active_rows for j in range(n)) + tuple(
        range(len(R)*n, full_dimension))
    independent_diagonal = None
    if len(active_rows) != len(R):
        target_squared = as_fraction(target_squared)
        if target_squared.denominator != 1:
            raise ValueError('adapted prototype requires integral squared widths')
        independent_diagonal = math.isqrt(target_squared.numerator)
        if independent_diagonal**2 != target_squared:
            raise ValueError('independent block requires an exact integer width')
        operator = StructuredTrapdoor(n, tuple(R[i] for i in active_rows),
            tuple(tuple(row[i] for i in active_rows) for row in adjoint),
            tuple(tuple(gram[i][j] for j in active_rows) for i in active_rows),
            norm_bound, frobenius)
    covariance = exact_integer_covariance(operator, target_squared, gadget_squared)
    prefix = len(active_rows)*n
    permutation = tuple(range(prefix, len(covariance))) + tuple(range(prefix))
    covariance = covariance[np.ix_(permutation, permutation)]
    active_indices = tuple(active_indices[i] for i in permutation)
    factor, components = _component_cholesky(covariance, len(adjoint)*n)
    certificate = _certify_factor(covariance, factor, mu, target_squared, components,
                                  diagonal_prefix=len(adjoint)*n)
    certificate.factorization_model += (
        f'; exact covariance components={len(components)}, '
        f'largest component={max(map(len, components))}; '
        'exact coordinate permutation; diagonal gadget block eliminated first; '
        'common inflation and original active-d residual bound')
    if independent_diagonal is not None:
        active_dimension = certificate.dimension
        certificate.dimension = full_dimension
        certificate.factorization_model += (
            f'; exact independent diagonal block; active dimension={active_dimension}, '
            f'full error-budget dimension={full_dimension}')
    del covariance
    return (factor.tobytes(order='C'), tuple(vars(certificate).values()),
            active_indices, independent_diagonal)


def prepare_adapted_perturbation(operator, target, gadget, *, component_bits,
                                 minimum_acceptance=Fraction(1, 64), max_proposals=4096):
    target, gadget = map(as_fraction, (target, gadget))
    if target <= 0 or gadget <= 0:
        raise ValueError('positive Gaussian widths required')
    if (not isinstance(max_proposals, int) or isinstance(max_proposals, bool)
            or max_proposals <= 0):
        raise ValueError('max_proposals must be a positive integer')
    if not 0 < minimum_acceptance <= 1:
        raise ValueError('minimum_acceptance must be in (0,1]')
    mu = target*target-gadget*gadget*(operator.norm_squared_bound+1)
    if mu <= 0:
        raise ValueError('target covariance lacks a positive spectral certificate')
    freeze_matrix = lambda matrix: tuple(tuple(tuple(poly) for poly in row) for row in matrix)
    encoded_factor, certificate_values, active_indices, independent_diagonal = _certified_geometry(
        operator.n, freeze_matrix(operator.R), freeze_matrix(operator.adjoint),
        freeze_matrix(operator.gram), operator.norm_squared_bound,
        operator.frobenius_squared, target*target, gadget*gadget)
    certificate = AdaptedCertificate(*certificate_values)
    factor = np.frombuffer(encoded_factor, dtype=np.float64).reshape(
        len(active_indices), len(active_indices))
    triangular = DyadicTriangular(factor, certificate, active_indices, independent_diagonal)
    return AdaptedPerturbation(operator, target, gadget, triangular, component_bits=component_bits,
                               minimum_acceptance=as_fraction(minimum_acceptance), max_proposals=max_proposals)


def prepare_adapted_plan(key, sigma, columns=1, *, statistical_bits=80,
                         gadget_sigma=None, minimum_acceptance=Fraction(1,64),
                         max_proposals=4096):

    from .mp12_reference import (MP12Plan, MP12ResourceLimitError, MP12SamplingReport, centered_tail_bound,
                                 sample_parameters, key_instance_digest)
    from .gaussian_reference import GadgetCosetSampler
    parameters=sample_parameters(key,columns,statistical_bits=statistical_bits,gadget_sigma=gadget_sigma)
    target=as_fraction(sigma)
    if target <= 0 or target*target < parameters['minimum_target_squared']:
        raise ValueError(f'MP12 target sigma is too small; this certified profile requires sigma >= {parameters["minimum_target_sigma"]}')
    ideal_tail=centered_tail_bound(key,target,columns,statistical_bits,
                                   parameters['structured_operator'].frobenius_squared)
    failure=parameters['error_bound']+ideal_tail
    distance=2*failure/(1-failure) if failure<1 else Fraction(1)
    if distance>Fraction(1,1 << statistical_bits):
        raise ValueError('target sigma/q does not certify centered-overflow and sampling error')
    gadget=GadgetCosetSampler(key.ring.q,key.base,key.gadget_digits,parameters['gadget_sigma'],
                             precision_bits=parameters['component_bits'])
    return _prepare_adapted_plan_from_parameters(key,target,columns,parameters,
        ideal_tail,failure,distance,gadget,statistical_bits=statistical_bits,
        minimum_acceptance=minimum_acceptance,max_proposals=max_proposals)


def _prepare_adapted_plan_from_parameters(key,target,columns,parameters,
        ideal_tail,failure,distance,gadget,*,statistical_bits,minimum_acceptance,max_proposals):
    from .mp12_reference import MP12Plan, MP12ResourceLimitError, MP12SamplingReport, key_instance_digest
    rg=parameters['gadget_sigma']
    try:
        adapted=prepare_adapted_perturbation(parameters['structured_operator'],target,rg,
                                            component_bits=parameters['component_bits'],
                                            minimum_acceptance=minimum_acceptance,max_proposals=max_proposals)
    except AdaptedAcceptanceCostError as error:
        raise MP12ResourceLimitError(str(error)) from error
    cost_failure=min(Fraction(1),columns*exp_interval(
        -max_proposals*(adapted.acceptance_lower_bound-adapted.proposal_distance_bound),
        parameters['component_bits']+32)[1])
    report=MP12SamplingReport(
        parameters['dimension'],parameters['gadget_dimension'],columns,target,rg,
        parameters['omega'],parameters['component_bits'],statistical_bits,distance,
        parameters['error_bound'],ideal_tail,failure,adapted.certificate.target_min_eigenvalue,
        parameters['norm_bound_squared'],'adapted_discrete_rejection',
        adapted.acceptance_lower_bound,adapted.accepted_distance_bound,cost_failure,max_proposals,
        adapted.certificate.summary(), key_instance_digest=key_instance_digest(key))
    return MP12Plan([],[],[],gadget,report,adapted)
