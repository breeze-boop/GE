from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, localcontext
from fractions import Fraction
import hashlib
from math import isqrt
from typing import Sequence

import numpy as np

from .gaussian_reference import (
    DiscreteGaussian1D, GadgetCosetSampler, _gram_schmidt, as_fraction,
    exp_interval, gadget_kernel_basis, pi_interval, smoothing_integer,
)
from .linalg import mat_sub, matrix_equal
from .ring import Poly
from .mp12_structured import (StructuredAcceptanceCostError, StructuredTrapdoor,
                              prepare_structured_perturbation)


class MP12ResourceLimitError(RuntimeError):
    pass


def key_instance_digest(key) -> bytes:

    digest = hashlib.sha256(b'LatGE-MP12-local-key-instance-v1')
    for value in (key.ring.q, key.ring.n, key.rows, key.prefix_cols,
                  key.gadget_digits, key.base):
        data = str(value).encode('ascii')
        digest.update(len(data).to_bytes(8,'little')); digest.update(data)
    for matrix in (key.A, key.R, key.G):
        digest.update(len(matrix).to_bytes(8,'little'))
        columns = len(matrix[0]) if matrix else 0
        n, width = key.ring.n, (key.ring.q.bit_length() + 7) // 8
        if columns and all(len(row) == columns and all(
                poly.ring == key.ring and poly.coeffs.shape == (n,)
                and poly.coeffs.dtype == np.dtype('int64') for poly in row)
                for row in matrix):
            values = np.asarray([[poly.coeffs for poly in row] for row in matrix])
            if int(values.min()) < 0 or int(values.max()).bit_length() > 8 * width:
                raise OverflowError('polynomial coefficient does not fit its byte encoding')
            poly_record = np.dtype([('length', '<u8'), ('data', 'u1', (n, width))])
            framed = np.empty(len(matrix), dtype=[('columns', '<u8'),
                ('polys', poly_record, (columns,))])
            framed['columns'] = columns
            framed['polys']['length'] = n * width
            words = np.ascontiguousarray(values, dtype='<u8')
            framed['polys']['data'] = words.view(np.uint8).reshape(
                len(matrix), columns, n, 8)[..., :width]
            digest.update(framed)
            continue
        for row in matrix:
            digest.update(len(row).to_bytes(8,'little'))
            for poly in row:
                data = poly.to_bytes()
                digest.update(len(data).to_bytes(8,'little')); digest.update(data)
    return digest.digest()


def _ceil_sqrt(value: Fraction) -> int:
    candidate = isqrt(value.numerator // value.denominator)
    return candidate if candidate * candidate >= value else candidate + 1


def _lift_integer_matrix(key) -> list[list[int]]:
    n = key.ring.n
    gadget = key.rows * key.gadget_digits
    prefix_dimension, gadget_dimension = key.prefix_cols * n, gadget * n
    dimension = prefix_dimension + gadget_dimension
    result = [[0] * gadget_dimension for _ in range(dimension)]
    for row in range(key.prefix_cols):
        for column in range(gadget):
            coefficients = [int(x) for x in key.R[row][column].centered()]
            for i in range(n):
                for j in range(n):
                    result[row * n + i][column * n + j] = coefficients[(i - j) % n] * (1 if i >= j else -1)
    for i in range(gadget_dimension):
        result[prefix_dimension + i][i] = 1
    return result


def exact_ldl(matrix: Sequence[Sequence[Fraction]]) -> tuple[list[list[Fraction]], list[Fraction]]:
    dimension = len(matrix)
    if not dimension or any(len(row) != dimension for row in matrix):
        raise ValueError('covariance must be a nonempty square matrix')
    if any(matrix[i][j] != matrix[j][i] for i in range(dimension) for j in range(i)):
        raise ValueError('covariance is not symmetric')
    lower = [[Fraction(0)] * dimension for _ in range(dimension)]
    diagonal = [Fraction(0)] * dimension
    for i in range(dimension):
        lower[i][i] = Fraction(1)
        diagonal[i] = Fraction(matrix[i][i]) - sum(lower[i][k] ** 2 * diagonal[k] for k in range(i))
        if diagonal[i] <= 0:
            raise ValueError('perturbation covariance is not positive definite')
        for j in range(i + 1, dimension):
            lower[j][i] = (Fraction(matrix[j][i]) - sum(lower[j][k] * lower[i][k] * diagonal[k] for k in range(i))) / diagonal[i]
    return lower, diagonal


@dataclass(frozen=True)
class MP12SamplingReport:
    coefficient_dimension: int
    gadget_dimension: int
    columns: int
    target_sigma: Fraction
    gadget_sigma: Fraction
    smoothing_width: int
    component_bits: int
    requested_statistical_bits: int
    statistical_distance_bound: Fraction
    unconditioned_sampling_distance_bound: Fraction
    ideal_centered_tail_bound: Fraction
    centered_overflow_failure_bound: Fraction
    minimum_conditional_parameter_squared: Fraction
    norm_bound_squared: int
    perturbation_backend: str = 'exact_ldl'
    perturbation_acceptance_lower_bound: Fraction | None = None
    perturbation_distance_bound: Fraction | None = None
    proposal_cost_failure_bound: Fraction = Fraction(0)
    max_proposals_per_column: int | None = None
    numerical_certificate: dict | None = None
    key_instance_digest: bytes = b''

    def summary(self) -> dict:
        with localcontext() as context:
            context.prec, context.rounding = 24, ROUND_CEILING
            error_upper = Decimal(self.statistical_distance_bound.numerator) / Decimal(self.statistical_distance_bound.denominator)
            failure_upper = Decimal(self.centered_overflow_failure_bound.numerator) / Decimal(self.centered_overflow_failure_bound.denominator)
        return {
            'algorithm': 'MP12 Algorithm 3; ' + self.perturbation_backend + ' discrete perturbation oracle',
            'coefficient_dimension': self.coefficient_dimension,
            'gadget_dimension': self.gadget_dimension,
            'columns': self.columns,
            'target_sigma': str(self.target_sigma),
            'gadget_sigma': str(self.gadget_sigma),
            'smoothing_width': self.smoothing_width,
            'component_bits': self.component_bits,
            'requested_statistical_bits': self.requested_statistical_bits,
            'statistical_distance_upper_bound': str(error_upper),
            'centered_overflow_failure_upper_bound': str(failure_upper),
            'distribution_scope': 'conditional on successful return; overflow/cost exhaustion raises without changing sigma',
            'minimum_conditional_parameter_squared': str(self.minimum_conditional_parameter_squared),
            'mlwe_security_estimate': 'not inferred from sampler correctness',
            'perturbation_acceptance_lower_bound': str(self.perturbation_acceptance_lower_bound),
            'perturbation_statistical_distance_bound': str(self.perturbation_distance_bound),
            'proposal_cost_failure_upper_bound': str(self.proposal_cost_failure_bound),
            'max_proposals_per_column': self.max_proposals_per_column,
            'numerical_certificate': self.numerical_certificate,
            'local_key_instance_digest': self.key_instance_digest.hex(),
        }


@dataclass
class MP12Plan:
    lower: list[list[Fraction]]
    diagonal: list[Fraction]
    lift: list[list[int]]
    gadget: GadgetCosetSampler
    report: MP12SamplingReport
    structured: object | None = None


def sample_parameters(key, columns: int = 1, *, statistical_bits: int = 80,
                      gadget_sigma=None) -> dict:
    from .trapdoor import verify_trapdoor
    if not verify_trapdoor(key):
        raise ValueError('invalid canonical trapdoor')
    if type(statistical_bits) is not int or not 16 <= statistical_bits <= 256:
        raise ValueError('statistical_bits must be between 16 and 256')
    if type(columns) is not int or columns <= 0:
        raise ValueError('columns must be a positive integer')
    dimension = (key.prefix_cols + key.rows * key.gadget_digits) * key.ring.n
    gadget_dimension = key.rows * key.gadget_digits * key.ring.n
    calls = (dimension + gadget_dimension) * columns
    component_bits = max(64, statistical_bits + calls.bit_length() + 12)
    omega, delta = smoothing_integer(component_bits)
    _, squared_norms = _gram_schmidt(gadget_kernel_basis(key.ring.q, key.base, key.gadget_digits))
    minimum_gadget = omega * max(2, _ceil_sqrt(max(squared_norms)))
    selected_gadget = Fraction(minimum_gadget) if gadget_sigma is None else as_fraction(gadget_sigma)
    if selected_gadget <= 0 or selected_gadget ** 2 < max(squared_norms) * omega * omega or selected_gadget ** 2 < 2 * omega * omega:
        raise ValueError('gadget sigma does not meet the GPV/MP12 smoothing conditions')
    operator = StructuredTrapdoor.from_key(key)
    norm_bound = operator.norm_squared_bound
    minimum_target_squared = (norm_bound + 1) * (selected_gadget ** 2 + 2 * omega * omega)
    error_bound = 64 * calls * delta + Fraction(calls, 1 << component_bits)
    if error_bound > Fraction(1, 1 << statistical_bits):
        raise ArithmeticError('composed statistical-distance budget was exceeded')
    return dict(dimension=dimension, gadget_dimension=gadget_dimension,
                component_bits=component_bits, omega=omega, delta=delta,
                gadget_sigma=selected_gadget, norm_bound_squared=norm_bound,
                minimum_target_squared=minimum_target_squared,
                minimum_target_sigma=_ceil_sqrt(minimum_target_squared),
                error_bound=error_bound, structured_operator=operator)


def centered_tail_bound(key, sigma: Fraction, columns: int,
                       statistical_bits: int, trapdoor_frobenius_squared: int) -> Fraction:

    dimension = (key.prefix_cols + key.rows * key.gadget_digits) * key.ring.n
    norm_bound = trapdoor_frobenius_squared + 1
    last_digit = (key.ring.q - 1) // key.base ** (key.gadget_digits - 1)
    preimage_norm = norm_bound * key.rows * key.ring.n * (
        (key.gadget_digits - 1) * (key.base - 1) ** 2 + last_digit ** 2)
    first_outside = key.ring.q // 2 + 1
    if first_outside ** 2 <= preimage_norm:
        raise ValueError('cannot certify centered-sample tail: canonical preimage bound is too large for q')
    pi_lower, _ = pi_interval(max(128, statistical_bits + 32))
    ratio = exp_interval(-pi_lower * Fraction(2 * first_outside + 1) / (sigma * sigma),
                         statistical_bits + 64)[1]
    if ratio >= 1:
        raise ValueError('cannot certify centered-sample tail: sigma is too large for q')
    prefactor = columns * 2 * dimension * (1 + sigma) ** (dimension - 1) / (1 - ratio)
    magnitude_bits = max(0, prefactor.numerator.bit_length() - prefactor.denominator.bit_length() + 1)
    precision = statistical_bits + magnitude_bits + 32
    numerator = exp_interval(-pi_lower * Fraction(first_outside ** 2 - preimage_norm) / (sigma * sigma),
                             precision)[1]
    return min(Fraction(1), prefactor * numerator)


def prepare_sampler(key, sigma, columns: int = 1, *, statistical_bits: int = 80,
                    gadget_sigma=None, max_dimension: int = 256,
                    backend: str = 'auto', minimum_acceptance=Fraction(1, 64),
                    max_proposals: int = 4096) -> MP12Plan:

    if backend not in ('auto', 'dense_ldl', 'structured_rejection', 'adapted_rejection'):
        raise ValueError('unknown MP12 perturbation backend')
    if not isinstance(max_dimension, int) or isinstance(max_dimension, bool) or max_dimension < 1:
        raise ValueError('max_dimension must be a positive integer')
    if backend == 'adapted_rejection':
        from .mp12_adapted import prepare_adapted_plan
        return prepare_adapted_plan(key, sigma, columns, statistical_bits=statistical_bits,
                                    gadget_sigma=gadget_sigma, minimum_acceptance=minimum_acceptance,
                                    max_proposals=max_proposals)
    parameters = sample_parameters(key, columns, statistical_bits=statistical_bits, gadget_sigma=gadget_sigma)
    dimension = parameters['dimension']
    selected_backend = ('dense_ldl' if dimension <= max_dimension else 'structured_rejection') if backend == 'auto' else backend
    if selected_backend == 'dense_ldl' and dimension > max_dimension:
        raise MP12ResourceLimitError(
            f'exact MP12 reference coefficient dimension {dimension} exceeds max_dimension={max_dimension}; '
            'select the structured_rejection backend to avoid cubic Fraction LDL')
    target = as_fraction(sigma)
    if target <= 0 or target * target < parameters['minimum_target_squared']:
        raise ValueError(f'MP12 target sigma is too small; this conservative profile requires sigma >= {parameters["minimum_target_sigma"]}')
    ideal_tail = centered_tail_bound(key, target, columns, statistical_bits,
                                    parameters['structured_operator'].frobenius_squared)
    failure = parameters['error_bound'] + ideal_tail
    distance = 2 * failure / (1 - failure) if failure < 1 else Fraction(1)
    if distance > Fraction(1, 1 << statistical_bits):
        raise ValueError('MP12 sigma/q profile cannot certify the requested centered-overflow and sampling-error budget')
    rg = parameters['gadget_sigma']
    oracle = GadgetCosetSampler(key.ring.q, key.base, key.gadget_digits, rg,
                                precision_bits=parameters['component_bits'])
    if selected_backend == 'structured_rejection':
        try:
            structured = prepare_structured_perturbation(
                parameters['structured_operator'], target, rg, dimension=dimension,
                gadget_dimension=parameters['gadget_dimension'], omega=parameters['omega'],
                delta=parameters['delta'], component_bits=parameters['component_bits'],
                minimum_acceptance=as_fraction(minimum_acceptance), max_proposals=max_proposals)
        except StructuredAcceptanceCostError as error:
            if backend == 'auto':
                from .mp12_adapted import _prepare_adapted_plan_from_parameters
                return _prepare_adapted_plan_from_parameters(key, target, columns, parameters,
                    ideal_tail, failure, distance, oracle, statistical_bits=statistical_bits,
                    minimum_acceptance=minimum_acceptance, max_proposals=max_proposals)
            raise MP12ResourceLimitError(str(error)) from error
        minimum_parameter = target*target-rg*rg*(parameters['norm_bound_squared']+1)
        cost_failure = min(Fraction(1), columns * exp_interval(
            -max_proposals*(structured.acceptance_lower_bound-structured.proposal_distance_bound),
            parameters['component_bits'] + 32)[1])
        report = MP12SamplingReport(
            dimension, parameters['gadget_dimension'], columns, target, rg,
            parameters['omega'], parameters['component_bits'], statistical_bits,
            distance, parameters['error_bound'], ideal_tail, failure,
            minimum_parameter, parameters['norm_bound_squared'], 'structured_rejection',
            structured.acceptance_lower_bound, structured.accepted_distance_bound,
            cost_failure, max_proposals, key_instance_digest=key_instance_digest(key))
        return MP12Plan([], [], [], oracle, report, structured)
    lift = _lift_integer_matrix(key)
    covariance = [[(target * target if i == j else Fraction(0))
                   - rg * rg * sum(a * b for a, b in zip(lift[i], lift[j]))
                   for j in range(dimension)] for i in range(dimension)]
    lower, diagonal = exact_ldl(covariance)
    if min(diagonal) < parameters['omega'] ** 2:
        raise ValueError('a perturbation conditional Gaussian is below the smoothing bound')
    report = MP12SamplingReport(dimension, parameters['gadget_dimension'], columns,
                                target, rg, parameters['omega'], parameters['component_bits'],
                                statistical_bits, distance, parameters['error_bound'],
                                ideal_tail, failure, min(diagonal),
                                parameters['norm_bound_squared'],
                                key_instance_digest=key_instance_digest(key))
    return MP12Plan(lower, diagonal, lift, oracle, report)


def _sample_perturbation(plan: MP12Plan, *, randbelow) -> list[int]:
    if plan.structured is not None:
        return plan.structured.sample(randbelow=randbelow)
    residuals, result = [], []
    for i, variance in enumerate(plan.diagonal):
        center = sum(plan.lower[i][j] * residuals[j] for j in range(i))
        sampler = DiscreteGaussian1D(center=center, sigma_squared=variance,
                                     precision_bits=plan.report.component_bits)
        value = sampler.sample(randbelow=randbelow)
        result.append(value)
        residuals.append(Fraction(value) - center)
    return result


def sample_with_plan(key, U: Sequence[Sequence[Poly]], plan: MP12Plan):
    if plan.report.key_instance_digest != key_instance_digest(key):
        raise ValueError('sampling plan does not match these key matrices and gadget parameters')
    n, columns = key.ring.n, plan.report.columns
    if len(U) != key.rows or any(len(row) != columns for row in U):
        raise ValueError('target dimensions do not match the sampling plan')
    if any(not isinstance(poly, Poly) or poly.ring != key.ring for row in U for poly in row):
        raise ValueError('target has the wrong ring')
    from .sampler_native import SystemRandomBuffer
    randbelow = SystemRandomBuffer().randbelow
    total_rows = key.prefix_cols + key.rows * key.gadget_digits
    perturbations = np.asarray([
        _sample_perturbation(plan, randbelow=randbelow) for _ in range(columns)], dtype=object)
    p_coefficients = np.asarray(perturbations % key.ring.q, dtype=np.int64).reshape(
        columns, total_rows, n)
    p_polys = [[Poly(key.ring, p_coefficients[column, row])
                for column in range(columns)] for row in range(total_rows)]
    from .linalg import _PreparedMatrix
    prepared_public = _PreparedMatrix(key.A)
    adjusted = mat_sub(U, prepared_public.mat_mul(p_polys))
    gadget_values = plan.gadget.sample_many(
        (int(adjusted[row][column].coeffs[coefficient])
         for column in range(columns) for row in range(key.rows) for coefficient in range(n)),
        randbelow=randbelow)
    gadget_columns = np.asarray(gadget_values, dtype=object).reshape(
        columns, key.rows, n, key.gadget_digits).transpose(0, 1, 3, 2).reshape(
        columns, plan.report.gadget_dimension).tolist()
    lifted_columns = (plan.structured.operator.lift_many(gadget_columns)
                      if plan.structured is not None else
                      [[sum(a * b for a, b in zip(row, z)) for row in plan.lift]
                       for z in gadget_columns])
    integers = perturbations + np.asarray(lifted_columns, dtype=object)
    half = key.ring.q // 2
    if np.any(integers < -half) or np.any(integers > half):
        raise ArithmeticError('integer MP12 sample exceeds the centered ring representation; increase q')
    coefficients = np.asarray(integers % key.ring.q, dtype=np.int64).reshape(
        columns, total_rows, n)
    samples = [[Poly(key.ring, coefficients[column, row])
                for column in range(columns)] for row in range(total_rows)]
    if not matrix_equal(prepared_public.mat_mul(samples), U):
        raise ArithmeticError('MP12 sampler failed the exact preimage equation')
    return samples, plan.report
