from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

from .linalg import centered_matrix, concat_cols, const_poly, mat_add, mat_mul, mat_sub, matrix_equal, zero_matrix
from .ring import Poly, Ring


@dataclass
class TrapdoorKey:

    ring: Ring
    rows: int
    prefix_cols: int
    gadget_digits: int
    base: int
    A: List[List[Poly]]
    R: List[List[Poly]]
    G: List[List[Poly]]
    A_centered: list
    R_centered: list
    generation_report: dict | None = None


@dataclass(frozen=True)
class TrapdoorDistribution:


    name: str
    sample_matrix: Callable[[Ring, int, int], Sequence[Sequence[Poly]]]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("trapdoor distribution needs a nonempty name")
        if not callable(self.sample_matrix):
            raise TypeError("trapdoor distribution needs a callable matrix sampler")


def gadget_matrix(ring: Ring, rows: int, digits: int, base: int) -> List[List[Poly]]:
    if rows <= 0 or digits <= 0 or base < 2:
        raise ValueError("invalid gadget dimensions/base")
    G = zero_matrix(ring, rows, rows * digits)
    for row in range(rows):
        for digit in range(digits):
            G[row][row * digits + digit] = const_poly(ring, base ** digit)
    return G


class MissingTrapdoorParameterError(RuntimeError):
    pass


def _checked_matrix(matrix, ring: Ring, rows: int, cols: int, name: str) -> None:
    if len(matrix) != rows or any(len(row) != cols for row in matrix):
        raise ValueError(f"{name} has invalid dimensions")
    if any(not isinstance(poly, Poly) or poly.ring != ring for row in matrix for poly in row):
        raise ValueError(f"{name} has an invalid polynomial ring")


def _matrix_parameters(matrix: Sequence[Sequence[Poly]], name: str) -> tuple[Ring, int, int]:
    try:
        rows, columns = len(matrix), len(matrix[0])
        if not rows or not columns or not isinstance(matrix[0][0], Poly):
            raise ValueError(f"{name} must be a nonempty matrix of polynomials")
        ring = matrix[0][0].ring
        _checked_matrix(matrix, ring, rows, columns, name)
    except (TypeError, IndexError, AttributeError) as error:
        raise ValueError(f"{name} must be a nonempty rectangular matrix of polynomials") from error
    return ring, rows, columns


def _canonical_gadget_parameters(G: Sequence[Sequence[Poly]], ring: Ring,
                                 rows: int) -> tuple[int, int]:
    gadget_ring, gadget_rows, columns = _matrix_parameters(G, "G")
    if gadget_ring != ring or gadget_rows != rows or columns % rows:
        raise ValueError("G has the wrong ring or dimensions for A_left")
    digits = columns // rows
    if digits < 2:
        raise ValueError("the supported canonical gadget needs at least two digits to identify its base")
    base = int(G[0][1].coeffs[0])
    if not 2 <= base < ring.q or not matrix_equal(G, gadget_matrix(ring, rows, digits, base)):
        raise ValueError("only the canonical gadget G=I_alpha tensor [1,b,...,b^(digits-1)] is supported")
    return digits, base


def trapdoor_key_from_matrices(A: Sequence[Sequence[Poly]], R: Sequence[Sequence[Poly]]) -> TrapdoorKey:

    ring, rows, columns = _matrix_parameters(A, "A")
    secret_ring, prefix, gadget_columns = _matrix_parameters(R, "R")
    if secret_ring != ring or prefix >= columns or gadget_columns != columns - prefix:
        raise ValueError("A and R have incompatible dimensions or polynomial rings")
    public = [[poly.copy() for poly in row] for row in A]
    secret = [[poly.copy() for poly in row] for row in R]
    left = [row[:prefix] for row in public]
    right = [row[prefix:] for row in public]
    G = mat_add(mat_mul(left, secret), right)
    try:
        digits, base = _canonical_gadget_parameters(G, ring, rows)
    except ValueError as error:
        raise ValueError(f"A and R do not form a supported canonical G-trapdoor: {error}") from error
    return TrapdoorKey(ring, rows, prefix, digits, base, public, secret, G,
                       centered_matrix(public), centered_matrix(secret))


def construct_trapdoor(
    ring: Ring,
    A_left: Sequence[Sequence[Poly]],
    R: Sequence[Sequence[Poly]],
    gadget_digits: int,
    base: int,
) -> TrapdoorKey:

    if gadget_digits <= 0 or base < 2 or not A_left or not A_left[0]:
        raise ValueError("invalid trapdoor dimensions or gadget base")
    rows, prefix_cols = len(A_left), len(A_left[0])
    _checked_matrix(A_left, ring, rows, prefix_cols, "A_left")
    _checked_matrix(R, ring, prefix_cols, rows * gadget_digits, "R")
    left = [[poly.copy() for poly in row] for row in A_left]
    secret = [[poly.copy() for poly in row] for row in R]
    G = gadget_matrix(ring, rows, gadget_digits, base)
    right = mat_sub(G, mat_mul(left, secret))
    A = concat_cols(left, right)
    return TrapdoorKey(ring, rows, prefix_cols, gadget_digits, base,
                       A, secret, G, centered_matrix(A), centered_matrix(secret))


def _require_sigma(sigma: float | None, parameter: str) -> float:
    import math
    if sigma is None:
        raise MissingTrapdoorParameterError(f"the selected sampler requires explicit {parameter}")
    if isinstance(sigma, bool) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"{parameter} must be positive and finite")
    return float(sigma)


def _gaussian_trapdoor_profile(ring: Ring, rows: int, prefix_cols: int,
                                gadget_digits: int, sigma: float,
                                statistical_bits: int):
    from .gaussian_reference import DiscreteGaussian1D
    sigma = _require_sigma(sigma, 'trapdoor_sigma')
    if (any(type(v) is not int or v <= 0 for v in (rows,prefix_cols,gadget_digits))
            or type(statistical_bits) is not int or not 16 <= statistical_bits <= 256):
        raise ValueError('invalid Gaussian trapdoor dimensions or error budget')
    count = prefix_cols * rows * gadget_digits * ring.n
    component_bits = max(64, statistical_bits + count.bit_length() + 8)
    sampler = DiscreteGaussian1D(sigma, precision_bits=component_bits)
    if sampler.integer_weights is None:
        support_bound = max(abs(sampler.support[0]), abs(sampler.support[-1]))
    else:
        support_bound = max(abs(x) for x,weight in zip(sampler.support,sampler.integer_weights) if weight)
    return sampler, count, component_bits, support_bound


def gaussian_trapdoor_support_bound(ring: Ring, rows: int, prefix_cols: int,
                                    gadget_digits: int, sigma: float, *,
                                    statistical_bits: int = 80) -> int:

    return _gaussian_trapdoor_profile(ring,rows,prefix_cols,gadget_digits,sigma,
                                      statistical_bits)[3]


@dataclass(frozen=True)
class _GadgetGenerationContext:
    trapdoor_bound: int | None
    trapdoor_distribution: TrapdoorDistribution | None
    trapdoor_sigma: float | None
    statistical_bits: int


class ConfiguredGadget(list):

    def __init__(self, matrix, context):
        super().__init__([[poly.copy() for poly in row] for row in matrix])
        self.generation_context = context


def bind_gadget_context(G: Sequence[Sequence[Poly]], *,
                        trapdoor_bound: int | None = None,
                        trapdoor_distribution: TrapdoorDistribution | None = None,
                        trapdoor_sigma: float | None = None,
                        statistical_bits: int = 80) -> ConfiguredGadget:

    ring, rows, _ = _matrix_parameters(G, 'G')
    _canonical_gadget_parameters(G, ring, rows)
    return ConfiguredGadget(G, _GadgetGenerationContext(
        trapdoor_bound, trapdoor_distribution, trapdoor_sigma, statistical_bits))


def gen_trap(A_left: Sequence[Sequence[Poly]], G: Sequence[Sequence[Poly]]):
    key = _gen_trap_working_key(A_left, G)
    return key.A, key.R


def _gen_trap_working_key(A_left: Sequence[Sequence[Poly]], G: Sequence[Sequence[Poly]]):

    context = getattr(G, 'generation_context', None)
    if not isinstance(context, _GadgetGenerationContext):
        raise MissingTrapdoorParameterError(
            'GenTrap requires an explicitly bound G working context; use bind_gadget_context')
    return gen_trap_with_report(A_left, G,
        trapdoor_bound=context.trapdoor_bound,
        trapdoor_distribution=context.trapdoor_distribution,
        trapdoor_sigma=context.trapdoor_sigma,
        statistical_bits=context.statistical_bits)


def gen_trap_with_report(A_left: Sequence[Sequence[Poly]], G: Sequence[Sequence[Poly]], *,
                         trapdoor_bound: int | None = None,
                         trapdoor_distribution: TrapdoorDistribution | None = None,
                         trapdoor_sigma: float | None = None,
                         statistical_bits: int = 80) -> TrapdoorKey:


    from fractions import Fraction
    from .trapdoor_security import gentrap_support_bound
    ring, rows, prefix_cols = _matrix_parameters(A_left, "A_left")
    gadget_digits, base = _canonical_gadget_parameters(G, ring, rows)
    if trapdoor_distribution is None and trapdoor_sigma is None:
        raise MissingTrapdoorParameterError(
            "select an explicit trapdoor_distribution D or the Gaussian trapdoor_sigma profile")
    if trapdoor_distribution is not None and trapdoor_sigma is not None:
        raise ValueError("select only one trapdoor distribution: D or the Gaussian sigma profile")
    if trapdoor_distribution is not None and not isinstance(trapdoor_distribution, TrapdoorDistribution):
        raise TypeError("trapdoor_distribution must be a TrapdoorDistribution")
    if trapdoor_bound is None:
        raise MissingTrapdoorParameterError("trapdoor_bound must be specified independently of encryption B")
    if not isinstance(trapdoor_bound, int) or isinstance(trapdoor_bound, bool) or trapdoor_bound <= 0:
        raise ValueError("trapdoor_bound must be a positive integer")
    if type(statistical_bits) is not int or not 16 <= statistical_bits <= 256:
        raise ValueError("statistical_bits must be between 16 and 256")
    if trapdoor_bound > ring.q // 2:
        raise ValueError("trapdoor_bound exceeds the centered ring representation")
    if trapdoor_distribution is not None:
        secret = trapdoor_distribution.sample_matrix(ring, prefix_cols, rows * gadget_digits)
        _checked_matrix(secret, ring, prefix_cols, rows * gadget_digits, "sampled R")
        observed_bound = max(abs(int(x)) for row in secret for poly in row for x in poly.centered())
        if observed_bound > trapdoor_bound:
            raise ValueError("sampled R exceeds the declared trapdoor_bound; the draw is not retried")
        key = construct_trapdoor(ring, A_left, secret, gadget_digits, base)
        key.generation_report = {
            'distribution': trapdoor_distribution.name,
            'distribution_source': 'caller-supplied D; shape and observed coefficient checks only',
            'declared_bound': trapdoor_bound,
            'observed_coefficient_bound': observed_bound,
            'requested_statistical_bits': statistical_bits,
            'sampling_error_bound': None,
            'sampling_error_status': 'not certified for the caller-supplied sampler',
            'public_key_pseudorandomness': 'requires a separate ring regularity/MLWE parameter analysis',
            'statistical_uniformity_support_test': gentrap_support_bound(
                ring.q, ring.n, rows, prefix_cols, rows*gadget_digits,
                trapdoor_bound).summary(statistical_bits),
        }
        return key
    sigma = _require_sigma(trapdoor_sigma, "trapdoor_sigma")
    sampler, count, component_bits, support_bound = _gaussian_trapdoor_profile(
        ring,rows,prefix_cols,gadget_digits,sigma,statistical_bits)
    if support_bound > trapdoor_bound:
        raise ValueError(f"declared trapdoor_bound={trapdoor_bound} is too small for this Gaussian profile; require at least {support_bound}")
    from .sampler_native import SystemRandomBuffer
    randbelow = SystemRandomBuffer().randbelow
    coefficients = np.asarray(sampler.sample_many(count, randbelow=randbelow),
                              dtype=np.int64).reshape(prefix_cols, rows*gadget_digits, ring.n)
    secret = [[Poly(ring, poly) for poly in row] for row in coefficients]
    key = construct_trapdoor(ring, A_left, secret, gadget_digits, base)
    if count * sampler.total_variation_bound > Fraction(1, 1 << statistical_bits):
        raise ArithmeticError("GenTrap Gaussian distribution budget was exceeded")
    key.generation_report = {
        'distribution': 'independent coefficient D_Z,s with explicit tail/quantization budget',
        'sigma': str(trapdoor_sigma), 'declared_bound': trapdoor_bound,
        'support_bound': support_bound, 'statistical_bits': statistical_bits,
        'component_bits': component_bits,
        'public_key_pseudorandomness': 'requires a separate ring regularity/MLWE parameter analysis',
        'statistical_uniformity_support_test': gentrap_support_bound(
            ring.q, ring.n, rows, prefix_cols, rows*gadget_digits,
            support_bound).summary(statistical_bits),
    }
    return key


def sample_d_strict(key: TrapdoorKey, U: Sequence[Sequence[Poly]],
                    sigma: float | None, gadget_kernel_sigma: float | None = None, *, statistical_bits: int = 80,
                    max_dimension: int = 256, backend: str = 'auto',
                    minimum_acceptance=1/64, max_proposals: int = 4096) -> List[List[Poly]]:

    _require_sigma(sigma, "preimage sigma")
    return sample_d_strict_with_report(key, U, sigma,
                                      gadget_kernel_sigma=gadget_kernel_sigma,
                                      statistical_bits=statistical_bits,
                                      max_dimension=max_dimension, backend=backend,
                                      minimum_acceptance=minimum_acceptance, max_proposals=max_proposals)[0]


def sample_d_strict_with_report(key: TrapdoorKey, U: Sequence[Sequence[Poly]],
                               sigma: float | None, gadget_kernel_sigma=None, *,
                               statistical_bits: int = 80, max_dimension: int = 256,
                               backend: str = 'auto', minimum_acceptance=1/64,
                               max_proposals: int = 4096):
    from .mp12_reference import prepare_sampler, sample_with_plan
    _require_sigma(sigma, "preimage sigma")
    if not U or not U[0]:
        raise ValueError("target U must have at least one column")
    _checked_matrix(U, key.ring, key.rows, len(U[0]), "U")
    plan = prepare_sampler(key, sigma, len(U[0]), statistical_bits=statistical_bits,
                            gadget_sigma=gadget_kernel_sigma, max_dimension=max_dimension,
                            backend=backend, minimum_acceptance=minimum_acceptance,
                            max_proposals=max_proposals)
    return sample_with_plan(key, U, plan)


def sample_d_extended(key: TrapdoorKey, extension_B: Sequence[Sequence[Poly]],
                      U: Sequence[Sequence[Poly]], sigma: float | None, **kwargs) -> List[List[Poly]]:

    sigma = _require_sigma(sigma, "certificate_sigma")
    extended, source_rows = extend_trapdoor(key, extension_B)
    sample = sample_d_strict(extended, U, sigma, **kwargs)
    return [sample[i] for i in source_rows]


def extend_trapdoor(key: TrapdoorKey, extension_B: Sequence[Sequence[Poly]]) -> tuple[TrapdoorKey, tuple[int, ...]]:
    from copy import deepcopy
    try:
        public, secret, gadget = deepcopy((key.A, key.R, key.G))
        key = TrapdoorKey(key.ring, key.rows, key.prefix_cols, key.gadget_digits,
                         key.base, public, secret, gadget, [], [])
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("invalid input trapdoor") from error
    if not verify_trapdoor(key):
        raise ValueError("invalid input trapdoor")
    if not extension_B or not extension_B[0]:
        raise ValueError("extension B must have at least one column")
    extra = len(extension_B[0])
    _checked_matrix(extension_B, key.ring, key.rows, extra, "extension B")
    extension = [[poly.copy() for poly in row] for row in extension_B]
    public = [row[:key.prefix_cols] + extra_row + row[key.prefix_cols:]
              for row, extra_row in zip(key.A, extension)]
    secret = key.R + zero_matrix(key.ring, extra, key.rows * key.gadget_digits)
    extended = TrapdoorKey(key.ring, key.rows, key.prefix_cols + extra,
                          key.gadget_digits, key.base, public, secret, key.G,
                          centered_matrix(public), centered_matrix(secret))
    prefix, gadget = key.prefix_cols, key.rows * key.gadget_digits
    source_rows = tuple(range(prefix)) + tuple(range(prefix + extra, prefix + extra + gadget)) + tuple(range(prefix, prefix + extra))
    return extended, source_rows


def sample_d(A: Sequence[Sequence[Poly]], u: Sequence[Poly],
             R: Sequence[Sequence[Poly]], s: float | None) -> List[Poly]:
    return sample_d_with_options(A, u, R, s)


def sample_d_with_options(A: Sequence[Sequence[Poly]], u: Sequence[Poly],
                         R: Sequence[Sequence[Poly]], s: float | None, **kwargs) -> List[Poly]:

    key = trapdoor_key_from_matrices(A, R)
    if (len(u) != key.rows or
            any(not isinstance(poly, Poly) or poly.ring != key.ring for poly in u)):
        raise ValueError("u must be a syndrome vector with the row dimension and ring of A")
    result = sample_d_strict(key, [[poly] for poly in u], s, **kwargs)
    return [row[0] for row in result]


def sample_pre(A: Sequence[Sequence[Poly]], R: Sequence[Sequence[Poly]],
               U: Sequence[Poly] | Sequence[Sequence[Poly]],
               sigma: float | None) -> List[Poly] | List[List[Poly]]:
    return sample_pre_with_options(A, R, U, sigma)


def sample_pre_with_options(A: Sequence[Sequence[Poly]], R: Sequence[Sequence[Poly]],
                           U: Sequence[Poly] | Sequence[Sequence[Poly]],
                           sigma: float | None = None, **kwargs) -> List[Poly] | List[List[Poly]]:

    if 'perturbation_sigma' in kwargs:
        raise TypeError("use sigma for the target SamplePre width; perturbation_sigma was a misleading legacy name")
    key = trapdoor_key_from_matrices(A, R)
    try:
        if not len(U):
            raise ValueError("U must be a nonempty syndrome vector or matrix")
        is_vector = isinstance(U[0], Poly)
    except (TypeError, IndexError) as error:
        raise ValueError("U must be a nonempty syndrome vector or matrix") from error
    if is_vector:
        if len(U) != key.rows or any(not isinstance(poly, Poly) or poly.ring != key.ring for poly in U):
            raise ValueError("U must be a syndrome vector with the row dimension and ring of A")
        target = [[poly] for poly in U]
    else:
        target = U
    result = sample_d_strict(key, target, sigma, **kwargs)
    return [row[0] for row in result] if is_vector else result


def verify_trapdoor(key: TrapdoorKey) -> bool:
    try:
        rows, prefix, columns = key.rows, key.prefix_cols, key.rows * key.gadget_digits
        if rows <= 0 or prefix <= 0 or key.gadget_digits <= 0 or key.base < 2:
            return False
        _checked_matrix(key.A, key.ring, rows, prefix + columns, "A")
        _checked_matrix(key.R, key.ring, prefix, columns, "R")
        expected_G = gadget_matrix(key.ring, rows, key.gadget_digits, key.base)
        if not matrix_equal(key.G, expected_G):
            return False
        left = [row[:prefix] for row in key.A]
        right = [row[prefix:] for row in key.A]
        return matrix_equal(mat_sub(expected_G, mat_mul(left, key.R)), right)
    except (ValueError, TypeError, IndexError, AttributeError):
        return False
