from __future__ import annotations

from dataclasses import dataclass, field, fields
from copy import deepcopy
import math

from .distributions import (BoundedErrorDistribution, UniformBounded,
                            error_distribution_metadata, require_error_distribution)


_PAPER_DIMENSIONS = {'n': 222, 'q': 524_309, 't': 64, 'e': 10, 'd': 10}
PAPER_SECURITY_BITS = 80
PAPER_GROUP_DEPTHS = tuple(range(4, 11))


_SELECTED_PROFILE = {
    'mode': 'selected_finite',
    'params': {
        'n': 222, 'q': 524309, 't': 64, 'e': 10, 'd': 10, 'depth': 10,
        'seed': None, 'error_bound': 1, 'sigma': 2.25, 'beta': 11,
        'chi': {'type': 'uniform_bounded', 'bound': 1},
        'max_memory_bytes': 4294967296,
    },
    'yang_settings': {
        'l1': 222, 'l2': 222, 'p': 2, 'sigma1': 12,
        'auxiliary_rows': 222, 'auxiliary_hiding_bits': 128,
        'gaussian_tail': 14, 'max_memory_bytes': 4294967296,
    },
    'security': {
        'claimed_security_bits': None,
        'finite_author_parameter_set': False,
        'Yang_soundness_conditions_hold': False,
    },
}


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    d, power = value - 1, 0
    while d % 2 == 0:
        d //= 2
        power += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        x = pow(base, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(power - 1):
            x = x * x % value
            if x == value - 1:
                break
        else:
            return False
    return True


@dataclass(frozen=True)
class FDGEParams:
    n: int = 222
    q: int = 524_309
    t: int = 64
    e: int = 10
    d: int = 10
    depth: int = 10
    seed: int | None = None
    error_bound: int | None = None
    sigma: float | None = None
    beta: int | None = None
    max_memory_bytes: int = 512 * 1024 * 1024
    chi: BoundedErrorDistribution | None = field(default=None, kw_only=True)

    def __post_init__(self):
        for name, expected in _PAPER_DIMENSIONS.items():
            value = getattr(self, name)
            if type(value) is not int or value != expected:
                raise ValueError(f'{name} must be {expected}')
        if type(self.depth) is not int or self.depth not in PAPER_GROUP_DEPTHS:
            raise ValueError('depth must be 4..10')

    @classmethod
    def paper_table2(cls, **explicit_overrides) -> "FDGEParams":
        return cls(**explicit_overrides)

    @property
    def k(self) -> int:
        return (self.q - 1).bit_length() if self.q > 1 else 0

    @property
    def nk(self) -> int:
        return self.n * self.k

    @property
    def m(self) -> int:
        return 2 * self.n * self.k

    @property
    def N(self) -> int:
        return 1 << self.depth

    @property
    def qhalf(self) -> int:
        return self.q // 2

    @property
    def missing_parameters(self) -> list[str]:
        return [name for name in ("error_bound", "sigma", "beta") if getattr(self, name) is None]

    @property
    def worst_case_noise_bound(self) -> int | None:
        if self.error_bound is None or self.beta is None:
            return None
        return self.error_bound * (1 + self.m * self.beta)

    @property
    def paper_correctness_margin_supported(self) -> bool | None:
        bound = self.worst_case_noise_bound

        return None if bound is None else 5 * bound < self.q

    @property
    def exact_rounding_margin_supported(self) -> bool | None:
        bound = self.worst_case_noise_bound
        return None if bound is None else 2 * bound < self.qhalf

    @property
    def correctness_bound_supported(self) -> bool | None:
        paper = self.paper_correctness_margin_supported
        return None if paper is None else paper and self.exact_rounding_margin_supported

    def validation_errors(self, *, require_distributions: bool = True) -> list[str]:
        errors = []
        for name in ("n", "q", "t", "e", "d", "depth", "max_memory_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                errors.append(f"{name} must be a positive integer")
        if errors:
            return errors
        if self.q > (1 << 63) - 1 or not _is_prime_64(self.q) or self.q == 2:
            errors.append('q must be an odd int64 prime')
        if self.t > self.m or self.d > self.t:
            errors.append("require d <= t <= m")
        if self.m >= self.q or self.nk >= self.q:
            errors.append('require m < q and nk < q')
        if require_distributions and self.missing_parameters:
            errors.append("Missing parameters: " + ", ".join(self.missing_parameters))
        if self.error_bound is not None and (not isinstance(self.error_bound, int)
                                             or isinstance(self.error_bound, bool)
                                             or self.error_bound < 1):
            errors.append('error_bound must be a positive integer')
        if self.sigma is not None and (not isinstance(self.sigma, (int, float))
                                       or isinstance(self.sigma, bool)
                                       or not math.isfinite(self.sigma) or self.sigma <= 0):
            errors.append('sigma must be finite and > 0')
        if self.beta is not None and (not isinstance(self.beta, int)
                                      or isinstance(self.beta, bool) or self.beta < 1):
            errors.append('beta must be a positive integer')
        if self.chi is not None:
            try:
                require_error_distribution(self.chi, self.error_bound)
            except ValueError as exc:
                errors.append(str(exc))
        if not errors and self.correctness_bound_supported is False:
            errors.append(f"Correctness bound failed: {self.worst_case_noise_bound}; q={self.q}")
        return errors

    def validate(self, *, require_distributions: bool = True) -> "FDGEParams":
        errors = self.validation_errors(require_distributions=require_distributions)
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def resolved_sigma(self) -> float:
        if self.sigma is None:
            raise ValueError('sigma is required')
        return self.sigma

    def resolved_beta(self) -> int:
        if self.beta is None:
            raise ValueError('beta is required')
        return self.beta

    def resolved_chi(self) -> BoundedErrorDistribution:
        return require_error_distribution(self.chi, self.error_bound)

    def to_dict(self) -> dict:
        from .mathutils import discrete_gaussian_tail_bound

        result = {item.name: getattr(self, item.name) for item in fields(self) if item.name != 'chi'}
        result['chi'] = error_distribution_metadata(self.chi)
        result.update(k=self.k, m=self.m, nk=self.nk, N=self.N if self.depth >= 0 else None,
                      missing_parameters=self.missing_parameters,
                      worst_case_noise_bound=self.worst_case_noise_bound,
                      correctness_bound_supported=self.correctness_bound_supported,
                      paper_correctness_margin_supported=self.paper_correctness_margin_supported,
                      exact_rounding_margin_supported=self.exact_rounding_margin_supported,
                      validation_errors=self.validation_errors(),
                      distribution_independent_validation_errors=self.validation_errors(require_distributions=False),
                      claimed_security_bits=None,
                      error_distribution=result['chi'],
                      secret_distribution=(None if self.sigma is None or self.beta is None else
                                           'exp(-pi*x^2/sigma^2), x in Z; beta: tail bound'),
                      sampler_implementation=(None if self.sigma is None or self.beta is None else
                          'interval_CDF/finite_PMF; not_constant_time'),
                      randomness='runtime_rng')
        if self.sigma is not None and self.beta is not None and self.sigma > 0 and self.beta >= 0:
            result["gaussian_probability_bound_outside_beta"] = discrete_gaussian_tail_bound(self.sigma, self.beta)
        return result


def load_selected_profile():
    from .yang import YangProtocolParams

    config = deepcopy(_SELECTED_PROFILE)
    values = dict(config['params'])
    chi = values.pop('chi')
    if not isinstance(chi, dict) or chi.get('type') != 'uniform_bounded':
        raise ValueError('Uniform error distribution required')
    values['chi'] = UniformBounded(chi['bound'])
    params = FDGEParams(**values).validate()
    params.resolved_chi()
    settings = YangProtocolParams(**config['yang_settings'])
    settings.validate(params.q)
    return params, settings, config
