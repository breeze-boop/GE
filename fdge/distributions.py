from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from numbers import Integral

import numpy as np


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    result = int(value)
    if result < minimum or result > np.iinfo(np.int64).max:
        raise ValueError(f'{name} must be between {minimum} and signed int64 maximum')
    return result


class BoundedErrorDistribution(ABC):
    @property
    @abstractmethod
    def support_bound(self) -> int:
        pass

    def validate_bound(self, bound: int) -> None:
        bound = _integer(bound, 'B', minimum=1)
        support_bound = _integer(self.support_bound, 'chi support bound')
        if support_bound > bound:
            raise ValueError('chi support must be within [-B,B]')

    @abstractmethod
    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


@dataclass(frozen=True)
class UniformBounded(BoundedErrorDistribution):
    bound: int

    def __post_init__(self):
        object.__setattr__(self, 'bound', _integer(self.bound, 'uniform support bound'))

    @property
    def support_bound(self):
        return self.bound

    def sample(self, rng, size):
        size = _integer(size, 'sample size')
        return rng.integers(-self.bound, self.bound + 1, size=size, dtype=np.int64)

    def to_dict(self):
        return {'type': 'uniform_bounded', 'bound': self.bound,
                'law': 'uniform integers in [-bound,bound]',
                'source': 'experimental_choice; paper_unspecified',
                'implementation': 'rng.integers; runtime_rng'}


def require_error_distribution(chi, bound: int | None) -> BoundedErrorDistribution:
    if chi is None:
        raise ValueError('Explicit chi distribution required')
    if not isinstance(chi, BoundedErrorDistribution):
        raise ValueError('chi must implement BoundedErrorDistribution')
    if bound is None:
        raise ValueError('error_bound B required')
    chi.validate_bound(bound)
    return chi


def error_distribution_metadata(chi: BoundedErrorDistribution | None) -> dict | None:
    if chi is None:
        return None
    if not isinstance(chi, BoundedErrorDistribution):
        raise ValueError('chi must implement BoundedErrorDistribution')
    metadata = chi.to_dict()
    if not isinstance(metadata, dict):
        raise ValueError('chi metadata must be a JSON object')
    try:
        return json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('chi metadata must be finite JSON') from exc


def sample_bounded_errors(chi, rng, size: int, bound: int) -> np.ndarray:
    chi = require_error_distribution(chi, bound)
    size = _integer(size, 'sample size')
    values = np.asarray(chi.sample(rng, size))
    if values.shape != (size,) or values.dtype.kind not in 'iu':
        raise ValueError('Invalid chi sample shape or dtype')
    if np.any(values > bound) or (values.dtype.kind == 'i' and np.any(values < -bound)):
        raise ValueError('chi sample outside [-B,B]')
    return values.astype(np.int64, copy=False)
