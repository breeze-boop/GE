from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import secrets
from typing import Sequence

from .ring import Poly, Ring, is_prime_u64


_ENCODING_DOMAIN = b"LatGE-structured-CRS-v1\x00"
_ENTRY_DOMAIN = b"LatGE-structured-CRS-entry-v1\x00"
_VIEW_DOMAIN = b"LatGE-structured-CRS-research-view-v1\x00"
_PROFILES = ("paper_iii_d", "research_generic")


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "little") + value


def _uniform_coefficients(context: bytes, ring: Ring) -> tuple[int, ...]:
    width = (ring.q.bit_length() + 7) // 8
    limit = ((1 << (8 * width)) // ring.q) * ring.q
    xof = hashlib.shake_256(context)
    size, consumed = max(64, width * ring.n), 0
    result = []
    while len(result) < ring.n:
        stream = xof.digest(size)
        while consumed + width <= size and len(result) < ring.n:
            candidate = int.from_bytes(stream[consumed:consumed + width], "little")
            consumed += width
            if candidate < limit:
                result.append(candidate % ring.q)
        size *= 2
    return tuple(result)


@dataclass(frozen=True)
class StructuredCRS:


    ring: Ring
    kappa1: int
    kappa2: int
    message_count: int
    auxiliary_count: int = 3
    seed: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    profile: str = "paper_iii_d"
    source_digest: bytes = b""
    cache_capacity: int = field(default=256, compare=False)
    _cache: OrderedDict = field(default_factory=OrderedDict, init=False,
                                repr=False, compare=False)
    _encoded: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ring, Ring) or self.ring.n & (self.ring.n - 1) or not is_prime_u64(self.ring.q):
            raise ValueError("CRS requires a power-of-two cyclotomic degree and prime modulus")
        for name in ("kappa1", "kappa2", "message_count"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 1 << 64:
                raise ValueError(f"{name} must be a positive uint64 integer")
        if type(self.auxiliary_count) is not int or not 0 <= self.auxiliary_count < 1 << 64:
            raise ValueError("auxiliary_count must be a nonnegative uint64 integer")
        if self.ring.n >= 1 << 64 or self.columns >= 1 << 64:
            raise ValueError("CRS dimensions exceed the canonical uint64 encoding")
        if type(self.seed) is not bytes or len(self.seed) != 32:
            raise ValueError("CRS seed must be exactly 32 public bytes")
        if self.profile not in _PROFILES:
            raise ValueError("unknown CRS profile")
        if type(self.source_digest) is not bytes or len(self.source_digest) not in (0, 32):
            raise ValueError("source_digest must be empty or a 32-byte parent-descriptor digest")
        if self.profile == "paper_iii_d" and self.source_digest:
            raise ValueError("a derived research CRS cannot be labelled paper_iii_d")
        if type(self.cache_capacity) is not int or not 0 <= self.cache_capacity <= 1024:
            raise ValueError("cache_capacity must be between 0 and 1024")
        integers = (self.ring.n, self.ring.q, self.kappa1, self.kappa2,
                    self.message_count, self.auxiliary_count,
                    self.total_messages, self.columns)
        encoded = (_ENCODING_DOMAIN + _frame(self.profile.encode("ascii")) +
                   b"".join(value.to_bytes(8, "little") for value in integers) +
                   _frame(self.seed) + _frame(self.source_digest))
        object.__setattr__(self, "_encoded", encoded)

    @property
    def total_messages(self) -> int:
        return self.message_count + self.auxiliary_count

    @property
    def columns(self) -> int:
        return self.kappa1 + self.total_messages + self.kappa2

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def encode(self) -> bytes:
        return self._encoded

    @classmethod
    def from_params(cls, params, *, seed: bytes | None = None,
                    auxiliary_count: int = 3, cache_capacity: int = 256):
        from .strict_layout import strict_paper_layout
        layout = strict_paper_layout(params)
        return cls(Ring(params.ell, params.q), params.kappa1, params.kappa2,
                   layout.packed_message_count, auxiliary_count,
                   secrets.token_bytes(32) if seed is None else seed,
                   "paper_iii_d", cache_capacity=cache_capacity)

    def research_view(self, message_count: int, *, auxiliary_count: int = 3,
                      cache_capacity: int | None = None):

        source = hashlib.sha256(self.encode()).digest()
        seed = hashlib.sha256(_VIEW_DOMAIN + _frame(self.encode())).digest()
        return StructuredCRS(self.ring, self.kappa1, self.kappa2,
                             message_count, auxiliary_count, seed,
                             "research_generic", source,
                             self.cache_capacity if cache_capacity is None else cache_capacity)

    def _random_entry(self, kind: str, row: int, col: int) -> Poly:
        key = (kind, row, col)
        coefficients = self._cache.get(key)
        if coefficients is None:
            context = (_ENTRY_DOMAIN + _frame(self.encode()) +
                       _frame(kind.encode("ascii")) + row.to_bytes(8, "little") +
                       col.to_bytes(8, "little"))
            coefficients = _uniform_coefficients(context, self.ring)
            if self.cache_capacity:
                self._cache[key] = coefficients
                if len(self._cache) > self.cache_capacity:
                    self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return Poly.from_ints(self.ring, coefficients)

    def entry(self, kind: str, row: int, col: int) -> Poly:
        if kind not in ("A1", "A2"):
            raise ValueError("matrix kind must be A1 or A2")
        height = self.kappa1 if kind == "A1" else self.total_messages
        if type(row) is not int or type(col) is not int or not 0 <= row < height or not 0 <= col < self.columns:
            raise ValueError("CRS entry index out of range")
        if kind == "A1":
            if col < self.kappa1:
                return self.ring.one() if row == col else self.ring.zero()
        else:
            if col < self.kappa1:
                return self.ring.zero()
            if col < self.kappa1 + self.total_messages:
                return self.ring.one() if row == col - self.kappa1 else self.ring.zero()
        return self._random_entry(kind, row, col)

    def _validate_vector(self, vec: Sequence[Poly]) -> None:
        if len(vec) != self.columns:
            raise ValueError("opening dimension mismatch")
        if any(not isinstance(poly, Poly) or poly.ring != self.ring for poly in vec):
            raise ValueError("opening contains an element from another ring")

    def binding(self, vec: Sequence[Poly]) -> list[Poly]:
        self._validate_vector(vec)
        result = []
        for row in range(self.kappa1):
            value = vec[row].copy()
            for col in range(self.kappa1, self.columns):
                value = value + self._random_entry("A1", row, col) * vec[col]
            result.append(value)
        return result

    def _message(self, index: int, vec: Sequence[Poly]) -> Poly:
        value = vec[self.kappa1 + index].copy()
        for col in range(self.kappa1 + self.total_messages, self.columns):
            value = value + self._random_entry("A2", index, col) * vec[col]
        return value

    def message(self, index: int, vec: Sequence[Poly]) -> Poly:
        if type(index) is not int or not 0 <= index < self.total_messages:
            raise ValueError("message index out of range")
        self._validate_vector(vec)
        return self._message(index, vec)

    def messages(self, vec: Sequence[Poly]) -> list[Poly]:
        self._validate_vector(vec)
        return [self._message(index, vec) for index in range(self.total_messages)]
