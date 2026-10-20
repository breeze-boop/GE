from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from functools import lru_cache


@lru_cache(maxsize=128)
def is_prime_u64(value: int) -> bool:
    if value < 2 or value >= 1 << 64:
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
        x = pow(base, odd, value)
        if x in (1, value - 1):
            continue
        for _ in range(power - 1):
            x = x * x % value
            if x == value - 1:
                break
        else:
            return False
    return True


def _factorize(n: int) -> List[int]:

    x = n
    factors: List[int] = []
    d = 2
    while d * d <= x:
        if x % d == 0:
            factors.append(d)
            while x % d == 0:
                x //= d
        d += 1 if d == 2 else 2
    if x > 1:
        factors.append(x)
    return factors

def _primitive_root(mod: int) -> int:

    phi = mod - 1
    fac = _factorize(phi)
    for g in range(2, mod):
        ok = True
        for p in fac:
            if pow(g, phi // p, mod) == 1:
                ok = False
                break
        if ok:
            return g
    raise ValueError("no primitive root found")

@lru_cache(maxsize=64)
def _ntt_ctx(n: int, q: int):

    if n <= 0 or n & (n - 1) or (q - 1) % (2 * n) != 0 or not is_prime_u64(q):
        return None


    g = _primitive_root(q)
    psi = pow(g, (q - 1) // (2 * n), q)

    omega = (psi * psi) % q
    inv_omega = pow(omega, q - 2, q)
    inv_n = pow(n, q - 2, q)


    psi_pows = [1] * n
    inv_psi = pow(psi, q - 2, q)
    inv_psi_pows = [1] * n
    for i in range(1, n):
        psi_pows[i] = (psi_pows[i - 1] * psi) % q
        inv_psi_pows[i] = (inv_psi_pows[i - 1] * inv_psi) % q

    return {
        "n": n,
        "q": q,
        "omega": omega,
        "inv_omega": inv_omega,
        "inv_n": inv_n,
        "psi_pows": psi_pows,
        "inv_psi_pows": inv_psi_pows,
    }

def _bitrev_perm(a: List[int]) -> None:

    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

def _ntt_inplace(a: List[int], omega: int, q: int) -> None:

    n = len(a)
    _bitrev_perm(a)
    length = 2
    while length <= n:
        wlen = pow(omega, n // length, q)
        for i in range(0, n, length):
            w = 1
            half = length // 2
            for j in range(half):
                u = a[i + j]
                v = (a[i + j + half] * w) % q
                a[i + j] = (u + v) % q
                a[i + j + half] = (u - v) % q
                w = (w * wlen) % q
        length <<= 1

def _intt_inplace(a: List[int], inv_omega: int, inv_n: int, q: int) -> None:

    n = len(a)
    _bitrev_perm(a)
    length = 2
    while length <= n:
        wlen = pow(inv_omega, n // length, q)
        for i in range(0, n, length):
            w = 1
            half = length // 2
            for j in range(half):
                u = a[i + j]
                v = (a[i + j + half] * w) % q
                a[i + j] = (u + v) % q
                a[i + j + half] = (u - v) % q
                w = (w * wlen) % q
        length <<= 1
    for i in range(n):
        a[i] = (a[i] * inv_n) % q

def _negacyclic_mul_ntt(a: Sequence[int], b: Sequence[int], n: int, q: int) -> List[int]:

    ctx = _ntt_ctx(n, q)
    if ctx is None:
        raise ValueError("NTT context unavailable for given (n,q)")

    psi_pows = ctx["psi_pows"]
    inv_psi_pows = ctx["inv_psi_pows"]
    omega = ctx["omega"]
    inv_omega = ctx["inv_omega"]
    inv_n = ctx["inv_n"]

    A = [(int(a[i]) % q) * psi_pows[i] % q for i in range(n)]
    B = [(int(b[i]) % q) * psi_pows[i] % q for i in range(n)]

    _ntt_inplace(A, omega, q)
    _ntt_inplace(B, omega, q)
    C = [(A[i] * B[i]) % q for i in range(n)]
    _intt_inplace(C, inv_omega, inv_n, q)

    out = [(C[i] * inv_psi_pows[i]) % q for i in range(n)]
    return out


def _mod_center(x: np.ndarray, q: int) -> np.ndarray:

    y = x % q
    y = np.where(y > q // 2, y - q, y)
    return y


@dataclass(frozen=True)
class Ring:
    n: int
    q: int

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n <= 0:
            raise ValueError("ring degree must be a positive integer")
        if not isinstance(self.q, int) or isinstance(self.q, bool) or not 2 <= self.q < 1 << 63:
            raise ValueError("ring modulus must lie in [2, 2^63)")

    def zero(self) -> "Poly":
        return Poly(self, np.zeros(self.n, dtype=np.int64))

    def one(self) -> "Poly":
        c = np.zeros(self.n, dtype=np.int64)
        c[0] = 1
        return Poly(self, c)


@dataclass
class Poly:
    ring: Ring
    coeffs: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.coeffs)
        if values.shape != (self.ring.n,):
            raise ValueError(f"coeffs must have shape ({self.ring.n},)")
        if values.dtype.kind in "iu":
            self.coeffs = (values % self.ring.q).astype(np.int64, copy=True)
            return
        if not all(isinstance(x, (int, np.integer)) for x in values):
            raise ValueError("polynomial coefficients must be integers")
        self.coeffs = np.array([int(x) % self.ring.q for x in values], dtype=np.int64)

    @staticmethod
    def from_ints(ring: Ring, coeffs: Iterable[int]) -> "Poly":
        values = list(coeffs)
        if not all(isinstance(x, (int, np.integer)) for x in values):
            raise ValueError("polynomial coefficients must be integers")
        arr = np.array([int(x) % ring.q for x in values], dtype=np.int64)
        if arr.size != ring.n:
            raise ValueError(f"need exactly {ring.n} coefficients")
        return Poly(ring, arr)

    def copy(self) -> "Poly":
        return Poly(self.ring, self.coeffs.copy())

    def centered(self) -> np.ndarray:
        return _mod_center(self.coeffs, self.ring.q)

    def __add__(self, other: "Poly") -> "Poly":
        self._check_same_ring(other)
        if self.ring.q >= 1 << 62:
            return Poly.from_ints(self.ring, (int(a) + int(b) for a, b in zip(self.coeffs, other.coeffs)))
        return Poly(self.ring, (self.coeffs + other.coeffs) % self.ring.q)

    def __sub__(self, other: "Poly") -> "Poly":
        self._check_same_ring(other)
        return Poly(self.ring, (self.coeffs - other.coeffs) % self.ring.q)

    def __neg__(self) -> "Poly":
        return Poly(self.ring, (-self.coeffs) % self.ring.q)

    def scalar_mul(self, c: int) -> "Poly":
        q = self.ring.q
        cc = int(c) % q
        return Poly(self.ring, np.array([(int(x) * cc) % q for x in self.coeffs], dtype=np.int64))

    def mul(self, other: "Poly") -> "Poly":

        self._check_same_ring(other)
        n = self.ring.n
        q = self.ring.q


        a_cent = self.centered().astype(np.int64, copy=False)
        b_cent = other.centered().astype(np.int64, copy=False)
        max_a = int(np.max(np.abs(a_cent))) if a_cent.size else 0
        max_b = int(np.max(np.abs(b_cent))) if b_cent.size else 0

        safe = (max_a == 0 or max_b == 0 or (n * max_a * max_b) < (1 << 62))
        if safe:
            conv = np.convolve(a_cent, b_cent)
            out = np.array(conv[:n], dtype=np.int64, copy=True)
            if n > 1:
                out[: n - 1] -= conv[n:]
            out %= q
            return Poly(self.ring, out)

        from .ring_native import supports, product
        if n >= 32 and supports(n, q, 1):
            out = product((self.coeffs % q).reshape(1, 1, n),
                          (other.coeffs % q).reshape(1, 1, n), q)
            return Poly(self.ring, out[0, 0])


        ctx = _ntt_ctx(n, q)
        if ctx is not None and (n & (n - 1) == 0):
            a = [int(x) % q for x in self.coeffs]
            b = [int(x) % q for x in other.coeffs]
            out = _negacyclic_mul_ntt(a, b, n, q)
            return Poly(self.ring, np.array(out, dtype=np.int64))


        a = [int(x) % q for x in self.coeffs]
        b = [int(x) % q for x in other.coeffs]
        out = [0] * n
        for i in range(n):
            ai = a[i]
            if ai == 0:
                continue
            for j in range(n):
                t = (ai * b[j]) % q
                k = i + j
                if k < n:
                    out[k] = (out[k] + t) % q
                else:
                    out[k - n] = (out[k - n] - t) % q
        return Poly(self.ring, np.array(out, dtype=np.int64))

    def __mul__(self, other: "Poly") -> "Poly":
        return self.mul(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Poly):
            return False
        if self.ring.n != other.ring.n or self.ring.q != other.ring.q:
            return False
        return bool(np.array_equal(self.coeffs % self.ring.q, other.coeffs % other.ring.q))

    def _check_same_ring(self, other: "Poly") -> None:
        if self.ring.n != other.ring.n or self.ring.q != other.ring.q:
            raise ValueError("ring mismatch")

    def to_bytes(self) -> bytes:
        q = self.ring.q
        blen = (q.bit_length() + 7) // 8
        if self.coeffs.dtype.kind in 'iu' and self.coeffs.dtype.itemsize <= 8 and blen <= 8:
            if int(self.coeffs.min()) < 0 or int(self.coeffs.max()).bit_length() > 8*blen:
                raise OverflowError('polynomial coefficient does not fit its byte encoding')
            if blen in (1, 2, 4, 8):
                return self.coeffs.astype(f'<u{blen}', copy=False).tobytes()
            words = np.ascontiguousarray(self.coeffs, dtype='<u8')
            return words.view(np.uint8).reshape(-1, 8)[:, :blen].tobytes()
        return b"".join(int(c).to_bytes(blen, "little") for c in self.coeffs)

    def inv_mod_xn1(self) -> "Poly":


        return poly_inv_mod_xn1(self)

    @staticmethod
    def from_bytes(ring: Ring, raw: bytes) -> "Poly":

        q = ring.q
        blen = (q.bit_length() + 7) // 8
        if len(raw) != blen * ring.n:
            raise ValueError("invalid byte length for Poly.from_bytes")
        coeffs = [int.from_bytes(raw[i * blen : (i + 1) * blen], "little") for i in range(ring.n)]
        if any(c >= q for c in coeffs):
            raise ValueError("non-canonical polynomial coefficient encoding")
        return Poly(ring, np.array(coeffs, dtype=np.int64))


def mat_vec_mul(A: Sequence[Sequence[Poly]], x: Sequence[Poly]) -> List[Poly]:
    from .linalg import mat_vec_mul as checked_mat_vec_mul
    return checked_mat_vec_mul(A, x)


def poly_mod_xn1(ring: Ring, coeffs: Sequence[int]) -> Poly:

    q = ring.q
    n = ring.n
    arr = [0] * n
    for i, c in enumerate(coeffs):
        c = int(c)
        if c == 0:
            continue
        k = i % n
        sign = -1 if ((i // n) % 2 == 1) else 1
        arr[k] = (arr[k] + sign * c) % q
    return Poly.from_ints(ring, arr)


def poly_trim(a: List[int]) -> List[int]:
    while a and a[-1] == 0:
        a.pop()
    return a


def poly_sub_mod(a: List[int], b: List[int], q: int) -> List[int]:
    n = max(len(a), len(b))
    out = [0] * n
    for i in range(n):
        out[i] = ((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) % q
    return poly_trim(out)


def poly_mul_mod(a: List[int], b: List[int], q: int) -> List[int]:
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            out[i + j] = (out[i + j] + ai * bj) % q
    return poly_trim(out)


def inv_mod(a: int, q: int) -> int:
    a %= q
    if a == 0:
        raise ZeroDivisionError("no inverse")

    try:
        return pow(a, -1, q)
    except ValueError as exc:
        raise ZeroDivisionError("element is not invertible modulo q") from exc


def poly_divmod_mod(a: List[int], b: List[int], q: int) -> Tuple[List[int], List[int]]:

    a = a[:]
    b = poly_trim(b[:])
    if not b:
        raise ZeroDivisionError("division by zero poly")
    a = poly_trim(a)
    if not a or len(a) < len(b):
        return [], a
    deg_a = len(a) - 1
    deg_b = len(b) - 1
    inv_lc = inv_mod(b[-1], q)
    qout = [0] * (deg_a - deg_b + 1)
    while a and len(a) - 1 >= deg_b:
        deg_a = len(a) - 1
        t_deg = deg_a - deg_b
        t_coeff = (a[-1] * inv_lc) % q
        qout[t_deg] = t_coeff

        for i in range(len(b)):
            idx = i + t_deg
            a[idx] = (a[idx] - t_coeff * b[i]) % q
        a = poly_trim(a)
    return poly_trim(qout), a


def poly_egcd_mod(a: List[int], b: List[int], q: int) -> Tuple[List[int], List[int], List[int]]:

    a = poly_trim(a[:])
    b = poly_trim(b[:])

    r0, r1 = a, b
    s0, s1 = [1], []
    t0, t1 = [], [1]
    while r1:
        qout, r2 = poly_divmod_mod(r0, r1, q)
        r0, r1 = r1, r2
        s0, s1 = s1, poly_sub_mod(s0, poly_mul_mod(qout, s1, q), q)
        t0, t1 = t1, poly_sub_mod(t0, poly_mul_mod(qout, t1, q), q)
    return r0, s0, t0


def poly_inv_mod_xn1(a: Poly) -> Poly:

    ring = a.ring
    q = ring.q
    n = ring.n

    f = [0] * (n + 1)
    f[0] = 1
    f[n] = 1
    g, s, _t = poly_egcd_mod(list(map(int, a.coeffs.tolist())), f, q)

    if not g:
        raise ZeroDivisionError("not invertible")
    if len(g) != 1:
        raise ZeroDivisionError("not invertible")
    g0 = g[0] % q
    if g0 == 0:
        raise ZeroDivisionError("not invertible")
    s = [(coeff * inv_mod(g0, q)) % q for coeff in s]
    inv = poly_mod_xn1(ring, s)

    check = (a * inv).coeffs
    if int(check[0] % q) != 1 or np.any(check[1:] % q):
        raise ZeroDivisionError("inverse check failed")
    return inv
