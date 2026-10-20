from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .randomness import uniform_bounded_coeffs, uniform_mod_coeffs
from .ring import Poly, Ring


def const_poly(ring: Ring, value: int) -> Poly:
    coeffs = [0] * ring.n
    coeffs[0] = int(value)
    return Poly.from_ints(ring, coeffs)


def zero_matrix(ring: Ring, rows: int, cols: int) -> List[List[Poly]]:
    return [[ring.zero() for _ in range(cols)] for _ in range(rows)]


def random_vec(ring: Ring, length: int) -> List[Poly]:
    coeffs = uniform_mod_coeffs(int(length) * ring.n, ring.q).reshape(int(length), ring.n)
    return [Poly(ring, coeffs[i]) for i in range(int(length))]


def random_matrix(ring: Ring, rows: int, cols: int) -> List[List[Poly]]:
    coeffs = uniform_mod_coeffs(int(rows) * int(cols) * ring.n, ring.q).reshape(int(rows), int(cols), ring.n)
    return [[Poly(ring, coeffs[i, j]) for j in range(int(cols))] for i in range(int(rows))]


def small_poly(ring: Ring, bound: int) -> Poly:
    return Poly(ring, uniform_bounded_coeffs(ring.n, bound))


def small_vec(ring: Ring, length: int, bound: int) -> List[Poly]:
    coeffs = uniform_bounded_coeffs(int(length) * ring.n, bound).reshape(int(length), ring.n)
    return [Poly(ring, coeffs[i]) for i in range(int(length))]


def ternary_vec(ring: Ring, length: int) -> List[Poly]:
    return small_vec(ring, length, 1)


def _matrix_shape(A: Sequence[Sequence[Poly]]) -> tuple[int, int, Ring]:
    if not A or not A[0]:
        raise ValueError("matrix must have at least one row and column")
    cols = len(A[0])
    ring = A[0][0].ring
    if any(len(row) != cols for row in A):
        raise ValueError("ragged matrix")
    if any(poly.ring != ring for row in A for poly in row):
        raise ValueError("ring mismatch")
    return len(A), cols, ring


def _check_vector(v: Sequence[Poly], length: int, ring: Ring) -> None:
    if len(v) != length:
        raise ValueError("dimension mismatch")
    if any(poly.ring != ring for poly in v):
        raise ValueError("ring mismatch")


def _coefficient_matrix(A, q):
    return np.asarray([[p.coeffs for p in row] for row in A], dtype=np.int64) % q


class _PreparedMatrix:

    def __init__(self, matrix):
        self.matrix = matrix
        self._left = None

    def _prepare(self, ring):
        from .ring_native import _PreparedLeft
        values = _coefficient_matrix(self.matrix, ring.q)
        if (self._left is None or self._left.shape != values.shape
                or self._left.canonical != values.tobytes()):
            self._left = _PreparedLeft(values)
        return self._left

    def mat_mul(self, right):
        rows, inner, ring = _matrix_shape(self.matrix)
        b_rows, columns, b_ring = _matrix_shape(right)
        if inner != b_rows or ring != b_ring:
            raise ValueError('matrix shape or ring mismatch')
        if (ring.n, ring.q) != (128, 4294967969):
            return mat_mul(self.matrix, right)
        values = self._prepare(ring).multiply(_coefficient_matrix(right, ring.q))
        return [[Poly(ring, values[i, j]) for j in range(columns)] for i in range(rows)]

    def transpose_vec(self, vector):
        rows, columns, ring = _matrix_shape(self.matrix)
        _check_vector(vector, rows, ring)
        if (ring.n, ring.q) != (128, 4294967969):
            return mat_transpose_vec_mul(self.matrix, vector)
        values = self._prepare(ring).multiply(
            _coefficient_matrix([[poly] for poly in vector], ring.q), transpose_left=True)
        return [Poly(ring, values[j, 0]) for j in range(columns)]


def mat_add(A: Sequence[Sequence[Poly]], B: Sequence[Sequence[Poly]]) -> List[List[Poly]]:
    if _matrix_shape(A) != _matrix_shape(B):
        raise ValueError("matrix shape or ring mismatch")
    return [[a + b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]


def mat_sub(A: Sequence[Sequence[Poly]], B: Sequence[Sequence[Poly]]) -> List[List[Poly]]:
    if _matrix_shape(A) != _matrix_shape(B):
        raise ValueError("matrix shape or ring mismatch")
    return [[a - b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]


def vec_add(a: Sequence[Poly], b: Sequence[Poly]) -> List[Poly]:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    return [left + right for left, right in zip(a, b)]


def mat_mul(A: Sequence[Sequence[Poly]], B: Sequence[Sequence[Poly]]) -> List[List[Poly]]:
    rows, mid, ring = _matrix_shape(A)
    b_rows, cols, b_ring = _matrix_shape(B)
    if mid != b_rows or ring != b_ring:
        raise ValueError("matrix shape or ring mismatch")
    from .ring_native import supports, product
    if ring.n >= 32 and supports(ring.n, ring.q, mid):
        values = product(_coefficient_matrix(A, ring.q),
                         _coefficient_matrix(B, ring.q), ring.q)
        return [[Poly(ring, values[i, j]) for j in range(cols)] for i in range(rows)]
    out = zero_matrix(ring, rows, cols)
    for i in range(rows):
        for j in range(cols):
            for k in range(mid):
                out[i][j] = out[i][j] + A[i][k] * B[k][j]
    return out


def _centered_arrays_matrix(A: Sequence[Sequence[Poly]]) -> List[List[np.ndarray]]:
    _matrix_shape(A)
    return [[poly.centered() for poly in row] for row in A]


def centered_matrix(A: Sequence[Sequence[Poly]]) -> List[List[np.ndarray]]:
    return _centered_arrays_matrix(A)


def mat_vec_mul(A: Sequence[Sequence[Poly]], x: Sequence[Poly]) -> List[Poly]:
    rows, cols, ring = _matrix_shape(A)
    _check_vector(x, cols, ring)
    from .ring_native import supports, product
    if ring.n >= 32 and supports(ring.n, ring.q, cols):
        values = product(_coefficient_matrix(A, ring.q),
                         _coefficient_matrix([[p] for p in x], ring.q), ring.q)
        return [Poly(ring, values[i, 0]) for i in range(rows)]
    out = []
    for i in range(rows):
        acc = ring.zero()
        for j in range(cols):
            acc = acc + A[i][j] * x[j]
        out.append(acc)
    return out


def mat_transpose_vec_mul(A: Sequence[Sequence[Poly]], x: Sequence[Poly]) -> List[Poly]:
    rows, cols, ring = _matrix_shape(A)
    _check_vector(x, rows, ring)
    from .ring_native import supports, product
    if ring.n >= 32 and supports(ring.n, ring.q, rows):
        values = product(_coefficient_matrix(A, ring.q),
                         _coefficient_matrix([[p] for p in x], ring.q), ring.q, transpose_left=True)
        return [Poly(ring, values[j, 0]) for j in range(cols)]
    out = []
    for j in range(cols):
        acc = ring.zero()
        for i in range(rows):
            acc = acc + A[i][j] * x[i]
        out.append(acc)
    return out


def concat_cols(A: Sequence[Sequence[Poly]], B: Sequence[Sequence[Poly]]) -> List[List[Poly]]:
    rows_a, _, ring_a = _matrix_shape(A)
    rows_b, _, ring_b = _matrix_shape(B)
    if rows_a != rows_b or ring_a != ring_b:
        raise ValueError("matrix shape or ring mismatch")
    return [list(left) + list(right) for left, right in zip(A, B)]


def as_column(v: Sequence[Poly]) -> List[List[Poly]]:
    return [[x] for x in v]


def first_column(A: Sequence[Sequence[Poly]]) -> List[Poly]:
    _matrix_shape(A)
    return [row[0] for row in A]


def vec_equal(a: Sequence[Poly], b: Sequence[Poly]) -> bool:
    return len(a) == len(b) and all(a[i] == b[i] for i in range(len(a)))


def matrix_equal(A: Sequence[Sequence[Poly]], B: Sequence[Sequence[Poly]]) -> bool:
    return len(A) == len(B) and all(vec_equal(A[i], B[i]) for i in range(len(A)))


def poly_inf_norm(p: Poly) -> int:
    centered = p.centered()
    return int(np.max(np.abs(centered))) if centered.size else 0


def vec_inf_norm(v: Sequence[Poly]) -> int:
    return max((poly_inf_norm(p) for p in v), default=0)
