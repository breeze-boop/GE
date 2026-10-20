from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
from functools import cached_property
from typing import Iterator

from .decomposition import g_beta, idec
from .paper_relation import PaperPublicStatement, PaperPrivateWitness, Variable
from .ring import Poly, is_prime_u64


@dataclass(frozen=True)
class ScalarBlock:
    name: str
    offset: int
    size: int
    kind: str
    shape: tuple[int, ...]
    source_index: int = -1


@dataclass(frozen=True)
class ScalarLinearConstraint:
    name: str
    terms: tuple[tuple[int, int], ...]
    target: int = 0


@dataclass(frozen=True)
class ScalarProductConstraint:
    name: str
    left: int
    right: int
    result: int

    def as_polynomial(self) -> 'ScalarPolynomialConstraint':
        return ScalarPolynomialConstraint(self.name, ((1, (self.left, self.right)), (-1, (self.result,))))


@dataclass(frozen=True)
class ScalarMembershipConstraint:
    name: str
    variable: int
    domain: str

    def as_polynomial(self) -> 'ScalarPolynomialConstraint':
        if self.domain == 'binary':
            power = 2
        elif self.domain == 'ternary':
            power = 3
        else:
            raise ValueError('unsupported membership domain')
        return ScalarPolynomialConstraint(self.name, ((1, (self.variable,) * power), (-1, (self.variable,))))


@dataclass(frozen=True)
class ScalarPolynomialConstraint:
    name: str
    terms: tuple[tuple[int, tuple[int, ...]], ...]
    target: int = 0


@dataclass(frozen=True)
class LoweredRelationCheck:
    valid: bool
    failed_constraints: tuple[str, ...]


def tau(poly: Poly) -> tuple[int, ...]:
    return tuple(int(value) for value in poly.coeffs)


def rot_coefficient(poly: Poly, row: int, column: int) -> int:
    n, q = poly.ring.n, poly.ring.q
    if not 0 <= row < n or not 0 <= column < n:
        raise IndexError('rotation matrix index out of bounds')
    degree = row - column
    return (int(poly.coeffs[degree]) if degree >= 0 else -int(poly.coeffs[degree + n])) % q


def _bound_plan(lower: int, upper: int) -> tuple[str, tuple[int, ...], int]:
    if lower == upper:
        return 'constant', (), lower
    if (lower, upper) == (0, 1):
        return 'binary', (), 0
    if (lower, upper) == (-1, 1):
        return 'ternary', (), 0
    if lower == -upper:
        return 'ternary_digits', tuple(g_beta(upper)), 0
    return 'binary_digits', tuple(g_beta(upper - lower)), lower


def _normal_terms(terms, modulus: int) -> tuple[tuple[int, int], ...]:
    combined: dict[int, int] = {}
    for coefficient, variable in terms:
        combined[variable] = (combined.get(variable, 0) + int(coefficient)) % modulus
    return tuple((coefficient, variable) for variable, coefficient in sorted(combined.items()) if coefficient)


@dataclass(frozen=True)
class LoweredRelation:
    statement: PaperPublicStatement
    blocks: tuple[ScalarBlock, ...]
    scalar_count: int
    product_blocks: tuple[ScalarBlock, ...]
    digit_blocks: tuple[ScalarBlock | None, ...]

    @property
    def modulus(self) -> int:
        return self.statement.ring.q

    @property
    def n(self) -> int:
        return self.statement.ring.n

    @cached_property
    def _base_blocks(self) -> dict[str, ScalarBlock]:
        return {block.name: block for block in self.blocks if block.kind == 'base'}

    def base_block(self, group: str) -> ScalarBlock:
        return self._base_blocks[group]

    def coefficient_id(self, variable: Variable, coefficient: int) -> int:
        block = self.base_block(variable.group)
        if not 0 <= variable.index < block.shape[0] or not 0 <= coefficient < self.n:
            raise IndexError('private coefficient index out of bounds')
        return block.offset + variable.index * self.n + coefficient

    def digest(self) -> bytes:
        digest = hashlib.sha256(b'LatGE-coefficient-relation-v1')
        def put(value):
            encoded = value if isinstance(value, bytes) else json.dumps(value, separators=(',', ':')).encode()
            digest.update(len(encoded).to_bytes(8, 'little'))
            digest.update(encoded)
        encodings = {}
        def polynomial_bytes(poly):
            key = id(poly)
            encoded = encodings.get(key)
            if encoded is None:
                encoded = encodings[key] = poly.to_bytes()
            return encoded
        put((self.n, self.modulus, self.scalar_count))
        put(self.statement.encoding)
        put(self.statement.zk_parameters)
        put(self.statement.zk_crs_seed)
        put(self.statement.private_layout)
        put((len(self.statement.linear_equations), len(self.statement.hidden_products), len(self.statement.bounds)))
        for equation in self.statement.linear_equations:
            put(equation.name)
            put(polynomial_bytes(equation.target))
            put(len(equation.terms))
            for coefficient, variable in equation.terms:
                put((variable.group, variable.index))
                put(polynomial_bytes(coefficient))
        for product in self.statement.hidden_products:
            put((product.name, (product.left.group, product.left.index),
                 (product.right.group, product.right.index), (product.result.group, product.result.index)))
        for bound in self.statement.bounds:
            put((bound.group, bound.size, bound.lower, bound.upper, bound.supports))
        return digest.digest()

    def iter_linear_constraints(self) -> Iterator[ScalarLinearConstraint]:
        q, n = self.modulus, self.n
        for equation in self.statement.linear_equations:
            for row in range(n):
                terms = []
                for coefficient, variable in equation.terms:
                    for column in range(n):
                        weight = rot_coefficient(coefficient, row, column)
                        if weight:
                            terms.append((weight, self.coefficient_id(variable, column)))
                yield ScalarLinearConstraint(f'{equation.name}:coeff[{row}]',
                    _normal_terms(terms, q), int(equation.target.coeffs[row]))
        for product, block in zip(self.statement.hidden_products, self.product_blocks):
            for coefficient in range(n):
                terms = [(1, self.coefficient_id(product.result, coefficient))]
                for i in range(n):
                    j = (coefficient - i) % n
                    sign = 1 if i + j < n else -1
                    terms.append((-sign, block.offset + i * n + j))
                yield ScalarLinearConstraint(f'{product.name}:convolution[{coefficient}]', _normal_terms(terms, q))
        for bound_index, bound in enumerate(self.statement.bounds):
            domain, weights, offset = _bound_plan(bound.lower, bound.upper)
            digit_block = self.digit_blocks[bound_index]
            for poly_index in range(bound.size):
                for coefficient in range(n):
                    variable = self.coefficient_id(Variable(bound.group, poly_index), coefficient)
                    prefix = f'bound:{bound.group}[{poly_index},{coefficient}]'
                    if domain == 'constant':
                        yield ScalarLinearConstraint(prefix, ((1, variable),), offset % q)
                    elif digit_block is not None:
                        start = digit_block.offset + (poly_index * n + coefficient) * len(weights)
                        terms = [(1, variable)] + [(-weight, start + digit) for digit, weight in enumerate(weights)]
                        yield ScalarLinearConstraint(prefix + ':recompose', _normal_terms(terms, q), offset % q)
                    if bound.supports:
                        low, high = bound.supports[poly_index]
                        if not low <= coefficient < high:
                            yield ScalarLinearConstraint(prefix + ':support', ((1, variable),))

    def iter_product_constraints(self) -> Iterator[ScalarProductConstraint]:
        for product, block in zip(self.statement.hidden_products, self.product_blocks):
            for i in range(self.n):
                left = self.coefficient_id(product.left, i)
                for j in range(self.n):
                    yield ScalarProductConstraint(f'{product.name}:multiply[{i},{j}]', left,
                        self.coefficient_id(product.right, j), block.offset + i * self.n + j)

    def iter_membership_constraints(self) -> Iterator[ScalarMembershipConstraint]:
        for index, bound in enumerate(self.statement.bounds):
            domain, weights, _ = _bound_plan(bound.lower, bound.upper)
            if domain == 'constant':
                continue
            block = self.digit_blocks[index]
            if block is None:
                base = self.base_block(bound.group)
                for position in range(base.size):
                    yield ScalarMembershipConstraint(f'bound:{bound.group}:member[{position}]', base.offset + position, domain)
            else:
                for position in range(block.size):
                    yield ScalarMembershipConstraint(f'{block.name}:member[{position}]', block.offset + position, domain.removesuffix('_digits'))

    def iter_polynomial_constraints(self) -> Iterator[ScalarPolynomialConstraint]:
        for constraint in self.iter_product_constraints():
            yield constraint.as_polynomial()
        for constraint in self.iter_membership_constraints():
            yield constraint.as_polynomial()

    def statistics(self) -> dict[str, int]:
        linear = self.n * (len(self.statement.linear_equations) + len(self.product_blocks))
        memberships = 0
        for index, bound in enumerate(self.statement.bounds):
            domain, weights, _ = _bound_plan(bound.lower, bound.upper)
            if domain in ('constant', 'binary_digits', 'ternary_digits'):
                linear += bound.size * self.n
            if domain != 'constant':
                memberships += bound.size * self.n * (len(weights) or 1)
            linear += sum(self.n - high + low for low, high in bound.supports)
        return {'scalar_variables': self.scalar_count,
                'base_scalar_variables': sum(block.size for block in self.blocks if block.kind == 'base'),
                'linear_constraints': linear,
                'product_constraints': len(self.product_blocks) * self.n ** 2,
                'membership_constraints': memberships}


def lower_public_relation(statement: PaperPublicStatement) -> LoweredRelation:
    n, q = statement.ring.n, statement.ring.q
    if q <= 3 or not is_prime_u64(q):
        raise ValueError('binary/ternary field constraints require prime q > 3')
    blocks, groups, position = [], {}, 0
    for name, size in statement.private_layout:
        if not isinstance(name, str) or not name or name in groups or type(size) is not int or size <= 0:
            raise ValueError('invalid or duplicate private layout group')
        block = ScalarBlock(name, position, size * n, 'base', (size, n))
        blocks.append(block)
        groups[name] = block
        position += block.size
    def validate_reference(reference):
        if reference.group not in groups or type(reference.index) is not int or not 0 <= reference.index < groups[reference.group].shape[0]:
            raise ValueError('unknown or out-of-range private variable')
    for equation in statement.linear_equations:
        if equation.target.ring != statement.ring:
            raise ValueError('linear equation target ring mismatch')
        for coefficient, variable in equation.terms:
            if coefficient.ring != statement.ring:
                raise ValueError('public coefficient ring mismatch')
            validate_reference(variable)
    products = []
    for index, product in enumerate(statement.hidden_products):
        for reference in (product.left, product.right, product.result):
            validate_reference(reference)
        block = ScalarBlock(f'product_aux[{index}]', position, n * n, 'product', (n, n), index)
        blocks.append(block)
        products.append(block)
        position += block.size
    digits, seen_bounds = [], set()
    for index, bound in enumerate(statement.bounds):
        if bound.group not in groups or bound.group in seen_bounds or bound.size != groups[bound.group].shape[0]:
            raise ValueError('invalid, duplicate, or incomplete group bound')
        seen_bounds.add(bound.group)
        if type(bound.lower) is not int or type(bound.upper) is not int or not -(q // 2) <= bound.lower <= bound.upper <= q // 2:
            raise ValueError('bound must fit the unique centered representatives')
        if bound.supports and (len(bound.supports) != bound.size or any(type(low) is not int or type(high) is not int or not 0 <= low <= high <= n for low, high in bound.supports)):
            raise ValueError('invalid coefficient support')
        domain, weights, _ = _bound_plan(bound.lower, bound.upper)
        if not weights:
            digits.append(None)
            continue
        block = ScalarBlock(f'range_digits:{bound.group}', position, bound.size * n * len(weights),
                            'digits', (bound.size, n, len(weights)), index)
        blocks.append(block)
        digits.append(block)
        position += block.size
    return LoweredRelation(statement, tuple(blocks), position, tuple(products), tuple(digits))


class LazyPrivateAssignment:
    def __init__(self, relation: LoweredRelation, witness: PaperPrivateWitness):
        self.relation = relation
        self._starts = tuple(block.offset for block in relation.blocks)
        self._values = {}
        expected = dict(relation.statement.private_layout)
        if set(witness.values) != set(expected):
            raise ValueError('private witness groups do not match layout')
        for group, size in expected.items():
            vector = witness.values[group]
            if len(vector) != size or any(not isinstance(poly, Poly) or poly.ring != relation.statement.ring for poly in vector):
                raise ValueError('invalid private witness shape or ring')
            self._values[group] = tuple(tuple(int(x) for x in poly.centered()) for poly in vector)
        for bound in relation.statement.bounds:
            for index, polynomial in enumerate(self._values[bound.group]):
                if any(not bound.lower <= value <= bound.upper for value in polynomial):
                    raise ValueError('private witness coefficient violates bound')
                if bound.supports:
                    low, high = bound.supports[index]
                    if any(polynomial[:low]) or any(polynomial[high:]):
                        raise ValueError('private witness violates coefficient support')

    def __len__(self) -> int:
        return self.relation.scalar_count

    def __getitem__(self, variable: int) -> int:
        if type(variable) is not int or not 0 <= variable < len(self):
            raise IndexError('private scalar variable out of bounds')
        relation, n, q = self.relation, self.relation.n, self.relation.modulus
        block = relation.blocks[bisect_right(self._starts, variable) - 1]
        local = variable - block.offset
        if block.kind == 'base':
            return self._values[block.name][local // n][local % n] % q
        if block.kind == 'product':
            product = relation.statement.hidden_products[block.source_index]
            a = self._values[product.left.group][product.left.index][local // n]
            b = self._values[product.right.group][product.right.index][local % n]
            return a * b % q
        if block.kind == 'digits':
            bound = relation.statement.bounds[block.source_index]
            domain, weights, offset = _bound_plan(bound.lower, bound.upper)
            original, digit = divmod(local, len(weights))
            value = self._values[bound.group][original // n][original % n]
            if domain == 'ternary_digits':
                return ((-1 if value < 0 else 1) * idec(abs(value), bound.upper)[digit]) % q
            return idec(value - offset, bound.upper - bound.lower)[digit]
        raise ValueError('unknown scalar block kind')

    def __iter__(self):
        for variable in range(len(self)):
            yield self[variable]


def lower_private_witness(relation: LoweredRelation, witness: PaperPrivateWitness) -> LazyPrivateAssignment:
    return LazyPrivateAssignment(relation, witness)


