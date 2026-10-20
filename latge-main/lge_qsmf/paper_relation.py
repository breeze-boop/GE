from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .decomposition import g_beta, rdec
from .hash_to_ring import H1_mat, H2_vec
from .ring import Poly, Ring
from .structured_crs import StructuredCRS


@dataclass(frozen=True)
class Variable:
    group: str
    index: int


@dataclass(frozen=True)
class LinearEquation:
    name: str
    terms: tuple[tuple[Poly, Variable], ...]
    target: Poly


@dataclass(frozen=True)
class HiddenProduct:
    name: str
    left: Variable
    right: Variable
    result: Variable


@dataclass(frozen=True)
class CoefficientBound:
    group: str
    size: int
    lower: int
    upper: int
    supports: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class PaperPublicStatement:
    ring: Ring
    encoding: bytes
    linear_equations: tuple[LinearEquation, ...]
    hidden_products: tuple[HiddenProduct, ...]
    bounds: tuple[CoefficientBound, ...]
    private_layout: tuple[tuple[str, int], ...]
    zk_parameters: tuple[tuple[str, int | None], ...] = ()
    zk_crs_seed: bytes = b''
    zk_crs: StructuredCRS | None = None

    def digest(self) -> bytes:
        digest = hashlib.sha256(b'LatGE-paper-relation-v3')
        def add(value):
            digest.update(_framed(value))
        encodings = {}
        def polynomial_bytes(poly):
            key = id(poly)
            encoded = encodings.get(key)
            if encoded is None:
                encoded = encodings[key] = poly.to_bytes()
            return encoded
        add(self.encoding)
        add(json.dumps((self.ring.n, self.ring.q, self.private_layout,
                        self.zk_parameters), separators=(',', ':')).encode())
        add(self.zk_crs_seed)
        add(b'' if self.zk_crs is None else self.zk_crs.encode())
        for eq in self.linear_equations:
            add(eq.name.encode()); add(polynomial_bytes(eq.target))
            for coefficient, variable in eq.terms:
                add(polynomial_bytes(coefficient))
                add(json.dumps((variable.group, variable.index)).encode())
            add(b'end-equation')
        for product in self.hidden_products:
            add(json.dumps((product.name, (product.left.group, product.left.index),
                            (product.right.group, product.right.index),
                            (product.result.group, product.result.index))).encode())
        for bound in self.bounds:
            add(json.dumps((bound.group, bound.size, bound.lower, bound.upper,
                            bound.supports)).encode())
        return digest.digest()


@dataclass
class PaperPrivateWitness:
    values: dict[str, list[Poly]]


@dataclass(frozen=True)
class RelationCheck:
    valid: bool
    failed_constraints: tuple[str, ...]


def _framed(data: bytes) -> bytes:
    return len(data).to_bytes(8, 'little') + data


def build_public_statement(gpk: Any, ct: Any) -> PaperPublicStatement:
    from .scheme import (_pack_matrix, _pack_vec, _tag_boundaries,
                         ciphertext_shape_ok, _vector_shape, RISISStatement)
    gpk.validate()
    pp, p, ring = gpk.pp, gpk.pp.params, gpk.pp.ring_q
    if not ciphertext_shape_ok(pp, ct):
        raise ValueError('invalid ciphertext shape')
    relation = ct.relation
    if (not isinstance(relation, RISISStatement) or
            not _vector_shape(relation.AR, p.k, ring) or
            not isinstance(relation.uR, Poly) or relation.uR.ring != ring):
        raise ValueError('missing public RISIS statement (AR,uR)')
    boundaries = _tag_boundaries(pp)
    one, zero = ring.one(), ring.zero()
    scale = one.scalar_mul(p.q // 4)
    weights = g_beta((p.q - 1) // 2)
    if len(weights) != p.s:
        raise ValueError('inconsistent paper decomposition length')
    equations: list[LinearEquation] = []
    products: list[HiddenProduct] = []
    V = Variable
    def equation(name, terms, target):
        equations.append(LinearEquation(name, tuple(terms), target))

    for j in range(p.d):
        for l in range(3 * p.alpha):
            products.append(HiddenProduct(f'tag_product[{j},{l}]', V('tag', j),
                V('v', 5*p.alpha+l), V('tag_v', j*3*p.alpha+l)))
    for i in range(p.alpha):
        terms = [(gpk.pk_gm.A[i][j], V('v', j)) for j in range(5*p.alpha)]
        terms += [(pp.A_tags[0][i][j], V('v', 5*p.alpha+j)) for j in range(3*p.alpha)]
        terms += [(pp.A_tags[d+1][i][j], V('tag_v', d*3*p.alpha+j))
                  for d in range(p.d) for j in range(3*p.alpha)]
        terms += [(-pp.B2[i][j], V('w', j)) for j in range(p.s*p.alpha)]
        equation(f'certificate[{i}]', terms, pp.u[i])
        terms = [(pp.B0[i][j], V('r', j)) for j in range(p.k*p.alpha)]
        terms += [(pp.B1[i][j], V('h', j)) for j in range(p.s)]
        terms += [(one.scalar_mul(-weight), V('w', i*p.s+j)) for j, weight in enumerate(weights)]
        equation(f'commitment_decomposition[{i}]', terms, zero)
    equation('recipient_decomposition', [(one, V('A_user', 7*p.alpha1-1))] +
             [(one.scalar_mul(-weight), V('h', j)) for j, weight in enumerate(weights)], zero)

    H = H1_mat(ring, ct.vk_ots, p.alpha1, p.k)
    h = H2_vec(ring, ct.vk_ots, p.alpha1)
    for j in range(7*p.alpha1):
        for i in range(p.alpha1):
            index = i*7*p.alpha1+j
            products.append(HiddenProduct(f'recipient_product[{i},{j}]',
                V('A_user', index), V('z_rec', i), V('A_z', index)))
        equation(f'encryption_recipient_1[{j}]',
            [(one, V('A_z', i*7*p.alpha1+j)) for i in range(p.alpha1)] + [(one, V('y_rec', j))], ct.c_rec1[j])
    for j in range(p.k):
        equation(f'encryption_recipient_2[{j}]',
            [(H[i][j], V('z_rec', i)) for i in range(p.alpha1)] +
            [(one, V('x_rec', j)), (scale, V('m', j))], ct.c_rec2[j])
    equation('RISIS', [(relation.AR[j], V('m', j)) for j in range(p.k)], relation.uR)
    for j in range(7*p.alpha1):
        equation(f'encryption_opening_1[{j}]',
            [(gpk.pk_oa.A[i][j], V('z_oa', i)) for i in range(p.alpha1)] + [(one, V('y_oa', j))], ct.c_oa1[j])
    equation('encryption_opening_2', [(h[i], V('z_oa', i)) for i in range(p.alpha1)] +
             [(one, V('x_oa', 0)), (scale, V('h', 0))], ct.c_oa2)

    sizes = {'A_user': 7*p.alpha1*p.alpha1, 'tag': p.d, 'v': 8*p.alpha,
             'r': p.k*p.alpha, 'h': p.s, 'w': p.s*p.alpha, 'm': p.k,
             'x_rec': p.k, 'y_rec': 7*p.alpha1, 'z_rec': p.alpha1,
             'x_oa': 1, 'y_oa': 7*p.alpha1, 'z_oa': p.alpha1,
             'tag_v': p.d*3*p.alpha, 'A_z': 7*p.alpha1*p.alpha1}
    bounds = [CoefficientBound('tag', p.d, 0, 1, tuple(zip(boundaries, boundaries[1:]))),
              CoefficientBound('v', 8*p.alpha, -p.beta, p.beta)]
    bounds += [CoefficientBound(name, sizes[name], -1, 1) for name in ('r', 'h', 'w', 'm')]
    bounds += [CoefficientBound(name, sizes[name], -p.B, p.B)
               for name in ('x_rec','y_rec','z_rec','x_oa','y_oa','z_oa')]
    params_encoding = json.dumps(vars(p), sort_keys=True, separators=(',', ':')).encode()
    components = [params_encoding, _pack_matrix(gpk.pk_gm.A), _pack_matrix(gpk.pk_oa.A)]
    components += [_pack_matrix(A) for A in pp.A_tags]
    components += [_pack_matrix(pp.B0), _pack_matrix(pp.B1), _pack_matrix(pp.B2), _pack_vec(pp.u),
                   _pack_vec(relation.AR), relation.uR.to_bytes(), bytes(pp.ots_hash_seed), ct.vk_ots,
                   ct.label, _pack_vec(ct.c_rec1), _pack_vec(ct.c_rec2), _pack_vec(ct.c_oa1),
                   ct.c_oa2.to_bytes(), ct.sig]
    components += [pp.parcom.encode()]
    encoding = b''.join(_framed(item) for item in components)
    zk_parameters = (('ell', p.ell), ('kappa1', p.kappa1), ('kappa2', p.kappa2),
                     ('rejection_delta', getattr(p, 'zk_rejection_delta', None)),
                     ('conjugates', getattr(p, 'zk_conjugates', 4)),
                     ('research_reference', getattr(p, 'zk_research_reference', False)))
    crs_seed = hashlib.sha256(b'LatGE-A5-CRS' + pp.parcom.encode()).digest()
    return PaperPublicStatement(ring, encoding, tuple(equations), tuple(products), tuple(bounds), tuple(sizes.items()), zk_parameters, crs_seed, pp.parcom)


def build_private_witness(gpk: Any, witness: dict) -> PaperPrivateWitness:
    from .scheme import _certificate_target, _tag_polynomials
    pp, p = gpk.pp, gpk.pp.params
    A, cert = witness['pk_user'].A, witness['cert']
    h = rdec(A[0][-1], (p.q - 1) // 2)
    w, _ = _certificate_target(pp, h, cert.r_i)
    tag = _tag_polynomials(pp, cert.tag)
    vals = {'A_user': [a for row in A for a in row], 'tag': tag,
            'v': cert.v_i, 'r': cert.r_i, 'h': h, 'w': w, 'm': witness['m'],
            **{name: list(witness[name]) for name in ('x_rec','y_rec','z_rec','y_oa','z_oa')},
            'x_oa': [witness['x_oa']]}
    from .proof_timing import excluded_from_prover_time
    with excluded_from_prover_time():
        vals['tag_v'] = [t * v for t in tag for v in cert.v_i[5*p.alpha:]]
        vals['A_z'] = [a * witness['z_rec'][i] for i, row in enumerate(A) for a in row]
    return PaperPrivateWitness(vals)


def check_private_relation(statement: PaperPublicStatement, witness: PaperPrivateWitness) -> RelationCheck:
    failures = []
    values = witness.values
    for name, size in statement.private_layout:
        v = values.get(name)
        if not isinstance(v, (list, tuple)) or len(v) != size or any(not isinstance(x, Poly) or x.ring != statement.ring for x in v):
            failures.append('shape:' + name)
    if failures:
        return RelationCheck(False, tuple(failures))
    for bound in statement.bounds:
        for i, poly in enumerate(values[bound.group]):
            cc = poly.centered()
            if any(int(x) < bound.lower or int(x) > bound.upper for x in cc):
                failures.append(f'bound:{bound.group}[{i}]')
            if bound.supports:
                lo, hi = bound.supports[i]
                if any(int(x) != 0 for x in cc[:lo]) or any(int(x) != 0 for x in cc[hi:]):
                    failures.append(f'tag_support[{i}]')
    def get(ref):
        return values[ref.group][ref.index]
    for product in statement.hidden_products:
        if get(product.left) * get(product.right) != get(product.result):
            failures.append(product.name)
    for equation in statement.linear_equations:
        total = statement.ring.zero()
        for coefficient, variable in equation.terms:
            total = total + coefficient * get(variable)
        if total != equation.target:
            failures.append(equation.name)
    return RelationCheck(not failures, tuple(failures))
