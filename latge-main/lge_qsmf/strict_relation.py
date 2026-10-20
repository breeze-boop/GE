from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json

from .decomposition import g_beta, rdec
from .hash_to_ring import H1_mat, H2_vec
from .paper_relation import PaperPrivateWitness, build_public_statement
from .relation_lowering import ScalarPolynomialConstraint, ScalarProductConstraint
from .ring import Poly, Ring
from .strict_layout import StrictPaperLayout, strict_paper_layout


PROFILE = 'main-iii-d-dimension-corrected-eliminated-v1'
_TERNARY_BLOCKS = ('v1_digits', 'v2_digits', 'w', 'r', 'message',
    'x_rec_digits', 'y_rec_digits', 'z_rec_digits', 'h', 'x_oa_digits',
    'y_oa_digits', 'z_oa_digits', 'h_first')


def _poly(poly):
    return tuple(int(x) for x in poly.coeffs)


def _vector(vector):
    return tuple(_poly(poly) for poly in vector)


def _matrix(matrix):
    return tuple(_vector(row) for row in matrix)


def _rot(poly, row, column):
    difference = row - column
    return poly[difference] if difference >= 0 else -poly[difference + len(poly)]


@dataclass(frozen=True)
class StrictLinearConstraint:
    relation: 'StrictRelation'
    paper_row: int
    position: int

    @property
    def name(self):
        name = self.relation.layout.rows[self.paper_row - 1].name
        if self.paper_row == 1:
            name = 'certificate_minus_B2_w_equals_u'
        return f'paper_row_{self.paper_row}:{name}[{self.position}]'

    @property
    def target(self):
        return self.relation.linear_target(self.paper_row, self.position)

    @property
    def terms(self):
        return self.relation.linear_terms(self.paper_row, self.position)


@dataclass(frozen=True)
class StrictRelation:
    params: object
    layout: StrictPaperLayout
    public_statement_digest: bytes
    public: tuple

    @property
    def ring(self):
        return Ring(self.params.n, self.params.q)

    @property
    def n(self):
        return self.params.n

    @property
    def modulus(self):
        return self.params.q

    @property
    def scalar_count(self):
        return self.layout.scalar_count

    def block(self, name):
        for block in self.layout.witness:
            if block.name == name:
                return block
        raise KeyError(name)

    def variable(self, name, local):
        block = self.block(name)
        if type(local) is not int or not 0 <= local < block.scalar_length:
            raise IndexError('strict witness block index out of range')
        return block.offset + local

    def coefficient(self, name):
        for key, value in self.public:
            if key == name:
                return value
        raise KeyError(name)

    def digest(self):
        descriptor = (PROFILE, self.public_statement_digest.hex(), vars(self.params),
                      self.public, tuple((b.name, b.offset, b.scalar_length)
                                         for b in self.layout.witness),
                      tuple(c.code for c in self.layout.conflicts))
        return hashlib.sha256(json.dumps(descriptor, sort_keys=True,
                              separators=(',', ':')).encode()).digest()

    def statistics(self):
        return {'profile': PROFILE, 'literal_paper_layout': False,
                'scalar_variables': self.scalar_count,
                'packed_messages': self.layout.packed_message_count,
                'linear_constraints': self.layout.row_count - self.params.alpha*self.n,
                'product_constraints': self.layout.product_vector_count*self.n,
                'ternary_constraints': self.layout.ternary_vector_count*self.n,
                'tag_constraints': self.params.d*self.n,
                'eliminated_public_values': ('u_prime', 'u_double_prime'),
                'eliminated_paper_row': 3,
                'source_conflicts': tuple(c.code for c in self.layout.conflicts)}

    def iter_linear_constraints(self):
        for block in self.layout.rows:
            if block.index != 3:
                for position in range(block.scalar_length):
                    yield StrictLinearConstraint(self, block.index, position)

    def linear_target(self, row, position):
        p, n = self.params, self.n
        if not 1 <= row <= 12 or row == 3 or not 0 <= position < self.layout.rows[row-1].scalar_length:
            raise IndexError('unknown normalized public row')
        name = {1:'u', 7:'c_rec1', 9:'c_rec2', 10:'uR', 11:'c_oa1', 12:'c_oa2'}.get(row)
        if name is None:
            return 0
        value = self.coefficient(name)
        return value[position] if row in (10, 12) else value[position//n][position%n]

    def _ring_terms(self, coefficient, block, polynomial, out, multiplier=1):
        offset = self.block(block).offset + polynomial*self.n
        for column in range(self.n):
            weight = multiplier*_rot(coefficient, out, column)
            if weight % self.modulus:
                yield weight, offset+column

    def _digit_terms(self, coefficient, block, polynomial, out, weights):
        for digit, weight in enumerate(weights):
            yield from self._ring_terms(coefficient, block,
                                       polynomial*len(weights)+digit, out, weight)

    def linear_terms(self, row, position):
        self.linear_target(row, position)
        p, n, k0, k1 = self.params, self.n, self.layout.k0, self.layout.k1
        component, out = divmod(position, n)
        gb, ge, gh = g_beta(p.beta), g_beta(p.B), g_beta((p.q-1)//2)
        V = self.variable
        if row == 1:
            for j, a in enumerate(self.coefficient('A')[component]):
                yield from self._digit_terms(a, 'v1_digits', j, out, gb)
            tags = self.coefficient('A_tags')
            for j, a in enumerate(tags[0][component]):
                yield from self._digit_terms(a, 'v2_digits', j, out, gb)
            for tag in range(p.d):
                offset = self.block(f'tag_product[{tag+1}]').offset
                for poly, a in enumerate(tags[tag+1][component]):
                    for digit, weight in enumerate(gb):
                        start = offset+(poly*k0+digit)*n*n
                        for i in range(n):
                            for z in range(n):
                                sign = 1 if i+z < n else -1
                                c = weight*sign*_rot(a, out, (i+z)%n)
                                if c % p.q:
                                    yield c, start+i*n+z
            for j, a in enumerate(self.coefficient('B2')[component]):
                yield from self._ring_terms(a, 'w', j, out, -1)
        elif row in (2, 8):
            copy, original = (('v2_coefficient_copies', 'v2_digits') if row == 2
                              else ('z_rec_coefficient_copies', 'z_rec_digits'))
            poly_digit, residual = divmod(position, n*n)
            source_coefficient = residual//n
            yield 1, V(original, poly_digit*n+source_coefficient)
            yield -1, V(copy, position)
        elif row == 4:
            yield 1, V('A_user_column_major', (7*p.alpha1-1)*p.alpha1*n+out)
            for digit, weight in enumerate(gh):
                yield -weight, V('h', digit*n+out)
        elif row == 5:
            yield 1, V('h', out)
            yield -1, V('h_first', out)
        elif row == 6:
            for j, a in enumerate(self.coefficient('B0')[component]):
                yield from self._ring_terms(a, 'r', j, out)
            for j, a in enumerate(self.coefficient('B1')[component]):
                yield from self._ring_terms(a, 'h', j, out)
            for digit, weight in enumerate(gh):
                yield -weight, V('w', (component*p.s+digit)*n+out)
        elif row == 7:
            for i in range(p.alpha1):
                for digit, weight in enumerate(ge):
                    start = ((component*p.alpha1+i)*k1+digit)*n*n
                    for source in range(n):
                        z = (out-source)%n
                        sign = 1 if source+z < n else -1
                        yield sign*weight, V('recipient_products', start+source*n+z)
            for digit, weight in enumerate(ge):
                yield weight, V('y_rec_digits', (component*k1+digit)*n+out)
        elif row in (9, 11, 12):
            if row == 9:
                matrix, block, noise = self.coefficient('H1'), 'z_rec_digits', 'x_rec_digits'
            elif row == 11:
                matrix, block, noise = self.coefficient('A_oa'), 'z_oa_digits', 'y_oa_digits'
            else:
                matrix = tuple((a,) for a in self.coefficient('H2'))
                block, noise = 'z_oa_digits', 'x_oa_digits'
            for i in range(p.alpha1):
                yield from self._digit_terms(matrix[i][component], block, i, out, ge)
            for digit, weight in enumerate(ge):
                yield weight, V(noise, (component*k1+digit)*n+out)
            if row == 9:
                yield p.q//4, V('message', component*n+out)
            if row == 12:
                yield p.q//4, V('h_first', out)
        elif row == 10:
            for j, a in enumerate(self.coefficient('AR')):
                yield from self._ring_terms(a, 'message', j, out)

    def iter_product_constraints(self):
        p, n, k0, k1 = self.params, self.n, self.layout.k0, self.layout.k1
        V = self.variable
        for tag in range(p.d):
            for h in range(3*p.alpha*k0):
                for i in range(n):
                    for z in range(n):
                        local = h*n*n+i*n+z
                        yield ScalarProductConstraint(f'tag_product[{tag},{h},{i},{z}]',
                            V('v2_coefficient_copies', local), V('tag', tag*n+z),
                            V(f'tag_product[{tag+1}]', local))
        for column in range(7*p.alpha1):
            for row in range(p.alpha1):
                for digit in range(k1):
                    for i in range(n):
                        for z in range(n):
                            copy = (row*k1+digit)*n*n+i*n+z
                            product = ((column*p.alpha1+row)*k1+digit)*n*n+i*n+z
                            yield ScalarProductConstraint(f'recipient_product[{column},{row},{digit},{i},{z}]',
                                V('z_rec_coefficient_copies', copy),
                                V('A_user_column_major', (column*p.alpha1+row)*n+z),
                                V('recipient_products', product))

    def iter_membership_constraints(self):
        for name in _TERNARY_BLOCKS:
            block = self.block(name)
            for local in range(block.scalar_length):
                index = block.offset+local
                yield ScalarPolynomialConstraint(f'ternary:{name}[{local}]',
                    ((1, (index,index,index)), (-1, (index,))))
        boundaries = self.params.tag_boundaries
        for tag in range(self.params.d):
            for position in range(self.n):
                index = self.variable('tag', tag*self.n+position)
                indicator = int(boundaries[tag] <= position < boundaries[tag+1])
                yield ScalarPolynomialConstraint(f'tag_support[{tag},{position}]',
                    ((1, (index,index)), (-indicator, (index,))))

    def iter_polynomial_constraints(self):
        for constraint in self.iter_product_constraints():
            yield constraint.as_polynomial()
        yield from self.iter_membership_constraints()


def compile_strict_relation(gpk, ciphertext) -> StrictRelation:
    public = build_public_statement(gpk, ciphertext)
    return _compile_strict_relation_from_public(gpk, ciphertext, public)


def _compile_strict_relation_from_public(gpk, ciphertext, public) -> StrictRelation:

    pp, p = gpk.pp, gpk.pp.params
    layout = strict_paper_layout(p)
    if p.beta > p.q//2 or p.B > p.q//2:
        raise ValueError('paper bounds must fit centered representatives')
    constants = (
        ('A', _matrix(gpk.pk_gm.A)), ('A_tags', tuple(_matrix(a) for a in pp.A_tags)),
        ('B0', _matrix(pp.B0)), ('B1', _matrix(pp.B1)), ('B2', _matrix(pp.B2)),
        ('A_oa', _matrix(gpk.pk_oa.A)),
        ('H1', _matrix(H1_mat(pp.ring_q, ciphertext.vk_ots, p.alpha1, p.k))),
        ('H2', _vector(H2_vec(pp.ring_q, ciphertext.vk_ots, p.alpha1))),
        ('AR', _vector(ciphertext.relation.AR)), ('uR', _poly(ciphertext.relation.uR)),
        ('u', _vector(pp.u)), ('c_rec1', _vector(ciphertext.c_rec1)),
        ('c_rec2', _vector(ciphertext.c_rec2)), ('c_oa1', _vector(ciphertext.c_oa1)),
        ('c_oa2', _poly(ciphertext.c_oa2)))
    return StrictRelation(p, layout, public.digest(), constants)


class StrictPrivateAssignment:
    def __init__(self, relation, witness):
        self.relation = relation
        self._starts = tuple(block.offset for block in relation.layout.witness)
        p, n = relation.params, relation.n
        values = witness.values
        from .proof_timing import excluded_from_prover_time
        with excluded_from_prover_time():
            sizes = {'A_user':7*p.alpha1*p.alpha1, 'tag':p.d, 'v':8*p.alpha,
                'r':p.k*p.alpha, 'h':p.s, 'w':p.s*p.alpha, 'm':p.k,
                'x_rec':p.k, 'y_rec':7*p.alpha1, 'z_rec':p.alpha1,
                'x_oa':1, 'y_oa':7*p.alpha1, 'z_oa':p.alpha1}
            if not set(sizes) <= set(values) or set(values)-set(sizes)-{'tag_v','A_z'}:
                raise ValueError('unexpected or missing Eq.(9) private groups')
            for name, size in sizes.items():
                if len(values[name]) != size or any(not isinstance(a,Poly) or a.ring != relation.ring for a in values[name]):
                    raise ValueError(f'wrong private shape/ring: {name}')
            expected = {'tag_v':[t*v for t in values['tag'] for v in values['v'][5*p.alpha:]],
                        'A_z':[a*values['z_rec'][i//(7*p.alpha1)] for i,a in enumerate(values['A_user'])]}
            for name in expected:
                if name in values and list(values[name]) != expected[name]:
                    raise ValueError(f'inconsistent derived Eq.(9) auxiliary: {name}')
        def flat(vector):
            return tuple(int(x) for a in vector for x in a.coeffs)
        def digits(vector, bound):
            return flat([part for a in vector for part in rdec(a,bound)])
        direct = {'tag':flat(values['tag']), 'v1_digits':digits(values['v'][:5*p.alpha],p.beta),
                  'v2_digits':digits(values['v'][5*p.alpha:],p.beta),
                  'message':flat(values['m']), 'h_first':flat(values['h'][:1]),
                  'A_user_column_major':flat([values['A_user'][row*7*p.alpha1+col]
                    for col in range(7*p.alpha1) for row in range(p.alpha1)])}
        for name in ('r','h','w'):
            direct[name] = flat(values[name])
        for name in ('x_rec','y_rec','z_rec','x_oa','y_oa','z_oa'):
            direct[name+'_digits'] = digits(values[name],p.B)
        self._direct = direct
        if any(len(data) != relation.block(name).scalar_length for name,data in direct.items()):
            raise ValueError('strict private block dimensions disagree')

    @property
    def stored_scalar_count(self):
        return sum(len(v) for v in self._direct.values())

    def __len__(self):
        return self.relation.scalar_count

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError('private scalar index out of range')
        relation, n = self.relation, self.relation.n
        p, k1 = relation.params, relation.layout.k1
        block = relation.layout.witness[bisect_right(self._starts,index)-1]
        local = index-block.offset
        if block.name in self._direct:
            return self._direct[block.name][local]
        h, residual = divmod(local,n*n)
        i,z = divmod(residual,n)
        if block.name in ('v2_coefficient_copies','z_rec_coefficient_copies'):
            original = 'v2_digits' if block.name.startswith('v2') else 'z_rec_digits'
            return self._direct[original][h*n+i]
        if block.name.startswith('tag_product['):
            tag = int(block.name.removeprefix('tag_product[').removesuffix(']'))-1
            return self._direct['v2_digits'][h*n+i]*self._direct['tag'][tag*n+z] % p.q
        if block.name == 'recipient_products':
            key_entry, digit = divmod(h,k1)
            row = key_entry % p.alpha1
            return self._direct['z_rec_digits'][(row*k1+digit)*n+i]*self._direct['A_user_column_major'][key_entry*n+z] % p.q
        raise ValueError('unknown strict witness block')

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def lower_strict_witness(relation, witness: PaperPrivateWitness):
    return StrictPrivateAssignment(relation, witness)


@dataclass(frozen=True)
class StrictRelationCheck:
    valid: bool
    failed_constraints: tuple[str, ...]


