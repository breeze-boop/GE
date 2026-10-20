from __future__ import annotations

from dataclasses import replace
import hashlib
import numpy as np

from .decomposition import g_beta, idec
from .paper_relation import PaperPrivateWitness, check_private_relation
from .paper_zkp_backend import (PaperZKPBackend, PackedPolynomialConstraint,
    pack_slots, ntt_slots, _frame)
from .relation_lowering import lower_public_relation, lower_private_witness, LazyPrivateAssignment
from .ring import Ring, Poly
from .strict_streaming import (StreamingStrictRelation, RepeatableStream,
    StreamingLinearConstraint, _transpose_rot_matrix, _slot_product_mod, _challenge_array)
from .strict_slot_ops import _alpha_slots, _finish, _validate_vector
from .slot_native import dot_sums, expansion_sums, homogeneous_sums
from .streaming_zkp_backend import StrictWitnessSlots


class JoinWitnessSlots(StrictWitnessSlots):
    def __init__(self, relation, assignment):
        if not isinstance(relation, StreamingJoinRelation):
            raise TypeError('locally compiled JOIN relation required')
        if len(assignment) != relation.lowered.scalar_count:
            raise ValueError('wrong JOIN scalar assignment length')
        self.ring = relation.ring
        self.assignment = assignment
        self.length = relation.message_count
        self._join_relation = relation
        self._snapshot = None
        self._slots = None

    def read(self, start, count):
        if type(start) is not int or type(count) is not int or start < 0 or count < 0 or start+count > len(self):
            raise IndexError('private slot block out of range')
        relation, assignment = self._join_relation, self.assignment
        if type(assignment) is not LazyPrivateAssignment or assignment.relation is not relation.lowered:
            return super().read(start, count)
        source = assignment._values['R']
        if source is not self._snapshot:
            q = self.ring.q
            base = np.asarray(source, dtype=np.int64).reshape(-1)
            canonical = (base % q).astype(np.uint64)
            if relation.weights:
                unique, indices = np.unique(base, return_inverse=True)
                bound = relation.statement.bounds[0].upper
                table = np.asarray([[(1 if value >= 0 else -1)*bit % q
                    for bit in idec(abs(int(value)), bound)] for value in unique], dtype=np.uint64)
                canonical = np.concatenate((canonical, table[indices].reshape(-1)))
            self._slots = canonical.reshape(self.length, self.ring.n)
            self._snapshot = source
        return self._slots[start:start+count].copy()


class StreamingJoinRelation(StreamingStrictRelation):
    crs_profile = 'research_generic'

    def __init__(self, statement):
        if (len(statement.bounds) != 1 or statement.hidden_products
                or statement.private_layout != (('R', statement.bounds[0].size),)):
            raise ValueError('JOIN requires one bounded R group and no hidden products')
        bound = statement.bounds[0]
        if bound.group != 'R' or bound.lower != -bound.upper or bound.upper < 1 or bound.supports:
            raise ValueError('JOIN requires a positive symmetric coefficient bound')
        self.statement = statement
        self.lowered = lower_public_relation(statement)
        parameters = PaperZKPBackend._parameters(statement)
        self.ring = Ring(parameters.degree, parameters.modulus)
        if self.lowered.scalar_count % self.ring.n:
            raise ValueError('JOIN packing must have complete slot rows')
        self.message_count = self.lowered.scalar_count // self.ring.n
        parent = statement.zk_crs
        if parent is None:
            raise ValueError('JOIN requires its group public commitment CRS')
        self.parameters = replace(parameters, commitment_crs=parent.research_view(self.message_count))
        self.parameters.validate()
        self.base_count = bound.size * statement.ring.n
        self.weights = () if bound.upper == 1 else tuple(g_beta(bound.upper))
        self.member_start = self.base_count if self.weights else 0
        self.member_count = self.base_count * (len(self.weights) or 1)
        if self.member_start % self.ring.n or self.member_count % self.ring.n:
            raise ValueError('unaligned JOIN membership block')
        stats = self.lowered.statistics()
        self.linear = RepeatableStream(stats['linear_constraints'],
            lambda: (StreamingLinearConstraint(row, self.ring.n, self.ring.q)
                     for row in self.lowered.iter_linear_constraints()))
        self.polynomial = RepeatableStream(self.member_count // self.ring.n, self._polynomials)
        self.statement_digest = self.lowered.digest()
        self._context = hashlib.sha256(
            b'LatGE-LNS20-JOIN-streaming-v1\x00' + _frame(self.parameters.encode())
            + _frame(self.statement_digest)
            + b''.join(x.to_bytes(8, 'little') for x in
                       (self.message_count, len(self.linear), len(self.polynomial)))).digest()

    def _polynomials(self):
        one = self.ring.one()
        negative = one.scalar_mul(-1)
        for j in range(self.member_start // self.ring.n, self.message_count):
            yield PackedPolynomialConstraint(((one, (j, j, j)), (negative, (j,))))

    def assignment(self, private):
        return lower_private_witness(self.lowered, private)

    def _join_matrix(self):

        equations = self.statement.linear_equations
        prefix = len(equations[0].terms) if equations else 0
        size = self.statement.bounds[0].size
        if not prefix or size % prefix:
            return None
        columns = size // prefix
        if len(equations) % columns:
            return None
        matrix = []
        for i in range(len(equations)//columns):
            reference = equations[i*columns].terms
            for j in range(columns):
                terms = equations[i*columns+j].terms
                if len(terms) != prefix:
                    return None
                for h, (coefficient, variable) in enumerate(terms):
                    if (variable.group != 'R' or variable.index != h*columns+j
                            or (coefficient is not reference[h][0] and coefficient != reference[h][0])):
                        return None
            matrix.append([coefficient.coeffs for coefficient, _ in reference])
        return np.asarray(matrix, dtype=np.int64), columns

    def linear_adjoint(self, gamma):
        q, n = self.ring.q, self.statement.ring.n
        it = iter(gamma)
        result = np.zeros(self.lowered.scalar_count, dtype=np.uint64)
        target = 0
        batch = self._join_matrix()
        if batch is not None:
            from .ring_native import product
            matrix, columns = batch
            gamma_rows = _challenge_array(it, len(self.statement.linear_equations)*n, q).reshape(-1, n)
            for row, equation in zip(gamma_rows, self.statement.linear_equations):
                target = (target + sum(int(a)*int(b) for a,b in zip(row, equation.target.coeffs))) % q
            adjoint = matrix.copy()
            adjoint[:,:,1:] = (-matrix[:,:,1:][:,:,::-1]) % q
            products = product(adjoint, gamma_rows.reshape(len(matrix), columns, n),
                               q, transpose_left=True)
            result[:self.base_count] = products.reshape(-1)
        else:
            for equation in self.statement.linear_equations:
                row = _challenge_array(it, n, q)
                target = (target + sum(int(a) * int(b) for a, b in zip(row, equation.target.coeffs))) % q
                products=(_transpose_rot_matrix(
                    [[coefficient.coeffs for coefficient,_ in equation.terms]],
                    row[None,:],self.statement.ring) if equation.terms else ())
                for (_, variable),value in zip(equation.terms,products):
                    offset = variable.index * n
                    result[offset:offset+n] = (result[offset:offset+n] + value.astype(np.uint64)) % q
        if self.weights:
            row = _challenge_array(it, self.base_count, q, dtype=np.uint64)
            result[:self.base_count] = (result[:self.base_count] + row) % q
            weights = np.asarray(self.weights, dtype=np.uint64)
            result[self.base_count:] = ((q - _slot_product_mod(row[:, None], weights, q)) % q).reshape(-1)
        if next(it, None) is not None:
            raise ValueError('too many JOIN linear challenges')
        return result.reshape(-1, self.ring.n), target

    def linear_aggregate(self, stream, vectors):
        dots, targets = [], []
        q, e = self.ring.q, self.ring.n
        for _, gamma in stream.gamma():
            adjoint, target = self.linear_adjoint(gamma)
            accumulated = np.zeros((len(vectors), e), dtype=np.uint64)
            for start in range(0, self.message_count, 1024):
                count = min(1024, self.message_count-start)
                accumulated=(accumulated+dot_sums(adjoint[start:start+count],
                    (vector.read(start,count) for vector in vectors),q))%q
            dots.append([pack_slots(self.ring, row) for row in accumulated])
            targets.append(self.ring.one().scalar_mul(target * pow(e, -1, q) % q))
        return dots, targets

    def check_witness_slots(self, vector, bulk, *, batch_size=1024):
        _validate_vector(self, vector)
        if bulk.ring != self.ring:
            raise ValueError('JOIN commitment ring mismatch')
        values = np.asarray(vector.read(0, self.message_count), dtype=np.uint64).reshape(-1)
        if np.any(values >= self.ring.q):
            return False
        n, q = self.statement.ring.n, self.ring.q
        base = values[:self.base_count].reshape(-1, n).astype(np.int64)
        batch = self._join_matrix()
        if batch is not None:
            from .ring_native import product
            matrix, columns = batch
            bound = self.statement.bounds[0]
            centered = np.where(base > q//2, base-q, base)
            if np.any((centered < bound.lower) | (centered > bound.upper)):
                return False
            actual = product(matrix, base.reshape(matrix.shape[1], columns, n), q)
            target = np.asarray([eq.target.coeffs for eq in self.statement.linear_equations],
                                dtype=np.int64).reshape(actual.shape)
            if not np.array_equal(actual, target):
                return False
        else:
            private = PaperPrivateWitness({'R': [Poly.from_ints(self.statement.ring, row) for row in base]})
            if not check_private_relation(self.statement, private).valid:
                return False
        members = values[self.member_start:]
        if np.any((members != 0) & (members != 1) & (members != q-1)):
            return False
        if self.weights:
            digits = members.reshape(self.base_count, len(self.weights))
            combined = _slot_product_mod(digits, np.asarray(self.weights, dtype=np.uint64), q).sum(axis=1) % q
            if not np.array_equal(combined, values[:self.base_count]):
                return False
        return True

    def _batches(self, batch_size):
        if type(batch_size) is not int or not 1 <= batch_size <= 4096:
            raise ValueError('JOIN batch size must be in 1..4096')
        first = self.member_start // self.ring.n
        for start in range(first, self.message_count, batch_size):
            yield start, min(batch_size, self.message_count-start)

    def aggregate_expansions_slots(self, masks, messages, alpha, phi, bulk, *, batch_size=1024):
        ring, k = self.ring, self.parameters.automorphisms
        q = ring.q
        if len(masks) != k or bulk.ring != ring:
            raise ValueError('JOIN expansion shape mismatch')
        for vector in (messages, *masks):
            _validate_vector(self, vector)
        result = np.zeros((k, 3, ring.n), dtype=np.uint64)
        it = iter(alpha)
        for start, count in self._batches(batch_size):
            a = _alpha_slots(it, count, bulk)
            m = messages.read(start, count)
            for i, mask in enumerate(masks):
                y = mask.read(start, count)
                result[i]=(result[i]+expansion_sums('ternary',a,m,y,q))%q
        if next(it, None) is not None:
            raise ValueError('too many JOIN polynomial challenges')
        return _finish(result, phi, ring, k)

    def aggregate_homogeneous_slots(self, vectors, challenges, alpha, phi, bulk, *, batch_size=1024):
        ring, k = self.ring, self.parameters.automorphisms
        q = ring.q
        if len(vectors) != k or len(challenges) != k or bulk.ring != ring:
            raise ValueError('JOIN homogeneous shape mismatch')
        for vector in vectors:
            _validate_vector(self, vector)
        c = np.asarray([ntt_slots(value) for value in challenges], dtype=np.uint64)
        result = np.zeros((k, 1, ring.n), dtype=np.uint64)
        it = iter(alpha)
        for start, count in self._batches(batch_size):
            a = _alpha_slots(it, count, bulk)
            for i, vector in enumerate(vectors):
                x = vector.read(start, count)
                result[i,0]=(result[i,0]+homogeneous_sums('ternary',a,x,c[i],q))%q
        if next(it, None) is not None:
            raise ValueError('too many JOIN polynomial challenges')
        return _finish(result, phi, ring, k)[0]


def prove_join(statement, private):
    relation = StreamingJoinRelation(statement)
    messages = JoinWitnessSlots(relation, relation.assignment(private))
    core = PaperZKPBackend()._streaming_core()
    return core.prove(relation, messages)


def verify_join(statement, transcript):
    relation = StreamingJoinRelation(statement)
    return PaperZKPBackend()._streaming_core().verify(relation, transcript)
