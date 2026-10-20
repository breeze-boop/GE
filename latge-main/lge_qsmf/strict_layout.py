from __future__ import annotations

from dataclasses import asdict, dataclass


class PaperLayoutInconsistency(ValueError):
    pass


@dataclass(frozen=True)
class LayoutBlock:
    index: int
    name: str
    scalar_length: int
    offset: int


@dataclass(frozen=True)
class MatrixBlock:
    row: int
    column: int
    operator: str
    shape: tuple[int, int]


@dataclass(frozen=True)
class SourceConflict:
    code: str
    page: int
    printed: str
    dimension_consistent_reading: str


@dataclass(frozen=True)
class StrictPaperLayout:
    n: int
    ell: int
    k0: int
    k1: int
    witness: tuple[LayoutBlock, ...]
    rows: tuple[LayoutBlock, ...]
    matrix_blocks: tuple[MatrixBlock, ...]
    t_formula: int
    printed_row_units: int
    ternary_vector_count: int
    printed_product_vector_count: int
    product_vector_count: int
    tag_vector_count: int
    conflicts: tuple[SourceConflict, ...]

    @property
    def scalar_count(self) -> int:
        return sum(block.scalar_length for block in self.witness)

    @property
    def row_count(self) -> int:
        return sum(block.scalar_length for block in self.rows)

    @property
    def t_from_blocks(self) -> int:
        return self.scalar_count // self.n

    @property
    def packed_message_count(self) -> int:
        if self.scalar_count % self.ell:
            raise PaperLayoutInconsistency('tn is not divisible by ell')
        return self.scalar_count // self.ell

    def validate_shapes(self) -> None:
        for blocks in (self.witness, self.rows):
            offset = 0
            for index, block in enumerate(blocks, 1):
                if block.index != index or block.offset != offset or block.scalar_length <= 0:
                    raise PaperLayoutInconsistency('non-contiguous or invalid block layout')
                if block.scalar_length % self.n:
                    raise PaperLayoutInconsistency('block is not a whole number of n-vectors')
                offset += block.scalar_length
        if self.scalar_count != self.n * self.t_formula:
            raise PaperLayoutInconsistency('printed t formula does not equal witness block sum')
        seen = set()
        for block in self.matrix_blocks:
            if not 1 <= block.row <= len(self.rows) or not 1 <= block.column <= len(self.witness):
                raise PaperLayoutInconsistency('matrix block index out of range')
            position = (block.row, block.column)
            if position in seen:
                raise PaperLayoutInconsistency('duplicate matrix block position')
            seen.add(position)
            expected = (self.rows[block.row - 1].scalar_length,
                        self.witness[block.column - 1].scalar_length)
            if block.shape != expected:
                raise PaperLayoutInconsistency(f'{position}: operator shape {block.shape} != layout {expected}')
        self.packed_message_count

    def require_literal_consistency(self) -> None:
        self.validate_shapes()
        if self.conflicts:
            raise PaperLayoutInconsistency('; '.join(c.code for c in self.conflicts))

    def summary(self) -> dict:
        self.validate_shapes()
        return {
            'source': 'Main.pdf pp. 5--8, III-B/C/D, Eq. (6), (8), (9), (10)',
            'scope': 'dimension_consistent_reading; metadata only, no ZKP or dense matrix',
            'literal_source_consistent': not self.conflicts,
            'n': self.n, 'ell': self.ell, 'k0': self.k0, 'k1': self.k1,
            't_formula': self.t_formula, 't_from_blocks': self.t_from_blocks,
            'scalar_count': self.scalar_count, 'packed_message_count': self.packed_message_count,
            'row_count': self.row_count, 'derived_row_units': self.row_count // self.n,
            'printed_row_units': self.printed_row_units,
            'ternary_vector_count_f': self.ternary_vector_count,
            'printed_product_vector_count_g': self.printed_product_vector_count,
            'derived_product_vector_count_g': self.product_vector_count,
            'binary_tag_vector_count': self.tag_vector_count,
            'witness_blocks': [asdict(b) for b in self.witness],
            'row_blocks': [asdict(b) for b in self.rows],
            'nonzero_matrix_blocks': [asdict(b) for b in self.matrix_blocks],
            'source_conflicts': [asdict(c) for c in self.conflicts],
        }


def _blocks(entries) -> tuple[LayoutBlock, ...]:
    result, offset = [], 0
    for index, (name, size) in enumerate(entries, 1):
        result.append(LayoutBlock(index, name, size, offset))
        offset += size
    return tuple(result)


def strict_paper_layout(params) -> StrictPaperLayout:
    names = ('n', 'ell', 'alpha', 'alpha1', 'd', 'beta', 'B', 'k', 's')
    values = {name: getattr(params, name) for name in names}
    if any(type(value) is not int or value <= 0 for value in values.values()):
        raise ValueError('paper dimensions and bounds must be positive integers')
    n, ell, a, a1, d, beta, noise, k, s = (values[name] for name in names)
    if n % ell:
        raise ValueError('ell must divide n for the paper block packing')
    k0, k1 = beta.bit_length(), noise.bit_length()
    entries = [('tag', d*n), ('v1_digits', 5*a*k0*n), ('v2_digits', 3*a*k0*n)]
    entries += [(f'tag_product[{j}]', 3*a*k0*n*n) for j in range(1, d+1)]
    entries += [
        ('v2_coefficient_copies', 3*a*k0*n*n), ('w', s*a*n), ('r', k*a*n),
        ('message', k*n), ('x_rec_digits', k*k1*n), ('y_rec_digits', 7*a1*k1*n),
        ('z_rec_digits', a1*k1*n), ('h', s*n),
        ('recipient_products', 7*a1*a1*k1*n*n),
        ('z_rec_coefficient_copies', a1*k1*n*n),
        ('x_oa_digits', k1*n), ('y_oa_digits', 7*a1*k1*n),
        ('z_oa_digits', a1*k1*n), ('h_first', n), ('A_user_column_major', 7*a1*a1*n),
    ]
    witness = _blocks(entries)
    rows = _blocks([
        ('certificate_equals_u_prime', a*n), ('copy_v2_digits', 3*a*k0*n*n),
        ('B2_w_equals_u_double_prime', a*n), ('certified_key_entry', n),
        ('h_first_selection', n), ('B0_r_plus_B1_h_decomposition', a*n),
        ('recipient_ciphertext_1', 7*a1*n), ('copy_z_rec_digits', a1*k1*n*n),
        ('recipient_ciphertext_2', k*n), ('RISIS', n),
        ('opening_ciphertext_1', 7*a1*n), ('opening_ciphertext_2', n),
    ])
    blocks = []
    def add(row, column, operator, height, width):
        blocks.append(MatrixBlock(row, column, operator, (height, width)))
    add(1, 2, 'rot(A H_{5alpha,beta})', a*n, 5*a*k0*n)
    add(1, 3, 'rot(A_0 H_{3alpha,beta})', a*n, 3*a*k0*n)
    for j in range(1, d+1):
        add(1, j+3, f'rot(A_{j} H_{{3alpha,beta}})(I_{{3alpha*k0}} tensor B_[n])', a*n, 3*a*k0*n*n)
    add(2, 3, 'I_{3alpha*k0} tensor D_[n]', 3*a*k0*n*n, 3*a*k0*n)
    add(2, d+4, '-I_{3alpha*k0*n^2}', 3*a*k0*n*n, 3*a*k0*n*n)
    add(3, d+5, 'rot(B2)', a*n, s*a*n)
    add(4, d+18, 'g2^T tensor I_n', n, 7*a1*a1*n)
    add(4, d+11, '-g^T tensor I_n', n, s*n)
    add(5, d+11, 'g1^T tensor I_n', n, s*n)
    add(5, d+17, '-I_n', n, n)
    add(6, d+6, 'rot(B0)', a*n, k*a*n)
    add(6, d+11, 'rot(B1)', a*n, s*n)
    add(6, d+5, '-H_{alpha,(q-1)/2} tensor I_n', a*n, s*a*n)
    add(7, d+12, '(H_{7alpha1,B} tensor I_n)(I_{7alpha1} tensor B_prime_[n])', 7*a1*n, 7*a1*a1*k1*n*n)
    add(7, d+9, 'H_{7alpha1,B} tensor I_n', 7*a1*n, 7*a1*k1*n)
    add(8, d+10, 'I_{alpha1*k1} tensor D_[n]', a1*k1*n*n, a1*k1*n)
    add(8, d+13, '-I_{alpha1*k1*n^2}', a1*k1*n*n, a1*k1*n*n)
    add(9, d+10, 'rot(H_vk^T H_{alpha1,B})', k*n, a1*k1*n)
    add(9, d+8, 'H_{k,B} tensor I_n', k*n, k*k1*n)
    add(9, d+7, 'floor(q/4) I_{kn}', k*n, k*n)
    add(10, d+7, 'rot(A_R)', n, k*n)
    add(11, d+16, 'rot(A_oa^T H_{alpha1,B})', 7*a1*n, a1*k1*n)
    add(11, d+15, 'H_{7alpha1,B} tensor I_n', 7*a1*n, 7*a1*k1*n)
    add(12, d+16, 'rot(h_vk^T H_{alpha1,B})', n, a1*k1*n)
    add(12, d+14, 'H_{1,B} tensor I_n', n, k1*n)
    add(12, d+17, 'floor(q/4) I_n', n, n)
    t = (3*(d+1)*a*k0*n + (7*a1+1)*a1*k1*n + 8*a*k0 + 16*a1*k1
         + 7*a1*a1 + (a+1)*(s+k) + (k+1)*k1 + d+1)
    printed_rows = k*k*k1 + d*s*k0 + 2*k + 6
    f = (a+1)*(s+k) + 16*a1*k1 + 8*a*k0 + k*k1 + k1 + 1
    printed_g = 3*d*k0*n + 7*a1*a1*k1*n
    actual_g = 3*d*a*k0*n + 7*a1*a1*k1*n
    conflicts = [
        SourceConflict('CASE1_WEIGHT_INDEX', 6, 'g_beta0^T tensor g_beta0^T', 'g_beta0^T tensor g_beta1^T'),
        SourceConflict('CASE2_KRONECKER_DIMENSION', 6, 'A tensor H_{m,beta}', 'A H_{m,beta}; equivalently A tensor g_beta^T'),
        SourceConflict('EXAMPLE_ROT_SIGN', 7, 'rot(a)=[[a0,a1],[-a1,a0]]', 'II-A p3 requires [[a0,-a1],[a1,a0]]'),
        SourceConflict('M_GRID_EXTENT', 8, 'matrix drawing ends at row 11, column d+17', 'listed equations require row 12, column d+18'),
        SourceConflict('M8_COPY_COLUMN', 8, 'M_{8,d+9}=-I_{alpha*k1*n^2}', 'M_{8,d+13}=-I_{alpha1*k1*n^2}'),
        SourceConflict('M6_RECOMPOSITION_SHAPE', 8, 'M_{6,d+5}=M_{4,d+11}=-g_prime^T', 'row 6 needs I_alpha tensor (-g^T tensor I_n)'),
        SourceConflict('M12_ROW_TRANSPOSE', 8, 'M_{12,d+16}=H_{vk,1}^T', 'row operator rot(h_vk^T H_{alpha1,B}) has shape n by alpha1*k1*n'),
    ]
    derived_rows = sum(row.scalar_length for row in rows) // n
    if printed_rows != derived_rows:
        conflicts.append(SourceConflict('ROW_COUNT_FORMULA', 8,
            f'K=k^2*k1+d*s*k0+2*k+6={printed_rows}',
            f'K=3alpha+3alpha*k0*n+14alpha1+alpha1*k1*n+k+4={derived_rows}'))
    if printed_g != actual_g:
        conflicts.append(SourceConflict('PRODUCT_COUNT_ALPHA', 8,
            f'g=3d*k0*n+7alpha1^2*k1*n={printed_g}',
            f'g=3d*alpha*k0*n+7alpha1^2*k1*n={actual_g}'))
    layout = StrictPaperLayout(n, ell, k0, k1, witness, rows, tuple(blocks), t,
        printed_rows, f, printed_g, actual_g, d, tuple(conflicts))
    layout.validate_shapes()
    return layout
