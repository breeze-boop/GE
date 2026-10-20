from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import time
from types import SimpleNamespace

import numpy as np

from . import metal_matrix
from .commitments import LatticeCommitmentKey, framed_hash, sample_standard_discrete_gaussian
from .cost_model import _MatrixFixture, target_dimensions
from .fast_uniform import uniform_source, uniform_field
from .linear_hash import _same_key_decomposition
from .mathutils import _affine_canonical_equal, _matmul_mod_prepared_left, require_memory
from .randomness import CryptographicRNG
from .relation_work import RelationWork
from .relations import _canonical_matvec_input, _immutable_multiplication_triples
from .schedule_model import fit_parallel_schedule
from .yang import (_prover_first_phase, _prover_second_phase, _prover_witness_context_phase,
                   _quadratic_masks, _quadratic_response, _quadratic_witness_layout,
                   _verifier_equation_phase)


class _UncachedKey:
    def __init__(self, params, settings, A, width, resident_bytes=0):
        l1, l2 = settings.l1, settings.l2
        top_shape, bottom_shape = (l1, width + l2), (width, l2)
        peak = resident_bytes + 4 * (math.prod(top_shape) + 2 * math.prod(bottom_shape))
        peak += 192 * 1024**2 + 160 * width
        require_memory(peak, min(params.max_memory_bytes, 4 * 1024**3),
                       'parallel calibration workspace')
        backend = metal_matrix._backend()
        if backend is None or 4 * max(map(math.prod, (top_shape, bottom_shape))) > backend.max_buffer_length:
            raise MemoryError('Parallel fixture exceeds buffer limit')
        self.key = LatticeCommitmentKey(params.q, l1, l2, width,
            _MatrixFixture(A, top_shape), _MatrixFixture(A.T, bottom_shape))
        self.q = params.q
        self.randomness_size = self.key.randomness_size

    def commit(self, message, randomness):
        with metal_matrix._uncached_workspaces():
            return self.key.commit(message, randomness)

    def commit_pair(self, first, second):
        with metal_matrix._uncached_workspaces():
            return self.key.commit_pair(first, second)


class _Operator:
    def __init__(self, relation, width, F_columns, matrix=None, prepared=False):
        self.relation, self.width = relation, width
        self.F_columns, self.matrix, self.prepared = F_columns, matrix, prepared
        self.expected_bits = np.frombuffer(bytes([1]) * F_columns, dtype=np.uint8)

    def __matmul__(self, vector):
        value = _canonical_matvec_input(vector, self.width, self.relation.params.q)

        def F_action():
            key = value[:self.F_columns]
            if self.prepared:
                half = self.F_columns // 2
                if _same_key_decomposition((key[:half], key[half:]), self.expected_bits):
                    return self.relation.zero_F
            if self.matrix is None:
                raise AssertionError('Cached decomposition mismatch')
            return _matmul_mod_prepared_left(self.matrix, key, self.relation.params.q)

        return self.relation.apply(F_action)


class _RecordingFuture:
    def __init__(self, future, record):
        self._future, self._record = future, record

    def __getattr__(self, name):
        return getattr(self._future, name)

    def result(self, timeout=None):
        if 'main_end_ns' not in self._record:
            self._record['main_end_ns'] = time.perf_counter_ns()
        try:
            return self._future.result(timeout=timeout)
        finally:
            self._record['wait_end_ns'] = time.perf_counter_ns()


class _RecordingExecutor:
    def __init__(self, executor):
        self.executor, self.record = executor, {}

    def submit(self, operation, *args, **kwargs):
        if self.record:
            raise ValueError('Expected one parallel branch')
        self.record['submit_ns'] = time.perf_counter_ns()

        def invoke():
            self.record['worker_start_ns'] = time.perf_counter_ns()
            try:
                return operation(*args, **kwargs)
            finally:
                self.record['worker_end_ns'] = time.perf_counter_ns()

        future = self.executor.submit(invoke)
        self.record['main_start_ns'] = time.perf_counter_ns()
        return _RecordingFuture(future, self.record)

    def observe(self, started):
        record = self.record
        required = ('worker_start_ns', 'worker_end_ns', 'main_start_ns', 'main_end_ns', 'wait_end_ns')
        if any(name not in record for name in required):
            raise AssertionError('Parallel branch incomplete')
        first = record['worker_end_ns'] - record['worker_start_ns']
        second = record['main_end_ns'] - record['main_start_ns']
        if min(first, second) < 0:
            raise AssertionError('parallel branch timestamps are not monotonic')
        return {'concurrent_a_ms': first / 1e6, 'concurrent_b_ms': second / 1e6,
                'worker_start_offset_ms': (record['worker_start_ns'] - started) / 1e6,
                'main_start_offset_ms': (record['main_start_ns'] - started) / 1e6,
                'submission_offset_ms': (record['submit_ns'] - started) / 1e6,
                'wait_end_offset_ms': (record['wait_end_ns'] - started) / 1e6}


def _immutable(value):
    return np.frombuffer(np.asarray(value, dtype=np.int64).tobytes(), dtype=np.int64)


def _same(left, right):
    if isinstance(right, tuple):
        return isinstance(left, tuple) and len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right))
    if isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and np.array_equal(left, right)
    return left == right


def _measure(phase, shape, work, prepare, check_combined, repeats, warmups):
    modes = ('branch_a', 'branch_b', 'joint') + (('post',) if phase == 'v_equation' else ())
    reference = {}
    for mode in modes:
        prepared = prepare(mode)
        operation, normalize = prepared[:2]
        reference[mode] = normalize(operation())
    check_combined(reference)
    samples = {mode: [] for mode in modes}
    concurrent = {name: [] for name in ('concurrent_a_ms', 'concurrent_b_ms',
                  'worker_start_offset_ms', 'main_start_offset_ms',
                  'submission_offset_ms', 'wait_end_offset_ms')}
    for iteration in range(warmups + repeats):
        order = modes[iteration % len(modes):] + modes[:iteration % len(modes)]
        if iteration % 2:
            order = tuple(reversed(order))
        for mode in order:
            prepared = prepare(mode)
            operation, normalize = prepared[:2]
            observe = prepared[2] if len(prepared) == 3 else None
            started = time.perf_counter_ns()
            value = operation()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            observed = observe(started) if observe is not None else None
            actual = normalize(value)
            if not _same(actual, reference[mode]):
                raise AssertionError(f'{phase} parallel calibration changed output or randomness')
            if iteration >= warmups:
                samples[mode].append(elapsed)
                if observed is not None:
                    for name, duration in observed.items():
                        concurrent[name].append(duration)
            del value, actual, operation, normalize, prepared, observe, observed
    if any(len(values) != repeats for values in concurrent.values()):
        raise AssertionError('Parallel observation count mismatch')
    return {'shape': shape, 'work': work, 'warmup_samples': warmups,
            'sample_count': repeats, **{name + '_ms': values for name, values in samples.items()},
            **concurrent, 'branch_timestamps_in_joint_scope': True}


def _triples(params, width, count):
    extra = count - width
    if not 0 < extra < width // 2:
        raise ValueError('Mixed constraint layout required')
    return _immutable_multiplication_triples(width,
        ((extra, width - extra, 0, extra, True, params.m * params.k),), np.uint32)


def _work(params, width, count, F_columns, rows):
    return {'W': width, 'L': count, 'F_columns': F_columns, 'relation_rows': rows,
            'q': params.q, 'n': params.n, 'm': params.m, 'k': params.k,
            'depth': params.depth, 'recipient_decode_rows': 2 * params.n * params.m}


def _context(params, settings, relation, width, count, F_columns, repeats, warmups, shape):
    triples = _triples(params, width, count)
    x = _immutable(np.ones(width, dtype=np.int64))
    operator = _Operator(relation, width, F_columns, prepared=True)
    y = _immutable(operator @ x)
    public = SimpleNamespace(M=operator, y=y, triples=triples, statement_digest='21' * 32)
    pp = SimpleNamespace(digest=bytes.fromhex('43' * 32), witness_key=SimpleNamespace(q=params.q))
    auxiliary = b'parallel calibration'

    def digest():
        return framed_hash(b'Yang P1 FDGE statement context v1',
                           bytes.fromhex(public.statement_digest), y, triples, pp.digest, auxiliary)

    def prepare(mode):
        source = CryptographicRNG(bytes.fromhex('19' * 32))
        if mode == 'branch_a':
            return digest, lambda value: value
        if mode == 'branch_b':
            def operation():
                if not np.array_equal(operator @ x, y):
                    raise AssertionError('context linear equation changed')
                valid, plan = _quadratic_witness_layout(triples, x, params.q)
                if not valid or plan is None:
                    raise AssertionError('context witness constraints changed')
                return plan
            return operation, lambda value: (value is not None, source._random_bytes(64))
        holder = {}
        def operation():
            with ThreadPoolExecutor(max_workers=1) as pool:
                executor = _RecordingExecutor(pool)
                holder['executor'] = executor
                return _prover_witness_context_phase(public, x, pp, auxiliary, source,
                    executor=executor, _collect_layout=True)
        return (operation, lambda value: (value[0], value[2] is not None, value[1]._random_bytes(64)),
                lambda started: holder['executor'].observe(started))

    def combined(ref):
        if not _same(ref['joint'], (ref['branch_a'], *ref['branch_b'])):
            raise AssertionError('Context branch mismatch')

    return _measure('context', shape, _work(params, width, count, F_columns, len(y)),
                    prepare, combined, repeats, warmups)


def _first(params, settings, dims, A, width, count, repeats, warmups, shape):
    key = _UncachedKey(params, settings, A, width)
    triples = _triples(params, width, count)
    x = np.ones(width, dtype=np.int64)
    valid, layout = _quadratic_witness_layout(triples, x, params.q)
    if not valid or layout is None or count + settings.l1 + settings.l2 < 1048576:
        raise ValueError('First-phase execution branch changed')
    rng = CryptographicRNG(bytes.fromhex('29' * 32))
    with uniform_source(rng) as source:
        r = uniform_field(source, (width,), params.q)
    s1 = sample_standard_discrete_gaussian(rng, key.randomness_size, settings.sigma1, settings.gaussian_tail)
    s2 = sample_standard_discrete_gaussian(rng, key.randomness_size, dims['yang_sigma2'], settings.gaussian_tail)
    length = count + settings.l1 + settings.l2

    def prepare(mode):
        source = CryptographicRNG(bytes.fromhex('27' * 32))
        if mode == 'branch_a':
            return lambda: key.commit_pair((x, s1), (r, s2)), lambda value: value
        if mode == 'branch_b':
            def operation():
                s3 = sample_standard_discrete_gaussian(source, length, settings.sigma1, settings.gaussian_tail)
                s4 = sample_standard_discrete_gaussian(source, length, dims['yang_sigma2'], settings.gaussian_tail)
                a, b = _quadratic_masks(triples, r, x, params.q, _layout_plan=layout)
                return s3, s4, a, b
            return operation, lambda value: (*value, source._random_bytes(64))
        holder = {}
        def operation():
            pool = ThreadPoolExecutor(max_workers=1)
            holder['pool'] = pool
            executor = _RecordingExecutor(pool)
            holder['executor'] = executor
            try:
                return _prover_first_phase(key, triples, x, r, s1, s2, source, length,
                    settings.sigma1, dims['yang_sigma2'], settings.gaussian_tail,
                    executor=executor, _layout_plan=layout)
            except BaseException:
                pool.shutdown(wait=True)
                raise
        def normalize(value):
            holder.pop('pool').shutdown(wait=True)
            return (*value, source._random_bytes(64))
        return operation, normalize, lambda started: holder['executor'].observe(started)

    def combined(ref):
        if not _same(ref['joint'], (*ref['branch_a'], *ref['branch_b'])):
            raise AssertionError('First-phase branch mismatch')

    result = _measure('p_first', shape, _work(params, width, count, 0, 0),
                      prepare, combined, repeats, warmups)
    result.update(executor_start_included=True, executor_shutdown_included=False)
    return result


def _equations(phase, params, settings, dims, A, relation, width, count, F_columns,
               repeats, warmups, shape):
    matrix = _MatrixFixture(A, (params.n, F_columns)).__array__(np.float64)
    matrix.flags.writeable = False
    resident = relation.metadata['fixture_bytes'] + matrix.nbytes
    size = count if phase == 'p_second' else width
    key = _UncachedKey(params, settings, A, size, resident)
    operator = _Operator(relation, width, F_columns, matrix=matrix, prepared=phase == 'p_second')
    triples = _triples(params, width, count)
    rng = CryptographicRNG(bytes.fromhex('37' * 32))
    with uniform_source(rng) as source:
        vector = uniform_field(source, (width,), params.q)
    vector[0] = params.q - 1
    if phase == 'v_equation':
        vector = _immutable(vector)
    s1 = sample_standard_discrete_gaussian(rng, key.randomness_size, settings.sigma1, settings.gaussian_tail)
    s2 = sample_standard_discrete_gaussian(rng, key.randomness_size, dims['yang_sigma2'], settings.gaussian_tail)
    if phase == 'p_second':
        a, b = _quadratic_masks(triples, vector, np.ones(width, dtype=np.int64), params.q)

        def prepare(mode):
            if mode == 'branch_a':
                return lambda: key.commit_pair((a, s1), (b, s2)), lambda value: value
            if mode == 'branch_b':
                return lambda: operator @ vector, lambda value: value
            pool = ThreadPoolExecutor(max_workers=1)
            pool.submit(lambda: None).result()
            executor = _RecordingExecutor(pool)
            def operation():
                try:
                    return _prover_second_phase(key, operator, vector, a, b, s1, s2, executor=executor)
                finally:
                    pool.shutdown(wait=True)
            return operation, lambda value: value, executor.observe

        def combined(ref):
            if not _same(ref['joint'], (ref['branch_b'], *ref['branch_a'])):
                raise AssertionError('Second-phase branch mismatch')
    else:
        alpha = -settings.p
        z1 = _immutable(s2 + alpha * s1)
        del s1, s2
        t = _immutable(operator @ vector)
        y = _immutable(np.zeros_like(t))
        c1 = _immutable(np.zeros(key.key.commitment_size, dtype=np.int64))
        c2 = _immutable(key.commit(vector, z1))

        def prepare(mode):
            if mode == 'branch_a':
                return lambda: key.commit(vector, z1), lambda value: value
            if mode == 'post':
                return lambda: _affine_canonical_equal(c2, c1, alpha, c2, params.q), lambda value: value
            if mode == 'branch_b':
                def operation():
                    if not _affine_canonical_equal(operator @ vector, y, alpha, t, params.q):
                        raise AssertionError('Verifier equation failed')
                    return _quadratic_response(triples, vector, alpha, params.q)
                return operation, lambda value: value
            holder = {}
            def operation():
                with ThreadPoolExecutor(max_workers=1) as pool:
                    executor = _RecordingExecutor(pool)
                    holder['executor'] = executor
                    return _verifier_equation_phase(key, operator, triples, vector, z1,
                        alpha, y, t, c1, c2, executor=executor)
            return operation, lambda value: value, lambda started: holder['executor'].observe(started)

        def combined(ref):
            if (not ref['post'] or not _same(ref['branch_a'], c2)
                    or not _same(ref['joint'], ref['branch_b'])):
                raise AssertionError('Verifier branch mismatch')

    result = _measure(phase, shape, _work(params, width, count, F_columns,
                      relation.metadata['relation_rows']), prepare, combined, repeats, warmups)
    result.update(executor_start_included=phase == 'v_equation', executor_shutdown_included=True)
    return result


def calibrate_parallel_schedule(params, settings, A, *, repeats=10, warmups=2, on_progress=None):
    params.validate()
    settings.validate(params.q)
    if repeats != 10 or warmups != 2:
        raise ValueError('Schedule requires warmup=2,n=10')
    dims = target_dimensions(params, settings)
    records = {name: [] for name in ('context', 'p_first', 'p_second', 'v_equation')}
    order_rng = np.random.default_rng(84219)

    def add(phase, result):
        records[phase].append(result)
        if on_progress:
            on_progress({'phase': 'parallel_schedule', 'stage': phase,
                         'index': len(records[phase]), 'shapes': 6})

    relation = RelationWork(params, A, 'binary')
    context_shapes = ((524288, 700000, 131072), (1048576, 1400000, 262144),
                      (2097152, 2800000, 524288), (4194304, 5600000, 1048576),
                      (2097152, 2500000, 1048576), (4194304, 6200000, 524288))
    for index in order_rng.permutation(len(context_shapes)):
        width, count, F_columns = context_shapes[index]
        add('context', _context(params, settings, relation, width, count, F_columns,
                               repeats, warmups, f'context-{index}'))
    del relation
    first_shapes = ((720000, 1060000), (800000, 1100000), (880000, 1160000),
                    (960000, 1420000), (880000, 1300000), (960000, 1220000))
    equation_shapes = ((160000, 230000, 40000), (192000, 250000, 80000),
                       (224000, 310000, 120000), (240000, 350000, 80000),
                       (224000, 280000, 160000), (192000, 280000, 140000))
    for index in order_rng.permutation(len(first_shapes)):
        width, count = first_shapes[index]
        add('p_first', _first(params, settings, dims, A, width, count,
                             repeats, warmups, f'p_first-{index}'))
    relation = RelationWork(params, A, 'field')
    for phase in ('p_second', 'v_equation'):
        for index in order_rng.permutation(len(equation_shapes)):
            width, count, F_columns = equation_shapes[index]
            add(phase, _equations(phase, params, settings, dims, A, relation, width,
                count, F_columns, repeats, warmups, f'{phase}-{index}'))
    return {'phases': {name: fit_parallel_schedule(values) for name, values in records.items()},
            'records': records, 'warmup_samples': warmups, 'sample_count': repeats,
            'independent_workloads': True, 'full_target_executed': False,
            'prover_executor_lifecycle_included_once': True,
            'shape_order_seed': 84219}
