from __future__ import annotations

from dataclasses import replace
import hashlib
import statistics
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .decomposition import weights
from .mathutils import add_mod
from .params import PAPER_GROUP_DEPTHS
from .relations import (_accumulator_linear_operands, _mm, _ordinary_decode,
                        _ordinary_encode, _scale, _weighted)


def _fixture_rng(kind, name):
    seed = int.from_bytes(hashlib.sha256(
        ('relation-work:' + kind + ':' + name).encode()).digest(), 'little')
    return np.random.default_rng(seed)


class RelationWork:
    def __init__(self, params, A, kind='field'):
        params.validate()
        if kind not in ('field', 'binary'):
            raise ValueError('relation workload kind must be field or binary')
        n, m, k, nk, ell, q = (params.n, params.m, params.k, params.nk,
                              params.depth, params.q)
        if (not isinstance(A, np.ndarray) or A.shape != (n, m)
                or A.dtype != np.dtype(np.int64) or np.any(A < 0) or np.any(A >= q)):
            raise ValueError('Canonical relation matrix required')
        self.params, self.kind = params, kind
        upper = 2 if kind == 'binary' else q
        self.parts = {}

        def part(name, size):
            value = _fixture_rng(kind, name).integers(0, upper, size=size, dtype=np.int64)
            value.flags.writeable = False
            self.parts[name] = value

        self.A = np.array(A, copy=True, order='C')
        self.Ar = (A + 1) % q
        self.Ao = (A + 2) % q
        self.Uo = tuple(np.ascontiguousarray((A[:, :ell] + offset) % q)
                        for offset in (3, 4))
        self.patterns = _fixture_rng('public', 'patterns').integers(
            0, 2, size=(params.e, params.t), dtype=np.int64)
        self.zero_F = np.zeros(n, dtype=np.int64)
        for array in (self.A, self.Ar, self.Ao, *self.Uo, self.patterns, self.zero_F):
            array.flags.writeable = False
        part('identity', ell)
        part('p', nk)
        part('nonzero_slack', weights(nk - 1).size)
        part('message', m)
        policy_rows = params.e * (m - params.t + 1)
        part('policy_slack', policy_rows * weights(m - params.d).size)
        for level in range(ell):
            if level:
                part(f'node{level}', nk)
            for prefix in ('sibling', 'jnode', 'jsibling'):
                part(f'{prefix}{level}', nk)
        noise_digits = weights(2 * params.error_bound).size
        for index in range(2):
            part(f's_rec{index}', n)
            part(f'x_rec{index}', m * noise_digits)
            part(f'y_rec{index}', m * noise_digits)
            part(f'a_times_s{index}', n * m * k)
            part(f's_oa{index}', n)
            part(f'x_oa{index}', m * noise_digits)
            part(f'y_oa{index}', ell * noise_digits)
        rows = ell * n + 1 + n + 6 * m + 2 * ell + policy_rows
        self.metadata = {
            'q': q, 'n': n, 'm': m, 'k': k, 'depth': ell, 'N': 1 << ell,
            'relation_rows': rows, 'recipient_product_shapes': [[n, m, k]] * 2,
            'independent_recipient_inputs': True,
            'independent_public_role_arrays': True,
            'shared_fixed_inputs_across_depths': True,
            'input_domain': kind,
            'input_normalization_included': False,
            'output_modulo_and_concatenation_included': True,
            'F_product_included': False,
            'F_zero_subtraction_included': True,
            'valid_protocol_witness': False,
            'fixture_preparation_included': False,
            'fixture_bytes': sum(value.nbytes for value in self.parts.values())
                + sum(value.nbytes for value in
                      (self.A, self.Ar, self.Ao, *self.Uo, self.patterns, self.zero_F)),
        }

    def apply(self, F_action=None):
        p = self.params
        n, m, k, nk, ell, q, bound = (p.n, p.m, p.k, p.nk, p.depth, p.q, p.error_bound)
        parts, outputs = self.parts, []

        def output(value):
            outputs.append(np.asarray(value, dtype=np.int64).reshape(-1) % q)

        A0, A1 = self.A[:, :nk], self.A[:, nk:]
        for level in range(ell):
            node_name = 'p' if level == 0 else f'node{level}'
            left, right = _accumulator_linear_operands(
                parts[node_name], parts[f'jnode{level}'],
                parts[f'sibling{level}'], parts[f'jsibling{level}'], q)
            value = add_mod(_mm(A0, left, q), _mm(A1, right, q), q=q)
            if level + 1 < ell:
                value -= _ordinary_decode(parts[f'node{level + 1}'], n, k, q)
            output(value % q)
            del left, right, value
        output(np.array([(int(_mm(np.ones(nk, dtype=np.int64), parts['p'], q))
                          - int(_mm(weights(nk - 1), parts['nonzero_slack'], q))) % q]))
        leaf = _ordinary_decode(parts['p'], n, k, q)
        syndrome = self.zero_F if F_action is None else F_action()
        if np.asarray(syndrome).shape != (n,):
            raise ValueError('injected F product must have n coordinates')
        output((leaf - syndrome) % q)
        del leaf, syndrome
        for index in range(2):
            output(add_mod(_mm(self.Ar.T, parts[f's_rec{index}'], q),
                           _weighted(parts[f'x_rec{index}'], 2 * bound, m, q), q=q))
            products = parts[f'a_times_s{index}'].reshape(n, m, k)
            decoded = _mm(products.reshape(n * m, k), weights(q - 1), q).reshape(n, m)
            output(add_mod(_mm(np.ones(n, dtype=np.int64), decoded, q),
                           _weighted(parts[f'y_rec{index}'], 2 * bound, m, q),
                           _scale(parts['message'], q // 2, q), q=q))
            del products, decoded
            output(add_mod(_mm(self.Ao.T, parts[f's_oa{index}'], q),
                           _weighted(parts[f'x_oa{index}'], 2 * bound, m, q), q=q))
            output(add_mod(_mm(self.Uo[index].T, parts[f's_oa{index}'], q),
                           _weighted(parts[f'y_oa{index}'], 2 * bound, ell, q),
                           _scale(parts['identity'], q // 2, q), q=q))
        windows = sliding_window_view(parts['message'], p.t)
        linear = np.concatenate([_mm(windows, 1 - 2 * pattern, q)
                                 for pattern in self.patterns])
        output((linear - _weighted(parts['policy_slack'], m - p.d,
                                    p.e * (m - p.t + 1), q)) % q)
        return np.concatenate(outputs)


def calibrate_relation_work(params, settings, A, repeats=10, warmups=2, on_progress=None):
    params.validate()
    settings.validate(params.q)
    if (type(repeats) is not int or repeats < 1
            or type(warmups) is not int or warmups < 0):
        raise ValueError('invalid relation-work calibration counts')

    def measure(name, operation, expected, **metadata):
        for _ in range(warmups):
            actual = operation()
            if not np.array_equal(actual, expected):
                raise AssertionError('relation-work warmup changed its result')
            del actual
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            actual = operation()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if not np.array_equal(actual, expected):
                raise AssertionError('relation-work sample changed its result')
            samples.append(elapsed)
            del actual
        return {'kernel': name, 'units_per_sample': 1, 'unit': 'calls',
                'fixed_work_calls': True, 'measured_this_run': True,
                'rate_kind': 'measured_full_fixed_shape_call', 'q': params.q,
                'warmup_samples': warmups, 'sample_count': repeats,
                'samples_ms': samples, 'mean_ms': statistics.mean(samples),
                'std_ms': statistics.stdev(samples) if repeats > 1 else None,
                'empirical_min_ms': min(samples), 'empirical_max_ms': max(samples),
                **metadata}

    rng = np.random.default_rng(747931)
    depth_order = list(PAPER_GROUP_DEPTHS)
    rng.shuffle(depth_order)
    by_depth = {}
    for depth in depth_order:
        current = replace(params, depth=depth)
        by_depth[depth] = {}
        kinds = ['binary', 'field']
        rng.shuffle(kinds)
        for kind in kinds:
            fixture = RelationWork(current, A, kind)
            expected = fixture.apply()
            if expected.shape != (fixture.metadata['relation_rows'],):
                raise AssertionError('Relation row count mismatch')
            sample = measure('relation_non_F_' + kind, fixture.apply, expected,
                             **fixture.metadata)
            by_depth[depth][kind] = sample
            del fixture, expected
        if on_progress:
            on_progress({'phase': 'relation_work', 'depth': depth, 'N': 1 << depth})
    n, k, q = params.n, params.k, params.q
    coordinates = rng.integers(0, 2, size=(n, k), dtype=np.int64)
    weights2 = 1 << np.arange(k, dtype=np.int64)
    decoded = np.sum(coordinates * weights2, axis=1, dtype=np.int64) % q
    ordinary_decode = measure('ordinary_decode',
        lambda: _ordinary_decode(coordinates.reshape(-1), n, k, q), decoded,
        input_shape=[n * k], output_shape=[n], implementation='_ordinary_decode')
    values = rng.integers(0, q, size=n, dtype=np.int64)
    encoded = ((values[:, None] >> np.arange(k)) & 1).reshape(-1)
    ordinary_encode = measure('ordinary_encode', lambda: _ordinary_encode(values, k), encoded,
        input_shape=[n], output_shape=[n * k], implementation='_ordinary_encode')
    return {'kind': 'original_shape_non_F_relation_calibration',
            'by_depth': by_depth,
            'kernels': {'ordinary_decode': ordinary_decode, 'ordinary_encode': ordinary_encode},
            'warmup_samples': warmups, 'sample_count': repeats,
            'depth_order': depth_order, 'includes_fixture_preparation': False}
