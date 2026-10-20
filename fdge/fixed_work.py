from __future__ import annotations

import math
import hashlib
import statistics
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .accumulator import LNWXAccumulator
from .commitments import auxiliary_digest, _auxiliary_digest_state
from .decomposition import vdec, weights
from .mathutils import _matmul_mod_canonical_left, matmul_mod, rand_discrete_gaussian
from .params import PAPER_GROUP_DEPTHS
from .randomness import CryptographicRNG
from .relations import _accumulator_linear_operands, _weighted, _mm
from .yang import _snapshot_integer_vector


def calibrate_fixed_work(params, settings, Arec, repeats=10, warmups=2):
    from .cost_model import target_dimensions

    params.validate()
    settings.validate(params.q)
    if type(repeats) is not int or repeats < 1 or type(warmups) is not int or warmups < 0:
        raise ValueError('invalid fixed-work calibration counts')
    if (not isinstance(Arec, np.ndarray) or Arec.shape != (params.n, params.m)
            or Arec.dtype != np.dtype(np.int64) or not Arec.flags.c_contiguous
            or np.any(Arec < 0) or np.any(Arec >= params.q)):
        raise ValueError('fixed-work calibration requires canonical Arec')
    dims = target_dimensions(params, settings)
    q, n, m, nk = params.q, params.n, params.m, params.nk
    rng = CryptographicRNG()
    kernels = {}

    def same(actual, expected):
        if isinstance(expected, tuple):
            return (isinstance(actual, tuple) and len(actual) == len(expected)
                    and all(np.array_equal(a, b) for a, b in zip(actual, expected)))
        if expected is None:
            return actual is None
        return np.array_equal(actual, expected)

    def measure(name, operation, expected, units, unit, **metadata):
        for _ in range(warmups):
            actual = operation()
            if not same(actual, expected):
                raise AssertionError('Fixed-work warmup mismatch')
            del actual
        samples = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            actual = operation()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if not same(actual, expected):
                raise AssertionError('Fixed-work result mismatch')
            samples.append(elapsed)
            del actual
        return {'kernel': name, 'units_per_sample': int(units), 'unit': unit,
                'fixed_work_calls': True, 'measured_this_run': True,
                'rate_kind': 'measured_full_fixed_shape_call', 'q': q,
                'warmup_samples': warmups, 'sample_count': repeats,
                'samples_ms': samples, 'mean_ms': statistics.mean(samples),
                'std_ms': statistics.stdev(samples) if repeats > 1 else None,
                'empirical_min_ms': min(samples), 'empirical_max_ms': max(samples),
                **metadata}

    def product(name, matrix, operand, *, canonical=False, unit='matrix_entries', units=None):
        bound = matrix.shape[1] * int(np.max(np.abs(matrix))) * int(np.max(np.abs(operand)))
        if bound > np.iinfo(np.int64).max:
            raise ValueError('fixed-work integer oracle would overflow')
        expected = (matrix @ operand) % q
        implementation = _matmul_mod_canonical_left if canonical else matmul_mod
        return measure(name, lambda: implementation(matrix, operand, q), expected,
                       matrix.size if units is None else units, unit,
                       matrix_shape=list(matrix.shape), matrix_strides=list(matrix.strides),
                       operand_shape=list(operand.shape), rhs_count=1,
                       implementation=implementation.__name__,
                       fixture_preparation_included=False)

    field_nk = rng.integers(0, q, size=nk, dtype=np.int64)
    kernels['accumulator_matvec'] = product('accumulator_matvec', Arec[:, :nk], field_nk)
    path_children = rng.integers(0, 2, size=m, dtype=np.int64)
    kernels['path_reconstruction'] = product('path_reconstruction', Arec, path_children)
    field_n = rng.integers(0, q, size=n, dtype=np.int64)
    kernels['encryption_matvec'] = product('encryption_matvec', Arec.T, field_n)

    oa_by_depth = {}
    for depth in PAPER_GROUP_DEPTHS:
        matrix = np.ascontiguousarray(Arec[:, :depth]).T
        sample = product('oa_matvec', matrix, field_n)
        sample.update(depth=depth, N=1 << depth)
        oa_by_depth[depth] = sample
    kernels['oa_matvec'] = oa_by_depth[params.depth]

    plan = dims['yang_memory_plan']
    auxiliary_width = plan['auxiliary_message_bits'] + plan['auxiliary_randomness_bits']
    if settings.auxiliary_rows > n or auxiliary_width > m:
        raise ValueError('Auxiliary fixture exceeds Arec')
    auxiliary_matrix = np.frombuffer(
        np.ascontiguousarray(Arec[:settings.auxiliary_rows, :auxiliary_width]).tobytes(),
        dtype=np.int64).reshape(settings.auxiliary_rows, auxiliary_width)
    auxiliary_bits = rng.integers(0, 2, size=auxiliary_width, dtype=np.int64)
    kernels['auxiliary_binary'] = product('auxiliary_binary', auxiliary_matrix,
                                          auxiliary_bits, canonical=True)

    operands = tuple(rng.integers(0, q, size=nk, dtype=np.int64) for _ in range(4))
    node, jnode, sibling, jsibling = operands
    difference = (jnode - jsibling) % q
    expected = ((node - difference) % q, (sibling + difference) % q)
    kernels['accumulator_linear_operands'] = measure(
        'accumulator_linear_operands', lambda: _accumulator_linear_operands(*operands, q),
        expected, nk, 'coordinates', operand_shape=[nk], rhs_count=4,
        implementation='_accumulator_linear_operands', fixture_preparation_included=False)

    gaussian_column = rand_discrete_gaussian(rng, (m,), params.sigma, params.beta,
                                             params.max_memory_bytes)
    kernels['gpv_AE_columns'] = product('gpv_AE_columns', Arec, gaussian_column,
                                         unit='output_columns', units=1)

    accumulator = object.__new__(LNWXAccumulator)
    accumulator.params = params
    leaf = rng.integers(0, 2, size=nk, dtype=np.uint8)
    equal_leaf = leaf.copy()
    kernels['node_bits'] = measure(
        'node_bits', lambda: accumulator._check_bits(leaf), None, 1, 'calls',
        node_coordinates=nk, dtype=leaf.dtype.name, implementation='LNWXAccumulator._check_bits')
    kernels['node_equal'] = measure(
        'node_equal', lambda: np.array_equal(leaf, equal_leaf), True, 1, 'calls',
        node_coordinates=nk, dtype=leaf.dtype.name, implementation='numpy.array_equal')

    return {'kind': 'fixed_shape_protocol_work_calibration', 'kernels': kernels,
            'oa_by_depth': oa_by_depth,
            'published_path': {name: kernels[name] for name in ('node_bits', 'node_equal')},
            'warmup_samples': warmups, 'sample_count': repeats,
            'includes_fixture_preparation': False}


def calibrate_encoding_work(params, settings, fixture_scale=1.0, repeats=10, warmups=2):
    params.validate()
    settings.validate(params.q)
    if (type(repeats) is not int or repeats < 1 or type(warmups) is not int or warmups < 0
            or not isinstance(fixture_scale, (int, float)) or not math.isfinite(fixture_scale)
            or not 0.125 <= fixture_scale <= 1):
        raise ValueError('invalid encoding calibration counts or scale')
    rows = round(131072 * fixture_scale)
    q = params.q
    rng = CryptographicRNG()
    kernels = {}

    def measure(name, operation, check, units, unit, **metadata):
        for _ in range(warmups):
            actual = operation()
            check(actual)
            del actual
        samples = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            actual = operation()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            check(actual)
            samples.append(elapsed)
            del actual
        return {'kernel': name, 'units_per_sample': int(units), 'unit': unit,
                'measured_this_run': True, 'rate_kind': 'measured_current_encoding_call',
                'q': q, 'fixture_scale': fixture_scale, 'warmup_samples': warmups,
                'sample_count': repeats, 'samples_ms': samples,
                'mean_ms': statistics.mean(samples),
                'std_ms': statistics.stdev(samples) if repeats > 1 else None,
                'empirical_min_ms': min(samples), 'empirical_max_ms': max(samples),
                'fixture_preparation_included': False, **metadata}

    values = rng.integers(0, q, size=rows, dtype=np.int64)
    decomposition_weights = weights(q - 1)

    def check_decomposition(actual):
        if (actual.shape != (rows * len(decomposition_weights),)
                or actual.dtype != np.dtype(np.uint8) or np.any(actual > 1)
                or not np.array_equal(actual.reshape(rows, -1) @ decomposition_weights, values)):
            raise AssertionError('Weighted decomposition mismatch')

    kernels['weighted_decomposition'] = measure(
        'weighted_decomposition', lambda: vdec(values, q - 1), check_decomposition,
        rows, 'field_elements', beta=q - 1, digits=len(decomposition_weights),
        input_shape=[rows], output_shape=[rows * len(decomposition_weights)],
        implementation='vdec')

    for kind, beta in (('noise', 2 * params.error_bound), ('policy', params.m - params.d)):
        source = rng.integers(0, beta + 1, size=rows, dtype=np.int64)
        digit_weights = weights(beta)

        def check_bounded_decomposition(actual, source=source, digit_weights=digit_weights):
            if (actual.shape != (rows * len(digit_weights),)
                    or actual.dtype != np.dtype(np.uint8) or np.any(actual > 1)
                    or not np.array_equal(actual.reshape(rows, -1) @ digit_weights, source)):
                raise AssertionError('Bounded decomposition mismatch')

        name = 'weighted_decomposition_' + kind
        kernels[name] = measure(
            name, lambda source=source, beta=beta: vdec(source, beta),
            check_bounded_decomposition, rows, 'field_elements', beta=beta,
            digits=len(digit_weights), input_shape=[rows],
            output_shape=[rows * len(digit_weights)], implementation='vdec')

    for kind, beta in (('q', q - 1), ('noise', 2 * params.error_bound),
                       ('policy', params.m - params.d)):
        digit_weights = weights(beta)
        matrix = rng.integers(0, q, size=(rows, len(digit_weights)), dtype=np.int64)
        operand = matrix.reshape(-1)
        expected = np.sum(matrix * digit_weights, axis=1, dtype=np.int64) % q

        def check_decode(actual, expected=expected):
            if not np.array_equal(actual, expected):
                raise AssertionError('Weighted decoding mismatch')

        name = 'weighted_decode_' + kind
        kernels[name] = measure(
            name, lambda operand=operand, beta=beta: _weighted(operand, beta, rows, q),
            check_decode, rows, 'decoded_coordinates', beta=beta, digits=len(digit_weights),
            input_shape=list(matrix.shape), output_shape=[rows],
            input_domain='canonical_field', implementation='_weighted')

    message = rng.integers(0, q, size=params.m, dtype=np.int64)
    pattern = rng.integers(0, 2, size=params.t, dtype=np.int64)
    windows = sliding_window_view(message, params.t)
    signed_pattern = 1 - 2 * pattern
    expected = np.sum(windows * signed_pattern, axis=1, dtype=np.int64) % q

    def policy_product():
        return _mm(windows, 1 - 2 * pattern, q)

    def check_policy(actual):
        if not np.array_equal(actual, expected):
            raise AssertionError('Policy product mismatch')

    kernels['policy_window_product'] = measure(
        'policy_window_product', policy_product, check_policy, 1, 'pattern',
        fixed_work_calls=True, matrix_shape=list(windows.shape),
        matrix_strides=list(windows.strides), input_domain='canonical_field',
        implementation='_mm', includes_window_view=False,
        includes_signed_pattern=True)

    binary_message = rng.integers(0, 2, size=params.m, dtype=np.int64)
    binary_windows = sliding_window_view(binary_message, params.t)
    witness_distances = np.sum(np.abs(binary_windows - pattern), axis=1, dtype=np.int64)

    def check_distances(actual):
        if not np.array_equal(actual, witness_distances):
            raise AssertionError('Policy distance mismatch')

    kernels['policy_witness_distances'] = measure(
        'policy_witness_distances', lambda: np.count_nonzero(binary_windows != pattern, axis=1),
        check_distances, 1, 'pattern', fixed_work_calls=True,
        matrix_shape=list(binary_windows.shape), matrix_strides=list(binary_windows.strides),
        input_domain='binary', implementation='numpy.count_nonzero',
        includes_window_view=False)
    return {'kind': 'current_protocol_encoding_calibration', 'kernels': kernels,
            'fixture_scale': fixture_scale, 'warmup_samples': warmups,
            'sample_count': repeats, 'includes_fixture_preparation': False}


def calibrate_auxiliary_work(params, settings, fixture_scale=1.0, repeats=10, warmups=2):
    from .cost_model import target_dimensions

    if (type(repeats) is not int or repeats < 1 or type(warmups) is not int or warmups < 0
            or not isinstance(fixture_scale, (int, float)) or not math.isfinite(fixture_scale)
            or not 0.125 <= fixture_scale <= 1):
        raise ValueError('invalid auxiliary calibration counts or scale')
    dims = target_dimensions(params, settings)
    q, width, count = params.q, dims['witness_coordinates'], dims['quadratic_constraints']
    target_lengths = (dims['relation_rows'], settings.l1 + width, settings.l1 + width,
                      settings.l1 + count, settings.l1 + count)
    scale = 4194304 * fixture_scale / sum(target_lengths)
    lengths = tuple(max(1, round(length * scale)) for length in target_lengths)
    rng = CryptographicRNG()
    vectors = tuple(rng.integers(0, q, size=length, dtype=np.int64) for length in lengths)
    context = bytes.fromhex('39' * 32)
    reference = hashlib.sha256(b'FDGE-aCommit-CRH-v1\0')
    reference.update(q.to_bytes(8, 'little'))
    reference.update(len(vectors).to_bytes(8, 'little'))
    reference.update(context)
    encoded_dtype = np.dtype('<u4' if q <= 2**32 else '<u8')
    for vector in vectors:
        reference.update(vector.size.to_bytes(8, 'little'))
        reference.update(vector.astype(encoded_dtype).tobytes())
    expected = reference.digest()

    def snapshot_hash():
        state = _auxiliary_digest_state(context, q, 5)
        copies = tuple(_snapshot_integer_vector(vector, length, name, q, digest=state)
                       for vector, length, name in zip(vectors, lengths, ('t', 'c1', 'c2', 'c3', 'c4')))
        return copies, state.digest()

    def check_hash(actual):
        if actual != expected:
            raise AssertionError('Auxiliary hash mismatch')

    def check_snapshots(actual):
        copies, digest = actual
        check_hash(digest)
        if (len(copies) != 5 or any(copy.dtype != np.dtype(np.int64) or copy.flags.writeable
                                  or np.shares_memory(copy, original)
                                  or not np.array_equal(copy, original)
                                  for copy, original in zip(copies, vectors))):
            raise AssertionError('Invalid auxiliary snapshots')

    kernels = {}
    for name, operation, check in (
            ('auxiliary_message_hash', lambda: auxiliary_digest(vectors, context, q), check_hash),
            ('auxiliary_snapshot_hash', snapshot_hash, check_snapshots)):
        for _ in range(warmups):
            actual = operation()
            check(actual)
            del actual
        samples = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            actual = operation()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            check(actual)
            samples.append(elapsed)
            del actual
        kernels[name] = {
            'kernel': name, 'units_per_sample': sum(lengths), 'unit': 'field_elements',
            'measured_this_run': True, 'rate_kind': 'measured_five_vector_auxiliary_work',
            'q': q, 'fixture_scale': fixture_scale, 'warmup_samples': warmups,
            'sample_count': repeats, 'samples_ms': samples,
            'mean_ms': statistics.mean(samples),
            'std_ms': statistics.stdev(samples) if repeats > 1 else None,
            'empirical_min_ms': min(samples), 'empirical_max_ms': max(samples),
            'vector_count': 5, 'sample_vector_lengths': list(lengths),
            'target_vector_lengths': list(target_lengths), 'canonical_payload_dtype': encoded_dtype.str,
            'fixture_preparation_included': False,
            'includes_snapshots': name == 'auxiliary_snapshot_hash',
            'includes_shared_digest_framing': True}
    return {'kind': 'five_vector_auxiliary_calibration', 'kernels': kernels,
            'fixture_scale': fixture_scale, 'warmup_samples': warmups,
            'sample_count': repeats, 'includes_fixture_preparation': False}
