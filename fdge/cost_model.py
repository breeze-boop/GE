from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
import statistics
import time

import numpy as np

from .commitments import (auxiliary_digest, sample_standard_discrete_gaussian, _commitment_tail,
                          framed_hash, _auxiliary_digest_state, LatticeCommitmentKey)
from .linear_hash import (LinearPublicKeyHash, linear_hash_storage_plan,
                          _F_DISK_COLUMN_BLOCK, _same_key_decomposition)
from .mathutils import (_modular_product_plan, _matmul_mod_canonical_left, add_mod,
                        _matmul_mod_prepared_left, _prepared_product_plan,
                        matmul_mod, rand_discrete_gaussian, require_memory, scale_mod,
                        _add_canonical_mod, _sub_canonical_mod, _affine_canonical_mod,
                        _affine_canonical_equal, mod_q)
from .relations import (relation_resource_plan, _immutable_multiplication_triples,
                        _canonical_matvec_input, _accumulator_linear_operands)
from .randomness import CryptographicRNG
from .fast_uniform import uniform_backend, uniform_source, uniform_field
from . import metal_matrix
from .yang import (YangProtocolParams, _product, _within_norm, _quadratic_masks,
                   _quadratic_response, _quadratic_witness_valid, _quadratic_witness_layout,
                   _snapshot_integer_vector, _constraint_indices_in_range,
                   _prover_first_phase, _verifier_first_phase, _prover_witness_context_phase,
                   _prover_second_phase, _response_opening, _verifier_equation_phase,
                   resource_plan as yang_resource_plan)


class _MatrixFixture:
    def __init__(self, base, shape):
        self.base, self.shape = (base, shape)

    def __array__(self, dtype=None, copy=None):
        result = np.empty(self.shape, dtype=np.uint32 if dtype is None else dtype)
        for row in range(0, self.shape[0], self.base.shape[0]):
            rows = min(self.base.shape[0], self.shape[0] - row)
            for column in range(0, self.shape[1], self.base.shape[1]):
                columns = min(self.base.shape[1], self.shape[1] - column)
                result[row:row + rows, column:column + columns] = self.base[:rows, :columns]
        return result


def target_dimensions(params, settings: YangProtocolParams | None = None) -> dict:
    params.validate()
    if settings is None:
        raise ValueError('Commitment parameters required')
    settings.validate(params.q)
    if params.missing_parameters:
        raise ValueError('B, sigma, beta required')
    if params.error_bound < 1 or params.sigma <= 0 or params.beta < 1:
        raise ValueError('B, sigma, beta must be > 0')
    if not 0 < params.d <= params.t <= params.m < params.q:
        raise ValueError("require 0 < d <= t <= m < q")
    relation = relation_resource_plan(params)
    width, count, rows = (relation[k] for k in ("witness_coordinates", "triples", "rows"))
    yang = yang_resource_plan(params.q, rows, width, count, settings)
    length = 2 * settings.l1 + 2 * settings.l2 + width + count
    sigma2 = 2 * settings.p * math.sqrt(length) * math.log2(length) * settings.sigma1
    explicit_q_factor = (16 * settings.p * math.sqrt(max(settings.l1 + settings.l2 + width,
                                                       settings.l1 + settings.l2 + count))
                         * (sigma2 + settings.p * settings.sigma1))
    soundness_expression = 2 / (2 * settings.p + 1)
    return {
        "n": params.n, "q": params.q, "k": params.k, "m": params.m,
        "nk": params.nk, "depth": params.depth, "N": params.N,
        "F_rows": params.n, "F_columns": 2 * params.n * params.m * params.k,
        "F_entries": 2 * params.n**2 * params.m * params.k,
        "witness_coordinates": width, "quadratic_constraints": count,
        "relation_rows": rows, "noise_digits": (2 * params.error_bound).bit_length(),
        "policy_rows": params.e * (params.m - params.t + 1),
        "policy_digits": (params.m - params.d).bit_length(),
        "yang_randomness_length": length, "yang_sigma2": sigma2,
        "yang_rejection_M": math.exp(13.3 / math.log2(length) + 1 / (2 * math.log2(length)**2)),
        "yang_theorem_q_explicit_factor": explicit_q_factor,
        "yang_q_exceeds_explicit_factor": params.q >= explicit_q_factor,
        "yang_single_round_soundness_expression": soundness_expression,
        "yang_conditional_rounds_for_80_bits": math.ceil(80 / -math.log2(soundness_expression)),
        "yang_soundness_expression_validated": False,
        "relation_memory_plan": relation, "yang_memory_plan": yang,
        "fdge_validation_errors": params.validation_errors(),
        "security_validated": False,
    }


def _summary(samples):
    return {"samples_ms": samples, "mean_ms": statistics.mean(samples),
            "std_ms": statistics.stdev(samples) if len(samples) > 1 else None,
            "empirical_min_ms": min(samples), "empirical_max_ms": max(samples)}


def _commitment_implementation():
    import inspect
    from .commitments import _commitment_output

    functions = (LatticeCommitmentKey.commit, LatticeCommitmentKey.commit_pair,
                 _commitment_output, _commitment_tail, metal_matrix._can_fuse_commitment,
                 metal_matrix._commitment_products, metal_matrix._matmul_many,
                 metal_matrix._encode_commitment_tail, metal_matrix._write_canonical,
                 metal_matrix._read_product, metal_matrix._encode_product, metal_matrix._use_shoup,
                 _prover_first_phase, _prover_second_phase, _verifier_equation_phase)
    constants = tuple(getattr(metal_matrix, key) for key in
                      ('_Q', '_BLOCK', '_SHORT_ROWS_PER_SIMD', '_SHOUP_MIN_ENTRIES', '_SHOUP_MIN_ROWS'))
    source = repr(constants) + metal_matrix._SOURCE + ''.join(inspect.getsource(function) for function in functions)
    return {'commitment_impl_id': 'metal_complete_commitment_tail_v1',
            'commitment_impl_sha256': hashlib.sha256(source.encode()).hexdigest()}


def _complete_commitment_coverage(width, l1, l2, lanes):
    return {'matrix_entries': l1 * (width + l2) + width * l2,
            'rhs_count': lanes, 'randomness_coordinates': lanes * (l1 + width + l2),
            'message_coordinates': lanes * width, 'tail_coordinates': lanes * width,
            'header_coordinates': lanes * l1, 'output_coordinates': lanes * (l1 + width)}


def _commitment_physical_operations(width, l1, lanes, fused):
    return {'message_upload_bytes': 4 * lanes * width if fused else 0,
            'gpu_tail_coordinates': lanes * width if fused else 0,
            'cpu_tail_coordinates': 0 if fused else lanes * width,
            'lower_int64_intermediate_bytes': 0 if fused else 8 * lanes * width,
            'cpu_header_coordinates': lanes * l1,
            'final_output_bytes': 8 * lanes * (l1 + width)}


def _complete_phase_coverage(phase, sample):
    width, count, l1, l2 = (sample[key] for key in
                           ('sample_witness_width', 'sample_constraint_count', 'l1', 'l2'))
    if phase == 'p_first_phase_layout':
        return {'complete_B1_pair': _complete_commitment_coverage(width, l1, l2, 2),
                'yang_gaussian_small': count + l1 + l2,
                'yang_gaussian_large': count + l1 + l2, 'quadratic_masks_reused_layout': count}
    if phase == 'p_second_phase':
        return {'complete_B2_pair': _complete_commitment_coverage(count, l1, l2, 2),
                'F_streamed_float64_field': sample['n'] * sample['sample_F_columns']}
    if phase == 'v_equation_phase':
        return {'complete_B1_single': _complete_commitment_coverage(width, l1, l2, 1),
                'F_streamed_float64_field': sample['n'] * sample['sample_F_columns'],
                'quadratic_response': count, 'challenge_equal_add': sample['n'] + l1 + width}
    raise ValueError('Unknown commitment phase')


def _calibrate_complete_commitments(params, settings, dims, public_matrix, *, fixture_scale=1.0, repeats=10, warmups=2):
    q, l1, l2 = params.q, settings.l1, settings.l2
    if q != 524309 or not 0 < l2 <= metal_matrix._BLOCK:
        return {}
    sample_scale = float(np.interp(fixture_scale, (0.125, 0.25, 0.5, 1.0), (0.375, 0.5, 0.75, 1.0)))
    width = round(48 * params.m * sample_scale)
    count = round(width * dims['quadratic_constraints'] / dims['witness_coordinates'])
    if not width < count < 3 * width // 2:
        raise ValueError('Commitment constraint layout mismatch')
    entries = l1 * (count + l2) + count * l2
    peak = 12 * entries + 160 * (width + count) + 512 * 1024**2
    require_memory(peak, min(params.max_memory_bytes, 4 * 1024**3),
                   'commitment calibration workspace')
    backend = metal_matrix._backend()
    if backend is None or 4 * max(l1 * (count + l2), count * l2) > backend.max_buffer_length:
        raise MemoryError('Commitment fixture too large')

    class LegacyKey(LatticeCommitmentKey):
        pass

    source = CryptographicRNG(bytes.fromhex('57' * 32))
    extra = count - width
    triples = _immutable_multiplication_triples(
        width, ((extra, width - extra, 0, extra, True, params.m * params.k),), np.uint32)
    x = np.ones(width, dtype=np.int64)
    with uniform_source(source) as stream:
        r = uniform_field(stream, (width,), q)
    a, b = _quadratic_masks(triples, r, x, q)
    alpha = -settings.p
    z0 = _affine_canonical_mod(x, alpha, r, q)
    d = _quadratic_response(triples, z0, alpha, q)
    del triples
    binding = _commitment_implementation()
    kernels = {}
    for role, size, messages, response_message, subtract in (
            ('B1', width, (x, r), z0, False), ('B2', count, (a, b), d, True)):
        shapes = ((l1, size + l2), (size, l2))
        key = LatticeCommitmentKey(q, l1, l2, size,
                                  _MatrixFixture(public_matrix, shapes[0]),
                                  _MatrixFixture(public_matrix.T, shapes[1]))
        if not metal_matrix._can_fuse_commitment(key.top, key.bottom, q):
            raise AssertionError('Fused commitment required')
        legacy = object.__new__(LegacyKey)
        for field in ('q', 'l1', 'l2', 'width', 'top', 'bottom'):
            object.__setattr__(legacy, field, getattr(key, field))
        s1 = sample_standard_discrete_gaussian(source, key.randomness_size,
                                              settings.sigma1, settings.gaussian_tail)
        s2 = sample_standard_discrete_gaussian(source, key.randomness_size,
                                              dims['yang_sigma2'], settings.gaussian_tail)
        response, _ = _response_opening(s1, s2, alpha, subtract=subtract)
        for kind, lanes in (('pair', 2), ('single', 1)):
            def invoke(fused):
                implementation = key if fused else legacy
                with metal_matrix._uncached_workspaces():
                    if lanes == 2:
                        return implementation.commit_pair((messages[0], s1), (messages[1], s2))
                    return (implementation.commit(response_message, response),)

            reference = invoke(False)
            if lanes == 2:
                pair_reference = reference
            elif not np.array_equal(reference[0], _affine_canonical_mod(
                    pair_reference[0], alpha, pair_reference[1], q, subtract=subtract)):
                raise AssertionError('Commitment homomorphism failed')
            def check(actual):
                if (len(actual) != len(reference)
                        or not all(np.array_equal(left, right) for left, right in zip(reference, actual))):
                    raise AssertionError('Fused commitment mismatch')
            check(invoke(True))
            for _ in range(warmups):
                for fused in (True, False):
                    check(invoke(fused))
            samples = {False: [], True: []}
            for iteration in range(repeats):
                for fused in ((False, True) if iteration % 2 == 0 else (True, False)):
                    started = time.perf_counter_ns()
                    actual = invoke(fused)
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    check(actual)
                    samples[fused].append(elapsed)
                    del actual
            for fused, suffix in ((False, 'old'), (True, 'new')):
                name = 'complete_' + role + '_' + kind + '_' + suffix
                kernels[name] = {
                    'kernel': name, 'units_per_sample': 1, 'unit': 'commitment_' + kind,
                    'rate_kind': 'measured_complete_commitment_ab', 'measured_this_run': True,
                    **binding, 'commitment_variant': 'fused_gpu_tail' if fused else 'original_cpu_tail',
                    'sample_witness_width': width, 'sample_constraint_count': count,
                    'sample_fixture_scale': sample_scale,
                    'sample_commitment_width': size, 'commitment_role': role, 'rhs_count': lanes,
                    'q': q, 'n': params.n, 'l1': l1, 'l2': l2,
                    'sigma1': settings.sigma1, 'sigma2': dims['yang_sigma2'],
                    'gaussian_tail': settings.gaussian_tail, 'response_alpha': alpha,
                    'calibration_reference_depth': params.depth,
                    'reference_randomness_length': dims['yang_randomness_length'],
                    'matrix_shapes': [list(shape) for shape in shapes], 'matrix_storage_dtype': 'uint32',
                    'covered_components': _complete_commitment_coverage(size, l1, l2, lanes),
                    'physical_operations': _commitment_physical_operations(size, l1, lanes, fused),
                    'workspace_peak_bound_bytes': peak, 'workspace_budget_bytes': 4 * 1024**3,
                    'warmup_samples': warmups, 'same_prepared_key_and_inputs': True,
                    'workspace_cache': 'disabled_inside_complete_call',
                    **_summary(samples[fused])}
            del reference
        del key, legacy, s1, s2, response, pair_reference
    return kernels


def _calibrate_yang_gaussians(params, settings, dims, *, fixture_scale=1.0, repeats=10, warmups=2):
    sample_scale = float(np.interp(fixture_scale, (0.125, 0.25, 0.5, 1.0), (0.25, 0.375, 0.625, 1.0)))
    count = round(4194304 * sample_scale)
    require_memory(64 * count + 256 * 1024**2, params.max_memory_bytes,
                   'Gaussian calibration workspace')
    rng = CryptographicRNG()
    kernels = {}
    for kind, sigma in (('small', settings.sigma1), ('large', dims['yang_sigma2'])):
        samples = []
        for iteration in range(warmups + repeats):
            started = time.perf_counter_ns()
            actual = sample_standard_discrete_gaussian(rng, count, sigma, settings.gaussian_tail)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if (actual.shape != (count,) or actual.dtype != np.dtype(np.int64)
                    or np.any(actual < -math.ceil(settings.gaussian_tail * sigma))
                    or np.any(actual > math.ceil(settings.gaussian_tail * sigma))):
                raise AssertionError('Gaussian support mismatch')
            if iteration >= warmups:
                samples.append(elapsed)
            del actual
        name = 'yang_gaussian_' + kind
        kernels[name] = {'kernel': name, 'units_per_sample': count, 'unit': 'entries',
                         'rate_kind': 'measured_current_cryptographic_gaussian', 'measured_this_run': True,
                         'sigma': sigma, 'tail': settings.gaussian_tail,
                         'sample_fixture_scale': sample_scale,
                         'sample_gaussian_parallel_eligible': count >= 1048576,
                         'sampler': ('cached_finite_alias' if math.ceil(settings.gaussian_tail * sigma) <= 4096
                                     else 'batched_rejection'),
                         'randomness': uniform_backend(), 'warmup_samples': warmups, **_summary(samples)}
    return kernels


def _prover_layout_calibration(kernels):
    names = ('quadratic_witness_layout', 'quadratic_masks_reused_layout',
             'p_witness_context_layout_serial', 'p_witness_context_layout_joint',
             'p_first_phase_layout_serial', 'p_first_phase_layout_joint')
    if not all(name in kernels and kernels[name].get('constraint_layout_path') == 'same_call_immutable_plan'
               for name in names):
        return False
    return (all(kernels[name].get('layout_plan_creation_included') is True
                for name in ('quadratic_witness_layout', 'p_witness_context_layout_serial',
                             'p_witness_context_layout_joint'))
            and all(kernels[name].get('layout_plan_creation_included') is False
                    and kernels[name].get('layout_plan_validation_included') is True
                    for name in ('quadratic_masks_reused_layout', 'p_first_phase_layout_serial',
                                 'p_first_phase_layout_joint')))


def _complete_commitment_calibration(kernels):
    complete = tuple('complete_' + role + '_' + kind + '_' + variant
                     for role in ('B1', 'B2') for kind in ('pair', 'single') for variant in ('old', 'new'))
    phases = tuple(phase + '_' + mode for phase, modes in (
        ('p_first_phase_layout', ('serial', 'joint')), ('p_second_phase', ('serial', 'joint')),
        ('v_equation_phase', ('serial', 'joint', 'previous'))) for mode in modes)
    if not any(name.startswith('complete_') or sample.get('commitment_impl_id') is not None
               for name, sample in kernels.items()):
        return False
    if not all(name in kernels for name in complete + phases) or not _prover_layout_calibration(kernels):
        raise ValueError('Incomplete commitment calibration')
    binding = _commitment_implementation()
    first = kernels[complete[0]]
    parameters = ('q', 'n', 'l1', 'l2', 'sigma1', 'sigma2', 'gaussian_tail')
    if first.get('q') != 524309 or not 0 < first.get('l2', 0) <= metal_matrix._BLOCK:
        raise ValueError('Unsupported commitment calibration')
    for name in complete + phases:
        sample = kernels[name]
        values = sample.get('samples_ms', ())
        if (any(sample.get(key) != value for key, value in binding.items())
                or any(sample.get(key) != first.get(key) for key in parameters)
                or sample.get('measured_this_run') is not True or sample.get('warmup_samples') != 2
                or len(values) != 10 or any(not math.isfinite(value) or value <= 0 for value in values)
                or not math.isclose(sample.get('mean_ms', 0), statistics.mean(values), rel_tol=1e-12)
                or sample.get('empirical_min_ms') != min(values)
                or sample.get('empirical_max_ms') != max(values)):
            raise ValueError('Commitment calibration mismatch')
        if name in complete:
            role, kind, variant = name.split('_')[1:]
            width = sample.get('sample_witness_width') if role == 'B1' else sample.get('sample_constraint_count')
            lanes = 2 if kind == 'pair' else 1
            if (not isinstance(width, int) or width <= 0 or sample.get('sample_commitment_width') != width
                    or sample.get('matrix_storage_dtype') != 'uint32'
                    or sample.get('commitment_role') != role or sample.get('rhs_count') != lanes
                    or any(sample.get(key) != first.get(key) for key in
                           ('response_alpha', 'calibration_reference_depth', 'reference_randomness_length'))
                    or sample.get('same_prepared_key_and_inputs') is not True
                    or sample.get('workspace_cache') != 'disabled_inside_complete_call'
                    or sample.get('commitment_variant') != ('fused_gpu_tail' if variant == 'new' else 'original_cpu_tail')
                    or sample.get('covered_components') != _complete_commitment_coverage(width, first['l1'], first['l2'], lanes)
                    or sample.get('physical_operations') != _commitment_physical_operations(width, first['l1'], lanes, variant == 'new')
                    or sample.get('matrix_shapes') != [[first['l1'], width + first['l2']], [width, first['l2']]]):
                raise ValueError('Incomplete commitment coverage')
            counterpart = kernels[name.rsplit('_', 1)[0] + ('_old' if variant == 'new' else '_new')]
            if any(sample.get(key) != counterpart.get(key) for key in (
                    'sample_witness_width', 'sample_constraint_count', 'sample_commitment_width',
                    'response_alpha', 'covered_components', 'matrix_shapes')):
                raise ValueError('Commitment fixtures differ')
        else:
            phase = name.rsplit('_', 1)[0]
            if sample.get('covered_components') != _complete_phase_coverage(phase, sample):
                raise ValueError('Phase commitment coverage mismatch')
            counterpart = kernels[phase + '_serial']
            if any(sample.get(key) != counterpart.get(key) for key in
                   ('sample_witness_width', 'sample_constraint_count', 'sample_F_columns',
                    'sample_relation_rows', 'matrix_shapes', 'covered_components')):
                raise ValueError('Serial/joint fixture mismatch')
            if phase == 'v_equation_phase' and sample.get('equation_comparison') != 'fused_canonical_equal':
                raise ValueError('Missing equality calibration')
    for name, sigma in (('yang_gaussian_small', first['sigma1']), ('yang_gaussian_large', first['sigma2'])):
        sample = kernels.get(name, {})
        if (sample.get('warmup_samples') != 2 or len(sample.get('samples_ms', ())) != 10
                or sample.get('sigma') != sigma or sample.get('tail') != first['gaussian_tail']):
            raise ValueError('Gaussian calibration mismatch')
    if (any(kernels.get(name, {}).get('matrix_storage_dtype') != 'uint32'
            for name in ('yang_joint_pair', 'yang_joint_yang_large'))
            or not all(name in kernels for name in
                       ('F_streamed_float64_field', 'challenge_equal_add', 'challenge_equal_sub'))):
        raise ValueError('Matrix or equality calibration mismatch')
    return True


def _multiscale_stage(sample, units, term):
    fit = sample.get('scaling_fit')
    if fit is None:
        return term
    from .multiscale import predict_kernel

    prediction = predict_kernel(sample, units)
    term.update({key: prediction[key] for key in (
        'estimated_mean_ms', 'empirical_scaled_min_ms', 'empirical_scaled_max_ms',
        'extrapolation_ratio', 'model_kind', 'uncertainty_kind', 'fit_diagnostics')})
    term.update(
        calibrated_kernel_estimate_ms=prediction['estimated_mean_ms'],
        extrapolated_work_units=units,
        scaling_fit=fit,
        rate_kind='measured_multiscale_stage_extrapolation',
        formula='affine(all_scales,same_source)',
        ratio_diagnostic_only=True,
        covered_components_additive=False,
        reference_components_additive=False,
    )
    return term


def _apply_complete_commitment_ratios(terms, kernels, width, count, l1, l2, *, pair):
    if not _complete_commitment_calibration(kernels):
        return terms
    lanes, kind = (2, 'pair') if pair else (1, 'single')
    kept = list(terms)
    for role, size in (('B1', width), ('B2', count)):
        covered = [term for term in kept if term['label'].startswith((role + '.', role + ' '))]
        kept = [term for term in kept if not term['label'].startswith((role + '.', role + ' '))]
        expected = {'yang_joint_pair' if pair else 'yang_joint_yang_large': l1 * (size + l2) + size * l2,
                    'commitment_add_mod': lanes * l1, 'commitment_tail': lanes * size,
                    'vector_copy': lanes * l1}
        observed = {}
        for term in covered:
            observed[term['kernel']] = observed.get(term['kernel'], 0) + term['target_units']
        if observed != expected:
            raise ValueError('Commitment coverage mismatch')
        old_name, new_name = ('complete_' + role + '_' + kind + '_' + suffix for suffix in ('old', 'new'))
        old, new = kernels[old_name], kernels[new_name]
        ratio = new['mean_ms'] / old['mean_ms']
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError('Invalid commitment timing ratio')
        base = sum(term['estimated_mean_ms'] for term in covered)
        kept.append(_multiscale_stage(new, size, {'label': role + ' complete ' + kind + ' commitment', 'kernel': new_name,
                     'target_units': 1, 'unit': 'commitment_' + kind, 'sample_multiplier': 1,
                     'formula': 'sum(components)*T_new/T_old',
                     'estimated_mean_ms': base * ratio, 'calibrated_kernel_estimate_ms': base * ratio,
                     'matrix_input_scan_projection_ms': 0.0,
                     'empirical_scaled_min_ms': sum(term['empirical_scaled_min_ms'] for term in covered)
                         * new['empirical_min_ms'] / old['empirical_max_ms'],
                     'empirical_scaled_max_ms': sum(term['empirical_scaled_max_ms'] for term in covered)
                         * new['empirical_max_ms'] / old['empirical_min_ms'],
                     'rate_kind': 'measured_complete_commitment_ratio_transfer',
                     **{key: new[key] for key in ('commitment_impl_id', 'commitment_impl_sha256')},
                     'complete_new_to_old_ratio': ratio, 'new_faster': ratio < 1,
                     'old_samples_ms': list(old['samples_ms']), 'new_samples_ms': list(new['samples_ms']),
                     'target_coverage': _complete_commitment_coverage(size, l1, l2, lanes),
                     'physical_operations': _commitment_physical_operations(size, l1, lanes, True),
                     'reference_estimate_ms': base, 'reference_components': covered,
                     'reference_components_additive': False}))
    return kept


def _calibrate_joint_phases(params, settings, dims, public_matrix, *, fixture_scale=1.0, repeats=10, warmups=2):
    from concurrent.futures import ThreadPoolExecutor

    sample_scale = 1 + math.log2(fixture_scale) / 8
    width = round(128 * params.m * sample_scale)
    count = round(width * dims['quadratic_constraints'] / dims['witness_coordinates'])
    l1, l2 = settings.l1, settings.l2
    top_shape, bottom_shape = (l1, width + l2), (width, l2)
    top_entries, bottom_entries = math.prod(top_shape), math.prod(bottom_shape)
    construction_peak = 4 * (top_entries + 2 * bottom_entries) + 512 * 1024**2
    application_peak = 4 * (top_entries + bottom_entries) + 160 * (width + count) + 512 * 1024**2
    peak = max(construction_peak, application_peak)
    require_memory(peak, min(params.max_memory_bytes, 4 * 1024**3),
                   'joint-phase workspace')
    backend = metal_matrix._backend()
    if backend is None or max(4 * top_entries, 4 * bottom_entries) > backend.max_buffer_length:
        raise MemoryError('Joint fixture too large')
    if width < 131072 or count + l1 + l2 < 1048576 or not width < count < 3 * width // 2:
        raise ValueError('Invalid parallel fixture')

    key = LatticeCommitmentKey(params.q, l1, l2, width,
                              _MatrixFixture(public_matrix, top_shape),
                              _MatrixFixture(public_matrix.T, bottom_shape))

    binding = (_commitment_implementation()
               if metal_matrix._can_fuse_commitment(key.top, key.bottom, params.q) else {})

    class UncachedKey:
        q = params.q

        def commit(self, message, randomness):
            with metal_matrix._uncached_workspaces():
                return key.commit(message, randomness)

        def commit_pair(self, first, second):
            with metal_matrix._uncached_workspaces():
                return key.commit_pair(first, second)

    uncached_key = UncachedKey()
    extra = count - width
    triples = _immutable_multiplication_triples(
        width, ((extra, width - extra, 0, extra, True, params.m * params.k),), np.uint32)
    rng = CryptographicRNG(bytes.fromhex('19' * 32))
    x = np.ones(width, dtype=np.int64)
    valid, layout_plan = _quadratic_witness_layout(triples, x, params.q)
    if not valid or layout_plan is None:
        raise AssertionError('Validated P layout required')
    with uniform_source(rng) as source:
        r = uniform_field(source, (width,), params.q)
    s1 = sample_standard_discrete_gaussian(rng, key.randomness_size, settings.sigma1, settings.gaussian_tail)
    s2 = sample_standard_discrete_gaussian(rng, key.randomness_size, dims['yang_sigma2'], settings.gaussian_tail)
    kernels = {}
    def invoke(parallel, source):
        def apply(executor):
            return _prover_first_phase(uncached_key, triples, x, r, s1, s2, source,
                                       count + l1 + l2, settings.sigma1,
                                       dims['yang_sigma2'], settings.gaussian_tail, executor=executor,
                                       _layout_plan=layout_plan)
        if parallel:
            with ThreadPoolExecutor(max_workers=1) as pool:
                return apply(pool)
        return apply(None)

    reference_source = CryptographicRNG(bytes.fromhex('27' * 32))
    reference = invoke(False, reference_source)
    reference_next = reference_source._random_bytes(64)
    actual_source = CryptographicRNG(bytes.fromhex('27' * 32))
    actual = invoke(True, actual_source)
    if (len(reference) != len(actual)
            or not all(np.array_equal(a, b) for a, b in zip(reference, actual))
            or actual_source._random_bytes(64) != reference_next):
        raise AssertionError('Phase output/RNG mismatch')
    del actual
    for _ in range(warmups):
        for parallel in (True, False):
            source = CryptographicRNG(bytes.fromhex('27' * 32))
            actual = invoke(parallel, source)
            if (len(reference) != len(actual)
                    or not all(np.array_equal(a, b) for a, b in zip(reference, actual))
                    or source._random_bytes(64) != reference_next):
                raise AssertionError('Joint warmup mismatch')
            del actual
    samples = {False: [], True: []}
    for iteration in range(repeats):
        for parallel in ((False, True) if iteration % 2 == 0 else (True, False)):
            source = CryptographicRNG(bytes.fromhex('27' * 32))
            start = time.perf_counter_ns()
            actual = invoke(parallel, source)
            samples[parallel].append((time.perf_counter_ns() - start) / 1e6)
            if (len(reference) != len(actual)
                    or not all(np.array_equal(a, b) for a, b in zip(reference, actual))
                    or source._random_bytes(64) != reference_next):
                raise AssertionError('Joint output/RNG mismatch')
            del actual
    del reference
    coverage = {'B1_commit_pair': {'matrix_entries': top_entries + bottom_entries, 'rhs_count': 2},
                'commitment_add_mod': 2 * l1, 'commitment_tail': 2 * width,
                'vector_copy': 2 * l1, 'yang_gaussian_small': count + l1 + l2,
                'yang_gaussian_large': count + l1 + l2, 'quadratic_masks_reused_layout': count}
    if binding:
        coverage = _complete_phase_coverage('p_first_phase_layout', {
            'sample_witness_width': width, 'sample_constraint_count': count,
                    'sample_fixture_scale': sample_scale, 'l1': l1, 'l2': l2})
    ratio = statistics.mean(samples[True]) / statistics.mean(samples[False])
    for parallel, suffix in ((False, 'serial'), (True, 'joint')):
        phase = 'p_first_phase_layout'
        kernel = phase + '_' + suffix
        kernels[kernel] = {
            'kernel': kernel, 'units_per_sample': 1, 'unit': 'stage',
            'rate_kind': 'measured_existing_joint_phase_ab', 'measured_this_run': True, **binding,
            'sample_witness_width': width, 'sample_constraint_count': count,
                    'sample_fixture_scale': sample_scale,
            'q': params.q, 'n': params.n, 'l1': l1, 'l2': l2,
            'sigma1': settings.sigma1, 'sigma2': dims['yang_sigma2'],
            'gaussian_tail': settings.gaussian_tail,
            'sample_constraint_ratio': count / width,
            'target_constraint_ratio': dims['quadratic_constraints'] / dims['witness_coordinates'],
            'workspace_peak_bound_bytes': peak, 'workspace_budget_bytes': 4 * 1024**3,
            'matrix_shapes': [list(top_shape), list(bottom_shape)],
            'matrix_storage_dtype': 'uint32',
            'covered_components': coverage, 'joint_to_serial_ratio': ratio,
            'serial_kernel': phase + '_serial', 'joint_kernel': phase + '_joint',
            'joint_faster': ratio < 1, 'warmup_samples': warmups,
            'randomness': uniform_backend(),
            'sample_gaussian_parallel_eligible': count + l1 + l2 >= 1048576,
            'constraint_layout_path': 'same_call_immutable_plan',
            'layout_plan_creation_included': False,
            'layout_plan_validation_included': True,
            **_summary(samples[parallel])}
    return kernels


def _calibrate_second_phase(params, settings, dims, public_matrix, *, fixture_scale=1.0, repeats=10, warmups=2):
    from concurrent.futures import ThreadPoolExecutor

    sample_scale = float(np.interp(fixture_scale, (0.125, 0.25, 0.5, 1.0), (0.375, 0.5, 0.75, 1.0)))
    width = round(48 * params.m * sample_scale)
    count = round(width * dims['quadratic_constraints'] / dims['witness_coordinates'])
    f_columns = round(width * dims['F_columns'] / dims['witness_coordinates'])
    l1, l2, q = settings.l1, settings.l2, params.q
    shapes = ((l1, count + l2), (count, l2))
    entries = [math.prod(shape) for shape in shapes]
    peak = (4 * sum(entries) + 8 * max(entries) + 8 * params.n * f_columns
            + 160 * (width + count) + 512 * 1024**2)
    require_memory(peak, min(params.max_memory_bytes, 4 * 1024**3),
                   'B2/F calibration workspace')
    backend = metal_matrix._backend()
    if backend is None or 4 * max(entries) > backend.max_buffer_length:
        raise MemoryError('Equation fixture too large')

    class UncachedKey:
        def __init__(self, size, top, bottom):
            self.key = LatticeCommitmentKey(q, l1, l2, size,
                                           _MatrixFixture(public_matrix, top),
                                           _MatrixFixture(public_matrix.T, bottom))

        def commit_pair(self, first, second):
            with metal_matrix._uncached_workspaces():
                return self.key.commit_pair(first, second)

    B2 = UncachedKey(count, *shapes)
    binding = (_commitment_implementation()
               if metal_matrix._can_fuse_commitment(B2.key.top, B2.key.bottom, q) else {})
    matrix = _MatrixFixture(public_matrix, (params.n, f_columns)).__array__(np.float64)
    matrix.flags.writeable = False

    class LinearFixture:
        def __matmul__(self, vector):
            return _matmul_mod_prepared_left(matrix, vector[:f_columns], q)

    M = LinearFixture()
    source = CryptographicRNG(bytes.fromhex('37' * 32))
    extra = count - width
    triples = _immutable_multiplication_triples(
        width, ((extra, width - extra, 0, extra, True, params.m * params.k),), np.uint32)
    with uniform_source(source) as stream:
        r = uniform_field(stream, (width,), q)
    r = np.frombuffer(r.tobytes(), dtype=np.int64)
    a, b = _quadratic_masks(triples, r, np.ones(width, dtype=np.int64), q)
    s3 = sample_standard_discrete_gaussian(source, l1 + count + l2,
                                          settings.sigma1, settings.gaussian_tail)
    s4 = sample_standard_discrete_gaussian(source, l1 + count + l2,
                                          dims['yang_sigma2'], settings.gaussian_tail)
    kernels = {}
    phase = 'p_second_phase'
    def invoke(parallel):
        if parallel:
            with ThreadPoolExecutor(max_workers=1) as pool:
                return _prover_second_phase(B2, M, r, a, b, s3, s4, executor=pool)
        return _prover_second_phase(B2, M, r, a, b, s3, s4)

    reference = invoke(False)
    def check(actual):
        if not all(np.array_equal(left, right) for left, right in zip(reference, actual)):
            raise AssertionError('P second-phase mismatch')
    for _ in range(warmups):
        check(invoke(False))
        check(invoke(True))
    samples = {False: [], True: []}
    for iteration in range(repeats):
        for parallel in ((False, True) if iteration % 2 == 0 else (True, False)):
            started = time.perf_counter_ns()
            actual = invoke(parallel)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            check(actual)
            samples[parallel].append(elapsed)
            del actual
    for parallel, suffix in ((False, 'serial'), (True, 'joint')):
        name = phase + '_' + suffix
        kernels[name] = {
            'kernel': name, 'units_per_sample': 1, 'unit': 'stage',
            'rate_kind': 'measured_commitment_F_overlap', 'measured_this_run': True, **binding,
            'sample_witness_width': width, 'sample_constraint_count': count,
                    'sample_fixture_scale': sample_scale,
            'sample_F_columns': f_columns, 'q': q, 'n': params.n, 'l1': l1, 'l2': l2,
            'sigma1': settings.sigma1, 'sigma2': dims['yang_sigma2'],
            'gaussian_tail': settings.gaussian_tail,
            **({'covered_components': _complete_phase_coverage(phase, {
                'sample_witness_width': width, 'sample_constraint_count': count,
                    'sample_fixture_scale': sample_scale,
                'l1': l1, 'l2': l2, 'n': params.n, 'sample_F_columns': f_columns})} if binding else {}),
            'matrix_shapes': [list(shape) for shape in shapes],
            'workspace_peak_bound_bytes': peak, 'workspace_budget_bytes': 4 * 1024**3,
            'warmup_samples': warmups,
            **_summary(samples[parallel])}
    return kernels


def _calibrate_verifier_equations(params, settings, dims, public_matrix, *, fixture_scale=1.0, repeats=10, warmups=2):
    from concurrent.futures import ThreadPoolExecutor

    sample_scale = float(np.interp(fixture_scale, (0.125, 0.25, 0.5, 1.0), (0.375, 0.5, 0.75, 1.0)))
    width = round(48 * params.m * sample_scale)
    count = round(width * dims['quadratic_constraints'] / dims['witness_coordinates'])
    f_columns = round(width * dims['F_columns'] / dims['witness_coordinates'])
    l1, l2, q = settings.l1, settings.l2, params.q
    shapes = ((l1, width + l2), (width, l2))
    entries = [math.prod(shape) for shape in shapes]
    peak = (4 * sum(entries) + 8 * max(entries) + 8 * params.n * f_columns
            + 160 * (width + count) + 512 * 1024**2)
    require_memory(peak, min(params.max_memory_bytes, 4 * 1024**3),
                   'verifier calibration workspace')
    backend = metal_matrix._backend()
    if backend is None or 4 * max(entries) > backend.max_buffer_length:
        raise MemoryError('Verifier fixture too large')
    if width < 131072 or not width < count < 3 * width // 2 or not 0 < f_columns <= width:
        raise ValueError('Invalid verifier fixture')

    key = LatticeCommitmentKey(q, l1, l2, width,
                              _MatrixFixture(public_matrix, shapes[0]),
                              _MatrixFixture(public_matrix.T, shapes[1]))

    binding = (_commitment_implementation()
               if metal_matrix._can_fuse_commitment(key.top, key.bottom, q) else {})

    class UncachedKey:
        def __init__(self, key):
            self.q, self.key = key.q, key

        def commit(self, message, randomness):
            with metal_matrix._uncached_workspaces():
                return self.key.commit(message, randomness)

    B1 = UncachedKey(key)
    matrix = _MatrixFixture(public_matrix, (params.n, f_columns)).__array__(np.float64)
    matrix.flags.writeable = False

    class LinearFixture:
        def __matmul__(self, vector):
            return _matmul_mod_prepared_left(matrix, vector[:f_columns], q)

    M = LinearFixture()
    source = CryptographicRNG(bytes.fromhex('47' * 32))
    extra = count - width
    triples = _immutable_multiplication_triples(
        width, ((extra, width - extra, 0, extra, True, params.m * params.k),), np.uint32)
    with uniform_source(source) as stream:
        z0 = uniform_field(stream, (width,), q)
    z0 = np.frombuffer(z0.tobytes(), dtype=np.int64)
    alpha = -settings.p
    s1 = sample_standard_discrete_gaussian(source, key.randomness_size,
                                          settings.sigma1, settings.gaussian_tail)
    s2 = sample_standard_discrete_gaussian(source, key.randomness_size,
                                          dims['yang_sigma2'], settings.gaussian_tail)
    z1 = np.frombuffer((s2 + alpha * s1).tobytes(), dtype=np.int64)
    del s1, s2
    y = np.frombuffer(bytes(params.n * 8), dtype=np.int64)
    t = np.frombuffer((M @ z0).tobytes(), dtype=np.int64)
    c1 = np.frombuffer(bytes(key.commitment_size * 8), dtype=np.int64)
    c2 = np.frombuffer(B1.commit(z0, z1).tobytes(), dtype=np.int64)
    args = (B1, M, triples, z0, z1, alpha, y, t, c1, c2)

    def invoke(mode):
        if mode == 'serial':
            return _verifier_equation_phase(*args)
        if mode == 'joint':
            with ThreadPoolExecutor(max_workers=1) as pool:
                return _verifier_equation_phase(*args, executor=pool)
        if mode != 'previous':
            raise ValueError('Unknown verifier mode')
        if not _affine_canonical_equal(M @ z0, y, alpha, t, q):
            return None
        with ThreadPoolExecutor(max_workers=1) as pool:
            first, d = _verifier_first_phase(B1, triples, z0, z1, alpha, executor=pool)
        return d if _affine_canonical_equal(first, c1, alpha, c2, q) else None

    reference = _quadratic_response(triples, z0, alpha, q)
    def check(actual):
        if actual is None or not np.array_equal(actual, reference):
            raise AssertionError('Verifier equation mismatch')
    modes = ('previous', 'joint', 'serial')
    for _ in range(warmups):
        for mode in modes:
            check(invoke(mode))
    samples = {mode: [] for mode in modes}
    for iteration in range(repeats):
        order = modes[iteration % len(modes):] + modes[:iteration % len(modes)]
        if iteration % 2:
            order = tuple(reversed(order))
        for mode in order:
            started = time.perf_counter_ns()
            actual = invoke(mode)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            check(actual)
            samples[mode].append(elapsed)
            del actual
    coverage = {'F_streamed_float64_field': params.n * f_columns,
                'B1_commit': {'matrix_entries': sum(entries), 'rhs_count': 1},
                'commitment_add_mod': l1, 'commitment_tail': width, 'vector_copy': l1,
                'quadratic_response': count, 'challenge_equal_add': params.n + l1 + width}
    if binding:
        coverage = _complete_phase_coverage('v_equation_phase', {
            'sample_witness_width': width, 'sample_constraint_count': count,
                    'sample_fixture_scale': sample_scale,
            'l1': l1, 'l2': l2, 'n': params.n, 'sample_F_columns': f_columns})
    return {'v_equation_phase_' + mode: {
        'kernel': 'v_equation_phase_' + mode, 'units_per_sample': 1, 'unit': 'stage',
        'rate_kind': 'measured_verifier_equation_schedule', 'measured_this_run': True, **binding,
        'sample_witness_width': width, 'sample_constraint_count': count,
                    'sample_fixture_scale': sample_scale,
        'sample_F_columns': f_columns, 'sample_relation_rows': params.n,
        'sample_native_schedule_eligible': width >= 131072,
        'q': q, 'n': params.n, 'l1': l1, 'l2': l2,
        'sigma1': settings.sigma1, 'sigma2': dims['yang_sigma2'],
        'gaussian_tail': settings.gaussian_tail,
        'sample_constraint_ratio': count / width,
        'target_constraint_ratio': dims['quadratic_constraints'] / dims['witness_coordinates'],
        'sample_F_to_witness_ratio': f_columns / width,
        'target_F_to_witness_ratio': dims['F_columns'] / dims['witness_coordinates'],
        'matrix_shapes': [list(shape) for shape in shapes],
        'covered_components': coverage, 'equation_comparison': 'fused_canonical_equal',
        'workspace_peak_bound_bytes': peak, 'workspace_budget_bytes': 4 * 1024**3,
        'warmup_samples': warmups,
        'joint_to_serial_ratio': statistics.mean(samples['joint']) / statistics.mean(samples['serial']),
        'joint_to_previous_ratio': statistics.mean(samples['joint']) / statistics.mean(samples['previous']),
        **_summary(samples[mode])} for mode in modes}


def _calibrate_witness_context(params, settings, dims, *, layout_reuse=False, fixture_scale=1.0, repeats=10, warmups=2):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    width = round(4194304 * fixture_scale)
    count = round(width * dims['quadratic_constraints'] / dims['witness_coordinates'])
    rows = max(params.n, round(count * dims['relation_rows'] / dims['quadratic_constraints']))
    peak = 48 * (width + count) + 16 * rows + 512 * 1024**2
    require_memory(peak, min(params.max_memory_bytes, 4 * 1024**3),
                   'context calibration workspace')
    extra = count - width
    if not 0 < extra < width // 2:
        raise ValueError('Context layout mismatch')
    triples = _immutable_multiplication_triples(
        width, ((extra, width - extra, 0, extra, True, params.m * params.k),), np.uint32)
    y = np.frombuffer(bytes(rows * 8), dtype=np.int64)

    class EmptyLinearBlock:
        def __matmul__(self, vector):
            return y

    relation = SimpleNamespace(M=EmptyLinearBlock(), y=y, triples=triples,
                               statement_digest='21' * 32)
    pp = SimpleNamespace(digest=bytes.fromhex('43' * 32),
                         witness_key=SimpleNamespace(q=params.q))
    x = np.ones(width, dtype=np.int64)
    def invoke(parallel):
        source = CryptographicRNG(bytes.fromhex('19' * 32))
        if parallel:
            with ThreadPoolExecutor(max_workers=1) as pool:
                return _prover_witness_context_phase(relation, x, pp, b'calibration',
                                                     source, executor=pool, _collect_layout=layout_reuse)
        return _prover_witness_context_phase(relation, x, pp, b'calibration', source,
                                             _collect_layout=layout_reuse)

    def checked(result):
        if layout_reuse:
            context, source, plan = result
            if plan is None:
                raise AssertionError('Immutable context required')
            return context, source
        return result

    reference, reference_source = checked(invoke(False))
    reference_next = reference_source._random_bytes(64)
    actual, actual_source = checked(invoke(True))
    if actual != reference or actual_source._random_bytes(64) != reference_next:
        raise AssertionError('Context digest/RNG mismatch')
    for _ in range(warmups):
        for parallel in (False, True):
            result, source = checked(invoke(parallel))
            if result != reference or source._random_bytes(64) != reference_next:
                raise AssertionError('Context warmup mismatch')
    samples = {False: [], True: []}
    for iteration in range(repeats):
        for parallel in ((False, True) if iteration % 2 == 0 else (True, False)):
            start = time.perf_counter_ns()
            result = invoke(parallel)
            samples[parallel].append((time.perf_counter_ns() - start) / 1e6)
            result, source = checked(result)
            if result != reference or source._random_bytes(64) != reference_next:
                raise AssertionError('Context digest/RNG mismatch')
    phase = 'p_witness_context_layout' if layout_reuse else 'p_witness_context'
    witness_kernel = 'quadratic_witness_layout' if layout_reuse else 'quadratic_witness_check'
    return {phase + '_' + suffix: {
        'kernel': phase + '_' + suffix, 'units_per_sample': 1, 'unit': 'stage',
        'rate_kind': 'measured_hash_constraint_overlap', 'measured_this_run': True,
        'sample_witness_width': width, 'sample_constraint_count': count,
        'sample_relation_rows': rows,
        'sample_constraint_ratio': count / width,
        'target_constraint_ratio': dims['quadratic_constraints'] / dims['witness_coordinates'],
        'q': params.q, 'workspace_peak_bound_bytes': peak,
        'covered_components': {'canonical_index_hash': 3 * count,
                               'hash_bytes': 8 * rows,
                               witness_kernel: count},
        'linear_product_included': False, 'warmup_samples': warmups,
        **({'constraint_layout_path': 'same_call_immutable_plan',
            'layout_plan_creation_included': True} if layout_reuse else {}),
        **_summary(samples[parallel])}
        for parallel, suffix in ((False, 'serial'), (True, 'joint'))}


def calibrate_commitment_products(params, settings, public_matrix, *, fixture_scale=1.0,
                                  repeats=10, warmups=2):
    if (type(repeats) is not int or repeats < 1 or type(warmups) is not int or warmups < 0
            or not isinstance(fixture_scale, (int, float)) or not math.isfinite(fixture_scale)
            or not 0 < fixture_scale <= 1 or round(8 * fixture_scale) < 1):
        raise ValueError('Invalid calibration scale/counts')
    fixture_options = dict(fixture_scale=fixture_scale, repeats=repeats, warmups=warmups)
    params.validate()
    settings.validate(params.q)
    if (not isinstance(public_matrix, np.ndarray)
            or public_matrix.shape != (params.n, params.m)
            or public_matrix.dtype != np.dtype(np.int64)
            or not public_matrix.flags.c_contiguous
            or np.any(public_matrix < 0) or np.any(public_matrix >= params.q)):
        raise ValueError('Canonical Arec required')
    stream_bytes = 8 * round(32 * 1024**2 * fixture_scale)
    require_memory(max(stream_bytes + 16*params.n*params.m + 64*params.m,
                       24*8*params.n*params.m + 1024*params.m + 64*1024**2),
                   params.max_memory_bytes, 'commitment kernel calibration workspace')
    stream = np.arange(stream_bytes//8, dtype=np.int64)
    expected_sum = (stream.size-1)*stream.size//2
    if int(np.sum(stream, dtype=np.int64)) != expected_sum:
        raise AssertionError('Stream checksum mismatch')
    for _ in range(warmups):
        if int(np.sum(stream, dtype=np.int64)) != expected_sum:
            raise AssertionError('Stream checksum mismatch')
    read_samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        checksum = np.sum(stream, dtype=np.int64)
        read_samples.append((time.perf_counter_ns()-start)/1e6)
        if int(checksum) != expected_sum:
            raise AssertionError('Stream checksum mismatch')
    del stream
    read_scan_per_entry = 8*statistics.mean(read_samples)/stream_bytes
    integer_wide = public_matrix.view()
    integer_wide.flags.writeable = False
    cpu_tiles, matrix_tiles = round(8 * fixture_scale), round(64 * fixture_scale)
    wide = np.tile(public_matrix.astype(np.float64), (1, cpu_tiles))
    tall = np.ascontiguousarray(wide.T)
    wide.flags.writeable = tall.flags.writeable = False
    use_metal = metal_matrix.available(params.q)
    if use_metal:
        require_memory(20*matrix_tiles*params.n*params.m + 256*1024**2,
                       params.max_memory_bytes, 'Metal commitment calibration workspace')
        expanded = np.concatenate((np.tile(public_matrix.astype(np.uint32), (1, matrix_tiles)),
                                   public_matrix[:, :params.n].astype(np.uint32)), axis=1)
        commitment_wide = metal_matrix.prepare(expanded)
        del expanded
        expanded = np.tile(public_matrix.T.astype(np.uint32), (matrix_tiles, 1))
        commitment_tall = metal_matrix.prepare(expanded)
        del expanded
        device_buffer_limit = commitment_wide._metal_owner.backend.max_buffer_length
    else:
        commitment_wide, commitment_tall = wide, tall
        device_buffer_limit = None
    rng = np.random.default_rng(9177)
    dims = target_dimensions(params, settings)
    kernels = {}

    def measure(name, matrix, right):
        if matrix.shape[1]*(params.q-1)**2 > np.iinfo(np.int64).max:
            raise ValueError('Integer oracle overflow')
        oracle_right = np.column_stack(right) if isinstance(right, tuple) else right
        expected = (matrix.astype(np.int64) @ (oracle_right.astype(np.int64) % params.q)) % params.q
        gpu_matrix = metal_matrix.is_prepared(matrix)
        product = (metal_matrix.matmul if gpu_matrix else _matmul_mod_prepared_left
                   if matrix.dtype == np.dtype(np.float64) else _matmul_mod_canonical_left)
        def invoke():
            if gpu_matrix:
                with metal_matrix._uncached_workspaces():
                    return product(matrix, right, params.q)
            return product(matrix, right, params.q)
        for _ in range(warmups):
            actual = invoke()
            if not np.array_equal(actual, expected):
                raise AssertionError('Commitment product mismatch')
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            actual = invoke()
            samples.append((time.perf_counter_ns()-start)/1e6)
            if not np.array_equal(actual, expected):
                raise AssertionError('Commitment product mismatch')
        kernels[name] = {'kernel': name, 'units_per_sample': matrix.size,
                         'unit': 'matrix_entries', 'matrix_shape': list(matrix.shape),
                         'matrix_storage_dtype': matrix.dtype.name,
                         'rate_kind': ('measured_metal_shared_commitment_shape_proxy' if gpu_matrix
                                       else 'measured_prepared_commitment_shape_proxy' if matrix.dtype.kind=='f'
                                       else 'measured_canonical_left_shape_proxy'),
                         'measured_this_run': True,
                         'reference_scan_ms_per_entry': 0.0 if gpu_matrix else read_scan_per_entry*(2 if name=='yang_top_pair' else 1),
                         'matrix_input_passes': 2 if name=='yang_top_pair' and not gpu_matrix else 1,
                         **_summary(samples)}

    for kind in ('field', 'yang_small', 'yang_large'):
        for orientation, matrix in (('top', commitment_wide), ('bottom', commitment_tall)):
            if kind == 'field':
                right = rng.integers(0, params.q, size=matrix.shape[1], dtype=np.int64)
            else:
                sigma = settings.sigma1 if kind == 'yang_small' else dims['yang_sigma2']
                right = sample_standard_discrete_gaussian(
                    rng, matrix.shape[1], sigma, settings.gaussian_tail)
            measure('yang_'+orientation+'_'+kind, matrix, right)
    measure('auxiliary_binary', integer_wide, rng.integers(0, 2, size=params.m, dtype=np.int64))
    for name,matrix in (('yang_top_pair',commitment_wide),('yang_bottom_pair',commitment_tall)):
        pair = tuple(sample_standard_discrete_gaussian(rng, matrix.shape[1], sigma, settings.gaussian_tail)
                     for sigma in (settings.sigma1, dims['yang_sigma2']))
        measure(name, matrix, pair)
    if use_metal:
        joint_units = commitment_wide.size + commitment_tall.size
        for name, sigmas in (('yang_joint_yang_large', (dims['yang_sigma2'],)),
                             ('yang_joint_pair', (settings.sigma1, dims['yang_sigma2']))):
            operands = tuple(sample_standard_discrete_gaussian(
                rng, commitment_wide.shape[1], sigma, settings.gaussian_tail) for sigma in sigmas)
            top_right = operands[0] if len(operands) == 1 else operands
            bottom_right = (operands[0][-params.n:] if len(operands) == 1
                            else tuple(column[-params.n:] for column in operands))
            jobs = ((commitment_wide, top_right), (commitment_tall, bottom_right))
            expected = []
            for matrix, right in jobs:
                right = np.column_stack(right) if isinstance(right, tuple) else right
                expected.append((matrix.astype(np.int64) @ (right % params.q)) % params.q)
            def invoke_joint():
                with metal_matrix._uncached_workspaces():
                    return metal_matrix.matmul_many(jobs, params.q)
            for _ in range(warmups):
                actual = invoke_joint()
                if not all(np.array_equal(a, b) for a, b in zip(actual, expected)):
                    raise AssertionError('Batched commitment mismatch')
            samples = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                actual = invoke_joint()
                samples.append((time.perf_counter_ns()-start)/1e6)
                if not all(np.array_equal(a, b) for a, b in zip(actual, expected)):
                    raise AssertionError('Batched commitment mismatch')
            kernels[name] = {'kernel': name, 'units_per_sample': joint_units,
                             'unit': 'matrix_entries', 'matrix_shapes': [list(a.shape) for a, _ in jobs],
                             'matrix_storage_dtype': 'uint32', 'rhs_count': len(sigmas),
                             'rate_kind': 'measured_metal_joint_commitment_shape_proxy',
                             'measured_this_run': True, 'reference_scan_ms_per_entry': 0.0,
                             'matrix_input_passes': 1,
                             **_summary(samples)}
        del jobs, expected, actual, operands, top_right, bottom_right
    del matrix, commitment_wide, commitment_tall
    require_memory(16 * matrix_tiles * params.n * params.m + 32 * cpu_tiles * params.n * params.m
                   + 256 * 1024**2, params.max_memory_bytes,
                   'F calibration workspace')
    f_wide = np.tile(public_matrix.astype(np.float64), (1, matrix_tiles))
    f_wide.flags.writeable = False
    f_binary = rng.integers(0, 2, size=f_wide.shape[1], dtype=np.int64)
    measure('F_streamed_float64_binary', f_wide, f_binary)
    measure('F_streamed_float64_binary_u8', f_wide, f_binary.astype(np.uint8))
    measure('F_streamed_float64_field', f_wide,
            rng.integers(0, params.q, size=f_wide.shape[1], dtype=np.int64))
    del f_wide, f_binary
    binary = rng.integers(0, 2, size=wide.shape[1], dtype=np.int64)
    for name in ('F_streamed_float64_binary', 'F_streamed_float64_binary_u8',
                 'F_streamed_float64_field'):
        kernels[name].update(includes_block_accumulation=True,
                             application_backend='prepared_exact_float64')
    if use_metal:
        F_compact = wide.astype(np.uint32)
        F_compact.flags.writeable = False
        for kind, operand in (('binary', binary), ('binary_u8', binary.astype(np.uint8)),
                              ('field', rng.integers(0, params.q, size=wide.shape[1], dtype=np.int64))):
            expected = (F_compact.astype(np.int64) @ operand.astype(np.int64)) % params.q
            for _ in range(warmups):
                if not np.array_equal(metal_matrix.linear_product(F_compact, operand, params.q), expected):
                    raise AssertionError('F warmup mismatch')
            samples = []
            for _ in range(repeats):
                start = time.perf_counter_ns()
                actual = metal_matrix.linear_product(F_compact, operand, params.q)
                samples.append((time.perf_counter_ns()-start)/1e6)
                if not np.array_equal(actual, expected):
                    raise AssertionError('F product mismatch')
            name = 'F_streamed_uint32_' + kind
            kernels[name] = {'kernel': name, 'units_per_sample': F_compact.size,
                             'unit': 'matrix_entries', 'matrix_shape': list(F_compact.shape),
                             'matrix_storage_dtype': 'uint32',
                             'rate_kind': 'measured_metal_streaming_F_shape_proxy',
                             'includes_block_accumulation': True,
                             'measured_this_run': True, 'reference_scan_ms_per_entry': 0.0,
                             **_summary(samples)}
        del F_compact, expected, actual, operand
    def measure_encoding(name, function, units, unit):
        expected = function()
        for _ in range(warmups):
            actual = function()
            same = (all(np.array_equal(a, b) for a, b in zip(actual, expected))
                    if isinstance(expected, tuple) else np.array_equal(actual, expected))
            if not same:
                raise AssertionError('Auxiliary warmup mismatch')
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            actual = function()
            samples.append((time.perf_counter_ns()-start)/1e6)
            same = (all(np.array_equal(a, b) for a, b in zip(actual, expected))
                    if isinstance(expected, tuple) else np.array_equal(actual, expected))
            if not same:
                raise AssertionError('Auxiliary encoding mismatch')
        kernels[name] = {'kernel': name, 'units_per_sample': units, 'unit': unit,
                         'rate_kind': 'measured_auxiliary_encoding', 'measured_this_run': True,
                         **_summary(samples)}
    encoding_rows = max(1, round(params.n * fixture_scale))
    encoding_matrix = public_matrix[:encoding_rows]
    measure_encoding('prepared_matrix_conversion', lambda: encoding_matrix.astype(np.float64),
                     encoding_matrix.size, 'entries')
    kernels['prepared_matrix_conversion']['rate_kind'] = 'measured_matrix_preparation'
    if use_metal:
        measure_encoding('lattice_matrix_preparation', lambda: metal_matrix.prepare(encoding_matrix),
                         encoding_matrix.size, 'entries')
        kernels['lattice_matrix_preparation']['rate_kind'] = 'measured_metal_validation_allocation_copy'
    hash_matrix = metal_matrix.prepare(encoding_matrix) if use_metal else encoding_matrix.astype(np.float64)
    measure_encoding('prepared_matrix_hash', lambda: framed_hash(b'Yang matrix calibration', hash_matrix),
                     encoding_matrix.size, 'entries')
    kernels['prepared_matrix_hash']['rate_kind'] = 'measured_canonical_conversion_and_framed_sha256'
    for name in ('prepared_matrix_conversion', 'lattice_matrix_preparation', 'prepared_matrix_hash'):
        if name in kernels:
            kernels[name]['matrix_shape'] = list(encoding_matrix.shape)
    del hash_matrix, encoding_matrix
    payload = public_matrix.reshape(-1)[:max(1, round(public_matrix.size * fixture_scale))]
    context = bytes(32)
    measure_encoding('auxiliary_message_hash',
                     lambda: auxiliary_digest((payload,), context, params.q),
                     payload.size, 'field_elements')
    digest = auxiliary_digest((payload,), context, params.q)
    measure_encoding('auxiliary_digest_bits',
                     lambda: np.unpackbits(np.frombuffer(digest, dtype=np.uint8), bitorder='little'),
                     256, 'bits')
    gaussian_rng = CryptographicRNG()
    gaussian_count = round(4194304 * fixture_scale)
    mask_samples = []
    for iteration in range(warmups + repeats):
        start = time.perf_counter_ns()
        with uniform_source(gaussian_rng) as source:
            actual = uniform_field(source, (gaussian_count,), params.q)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if (actual.shape != (gaussian_count,) or actual.dtype != np.dtype(np.int64)
                or int(actual.min()) < 0 or int(actual.max()) >= params.q):
            raise AssertionError('Noncanonical Yang mask')
        if iteration >= warmups:
            mask_samples.append(elapsed)
        del actual
    kernels['yang_uniform_mask'] = {
        'kernel': 'yang_uniform_mask', 'units_per_sample': gaussian_count, 'unit': 'entries',
        'rate_kind': 'measured_current_cryptographic_uniform_field', 'measured_this_run': True,
        'randomness': uniform_backend(), 'warmup_samples': warmups, **_summary(mask_samples)}
    kernels.update(_calibrate_yang_gaussians(params, settings, dims, **fixture_options))
    vector_count = round(1048576 * fixture_scale)
    x = rng.integers(0, params.q, size=vector_count, dtype=np.int64)
    y = rng.integers(0, params.q, size=vector_count, dtype=np.int64)
    equal_x = x.copy()
    indices = np.arange(vector_count, dtype=np.int64)
    bits = rng.integers(0, 2, size=vector_count, dtype=np.int64)
    key_bits = np.frombuffer(bits.astype(np.uint8).tobytes(), dtype=np.uint8)
    key_parts = tuple(np.array_split(bits, 2))
    def matched_key():
        if not _same_key_decomposition(key_parts, key_bits):
            raise AssertionError('Incomplete key comparison')
        return True
    measure_encoding('prepared_key_match', matched_key, vector_count, 'bits')
    mismatched = tuple(part.copy() for part in key_parts)
    mismatched[0][0] = params.q - 1
    def unmatched_key():
        if _same_key_decomposition(mismatched, key_bits):
            raise AssertionError('Field mask matches cached key')
        return False
    measure_encoding('prepared_key_miss', unmatched_key, 1, 'calls')
    measure_encoding('immutable_bit_copy',
                     lambda: np.frombuffer(key_bits.tobytes(), dtype=np.uint8), vector_count, 'bits')
    del key_bits, key_parts, mismatched
    functions = {
        'vector_add_mod': lambda: add_mod(x, y, q=params.q),
        'vector_mod_copy': lambda: x % params.q,
        'vector_copy': lambda: x.copy(),
        'immutable_copy': lambda: np.frombuffer(x.tobytes(), dtype=np.int64),
        'indexed_gather': lambda: x[indices],
        'vector_equal': lambda: np.array_equal(x, equal_x),
        'vector_scale': lambda: scale_mod(x, settings.p, params.q),
        'quadratic_product': lambda: _product(x, y, params.q),
        'quadratic_binary': lambda: np.all(bits == bits * bits),
        'range_scan': lambda: (np.any(x < 0), np.any(x >= params.q)),
        'canonical_add_mod': lambda: _add_canonical_mod(x, y, params.q),
        'canonical_sub_mod': lambda: _sub_canonical_mod(x, y, params.q),
        'challenge_affine_add': lambda: _affine_canonical_mod(x, settings.p, y, params.q),
        'challenge_affine_sub': lambda: _affine_canonical_mod(x, -settings.p, y, params.q, subtract=True),
        'commitment_add_mod': lambda: _add_canonical_mod(mod_q(x, params.q), y, params.q),
    }
    for name, function in functions.items():
        measure_encoding(name, function, vector_count, 'coordinates')
        kernels[name]['rate_kind'] = 'measured_current_batched_vector_operation'
    for name, subtract in (('challenge_equal_add', False), ('challenge_equal_sub', True)):
        alpha = -settings.p if subtract else settings.p
        expected = _affine_canonical_mod(x, alpha, y, params.q, subtract=subtract)
        def compare(expected=expected, subtract=subtract, alpha=alpha):
            if not _affine_canonical_equal(expected, x, alpha, y, params.q, subtract=subtract):
                raise AssertionError('Complete equality scan required')
            return True
        for _ in range(warmups):
            compare()
        samples = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            compare()
            samples.append((time.perf_counter_ns() - started) / 1e6)
        kernels[name] = {'kernel': name, 'units_per_sample': vector_count, 'unit': 'coordinates',
                         'rate_kind': 'measured_complete_canonical_affine_equality',
                         'measured_this_run': True, 'warmup_samples': warmups,
                         'alpha': alpha, 'subtract': subtract,
                         **_summary(samples)}
    del expected
    canonical_input = np.frombuffer(x.tobytes(), dtype=np.int64)
    measure_encoding('canonical_matvec_input',
                     lambda: _canonical_matvec_input(canonical_input, vector_count, params.q),
                     vector_count, 'coordinates')
    kernels['canonical_matvec_input'].update(
        rate_kind='measured_immutable_canonical_input_scan')
    del canonical_input
    def input_reference():
        array = np.asarray(x)
        if array.shape != (vector_count,) or array.dtype.kind not in 'iub':
            raise ValueError(f'integer vector required; length={vector_count}')
        if (array.dtype == np.dtype(np.int64) and array.flags.c_contiguous
                and not array.flags.writeable):
            raise ValueError('Writable input required')
        return (array % params.q).astype(np.int64, copy=False)

    input_operations = (input_reference,
                        lambda: _canonical_matvec_input(x, vector_count, params.q))
    input_expected = x.copy()
    input_samples = ([], [])
    for iteration in range(warmups + repeats):
        for variant in ((0, 1) if iteration % 2 == 0 else (1, 0)):
            started = time.perf_counter_ns()
            actual = input_operations[variant]()
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if (not np.array_equal(actual, input_expected)
                    or np.shares_memory(actual, x)
                    or not np.array_equal(x, input_expected)):
                raise AssertionError('Paired input/result mismatch')
            if iteration >= warmups:
                input_samples[variant].append(elapsed)
            del actual
    for variant, name in enumerate(('canonical_mutable_matvec_input_reference',
                                    'canonical_mutable_matvec_input')):
        kernels[name] = {
            'kernel': name, 'units_per_sample': vector_count, 'unit': 'coordinates',
            'rate_kind': 'measured_paired_mutable_input', 'measured_this_run': True,
            'warmup_samples': warmups, 'paired_order': 'alternating_AB_BA',
            **_summary(input_samples[variant])}
    del input_expected
    measure_encoding('accumulator_linear_operands', lambda: _accumulator_linear_operands(
        x[:params.nk], y[:params.nk], y[:params.nk], x[:params.nk], params.q),
        params.nk, 'coordinates')
    kernels['accumulator_linear_operands'].update(
        rate_kind='measured_original_width_accumulator_operands')
    product_rows = min(params.n, max(1, round(8 * fixture_scale)))
    product_bits = rng.integers(0, 2, size=(product_rows, params.m, params.k), dtype=np.uint8)
    product_secrets = rng.integers(0, 2, size=product_rows, dtype=np.int64)
    product_output = np.empty(product_bits.shape, dtype=np.int64)
    measure_encoding('witness_product_write', lambda: np.multiply(
        product_bits, product_secrets[:, None, None], out=product_output, dtype=np.int64),
        product_bits.size, 'coordinates')
    if not np.array_equal(product_output, np.where(product_secrets[:, None, None], product_bits, 0)):
        raise AssertionError('Witness product mismatch')
    kernels['witness_product_write'].update(
        rate_kind='measured_direct_witness_product_write',
        sample_shape=list(product_bits.shape))
    del product_bits, product_secrets, product_output
    constraint_width = max(1, vector_count*dims['witness_coordinates']//dims['quadratic_constraints'])
    index_dtype = np.dtype(dims['relation_memory_plan']['triples_storage_dtype'])
    def make_constraint_table():
        constraint_indices = np.arange(constraint_width, dtype=index_dtype)
        table = np.empty((vector_count, 3), dtype=index_dtype)
        table[:constraint_width] = constraint_indices[:, None]
        for start in range(constraint_width, vector_count, constraint_width):
            stop = min(start+constraint_width, vector_count)
            tail = constraint_indices[:stop-start]
            table[start:stop, 0] = tail
            table[start:stop, 1] = tail
            table[start:stop, 2] = tail//(params.m*params.k)
        return table
    constraint_table = make_constraint_table()
    if use_metal:
        constraint_table = np.frombuffer(constraint_table.tobytes(), dtype=index_dtype).reshape(vector_count, 3)
    valid_witness = np.ones(constraint_width, dtype=np.int64)
    witness_kernel = 'quadratic_witness_layout' if use_metal else 'quadratic_witness_check'
    masks_kernel = 'quadratic_masks_reused_layout' if use_metal else 'quadratic_masks'
    def validate_all_constraints():
        if use_metal:
            result, plan = _quadratic_witness_layout(constraint_table, valid_witness, params.q)
            if plan is None:
                raise AssertionError('Immutable witness layout required')
        else:
            result = _quadratic_witness_valid(constraint_table, valid_witness, params.q)
        if not result:
            raise AssertionError('Valid constraint table required')
        return result
    measure_encoding(witness_kernel, validate_all_constraints, vector_count, 'constraints')
    kernels[witness_kernel].update(
        rate_kind='measured_complete_valid_witness_constraint_check',
        binary_prefix_count=constraint_width)
    layout_plan = None
    if use_metal:
        valid, layout_plan = _quadratic_witness_layout(constraint_table, valid_witness, params.q)
        if not valid or layout_plan is None:
            raise AssertionError('Validated mask layout required')
        kernels[witness_kernel].update(
            constraint_layout_path='same_call_immutable_plan', layout_plan_creation_included=True)
    del valid_witness
    build_width, products = params.depth, []
    for level in range(params.depth):
        node, sibling, jnode, jsibling = (build_width + i*params.nk for i in range(4))
        build_width += 4*params.nk
        products.extend(((params.nk, jnode, params.depth-1-level, node, False, 1),
                         (params.nk, jsibling, params.depth-1-level, sibling, False, 1)))
    recipient_count = max(1, round(16 * fixture_scale))*params.m*params.k + 17
    secret_count = math.ceil(recipient_count / (params.m*params.k))
    for _ in range(2):
        a, secret = build_width, build_width + recipient_count
        output = secret + secret_count
        products.append((recipient_count, output, a, secret, True, params.m*params.k))
        build_width = output + recipient_count
    build_count = build_width + sum(product[0] for product in products)
    measure_encoding('immutable_triple_build',
                     lambda: _immutable_multiplication_triples(build_width, products, index_dtype),
                     build_count, 'triples')
    kernels['immutable_triple_build'].update(
        rate_kind='measured_bounded_immutable_constraint_construction',
        storage_dtype=index_dtype.name, binary_prefix_count=build_width)
    measure_encoding('index_range_scan',
                     lambda: _constraint_indices_in_range(constraint_table, constraint_width),
                     constraint_table.size, 'indices')
    measure_encoding('canonical_index_hash',
                     lambda: framed_hash(b'constraint canonical calibration', constraint_table),
                     constraint_table.size, 'indices')
    for name in ('index_range_scan', 'canonical_index_hash'):
        kernels[name].update(rate_kind='measured_compact_constraint_layout_proxy',
                             storage_dtype=index_dtype.name, canonical_hash_dtype='<i8')
    for name, function in (
            (masks_kernel, lambda: _quadratic_masks(constraint_table, x[:constraint_width], bits[:constraint_width], params.q,
                                                   _layout_plan=layout_plan)),
            ('quadratic_response', lambda: _quadratic_response(constraint_table, x[:constraint_width], -settings.p, params.q))):
        measure_encoding(name, function, vector_count, 'constraints')
        kernels[name].update(rate_kind='measured_bounded_constraint_layout_proxy',
                             binary_prefix_count=constraint_width)
    if use_metal:
        kernels[masks_kernel].update(
            constraint_layout_path='same_call_immutable_plan', layout_plan_creation_included=False,
            layout_plan_validation_included=True)
    del constraint_table, layout_plan
    xx = sample_standard_discrete_gaussian(
        gaussian_rng, vector_count, dims['yang_sigma2'], settings.gaussian_tail)
    vv = sample_standard_discrete_gaussian(
        gaussian_rng, vector_count, settings.sigma1, settings.gaussian_tail)
    norm_bound = 2 * math.sqrt(vector_count) * (dims['yang_sigma2'] + settings.p * settings.sigma1)
    def auxiliary_snapshot_hash():
        state = _auxiliary_digest_state(context, params.q, 1)
        value = _snapshot_integer_vector(x, vector_count, 'calibration', params.q, digest=state)
        return value, state.digest()
    measure_encoding('auxiliary_snapshot_hash', auxiliary_snapshot_hash, vector_count, 'field_elements')
    measure_encoding('canonical_snapshot',
                     lambda: _snapshot_integer_vector(x, vector_count, 'calibration', params.q),
                     vector_count, 'field_elements')
    measure_encoding('norm_snapshot',
                     lambda: _snapshot_integer_vector(xx, vector_count, 'calibration', bound=norm_bound),
                     vector_count, 'coordinates')
    measure_encoding('commitment_tail', lambda: _commitment_tail(
        xx, x, y, params.q, out=np.empty(vector_count + settings.l1, dtype=np.int64)[settings.l1:]),
                     vector_count, 'coordinates')
    measure_encoding('exact_integer_norm', lambda: _within_norm(xx, norm_bound),
                     vector_count, 'coordinates')
    for name, subtract in (('response_and_rejection', False), ('response_and_rejection_subtract', True)):
        measure_encoding(name, lambda subtract=subtract: _response_opening(
            vv, xx, settings.p, subtract=subtract), vector_count, 'coordinates')
    if use_metal:
        kernels.update(_calibrate_complete_commitments(params, settings, dims, public_matrix, **fixture_options))
        kernels.update(_calibrate_joint_phases(params, settings, dims, public_matrix, **fixture_options))
        kernels.update(_calibrate_second_phase(params, settings, dims, public_matrix, **fixture_options))
        kernels.update(_calibrate_verifier_equations(params, settings, dims, public_matrix, **fixture_options))
    kernels.update(_calibrate_witness_context(params, settings, dims, layout_reuse=use_metal, **fixture_options))
    for sample in kernels.values():
        sample.setdefault('warmup_samples', warmups)
        sample['fixture_scale'] = fixture_scale
        sample['sample_count'] = len(sample['samples_ms'])
    return {'kind': 'current_prepared_commitment_kernel_calibration',
            'fixture_scale': fixture_scale, 'repeats': repeats, 'warmups': warmups,
            'cpu_matrix_tiles': cpu_tiles, 'F_matrix_tiles': matrix_tiles,
            'commitment_backend': 'metal_uint32' if use_metal else 'exact_float64',
            'device_single_buffer_limit_bytes': device_buffer_limit,
            'full_target_layout_executed': False,
            'kernels': kernels, 'read_stream_bytes': stream_bytes,
            'read_stream': _summary(read_samples),
            'reference_scan_ms_per_entry': read_scan_per_entry,
            'read_scan_used_as_lower_bound': False,
            'new_protocol_matrices_constructed': False,
            'scope': 'bounded_kernel_calibration'}


def calibrate_protocol_simulation(params, settings, public_matrix, *, fixture_scale=1.0, repeats=10, warmups=2):
    if (type(repeats) is not int or repeats < 1 or type(warmups) is not int or warmups < 0
            or not isinstance(fixture_scale, (int, float)) or not math.isfinite(fixture_scale)
            or not 0 < fixture_scale <= 1 or round(8 * fixture_scale) < 1):
        raise ValueError('Invalid calibration scale/counts')
    dims = target_dimensions(params, settings)
    n, m, q, v = params.n, params.m, params.q, max(params.nk, round(1048576 * fixture_scale))
    if (not isinstance(public_matrix, np.ndarray) or public_matrix.shape != (n, m)
            or public_matrix.dtype != np.dtype(np.int64)
            or np.any(public_matrix < 0) or np.any(public_matrix >= q)):
        raise ValueError('Canonical int64 Arec required')
    rng = np.random.default_rng(9173)
    kernels = {}
    prepared = np.frombuffer(public_matrix.astype(np.float64).tobytes(), dtype=np.float64).reshape(public_matrix.shape)
    prepared_tall = np.frombuffer(np.ascontiguousarray(public_matrix.T, dtype=np.float64).tobytes(),
                                  dtype=np.float64).reshape(public_matrix.T.shape)

    def measure(name, function, units, unit='coordinates'):
        for _ in range(warmups):
            result = function()
            del result
        samples = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            result = function()
            samples.append((time.perf_counter_ns() - started) / 1e6)
            del result
        kernels[name] = {'kernel': name, 'units_per_sample': units, 'unit': unit,
                         'rate_kind': 'measured_reusable_primitive', 'measured_this_run': True,
                         'warmup_samples': warmups, 'sample_count': repeats,
                         'fixture_scale': fixture_scale, **_summary(samples)}

    def proxy(name, source, *, factor=1, unit=None):
        original = kernels[source]
        kernels[name] = {**original, 'kernel': name,
                         'units_per_sample': original['units_per_sample'] / factor,
                         'unit': unit or original['unit'], 'rate_kind': 'proxy_from_reused_primitive',
                         'source_kernel': source, 'source_work_per_target_unit': factor}

    measure('uniform_mod', lambda: rng.integers(0, q, size=v, dtype=np.int64), v, 'entries')
    measure('gpv_gaussian', lambda: rand_discrete_gaussian(rng, (v,), params.sigma,
                                                         params.beta, params.max_memory_bytes), v, 'entries')
    for kind, sigma in (('small', settings.sigma1), ('large', dims['yang_sigma2'])):
        measure('yang_gaussian_' + kind,
                lambda sigma=sigma: sample_standard_discrete_gaussian(rng, v, sigma, settings.gaussian_tail),
                v, 'entries')
    for kind in ('binary', 'field', 'yang_small', 'yang_large'):
        def operand(length):
            if kind == 'binary':
                return rng.integers(0, 2, size=length, dtype=np.int64)
            if kind == 'field':
                return rng.integers(0, q, size=length, dtype=np.int64)
            sigma = settings.sigma1 if kind == 'yang_small' else dims['yang_sigma2']
            return sample_standard_discrete_gaussian(rng, length, sigma, settings.gaussian_tail)
        right, tall_right = operand(m), operand(n)
        measure('reused_A_' + kind, lambda: matmul_mod(public_matrix, right, q), n*m, 'matrix_entries')
        measure('reused_AT_' + kind, lambda: matmul_mod(public_matrix.T, tall_right, q), n*m, 'matrix_entries')
        if kind != 'binary':
            measure('yang_top_'+kind, lambda: _matmul_mod_prepared_left(prepared,right,q), n*m, 'matrix_entries')
            measure('yang_bottom_'+kind, lambda: _matmul_mod_prepared_left(prepared_tall,tall_right,q), n*m, 'matrix_entries')
    for kind in ('binary', 'field'):
        proxy('F_' + kind, 'reused_A_' + kind)
        for dtype in ('uint32', 'int64'):
            proxy('F_streamed_' + dtype + '_' + kind, 'reused_A_' + kind)
            if kind == 'binary':
                proxy('F_streamed_' + dtype + '_binary_u8', 'reused_A_binary')
    for name,matrix in (('yang_top_pair',prepared),('yang_bottom_pair',prepared_tall)):
        pair = tuple(sample_standard_discrete_gaussian(rng, matrix.shape[1], sigma, settings.gaussian_tail)
                     for sigma in (settings.sigma1, dims['yang_sigma2']))
        measure(name, lambda: _matmul_mod_prepared_left(matrix,pair,q), n*m, 'matrix_entries')
    measure('prepared_matrix_conversion', lambda: public_matrix.astype(np.float64), n*m, 'entries')
    measure('prepared_matrix_hash', lambda: framed_hash(b'Yang matrix calibration', prepared), n*m, 'entries')
    proxy('auxiliary_binary', 'reused_A_binary')
    proxy('accumulator_matvec', 'reused_A_field')
    proxy('encryption_matvec', 'reused_AT_binary')
    proxy('oa_matvec', 'reused_AT_binary')
    proxy('decryption_matvec', 'reused_A_field')
    gaussian_column = rand_discrete_gaussian(rng, (m,), params.sigma, params.beta, params.max_memory_bytes)
    measure('gpv_AE_columns', lambda: matmul_mod(public_matrix, gaussian_column, q), 1, 'output_columns')

    x = rng.integers(0, q, size=v, dtype=np.int64)
    y = rng.integers(0, q, size=v, dtype=np.int64)
    measure('canonical_copy', lambda: (np.any(x < 0), np.any(x >= q), x.copy()), v)
    measure('hash_bytes', lambda: hashlib.sha256(memoryview(x).cast('B')).digest(), x.nbytes, 'bytes')
    measure('vector_add_mod', lambda: add_mod(x, y, q=q), v)
    measure('vector_mod_copy', lambda: (x % q).astype(np.int64), v)
    measure('canonical_mutable_matvec_input', lambda: _canonical_matvec_input(x, v, q), v)
    measure('vector_copy', lambda: x.copy(), v)
    measure('immutable_copy', lambda: np.frombuffer(x.tobytes(), dtype=np.int64), v)
    indices = np.arange(v, dtype=np.int64)
    measure('indexed_gather', lambda: x[indices], v)
    measure('vector_equal', lambda: np.array_equal(x, y), v)
    measure('vector_scale', lambda: (settings.p*x) % q, v)
    measure('quadratic_product', lambda: _product(x, y, q), v)
    bits = rng.integers(0, 2, size=v, dtype=np.int64)
    measure('quadratic_binary', lambda: np.all(bits == bits*bits), v)
    measure('range_scan', lambda: (np.any(x < 0), np.any(x >= q)), v)
    xx = sample_standard_discrete_gaussian(rng, v, dims['yang_sigma2'], settings.gaussian_tail)
    vv = sample_standard_discrete_gaussian(rng, v, settings.sigma1, settings.gaussian_tail)
    measure('commitment_tail', lambda: _commitment_tail(xx, x, y, q), v)
    measure('commitment_add_mod', lambda: _add_canonical_mod(mod_q(x, q), y, q), v)
    measure('canonical_add_mod', lambda: _add_canonical_mod(x, y, q), v)
    measure('canonical_sub_mod', lambda: _sub_canonical_mod(x, y, q), v)
    measure('challenge_affine_add', lambda: _affine_canonical_mod(x, settings.p, y, q), v)
    measure('challenge_affine_sub', lambda: _affine_canonical_mod(x, -settings.p, y, q, subtract=True), v)
    calibration_triples = np.repeat(indices[:, None], 3, axis=1)
    measure('quadratic_masks', lambda: _quadratic_masks(calibration_triples, x, bits, q), v, 'constraints')
    measure('quadratic_response', lambda: _quadratic_response(calibration_triples, x, -settings.p, q), v, 'constraints')
    norm_bound = 2*math.sqrt(v)*(dims['yang_sigma2'] + settings.p*settings.sigma1)
    measure('exact_integer_norm', lambda: _within_norm(xx, norm_bound), v)
    for name, subtract in (('response_and_rejection', False), ('response_and_rejection_subtract', True)):
        measure(name, lambda subtract=subtract: _response_opening(
            vv, xx, settings.p, subtract=subtract), v)
    measure('accumulator_linear_operands', lambda: _accumulator_linear_operands(
        x[:params.nk], y[:params.nk], y[:params.nk], x[:params.nk], q), params.nk)
    product_rows = min(params.n, max(1, round(8 * fixture_scale)))
    product_bits = rng.integers(0, 2, size=(product_rows, params.m, params.k), dtype=np.uint8)
    product_secrets = rng.integers(0, 2, size=product_rows, dtype=np.int64)
    product_output = np.empty(product_bits.shape, dtype=np.int64)
    measure('witness_product_write', lambda: np.multiply(
        product_bits, product_secrets[:, None, None], out=product_output, dtype=np.int64),
        product_bits.size)
    if not np.array_equal(product_output, np.where(product_secrets[:, None, None], product_bits, 0)):
        raise AssertionError('Witness product mismatch')
    kernels['witness_product_write'].update(
        rate_kind='measured_direct_witness_product_write',
        sample_shape=list(product_bits.shape))
    del product_bits, product_secrets, product_output
    def one_weighted_digit():
        selected = x >= (q-1)//2
        return selected.astype(np.uint8), x - selected.astype(np.int64)*((q-1)//2)
    measure('weighted_digit', one_weighted_digit, v)
    measure('ordinary_digit', lambda: ((x >> 1) & 1).astype(np.uint8), v)
    proxy('weighted_decomposition', 'weighted_digit', factor=params.k, unit='field_elements')
    proxy('ordinary_decomposition', 'ordinary_digit', factor=params.k, unit='field_elements')
    for name, digits in (('q', params.k), ('noise', dims['noise_digits']), ('policy', dims['policy_digits'])):
        proxy('weighted_decode_' + name, 'reused_A_field', factor=digits, unit='decoded_coordinates')
    proxy('policy_window_product', 'reused_A_field', factor=(m-params.t+1)*params.t, unit='pattern')
    proxy('make_triples', 'vector_copy', factor=3, unit='triples')
    measure('auxiliary_message_hash', lambda: auxiliary_digest((x,), bytes(32), q), v, 'field_elements')
    digest = auxiliary_digest((x,), bytes(32), q)
    measure('auxiliary_digest_bits',
            lambda: np.unpackbits(np.frombuffer(digest, dtype=np.uint8), bitorder='little'),
            256, 'bits')
    return {'schema_version': 3, 'arithmetic_model': 'protocol_simulation_reused_matrix_v1',
            'kind': 'symbolic_protocol_simulation_with_reused_primitive_rates',
            'params': params.to_dict(), 'yang_settings': asdict(settings), 'dimensions': dims,
            'repeats': repeats, 'warmups': warmups, 'fixture_scale': fixture_scale, 'kernels': kernels,
            'inner_tile': m, 'F_inner_tile': _F_DISK_COLUMN_BLOCK,
            'new_protocol_matrices_constructed': False, 'proof_constructed': False,
            'reused_matrix_shape': [n, m], 'vector_entries': v,
            'reference_depth': params.depth, 'reference_sigma2': dims['yang_sigma2'],
            'limitations': [
                'target_matrices_and_proofs_not_constructed',
                'transferred_primitive_rates',
                'reference_depth_gaussian_rates',
                'storage_io_and_uncalibrated_overheads_excluded',
            ]}


def estimate_modules(params, settings: YangProtocolParams | None, calibration: dict) -> dict:
    if calibration.get('arithmetic_model') != 'protocol_simulation_reused_matrix_v1':
        raise ValueError('Multiscale calibration required')
    if settings is None:
        raise ValueError('Commitment parameters required')
    d = target_dimensions(params, settings)
    reference_plan = d["relation_memory_plan"]
    backend_plan = linear_hash_storage_plan(params, storage_dtype='float64',
                                           disk_backed=False, read_only=True)
    resident = backend_plan['matrix_bytes']
    workspace = max(backend_plan["application_explicit_buffer_bytes"] - resident,
                    4 * LinearPublicKeyHash.canonical_chunk_bytes)
    d["relation_memory_plan_reference_copied_int64"] = reference_plan
    d["relation_memory_plan"] = {**reference_plan,
        "F_copy_bytes": 0, "F_resident_matrix_bytes": resident,
        "F_workspace_bytes": workspace, "F_backend": "read_only_streamed_memory",
        "F_storage_dtype": 'float64', "F_block_columns": backend_plan["column_block_size"],
        "F_backend_resource_plan": backend_plan,
        "conservative_working_bytes": reference_plan["other_relation_arrays_bytes"] + resident + workspace,
        "model_scope": "readonly_resident_F_counted_not_allocated"}

    old = calibration["params"]
    for key in ("n", "q", "t", "e", "d", "error_bound", "sigma", "beta"):
        if old[key] != getattr(params, key):
            raise ValueError(f"Calibration mismatch: {key}")
    current_chi=params.to_dict()['chi']

    excluded={'implementation','source'}
    if ({key:value for key,value in old.get('chi',{}).items() if key not in excluded}
            != {key:value for key,value in current_chi.items() if key not in excluded}):
        raise ValueError('Chi calibration mismatch')
    if calibration["yang_settings"] != asdict(settings):
        raise ValueError('Yang calibration mismatch')
    kernels = calibration["kernels"]
    schedules = calibration["parallel_schedule"]["phases"]
    if not schedules:
        raise ValueError('Parallel calibration required')
    kernels = dict(kernels)
    for kind, sample in calibration['relation_work']['by_depth'][params.depth].items():
        kernels['relation_non_F_' + kind] = sample
    n, m, k, nk, ell, q = params.n, params.m, params.k, params.nk, params.depth, params.q
    w, c, rows = d["witness_coordinates"], d["quadratic_constraints"], d["relation_rows"]
    l1, l2 = settings.l1, settings.l2
    Fentries, Fcols = d["F_entries"], d["F_columns"]
    F_tile = calibration["F_inner_tile"]
    prepared_key_reuse = all(name in kernels for name in
                            ('prepared_key_match', 'prepared_key_miss', 'immutable_bit_copy'))
    complete_commitments = _complete_commitment_calibration(kernels)
    if complete_commitments:
        sample = kernels['complete_B1_pair_new']
        if (w < 131072 or any(sample[key] != value for key, value in
                                      (('q', q), ('n', n), ('l1', l1), ('l2', l2),
                                       ('sigma1', settings.sigma1), ('gaussian_tail', settings.gaussian_tail),
                                       ('response_alpha', -settings.p)))):
            raise ValueError('Commitment execution plan mismatch')
        from dataclasses import replace
        reference = target_dimensions(replace(params, depth=sample['calibration_reference_depth']), settings)
        if (sample['reference_randomness_length'] != reference['yang_randomness_length']
                or sample['sigma2'] != reference['yang_sigma2']):
            raise ValueError('Reference response distribution mismatch')
    modules = {}

    def term(kernel, units, formula, *, label=None, calls=1):
        sample = kernels[kernel]
        factor = units / sample["units_per_sample"]
        kernel_ms = sample["mean_ms"] * factor
        scan_projection_ms = units * sample.get('reference_scan_ms_per_entry',
                                                sample.get('matrix_entry_read_floor_ms', 0.0))
        result = {"label": label or kernel, "kernel": kernel, "target_units": int(units),
                "unit": sample["unit"], "formula": formula, "sample_multiplier": factor,
                "estimated_mean_ms": kernel_ms,
                "calibrated_kernel_estimate_ms": kernel_ms,
                "matrix_input_scan_projection_ms": scan_projection_ms,
                "empirical_scaled_min_ms": sample["empirical_min_ms"] * factor,
                "empirical_scaled_max_ms": sample["empirical_max_ms"] * factor,
                'rate_kind': sample.get('rate_kind', 'measured_target_kernel'),
                'proxy_source': sample.get('source_kernel')}
        if 'scaling_fit' in sample:
            from .multiscale import predict_kernel

            prediction = predict_kernel(sample, units / calls)
            for key in ('estimated_mean_ms', 'empirical_scaled_min_ms', 'empirical_scaled_max_ms'):
                prediction[key] *= calls
            result.update({key: value for key, value in prediction.items() if key != 'target_units'})
            result['calibrated_kernel_estimate_ms'] = prediction['estimated_mean_ms']
            result['independent_calls'] = calls
        return result

    def module(name, terms):
        result = {"status": "estimated", "execution_scope": "symbolic_operation_counts",
                  "components": terms, "estimated_mean_ms": sum(t["estimated_mean_ms"] for t in terms),
                  "empirical_scaled_min_ms": sum(t["empirical_scaled_min_ms"] for t in terms),
                  "empirical_scaled_max_ms": sum(t["empirical_scaled_max_ms"] for t in terms),
                  "includes_disk_or_swap_io": False, "is_confidence_interval": False}
        modules[name] = result
        return terms

    def parallel_phase(name, branch_a, branch_b, post=()):
        from .schedule_model import predict_parallel_schedule

        fields = ('estimated_mean_ms', 'empirical_scaled_min_ms', 'empirical_scaled_max_ms')
        def total(parts):
            return {key: sum(part[key] for part in parts) for key in fields}
        covered = [*branch_a, *branch_b, *post]
        if len({id(part) for part in covered}) != len(covered):
            raise AssertionError('Duplicate parallel operation')
        result = predict_parallel_schedule(schedules[name], total(branch_a), total(branch_b), total(post),
            target_work={'W': w, 'L': c, 'F_columns': 0 if name == 'p_first' else Fcols,
                         'relation_rows': 0 if name == 'p_first' else rows,
                         'q': q, 'n': n, 'm': m, 'k': k, 'depth': ell,
                         'recipient_decode_rows': 2*n*m},
            branch_components={'a': [part['kernel'] for part in branch_a],
                               'b': [part['kernel'] for part in branch_b]})
        return {'kernel': 'parallel_' + name, 'label': name, 'target_units': 1,
                'unit': 'scheduled_phase', 'formula': 'schedule(A,B,post)',
                'branch_a_components': branch_a, 'branch_b_components': branch_b,
                'post_components': list(post), 'covered_components_additive': False,
                **result}

    def replace_parallel(parts, name, branch_a, branch_b, post=()):
        covered = [*branch_a, *branch_b, *post]
        identifiers = {id(part) for part in covered}
        if sum(id(part) in identifiers for part in parts) != len(covered):
            raise AssertionError('Parallel operation coverage mismatch')
        return [part for part in parts if id(part) not in identifiers] + [
            parallel_phase(name, branch_a, branch_b, post)]

    def index_scan(units, label):
        return term('index_range_scan' if 'index_range_scan' in kernels else 'range_scan',
                    units, '3*triples', label=label)

    def statement_hash(entries, label):
        if 'canonical_index_hash' in kernels:
            return [term('canonical_index_hash', 3*c, '3*triples', label=label+' indices'),
                    term('hash_bytes', 8*entries, '8*other_statement_entries', label=label+' fields')]
        return [term('hash_bytes', 8*(entries+3*c), '8*(other_statement_entries+3*triples)', label=label)]

    def Fkernel(field, bits_input=False):
        kind = "field" if field else "binary"
        return 'F_streamed_float64_' + kind + ('_u8' if bits_input else '')

    def Fproducts(field, bits_input=False):
        return [term(Fkernel(field, bits_input), Fentries, "n * (2*n*m*k)",
                     label='F field-mask product' if field else 'F key-bit product')]

    def commitment(width, kind, label):
        kernel = "yang_top_" + kind
        products = ([term('yang_joint_' + kind, l1*(width+l2)+width*l2,
                         'l1*(width+l2)+width*l2', label=label + ' top/bottom batch')]
                    if 'yang_joint_' + kind in kernels else [term(kernel, l1 * (width + l2),
                     "l1*(width+l2)", label=label + " top"),
                term("yang_bottom_" + kind, width * l2, "width*l2", label=label + " bottom")])
        result = products + [
                term("commitment_add_mod", l1, "l1", label=label + ' header addition'),
                term("commitment_tail", width, "width", label=label + ' fused tail addition'),
                term("vector_copy", l1, "l1", label=label + ' header write')]
        return result

    public_entries = 3*n*m + 2*n*ell + nk + params.e*params.t + 6*m + 2*ell
    def Mvector(field, prepared_key=False, canonical_input=False):
        key_product = ([term('prepared_key_match', Fcols, '2*n*m*k',
                             label='complete input comparison before same-call F result reuse')]
                       if prepared_key and not field else
                       Fproducts(field))
        if prepared_key and field:
            key_product.append(term('prepared_key_miss', 1, 'one field-mask mismatch',
                                    label='prepared-key mismatch check'))
        rest = 'relation_non_F_' + ('field' if field else 'binary')
        return [*key_product, term(rest, 1, 'one original-size non-F call'),
                term('canonical_matvec_input' if canonical_input else 'canonical_mutable_matvec_input',
                     w, 'witness_width', label='operator input normalization')]

    f_setup = [term("uniform_mod", Fentries, "2*n^2*m*k"),
                       term("vector_copy", Fentries, "2*n^2*m*k",
                            label='F chunk storage'),
                       term("hash_bytes", 8*Fentries, "8*F_entries",
                            label='F generation digest')]
    f_setup.append(term('prepared_matrix_conversion', Fentries, 'F_entries',
                        label='exact F float64 preparation'))
    module("SETUP_F", f_setup)
    module("SETUP_GPV_OA", [term("gpv_gaussian", 2*m*ell, "2*m*depth"),
                           term("gpv_AE_columns", 2*ell, "2*depth")])
    module("SETUP_public_A", [term("uniform_mod", 3*n*m, "3*n*m")])
    lattice_entries = 2*l1*l2 + (l1+l2)*(w+c)
    ppentries = d['yang_memory_plan']['public_matrix_entries']
    lattice_setup = ([term('lattice_matrix_preparation', lattice_entries, '2*l1*l2+(l1+l2)*(width+triples)',
                          label='immutable uint32 B1/B2 preparation'),
                      term('range_scan', ppentries-lattice_entries, 'auxiliary_matrix_entries'),
                      term('immutable_copy', ppentries-lattice_entries, 'auxiliary_matrix_entries')]
                     if 'lattice_matrix_preparation' in kernels else [
                          term('prepared_matrix_conversion', lattice_entries,
                               '2*l1*l2+(l1+l2)*(width+triples)',
                               label='exact B1/B2 float64 preparation'),
                          term("range_scan", ppentries, "Yang public_matrix_entries",
                               label='public matrix validation'),
                          term("immutable_copy", ppentries, "Yang public_matrix_entries",
                               label='public matrix freezing')])
    module("SETUP_Yang", [term("uniform_mod", ppentries, "Yang public_matrix_entries"),
                          *lattice_setup,
                          term('prepared_matrix_hash', lattice_entries, 'B1_B2_entries',
                               label='int64 matrix conversion/digest'),
                          term("hash_bytes", 8*(ppentries-lattice_entries), "8*auxiliary_matrix_entries",
                               label='int64 auxiliary parameter digest')])
    ukgen_terms = [term("gpv_gaussian", 2*m*m, "2*m^2"),
                     term("gpv_AE_columns", 2*m, "2*m"),
                     term("weighted_decomposition", 2*n*m, "2*n*m", calls=2), *Fproducts(False, bits_input=True),
                     term("vector_copy", math.ceil(Fcols/8), "ceil(F_columns/8)",
                          label='uint8 decomposition assembly')]
    if not kernels[Fkernel(False, True)].get('includes_block_accumulation', False):
        ukgen_terms.append(term("vector_add_mod", n*math.ceil(Fcols/F_tile),
                               "n*ceil(F_columns/F_inner_tile)", label="F block accumulation"))
    module("UKGEN", ukgen_terms)
    module("UKGEN_GPV", ukgen_terms[:2])
    module("UKGEN_F", ukgen_terms[2:])
    if prepared_key_reuse:
        key_preparation = ukgen_terms[2:] + [
            term('immutable_copy', 2*n*m, '2*n*m', label='source-key snapshots', calls=2),
            term('range_scan', 2*n*m, '2*n*m', label='source-key validation', calls=2),
            term('vector_equal', 2*n*m, '2*n*m', label='complete source-key identity comparison', calls=2),
            term('immutable_bit_copy', Fcols, '2*n*m*k', label='immutable decomposition snapshot'),
            term('immutable_copy', n+math.ceil(nk/8), 'n+ceil(nk/8)', label='syndrome/leaf snapshots')]
        module('P_key_binding', key_preparation)
        d['prover_prepared_key_persistent_bytes'] = 16*n*m+Fcols+8*n+nk
    module("DEC_core", [term("decryption_matvec", m*m, "m^2"),
                        term("vector_mod_copy", m, "m")])
    public_compile = [term("canonical_copy", public_entries, "other_public_entries")]
    relation_digest_entries = public_entries + 3*c + rows
    public_compile += ([term('immutable_triple_build', c, 'quadratic_constraints',
                            label='immutable constraint construction')]
                       if 'immutable_triple_build' in kernels else
                       [term('make_triples', c, 'quadratic_constraints'),
                        term('immutable_index_copy' if 'immutable_index_copy' in kernels else 'immutable_copy',
                             3*c, '3*triples', label='immutable constraint indices')])
    public_compile += [term("immutable_copy", rows, "rows", label='public target snapshot'),
                       *statement_hash(relation_digest_entries-3*c, 'public relation digest'),
                       term('hash_bytes', 96, 'statement headers')]
    public_compile.append(term('ordinary_decode', 1, 'one root decode'))
    module("RELATION_public_compile", public_compile)
    witness = [term("weighted_decomposition", 2*n*m, "2*n*m", calls=2),
               term("vector_copy", w, "witness_width", label='witness population proxy'),
               term("witness_product_write" if "witness_product_write" in kernels else "vector_scale",
                    2*n*m*k, "2*n*m*k", label='witness bit products'),
               term('path_reconstruction' if 'path_reconstruction' in kernels else 'accumulator_matvec',
                    ell*n*m, 'depth*n*m', label='path reconstruction'),
               term('policy_witness_distances' if 'policy_witness_distances' in kernels else 'policy_window_product',
                    params.e, 'e', label='policy witness distances'),
               term('weighted_decomposition_policy' if 'weighted_decomposition_policy' in kernels else 'weighted_decomposition',
                    d['policy_rows'], 'policy_rows', label='policy slack encoding'),
               term('weighted_decomposition_noise' if 'weighted_decomposition_noise' in kernels else 'weighted_decomposition',
                    6*m, '6*m', label='m-coordinate noise encoding', calls=6),
               term('weighted_decomposition_noise' if 'weighted_decomposition_noise' in kernels else 'weighted_decomposition',
                    2*ell, '2*depth', label='identity noise encoding', calls=2)]
    witness.append(term('ordinary_encode', ell, 'depth node encodings'))
    module("RELATION_compile", public_compile + witness)
    if prepared_key_reuse:
        module('P_RELATION_compile', public_compile + witness[1:])
    module("RELATION_M_binary", Mvector(False))
    module("RELATION_M_field", Mvector(True))
    module("RELATION_check", Mvector(False) + [term("range_scan", w, "witness_width"),
                                               term("quadratic_binary", c, "triples"),
                                               term("indexed_gather", 3*c, "3*triples")])
    auxbits = d["yang_memory_plan"]["auxiliary_message_bits"]
    rhobits = d["yang_memory_plan"]["auxiliary_randomness_bits"]
    def aux():
        plan = _modular_product_plan(settings.auxiliary_rows, auxbits+rhobits, 1, q-1, 1, q)
        return [term("auxiliary_message_hash", d['yang_memory_plan']['auxiliary_payload_field_elements'],
                     "relation_rows+4*l1+2*witness_width+2*triples",
                     label='framed auxiliary payload digest'),
                term("auxiliary_digest_bits", auxbits, "256",
                     label='auxiliary digest bits'),
                term("vector_copy", 2*auxbits+rhobits,
                     "2*message_bits+randomness_bits",
                     label="auxiliary vector/context/randomness concatenations"),
                term("auxiliary_binary", settings.auxiliary_rows*(auxbits+rhobits),
                     "auxiliary_rows*(message_bits+randomness_bits)"),
                term("range_scan", auxbits+rhobits, "message_bits+randomness_bits",
                     label='auxiliary bit validation'),
                term("vector_add_mod", settings.auxiliary_rows*max(0, math.ceil((auxbits+rhobits)/plan['inner_block'])-1),
                     "auxiliary_rows*(ceil((message_bits+randomness_bits)/current_safe_inner)-1)",
                     label='auxiliary block accumulation')]
    length = d["yang_randomness_length"]
    layout_reuse = w >= 65536 and _prover_layout_calibration(kernels)
    witness_kernel = 'quadratic_witness_layout' if layout_reuse else 'quadratic_witness_check'
    masks_kernel = 'quadratic_masks_reused_layout' if layout_reuse else 'quadratic_masks'
    witness_check = ([term(witness_kernel, c, "triples",
                           label='blocked quadratic witness check')]
                     if witness_kernel in kernels else
                     [term("quadratic_product", c, "triples", label='quadratic witness check'),
                      term("indexed_gather", 3*c, "3*triples", label='witness index gathers'),
                      term("vector_equal", c, "triples", label='witness equation checks')])
    witness_snapshot = ([term("canonical_snapshot", w, "witness_width",
                              label='canonical witness snapshot')]
                        if 'canonical_snapshot' in kernels else
                        [term("range_scan", w, "witness_width", label='witness validation'),
                         term("immutable_copy", w, "witness_width", label='witness snapshot')])
    binary_work = Mvector(False, prepared_key_reuse, canonical_input=True)
    field_work = Mvector(True, prepared_key_reuse)
    context_hash = statement_hash(rows, 'prover context digest')
    linear_witness_check = term('vector_equal', rows, 'rows', label='linear witness check')
    witness_context = [parallel_phase('context', context_hash,
                                     binary_work + witness_check + [linear_witness_check])]
    gaussian_terms = [term(kernel, size + l1 + l2, 'opening_width+l1+l2',
                           label='opening randomness ' + role)
                      for role, size in (('B1', w), ('B2', c))
                      for kernel in ('yang_gaussian_small', 'yang_gaussian_large')]
    prover = (field_work + witness_context + witness_snapshot
              + [term(masks_kernel, c, "triples", label='a,b from reused operands'),
                 term('yang_uniform_mask' if 'yang_uniform_mask' in kernels else 'uniform_mod',
                      w, 'witness_width', label='mask r sampling'),
                 *gaussian_terms,
                 *([term('response_and_rejection', w+l1+l2, 'witness_width+l1+l2',
                         label='witness response/rejection'),
                    term('response_and_rejection_subtract', c+l1+l2, 'triples+l1+l2',
                         label='quadratic response/rejection')]
                   if 'response_and_rejection_subtract' in kernels else
                   [term('response_and_rejection', length, '2*l1+2*l2+witness_width+triples')]),
                 term("challenge_affine_add", w, "witness_width", label="masked witness z0"),
                 index_scan(3*c, "public constraint indices"),
                 term("range_scan", rows, "rows", label="public target validation"),
                 term("immutable_copy", settings.auxiliary_rows, "auxiliary_rows",
                      label='initial commitment snapshot')]
              + commitment(w, "yang_small", "B1.commit(x,s1)")
              + commitment(w, "yang_large", "B1.commit(r,s2)")
              + commitment(c, "yang_small", "B2.commit(a,s3)")
              + commitment(c, "yang_large", "B2.commit(b,s4)") + aux())
    prover = [component for component in prover
              if not component['kernel'].startswith(('yang_top_', 'yang_bottom_', 'yang_joint_'))]
    for label,width in (('B1',w),('B2',c)):
        prover += ([term('yang_joint_pair', l1*(width+l2)+width*l2,
                         'l1*(width+l2)+width*l2', label=label+' two-opening matrix batch')]
                   if 'yang_joint_pair' in kernels else [term('yang_top_pair', l1*(width+l2), 'l1*(width+l2)',
                        label=label+' two-opening top'),
                   term('yang_bottom_pair', width*l2, 'width*l2',
                        label=label+' two-opening bottom')])
    prover = _apply_complete_commitment_ratios(prover, kernels, w, c, l1, l2, pair=True)
    first_a = [part for part in prover if part['kernel'] == 'complete_B1_pair_new']
    first_b = [part for part in prover if part['kernel'] == masks_kernel
               or (part['kernel'].startswith('yang_gaussian_') and part['label'] == 'opening randomness B2')]
    second_a = [part for part in prover if part['kernel'] == 'complete_B2_pair_new']
    if len(first_a) != 1 or len(first_b) != 3 or len(second_a) != 1:
        raise AssertionError('Incomplete prover branch')
    prover = replace_parallel(prover, 'p_first', first_a, first_b)
    prover = replace_parallel(prover, 'p_second', second_a, field_work)
    module("P_protocol_attempt", prover)
    module("V_prechallenge", [index_scan(3*c, "public constraint indices"),
                              term("range_scan", rows, "rows", label="public target validation"),
                              *([term("canonical_snapshot", settings.auxiliary_rows, "auxiliary_rows",
                                      label='canonical commitment snapshot')]
                                if 'canonical_snapshot' in kernels else
                                [term("range_scan", settings.auxiliary_rows, "auxiliary_rows"),
                                 term("immutable_copy", settings.auxiliary_rows, "auxiliary_rows")]),
                              *statement_hash(rows, "one verifier-constructor context digest")])
    verifier_aux = aux()
    fused_snapshots = all(name in kernels for name in ('auxiliary_snapshot_hash', 'canonical_snapshot', 'norm_snapshot'))
    if fused_snapshots:
        verifier_aux = [component for component in verifier_aux if component['kernel'] != 'auxiliary_message_hash']
        snapshots = [term('auxiliary_snapshot_hash', d['yang_memory_plan']['auxiliary_payload_field_elements'],
                          'rows+4*l1+2*witness_width+2*triples', label='canonical payload snapshots/digest'),
                     term('canonical_snapshot', w, 'witness_width', label='canonical masked witness'),
                     term('norm_snapshot', w+l1+l2, 'l1+l2+witness_width', label='signed B1 snapshot/exact norm'),
                     term('norm_snapshot', c+l1+l2, 'l1+l2+triples', label='signed B2 snapshot/exact norm'),
                     term('immutable_copy', rhobits, 'rho_bits', label='auxiliary randomness snapshot'),
                     index_scan(3*c, 'public constraint indices'),
                     term('range_scan', rows, 'rows', label='public target validation')]
    else:
        snapshots = [term("exact_integer_norm", length, "2*l1+2*l2+witness_width+triples"),
                     term("immutable_copy", d["yang_memory_plan"]["response_snapshot_bytes"]//8,
                          "rows+6*l1+2*l2+4*width+3*triples+rho_bits", label='received response snapshots'),
                     index_scan(3*c, 'public constraint indices'),
                     term("range_scan", 2*rows+3*w+2*c+4*l1, "2*rows+3*width+2*triples+4*l1",
                          label='target/response validation')]
    equality_terms = [term('challenge_equal_add', rows, 'rows', label='linear response equality'),
                      term('challenge_equal_add', l1+w, 'l1+witness_width', label='response equality B1'),
                      term('challenge_equal_sub', l1+c, 'l1+triples', label='response equality B2'),
                      term('vector_equal', settings.auxiliary_rows, 'auxiliary_rows',
                           label='auxiliary commitment comparison')]
    verifier_field_work = Mvector(True, canonical_input=True)
    verifier = (verifier_field_work + verifier_aux
                + commitment(w, "yang_large", "B1.commit(z0,z1)")
                + commitment(c, "yang_large", "B2.commit(d,z2)")
                + snapshots + equality_terms
                + [term("quadratic_response", c, "triples", label='blocked quadratic response')])
    verifier = _apply_complete_commitment_ratios(verifier, kernels, w, c, l1, l2, pair=False)
    equation_a = [part for part in verifier if part['kernel'] == 'complete_B1_single_new']
    equation_b = verifier_field_work + [part for part in verifier
                 if part['kernel'] == 'quadratic_response' or part['label'] == 'linear response equality']
    equation_post = [part for part in verifier if part['label'] == 'response equality B1']
    if len(equation_a) != 1 or len(equation_post) != 1 or len(equation_b) != len(verifier_field_work)+2:
        raise AssertionError('Missing verifier branch')
    verifier = replace_parallel(verifier, 'v_equation', equation_a, equation_b, equation_post)
    module("V_protocol_response", verifier)
    return {"schema_version": 3, "arithmetic_model": calibration['arithmetic_model'],
            "kind": "symbolic_protocol_simulation",
            "arithmetic_backend": 'streamed_f',
            "public_F_storage": 'readonly_float64',
            "implementation_contract": {"new_protocol_matrices_constructed": False,
                                        "implemented_full_target_executor": False},
            "params": params.to_dict(), "yang_settings": asdict(settings), "dimensions": d,
            "modules": modules, "security_validated": False,
            "empirical_range_definition": "scaled_kernel_extrema_not_confidence_interval",
            "retry_model": {"attempts_measured_at_target": None,
                            "one_round_expected_attempts_expression": d["yang_rejection_M"],
                            "expression_validated_at_target": False},
            "repetition_model": {"main_protocol_rounds": 1,
                                 "conditional_rounds_for_80_bits": d["yang_conditional_rounds_for_80_bits"],
                                 "single_round_expression": d["yang_single_round_soundness_expression"],
                                 "theorem_conditions_validated": False},
            "limitations": calibration["limitations"] + [
                "target_memory_and_cache_scaling_unvalidated",
                "support_bound_block_counts",
                "proxy_rates_and_uncalibrated_protocol_overheads",
            ]}
