from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import random
from pathlib import Path

from .scaling import fit_measurements, predict_scaling


SCALES = (0.125, 0.25, 0.5, 1.0)


def _source_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob('*.py')}


def _work_axis(name, sample):
    if name.startswith('complete_'):
        return 'sample_commitment_width'
    if name.startswith(('p_witness_context',)):
        return 'sample_constraint_count'
    if name.startswith(('p_first_phase', 'p_second_phase', 'v_equation_phase')):
        return 'sample_witness_width'
    return 'units_per_sample'


def extend_commitment_calibration(calibration):
    for phase, kernel, axis in (
            ('p_first', 'complete_B1_pair_new', 'W'),
            ('p_second', 'complete_B2_pair_new', 'L'),
            ('v_equation', 'complete_B1_single_new', 'W')):
        sample = calibration['kernels'][kernel]
        original = sample.setdefault('primitive_scale_measurements', deepcopy(sample['scale_measurements']))
        standalone = [dict(deepcopy(point), source='complete_commitment_calibration')
                      for point in original]
        branch = []
        for record in calibration['parallel_schedule']['records'][phase]:
            if record['warmup_samples'] != 2 or len(record['branch_a_ms']) != 10:
                raise ValueError('Branch calibration requires warmup=2,n=10')
            branch.append({'units_per_sample': record['work'][axis],
                           'samples_ms': list(record['branch_a_ms']), 'shape': record['shape'],
                           'source': 'independent_branch_in_parallel_calibration'})
        sources = {
            'complete_commitment_calibration': standalone,
            'independent_branch_in_parallel_calibration': branch,
        }
        selected = 'independent_branch_in_parallel_calibration'
        sample['source_calibrations'] = {
            name: {'scale_measurements': points, 'scaling_fit': fit_measurements(points)}
            for name, points in sources.items()
        }
        sample['selected_calibration_source'] = selected
        sample['calibration_source_selection'] = 'same_independent_helper_and_fixture_as_scheduled_phase'
        sample['scale_measurements'] = deepcopy(sources[selected])
        sample['scaling_fit'] = deepcopy(sample['source_calibrations'][selected]['scaling_fit'])
        sample['scaling_fit']['diagnostics']['calibration_source'] = selected


def calibrate_multiscale(params, settings, public_matrix, *, on_progress=None):
    from .cost_model import calibrate_commitment_products, calibrate_protocol_simulation
    from .fixed_work import calibrate_fixed_work, calibrate_encoding_work, calibrate_auxiliary_work

    source_before = _source_hashes()
    reference_params = replace(params, depth=10)
    records_by_scale = {}
    scale_order = random.Random(20261015).sample(list(SCALES), len(SCALES))
    for index, scale in enumerate(scale_order, 1):
        if on_progress:
            on_progress({'phase': 'scale_start', 'index': index, 'scales': len(SCALES), 'scale': scale})
        calibration = calibrate_protocol_simulation(
            reference_params, settings, public_matrix, fixture_scale=scale, repeats=10, warmups=2)
        products = calibrate_commitment_products(
            reference_params, settings, public_matrix, fixture_scale=scale, repeats=10, warmups=2)
        calibration['kernels'].update(products['kernels'])
        encoding = calibrate_encoding_work(reference_params, settings, fixture_scale=scale,
                                           repeats=10, warmups=2)
        calibration['kernels'].update(encoding['kernels'])
        auxiliary = calibrate_auxiliary_work(reference_params, settings, fixture_scale=scale,
                                             repeats=10, warmups=2)
        calibration['kernels'].update(auxiliary['kernels'])
        calibration['commitment_product_calibration'] = products
        records_by_scale[scale] = calibration
        if on_progress:
            on_progress({'phase': 'scale_complete', 'index': index, 'scales': len(SCALES), 'scale': scale})
    records = [records_by_scale[scale] for scale in SCALES]
    merged = deepcopy(records[-1])
    for name, sample in merged['kernels'].items():
        axis = _work_axis(name, sample)
        points = []
        for record in records:
            measured = record['kernels'][name]
            if len(measured['samples_ms']) != 10 or measured.get('warmup_samples') != 2:
                raise ValueError(f'{name} lacks ten post-warmup samples')
            points.append({'units_per_sample': measured[axis], 'samples_ms': measured['samples_ms']})
        sample['scaling_fit'] = fit_measurements(points)
        sample['scaling_axis'] = axis
        sample['scale_measurements'] = points
    if on_progress:
        on_progress({'phase': 'fixed_work_start'})
    fixed = calibrate_fixed_work(reference_params, settings, public_matrix, repeats=10, warmups=2)
    for sample in fixed['kernels'].values():
        sample['scaling_fit'] = fit_measurements([sample])
    for sample in fixed['oa_by_depth'].values():
        sample['scaling_fit'] = fit_measurements([sample])
    merged['kernels'].update(fixed['kernels'])
    from .relation_work import calibrate_relation_work
    from .parallel_calibration import calibrate_parallel_schedule

    relation_work = calibrate_relation_work(reference_params, settings, public_matrix,
                                            repeats=10, warmups=2, on_progress=on_progress)
    for sample in relation_work['kernels'].values():
        sample['scaling_fit'] = fit_measurements([sample])
    for group in relation_work['by_depth'].values():
        for sample in group.values():
            sample['scaling_fit'] = fit_measurements([sample])
    merged['kernels'].update(relation_work['kernels'])
    merged['relation_work'] = relation_work
    merged['parallel_schedule'] = calibrate_parallel_schedule(
        reference_params, settings, public_matrix, repeats=10, warmups=2, on_progress=on_progress)
    extend_commitment_calibration(merged)
    if on_progress:
        on_progress({'phase': 'offline_work_start'})
    merged['kernels'].update(calibrate_offline_work(reference_params, settings))
    merged.update(multiscale=True, fixed_work=fixed, rate_source='current_multiscale_measurements',
                  calibration_performed_this_run=True, scales=SCALES,
                  repetitions_per_scale=10, warmups_per_scale=2, scale_order=scale_order)
    if source_before != _source_hashes():
        raise RuntimeError('Sources changed; rerun calibration')
    return merged


def calibrate_offline_work(params, settings):
    import inspect
    import time
    import numpy as np
    from .cost_model import target_dimensions, _summary
    from .relations import _immutable_multiplication_triples
    from .yang import _quadratic_masks, _quadratic_mask_part, _commitment_add_message
    params.validate()
    settings.validate(params.q)
    dims = target_dimensions(params, settings)
    code = inspect.getsource(_quadratic_mask_part) + inspect.getsource(_commitment_add_message)
    implementation = hashlib.sha256(code.encode()).hexdigest()
    records = {key: [] for key in ('offline_quadratic_square', 'online_quadratic_linear',
                                  'online_commitment_message_add', 'joint_mask_reference')}
    for scale in SCALES:
        count = round(1048576 * scale)
        width = count * dims['witness_coordinates'] // dims['quadratic_constraints']
        table = _immutable_multiplication_triples(width,
            ((count-width, 0, 0, 0, True, params.m*params.k),), np.uint32)
        rng = np.random.default_rng(18073)
        r = rng.integers(params.q, size=width, dtype=np.int64)
        x = rng.integers(2, size=width, dtype=np.int64)
        a, b = _quadratic_masks(table, r, x, params.q)
        base = rng.integers(params.q, size=settings.l1 + width, dtype=np.int64)
        expected = base.copy()
        expected[settings.l1:] = (expected[settings.l1:] + x) % params.q
        samples = {key: [] for key in records}
        names = tuple(records)
        for iteration in range(12):
            order = names[iteration % len(names):] + names[:iteration % len(names)]
            for name in order:
                work = base.copy() if name == 'online_commitment_message_add' else None
                started = time.perf_counter_ns()
                if name == 'offline_quadratic_square':
                    actual = _quadratic_mask_part(table, r, params.q)
                elif name == 'online_quadratic_linear':
                    actual = _quadratic_mask_part(table, r, params.q, x)
                elif name == 'online_commitment_message_add':
                    actual = _commitment_add_message(work, x, settings.l1, params.q)
                else:
                    actual = _quadratic_masks(table, r, x, params.q)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if name == 'joint_mask_reference':
                    valid = np.array_equal(actual[0], a) and np.array_equal(actual[1], b)
                else:
                    valid = np.array_equal(actual, {'offline_quadratic_square': b,
                        'online_quadratic_linear': a, 'online_commitment_message_add': expected}[name])
                if not valid:
                    raise AssertionError('Offline algebra mismatch')
                if iteration >= 2:
                    samples[name].append(elapsed)
                del actual, work
        for name in names:
            records[name].append({'units_per_sample': width if name == 'online_commitment_message_add' else count,
                                  'samples_ms': samples[name], 'binary_prefix_count': width})
    return {name: {'kernel': name, 'unit': 'coordinates' if name == 'online_commitment_message_add' else 'constraints',
                   'units_per_sample': points[-1]['units_per_sample'], **_summary(points[-1]['samples_ms']),
                   'warmup_samples': 2, 'sample_count': 10, 'scale_measurements': points,
                   'scaling_fit': fit_measurements(points), 'implementation_sha256': implementation,
                   'rate_kind': 'measured_bounded_one_use_offline_work',
                   'fixture_scope': 'same original q/l1 and native constraint strides; bounded work counts',
                   'matrix_or_full_target_executed': False}
            for name, points in records.items()}


def predict_kernel(sample, units):
    if units == 0:
        return {'estimated_mean_ms': 0.0, 'empirical_scaled_min_ms': 0.0,
                'empirical_scaled_max_ms': 0.0, 'rate_kind': 'zero_work'}
    fit = sample['scaling_fit']
    if sample.get('fixed_work_calls'):
        calls = units / sample['units_per_sample']
        if abs(calls - round(calls)) > 1e-9:
            raise ValueError(f"{sample['kernel']} requires whole fixed-shape calls")
        result = predict_scaling(fit, sample['units_per_sample'])
        for key in ('estimated_mean_ms', 'empirical_scaled_min_ms', 'empirical_scaled_max_ms'):
            result[key] *= calls
        result.update(rate_kind='measured_fixed_shape_call_count', call_count=int(round(calls)))
        return result
    if fit['model_kind'] == 'constant' and units != fit['sampled_max_units']:
        factor = units / sample['units_per_sample']
        return {'estimated_mean_ms': sample['mean_ms'] * factor,
                'empirical_scaled_min_ms': sample['empirical_min_ms'] * factor,
                'empirical_scaled_max_ms': sample['empirical_max_ms'] * factor,
                'rate_kind': 'unvalidated_fixed_shape_proxy', 'extrapolation_ratio': factor}
    result = predict_scaling(fit, units)
    if 'source_calibrations' in sample:
        sensitivity = {name: predict_scaling(record['scaling_fit'], units)
                       for name, record in sample['source_calibrations'].items()}
        result['empirical_scaled_min_ms'] = min(result['empirical_scaled_min_ms'],
            *(value['empirical_scaled_min_ms'] for value in sensitivity.values()))
        result['empirical_scaled_max_ms'] = max(result['empirical_scaled_max_ms'],
            *(value['empirical_scaled_max_ms'] for value in sensitivity.values()))
        result['fit_diagnostics']['source_predictions_ms'] = {
            name: value['estimated_mean_ms'] for name, value in sensitivity.items()}
        result['fit_diagnostics']['selected_calibration_source'] = sample['selected_calibration_source']
    result['rate_kind'] = 'measured_multiscale_extrapolation' if fit['distinct_scales'] > 1 else 'measured_same_size_call'
    return result


def compose_matrix_products(model, calibration):
    from .schedule_model import predict_parallel_schedule

    fields = ('estimated_mean_ms', 'empirical_scaled_min_ms', 'empirical_scaled_max_ms')
    kernels = calibration['kernels']
    l1, l2 = (model['yang_settings'][key] for key in ('l1', 'l2'))
    dimensions = model['dimensions']
    phases = {'parallel_context': 'context', 'parallel_p_first': 'p_first',
              'parallel_p_second': 'p_second', 'parallel_v_equation': 'v_equation'}
    visited = set()

    def total(parts):
        return {key: sum(part[key] for part in parts) for key in fields}

    def product(name, units, calls=1):
        prediction = predict_kernel(kernels[name], units)
        return {**prediction, **{key: prediction[key] * calls for key in fields},
                'kernel': name, 'target_units': units * calls, 'independent_calls': calls}

    def update(part):
        if id(part) in visited:
            return
        visited.add(id(part))
        name = part['kernel']
        if name.startswith('complete_B'):
            role, kind = name.split('_')[1:3]
            width = dimensions['witness_coordinates' if role == 'B1' else 'quadratic_constraints']
            lanes = 2 if kind == 'pair' else 1
            operand = 'pair' if lanes == 2 else 'yang_large'
            pieces = [product('yang_top_' + operand, l1 * (width + l2)),
                      product('yang_bottom_' + operand, width * l2),
                      product('commitment_add_mod', l1, lanes),
                      product('commitment_tail', width, lanes),
                      product('vector_copy', l1, lanes)]
            part.update(reference_complete_helper_estimate_ms=part['estimated_mean_ms'],
                        **total(pieces), calibrated_kernel_estimate_ms=total(pieces)['estimated_mean_ms'],
                        product_components=pieces, rate_kind='conditional_matrix_product_composition',
                        formula='T_top(l1*(width+l2)) + T_bottom(width*l2) + lanes*T_vector_tail(width) + header',
                        matrix_shapes=[[l1, width + l2], [width, l2]], rhs_count=lanes,
                        model_kind='sum_of_separate_matrix_products_and_vector_operations',
                        fit_diagnostics={'target_execution_validated': False,
                                         'full_size_cache_and_bandwidth_validated': False},
                        fused_tail_speedup_assumed=False)
        elif name in phases:
            groups = [part[key] for key in ('branch_a_components', 'branch_b_components', 'post_components')]
            for group in groups:
                for child in group:
                    update(child)
            a, b, post = groups
            prediction = predict_parallel_schedule(calibration['parallel_schedule']['phases'][phases[name]],
                total(a), total(b), total(post), target_work=part['fit_diagnostics']['target_work'],
                branch_components={'a': [child['kernel'] for child in a],
                                   'b': [child['kernel'] for child in b]})
            prediction['fit_diagnostics']['matrix_composition_transfer_validated'] = False
            part.update(prediction)
            part['rate_kind'] = 'conditional_matrix_product_schedule_transfer'

    for module in model['modules'].values():
        for part in module['components']:
            update(part)
        module.update(total(module['components']))
    model['PV_time_model'] = 'measured_individual_matrix_products_with_resident_memory_assumption'
    return model


def _pv_row(model, calibration, depth):
    groups = {'P': ('P_key_binding', 'P_RELATION_compile', 'P_protocol_attempt'),
              'V': ('RELATION_public_compile', 'V_prechallenge', 'V_protocol_response')}
    row = {'N': 1 << depth, 'depth': depth, 'modules': {}}
    for role, names in groups.items():
        components = [part for name in names for part in model['modules'][name]['components']]
        if role == 'P':
            for kernel, calls in (('node_bits', 2 * depth + 2), ('node_equal', depth + 1)):
                sample = calibration['fixed_work']['published_path'][kernel]
                components.append({'kernel': kernel, 'target_units': calls, **predict_kernel(sample, calls)})
        def descendants(parts):
            for part in parts:
                yield part
                for key in ('branch_a_components', 'branch_b_components', 'post_components'):
                    yield from descendants(part.get(key, ()))
        unvalidated = [part['kernel'] for part in descendants(components)
                       if part.get('rate_kind') == 'unvalidated_fixed_shape_proxy']
        if unvalidated:
            raise ValueError(f'Uncalibrated work-size transfers remain in {role}: {unvalidated}')
        row['modules'][role] = {key: sum(part[key] for part in components) for key in
                               ('estimated_mean_ms', 'empirical_scaled_min_ms', 'empirical_scaled_max_ms')}
        row['modules'][role]['components'] = components
    row['stages'] = _pv_stages(model, row)
    row['preprocessed'] = _pv_preprocessed(model, row, calibration)
    return row


def _pv_preprocessed(model, row, calibration):
    fields = ('estimated_mean_ms', 'empirical_scaled_min_ms', 'empirical_scaled_max_ms')
    def total(parts):
        return {key: sum(part[key] for part in parts) for key in fields}
    modules = model['modules']
    public = modules['RELATION_public_compile']['components']
    keys = modules['P_key_binding']['components']
    source_checks = [part for part in keys
                     if part.get('label') == 'complete source-key identity comparison']
    source_ids = {id(part) for part in source_checks}
    key_preparation = [part for part in keys if id(part) not in source_ids]
    contexts = [part for part in modules['P_protocol_attempt']['components']
                if part['kernel'] == 'parallel_context']
    if len(contexts) != 1 or contexts[0]['post_components']:
        raise ValueError('Context dependency layout mismatch')
    context = contexts[0]
    context_hash = context['branch_a_components']
    witness_checks = context['branch_b_components']
    verifier_hash = [part for part in modules['V_prechallenge']['components']
                     if part.get('label', '').startswith('one verifier-constructor context digest')]
    signature = lambda parts: [(part['kernel'], part['target_units'],
                                part['estimated_mean_ms']) for part in parts]
    if not context_hash or signature(context_hash) != signature(verifier_hash):
        raise ValueError('P/V public context mismatch')
    n, m = (model['dimensions'][key] for key in ('n', 'm'))
    snapshot = predict_kernel(calibration['kernels']['immutable_copy'], n*m)
    for key in fields:
        snapshot[key] *= 2
    snapshot.update(kernel='immutable_copy', label='immutable Arec/Aoa proof-context snapshots',
                    target_units=2*n*m, independent_calls=2)
    shared = [snapshot, *public, *context_hash]
    removed_p = {id(part) for part in (*public, *keys, context)}
    removed_v = {id(part) for part in (*public, *verifier_hash)}
    online_p = [part for part in row['modules']['P']['components'] if id(part) not in removed_p]
    online_v = [part for part in row['modules']['V']['components'] if id(part) not in removed_v]
    online_p += [*source_checks, *witness_checks]
    if len(online_v) + len(removed_v) != len(row['modules']['V']['components']):
        raise AssertionError('Verifier preparation coverage mismatch')
    reference_online = {'P': total(online_p), 'V': total(online_v)}
    def public_scan(part):
        return ((part['kernel'] == 'index_range_scan' and part.get('label') == 'public constraint indices')
                or (part['kernel'] == 'range_scan' and part.get('label') == 'public target validation'))
    public_validation = [part for part in online_p if public_scan(part)]
    if len(public_validation) != 2:
        raise ValueError('Dimension check coverage mismatch')
    shared += public_validation
    online_p = [part for part in online_p if not public_scan(part)
                and id(part) not in source_ids and part['kernel'] != 'canonical_matvec_input']
    online_v = [part for part in online_v if not public_scan(part)]
    from .schedule_model import predict_parallel_schedule
    for index, part in enumerate(online_v):
        if part['kernel'] != 'parallel_v_equation':
            continue
        branches = {key: list(part[key]) for key in
                    ('branch_a_components', 'branch_b_components', 'post_components')}
        original_b = branches['branch_b_components']
        branches['branch_b_components'] = [piece for piece in original_b
                                           if piece['kernel'] != 'canonical_matvec_input']
        if len(original_b) - len(branches['branch_b_components']) != 1:
            raise ValueError('Verifier normalization coverage mismatch')
        a, b, post = (branches[key] for key in
                      ('branch_a_components', 'branch_b_components', 'post_components'))
        prediction = predict_parallel_schedule(calibration['parallel_schedule']['phases']['v_equation'],
            total(a), total(b), total(post), target_work=part['fit_diagnostics']['target_work'],
            branch_components={name: [piece['kernel'] for piece in group]
                               for name, group in (('a', a), ('b', b))})
        online_v[index] = {**part, **branches, **prediction}
    scheduling_reference = {'P': total(online_p), 'V': total(online_v)}
    offline_reference = online_p
    online_p, online_v = _pv_payload_schedules(model, calibration, online_p, online_v)
    return {
        'shared_public': {**total(shared), 'components': shared},
        'prover_key': {**total(key_preparation), 'components': key_preparation},
        'P_preprocess': total([*shared, *key_preparation]),
        'V_preprocess': total(shared),
        'P_online': {**total(online_p), 'components': online_p},
        'V_online': {**total(online_v), 'components': online_v},
        'reference_online': reference_online,
        'payload_scheduling_reference': scheduling_reference,
        'payload_scheduling_calibrated': False,
        'one_use_offline': _pv_one_use_offline(model, calibration, offline_reference, online_v),
        'uncredited_native_layout_reuse': True,
        'cache_scope': 'one immutable PROOF_CONTEXT; one resident prover key per runtime',
        'new_statement_requires_public_preparation': True,
        'shared_preparation_counted_once': True,
        'uncalibrated_cache_metadata_and_hash_state_copy': True,
        'full_target_executed': False,
    }


def _pv_one_use_offline(model, calibration, reference_p, online_v):
    kernels = calibration['kernels']
    needed = ('offline_quadratic_square', 'online_quadratic_linear', 'online_commitment_message_add')
    if not all(name in kernels for name in needed):
        return None
    import inspect
    from .yang import _quadratic_mask_part, _commitment_add_message
    code = inspect.getsource(_quadratic_mask_part) + inspect.getsource(_commitment_add_message)
    implementation = hashlib.sha256(code.encode()).hexdigest()
    if any(kernels[name].get('implementation_sha256') != implementation for name in needed):
        raise ValueError('Offline calibration/helper mismatch')
    from .schedule_model import predict_parallel_schedule
    fields = ('estimated_mean_ms', 'empirical_scaled_min_ms', 'empirical_scaled_max_ms')
    def total(parts):
        return {key: sum(part[key] for part in parts) for key in fields}
    def term(name, units, label):
        return {**predict_kernel(kernels[name], units), 'kernel': name, 'target_units': units, 'label': label}
    dims = model['dimensions']
    w, count = dims['witness_coordinates'], dims['quadratic_constraints']
    first = next(part for part in reference_p if part['kernel'] == 'parallel_p_first')
    b = [part for part in first['branch_b_components']
         if part['kernel'].startswith('yang_gaussian_')]
    if len(b) != 2:
        raise ValueError('Both B2 samplers required')
    b += [term('offline_quadratic_square', count, 'witness-independent quadratic mask b')]
    a, post = first['branch_a_components'], first['post_components']
    predicted = predict_parallel_schedule(calibration['parallel_schedule']['phases']['p_first'],
        total(a), total(b), total(post), target_work=first['fit_diagnostics']['target_work'],
        branch_components={'a': [part['kernel'] for part in a], 'b': [part['kernel'] for part in b]})
    predicted['fit_diagnostics'].update(composition_transfer_validated=False,
        witness_independent_square_only=True, paired_zero_message_fixture_measured=False)
    offline_first = {**first, **predicted, 'kernel': 'parallel_offline_first',
                     'branch_a_components': a, 'branch_b_components': b, 'post_components': post}
    moved = [part for part in reference_p if part['kernel'] in
             ('yang_uniform_mask', 'yang_gaussian_small', 'yang_gaussian_large',
              'parallel_p_first', 'parallel_p_second')]
    if len(moved) != 5:
        raise ValueError('Offline matrix or sampler coverage mismatch')
    moved_ids = {id(part) for part in moved}
    offline = [offline_first if part is first else part for part in moved]
    online = [part for part in reference_p if id(part) not in moved_ids]
    online += [term('online_quadratic_linear', count, 'witness-dependent quadratic mask a'),
               term('online_commitment_message_add', w, 'add x to the one-use B1 base'),
               term('online_commitment_message_add', count, 'add a to the one-use B2 base')]
    return {'P_offline': {**total(offline), 'components': offline},
            'P_online': {**total(online), 'components': online},
            'P_per_attempt_total': total([*offline, *online]),
            'V_offline': dict.fromkeys(fields, 0.0),
            'V_online': {**total(online_v), 'components': online_v},
            'per_proof_and_one_use': True, 'private_witness_needed_offline': False,
            'online_timing_scope_changed': True, 'joint_offline_schedule_measured': False,
            'full_target_executed': False}


def _pv_payload_schedules(model, calibration, online_p, online_v):
    from .schedule_model import predict_parallel_schedule
    fields = ('estimated_mean_ms', 'empirical_scaled_min_ms', 'empirical_scaled_max_ms')
    def total(parts):
        return {key: sum(part[key] for part in parts) for key in fields}
    def one(parts, kernel):
        found = [part for part in parts if part['kernel'] == kernel]
        if len(found) != 1:
            raise ValueError(f'payload scheduling requires exactly one {kernel}')
        return found[0]
    def scheduled(part, phase, name, a, b, post):
        prediction = predict_parallel_schedule(calibration['parallel_schedule']['phases'][phase],
            total(a), total(b), total(post), target_work=part['fit_diagnostics']['target_work'],
            branch_components={'a': [piece['kernel'] for piece in a],
                               'b': [piece['kernel'] for piece in b]})
        prediction['fit_diagnostics'].update(
            payload_pipeline_measured=False,
            overlap_coefficient_source=phase,
            composition_transfer_validated=False)
        return {**part, **prediction, 'kernel': name,
                'branch_a_components': a, 'branch_b_components': b, 'post_components': post,
                'rate_kind': 'conditional_payload_pipeline_transfer'}
    dims = model['dimensions']
    prefix_units = dims['relation_rows'] + 2 * (dims['witness_coordinates'] + model['yang_settings']['l1'])
    whole_hash = one(online_p, 'auxiliary_message_hash')
    suffix_units = whole_hash['target_units'] - prefix_units
    if min(prefix_units, suffix_units) <= 0:
        raise ValueError('Incomplete payload vectors')
    def hash_part(units, label):
        return {**predict_kernel(calibration['kernels']['auxiliary_message_hash'], units),
                'kernel': 'auxiliary_message_hash', 'label': label, 'target_units': units,
                'split_hash_fixed_cost_retained_per_segment': True}
    prefix = hash_part(prefix_units, 'framed t,c1,c2 payload prefix')
    suffix = hash_part(suffix_units, 'same-state c3,c4 payload suffix')
    second = one(online_p, 'parallel_p_second')
    pipeline_p = scheduled(second, 'p_second', 'parallel_p_second_payload',
        second['branch_a_components'], [*second['branch_b_components'], prefix],
        [*second['post_components'], suffix])
    online_p = [pipeline_p if part is second else part for part in online_p if part is not whole_hash]
    equation = one(online_v, 'parallel_v_equation')
    payload = [part for part in online_v
               if part['kernel'] in ('auxiliary_snapshot_hash', 'auxiliary_digest_bits', 'auxiliary_binary')
               or part.get('label') in ('auxiliary vector/context/randomness concatenations',
                                       'auxiliary binary validation', 'auxiliary exact safe-block accumulation',
                                       'auxiliary commitment comparison')]
    if sum(part['kernel'] == 'auxiliary_snapshot_hash' for part in payload) != 1:
        raise ValueError('Incomplete verifier payload schedule')
    payload.sort(key=lambda part: part['kernel'] != 'auxiliary_snapshot_hash')
    pipeline_v = scheduled(equation, 'v_equation', 'parallel_v_payload_equation',
        equation['branch_a_components'], [*payload, *equation['branch_b_components']],
        equation['post_components'])
    payload_ids = {id(part) for part in payload}
    online_v = [pipeline_v if part is equation else part
                for part in online_v if id(part) not in payload_ids]
    return online_p, online_v


def print_pv_preprocessed(rows):
    if rows and all(row['preprocessed']['one_use_offline'] is not None for row in rows):
        print('s | estimates; resident RAM; no disk/swap', flush=True)
        print('N          shared      P_key   P_offline    P_online       P_sum    V_online', flush=True)
        for row in rows:
            data = row['preprocessed']
            split = data['one_use_offline']
            parts = [data['shared_public'], data['prover_key'], split['P_offline'],
                     split['P_online'], split['P_per_attempt_total'], split['V_online']]
            print(f"{row['N']:<7}" + ''.join(f"{part['estimated_mean_ms']/1000:12.3f}" for part in parts), flush=True)
        print('P_sum=offline+online; prep once; target unmeasured.', flush=True)
        return
    print('s | estimates | preprocessing / same-statement online calls', flush=True)
    print('N            P_prep     P_online       V_prep     V_online', flush=True)
    for row in rows:
        data = row['preprocessed']
        values = [data[key]['estimated_mean_ms']/1000 for key in
                  ('P_preprocess', 'P_online', 'V_preprocess', 'V_online')]
        print(f"{row['N']:<7}" + ''.join(f'{value:13.3f}' for value in values), flush=True)
    print('V_prep shared once; P_prep adds key.', flush=True)
    print('Conditional overlap; target unmeasured.', flush=True)
    print('Cache overhead uncalibrated.', flush=True)


def _pv_stages(model, row):
    modules = model['modules']
    fields = ('estimated_mean_ms', 'empirical_scaled_min_ms', 'empirical_scaled_max_ms')
    def total(parts):
        return {key: sum(part[key] for part in parts) for key in fields}
    public = modules['RELATION_public_compile']['components']
    compile_parts = modules['P_RELATION_compile']['components']
    if compile_parts[:len(public)] != public:
        raise AssertionError('Compiler prefix mismatch')
    result = {'P': {}, 'V': {}}
    result['P']['key_binding'] = total(modules['P_key_binding']['components'])
    result['P']['public_compile'] = total(public)
    result['P']['private_witness'] = total(compile_parts[len(public):])
    result['V']['public_compile'] = total(public)
    result['V']['prechallenge'] = total(modules['V_prechallenge']['components'])
    def classify(role, parts, predicates):
        remaining = list(parts)
        for label, predicate in predicates:
            selected = [part for part in remaining if predicate(part)]
            identifiers = {id(part) for part in selected}
            remaining = [part for part in remaining if id(part) not in identifiers]
            result[role][label] = total(selected)
        result[role]['other_checks'] = total(remaining)
    classify('P', modules['P_protocol_attempt']['components'], (
        ('context', lambda p: p['kernel'] == 'parallel_context'),
        ('mask_and_gaussian', lambda p: p['kernel'] in ('yang_uniform_mask', 'yang_gaussian_small', 'yang_gaussian_large')),
        ('first_parallel', lambda p: p['kernel'] == 'parallel_p_first'),
        ('second_parallel', lambda p: p['kernel'] == 'parallel_p_second'),
        ('response_rejection', lambda p: p['kernel'] in ('response_and_rejection', 'response_and_rejection_subtract', 'challenge_affine_add')),
        ('auxiliary_commitment', lambda p: p['kernel'].startswith('auxiliary_') or 'auxiliary' in p.get('label', '')),
    ))
    result['P']['path_checks'] = total([p for p in row['modules']['P']['components']
                                       if p['kernel'] in ('node_bits', 'node_equal')])
    classify('V', modules['V_protocol_response']['components'], (
        ('equation_parallel', lambda p: p['kernel'] == 'parallel_v_equation'),
        ('B2_commitment', lambda p: p['kernel'] == 'complete_B2_single_new'),
        ('snapshot_hash', lambda p: p['kernel'] == 'auxiliary_snapshot_hash'),
    ))
    for role, stages in result.items():
        for key in fields:
            expected = row['modules'][role][key]
            actual = sum(stage[key] for stage in stages.values())
            if abs(expected - actual) > 1e-8 * max(1.0, abs(expected)):
                raise AssertionError('stage accounting changed the module total')
    return result


def estimate_populations(params, settings, calibration):
    from .cost_model import estimate_modules
    from .params import PAPER_GROUP_DEPTHS

    rows = []
    for depth in PAPER_GROUP_DEPTHS:
        current = dict(calibration)
        current['kernels'] = dict(calibration['kernels'])
        current['kernels']['oa_matvec'] = calibration['fixed_work']['oa_by_depth'][depth]
        if 'relation_work' in calibration:
            for kind, sample in calibration['relation_work']['by_depth'][depth].items():
                current['kernels']['relation_non_F_' + kind] = sample
        model = estimate_modules(replace(params, depth=depth), settings, current)
        compose_matrix_products(model, current)
        row = _pv_row(model, current, depth)
        from .memory_model import assess_pv_memory, annotate_conditional_runtime
        annotate_conditional_runtime(row, assess_pv_memory(replace(params, depth=depth), settings))
        rows.append(row)
    return rows


def run_pv_sweep():
    from .params import load_selected_profile
    from .randomness import CryptographicRNG
    from .mathutils import rand_mod_matrix

    params, settings, _ = load_selected_profile()
    public_matrix = rand_mod_matrix(CryptographicRNG(), params.n, params.m, params.q)
    def progress(event):
        if event['phase'] == 'scale_start':
            print(f"Calibration {event['index']}/{event['scales']} | warmup=2, samples=10", flush=True)
        elif event['phase'] == 'relation_work':
            print(f"M calibration N={event['N']} | warmup=2, samples=10", flush=True)
        elif event['phase'].startswith('parallel') and event.get('index') in (1, event.get('shapes')):
            print(f"Schedule {event['stage']} {event['index']}/{event['shapes']} | warmup=2, samples=10", flush=True)
    calibration = calibrate_multiscale(params, settings, public_matrix, on_progress=progress)
    rows = estimate_populations(params, settings, calibration)
    print_pv_preprocessed(rows)
    return rows, calibration


if __name__ == '__main__':
    run_pv_sweep()
