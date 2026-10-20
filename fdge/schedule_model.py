from __future__ import annotations

import json
import math

import numpy as np


_TIME_KEYS = ('estimated_mean_ms', 'empirical_scaled_min_ms', 'empirical_scaled_max_ms')


def _numbers(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size != 10 or not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f'{name} requires ten finite nonnegative measurements')
    return result


def _shape(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def _branch_fit(points, name):
    isolated = np.asarray([point[name + '_ms'] for point in points])
    concurrent = np.asarray([point['concurrent_' + name + '_mean_ms'] for point in points])
    denominator = float(isolated @ isolated)
    numerator = float(isolated @ concurrent)
    if denominator <= 0 or numerator <= 0:
        raise ValueError('Positive concurrent branch times required')
    factor = numerator / denominator
    observed = np.concatenate([
        np.asarray(point['concurrent_' + name + '_ms']) / np.asarray(point['branch_' + name + '_ms'])
        for point in points])
    residual = concurrent - factor * isolated
    return {'factor': factor, 'empirical_min_factor': min(factor, float(np.min(observed))),
            'empirical_max_factor': max(factor, float(np.max(observed))),
            'fit_kind': 'through_origin_shape_means_standalone_to_concurrent',
            'sum_isolated_squared_ms2': denominator,
            'sum_isolated_concurrent_ms2': numerator,
            'shape_count': len(points), 'sample_count': 10 * len(points),
            'max_isolated_mean_ms': float(np.max(isolated)),
            'max_concurrent_mean_ms': float(np.max(concurrent)),
            'shape_mean_rmse_ms': float(np.sqrt(np.mean(residual * residual))),
            'repeat_ratios_are_unpaired_mode_spread': True}


def _fit(points):
    residual = np.concatenate([point['dispatch_residual_samples_ms'] for point in points])
    raw_mean = float(np.mean(residual))
    central = max(0.0, raw_mean)
    overlap = np.asarray([min(point['a_ms'], point['b_ms']) for point in points])
    excess = np.asarray([point['joint_mean_ms'] - point['post_mean_ms']
                         - max(point['a_ms'], point['b_ms']) - central for point in points])
    denominator = float(overlap @ overlap)
    if denominator <= 0:
        raise ValueError('Both branches require positive work')
    coefficient = max(0.0, float(overlap @ excess) / denominator)
    observed = np.concatenate([
        (np.asarray(point['joint_ms']) - np.asarray(point['post_ms']) - central
         - np.maximum(point['branch_a_ms'], point['branch_b_ms']))
        / np.minimum(point['branch_a_ms'], point['branch_b_ms'])
        for point in points])
    return {'model_kind': 'measured_overlap_local_schedule',
            'branch_a': _branch_fit(points, 'a'), 'branch_b': _branch_fit(points, 'b'),
            'overlap': {'coefficient': coefficient,
                        'empirical_min_coefficient': max(0.0, min(coefficient, float(np.min(observed)))),
                        'empirical_max_coefficient': max(coefficient, float(np.max(observed))),
                        'raw_min_coefficient': float(np.min(observed)),
                        'raw_max_coefficient': float(np.max(observed)),
                        'fit_kind': 'nonnegative_excess_joint_time_per_standalone_overlap',
                        'sum_overlap_squared_ms2': denominator,
                        'shape_mean_rmse_ms': float(np.sqrt(np.mean((excess - coefficient * overlap) ** 2))),
                        'shape_count': len(points), 'sample_count': 10 * len(points),
                        'repeat_ratios_are_unpaired_mode_spread': True},
            'dispatch': {'estimated_mean_ms': central,
                         'empirical_scaled_min_ms': max(0.0, float(np.min(residual))),
                         'empirical_scaled_max_ms': max(central, float(np.max(residual))),
                         'raw_mean_ms': raw_mean,
                         'raw_min_ms': float(np.min(residual)),
                         'raw_max_ms': float(np.max(residual)),
                         'negative_residual_samples': int(np.count_nonzero(residual < 0)),
                         'sample_count': int(residual.size)}}


def _at(fit, a_ms, b_ms, *, endpoint='mean'):
    fields = {'mean': ('coefficient', 'estimated_mean_ms'),
              'lower': ('empirical_min_coefficient', 'empirical_scaled_min_ms'),
              'upper': ('empirical_max_coefficient', 'empirical_scaled_max_ms')}
    coefficient, dispatch = fields[endpoint]
    return (max(a_ms, b_ms) + fit['overlap'][coefficient] * min(a_ms, b_ms)
            + fit['dispatch'][dispatch])


def _relative_error(actual, predicted):
    return abs(actual - predicted) / max(abs(actual), abs(predicted), 1e-12)


def _work_design(points):
    axes = sorted({key for point in points for key in point['work']})
    maxima = {key: max(point['work'].get(key, 0.0) for point in points) for key in axes}
    varying = [key for key in axes if min(point['work'].get(key, 0.0) for point in points) != maxima[key]]
    design = np.asarray([[1.0, *(point['work'].get(key, 0.0) / maxima[key] for key in varying)]
                         for point in points], dtype=np.float64)
    rank = int(np.linalg.matrix_rank(design))
    expected = 1 + len(varying)
    largest = max(points, key=(
        (lambda point: sum(point['work'].get(key, 0.0) / maxima[key] for key in varying))
        if varying else (lambda point: point['a_ms'] + point['b_ms'])))
    return {'work_design_rank': rank, 'work_design_expected_rank': expected,
            'work_design_varying_axes': varying, 'work_design_full_rank': rank == expected,
            'work_design_condition_number': float(np.linalg.cond(design)) if rank == expected else None,
            'independent_design_identified': rank == expected and rank >= 3,
            'largest_shape': largest['shape'],
            'largest_shape_selection': 'sum_of_normalized_varying_work_axes' if varying
                else 'sum_of_measured_branch_times'}


def fit_parallel_schedule(records):
    if not records:
        raise ValueError('Measured schedule shapes required')
    points = []
    seen = set()
    for record in records:
        shape = _shape(record['shape'])
        if shape in seen:
            raise ValueError('Parallel calibration shapes must be distinct')
        seen.add(shape)
        values = {name: _numbers(record[name], name) for name in
                  ('branch_a_ms', 'branch_b_ms', 'concurrent_a_ms', 'concurrent_b_ms', 'joint_ms')}
        if any(np.any(values[name] <= 0) for name in
               ('branch_a_ms', 'branch_b_ms', 'concurrent_a_ms', 'concurrent_b_ms')):
            raise ValueError('Positive measured branch work required')
        values['post_ms'] = _numbers(record.get('post_ms', [0.0] * 10), 'post_ms')
        for name in ('worker_start_offset_ms', 'main_start_offset_ms',
                     'submission_offset_ms', 'wait_end_offset_ms'):
            if name in record:
                values[name] = _numbers(record[name], name)
        residual = values['joint_ms'] - np.maximum(values['concurrent_a_ms'], values['concurrent_b_ms'])
        residual -= values['post_ms']
        work = {str(key): float(value) for key, value in record.get('work', {}).items()}
        if any(not math.isfinite(value) or value < 0 for value in work.values()):
            raise ValueError('Parallel work must be finite and >= 0')
        points.append({'shape': record['shape'], 'work': work,
                       'a_ms': float(np.mean(values['branch_a_ms'])),
                       'b_ms': float(np.mean(values['branch_b_ms'])),
                       'concurrent_a_mean_ms': float(np.mean(values['concurrent_a_ms'])),
                       'concurrent_b_mean_ms': float(np.mean(values['concurrent_b_ms'])),
                       'joint_mean_ms': float(np.mean(values['joint_ms'])),
                       'post_mean_ms': float(np.mean(values['post_ms'])),
                       'sample_count': len(values['joint_ms']),
                       'dispatch_residual_samples_ms': residual.tolist(),
                       **{name: array.tolist() for name, array in values.items()}})
    central = _fit(points)
    holdouts, alternate_fits = [], []
    for index, point in enumerate(points):
        training = points[:index] + points[index + 1:]
        if not training:
            continue
        fitted = _fit(training)
        alternate_fits.append(fitted)
        predicted = _at(fitted, point['a_ms'], point['b_ms']) + point['post_mean_ms']
        actual = point['joint_mean_ms']
        branch_checks = {}
        for name in ('a', 'b'):
            branch_predicted = fitted['branch_' + name]['factor'] * point[name + '_ms']
            branch_actual = point['concurrent_' + name + '_mean_ms']
            branch_checks[name] = {'actual_ms': branch_actual, 'predicted_ms': branch_predicted,
                                   'relative_error': _relative_error(branch_actual, branch_predicted)}
        holdouts.append({'shape': point['shape'], 'actual_ms': actual, 'predicted_ms': predicted,
                         'relative_error': _relative_error(actual, predicted),
                         'branch_errors': branch_checks,
                         'residual_per_critical_ms': abs(actual - predicted) / max(point['a_ms'], point['b_ms']),
                         'residual_per_overlap_ms': abs(actual - predicted) / min(point['a_ms'], point['b_ms']),
                         'target_quantity': 'complete_joint_helper'})
    design = _work_design(points)
    largest_holdout = next((item for item in holdouts if _shape(item['shape']) == _shape(design['largest_shape'])), None)
    errors = [item['relative_error'] for item in holdouts]
    balances = [min(point['a_ms'], point['b_ms']) / max(point['a_ms'], point['b_ms']) for point in points]
    branch_max_errors = {name: max((item['branch_errors'][name]['relative_error'] for item in holdouts), default=None)
                         for name in ('a', 'b')}
    return {**central, 'alternate_models': alternate_fits, 'points': points,
            'sampled_max_branch_a_ms': max(point['a_ms'] for point in points),
            'sampled_max_branch_b_ms': max(point['b_ms'] for point in points),
            'sampled_max_critical_ms': max(max(point['a_ms'], point['b_ms']) for point in points),
            'sampled_balance_min': min(balances), 'sampled_balance_max': max(balances),
            'holdout_residual_per_critical_ms': max(
                (item['residual_per_critical_ms'] for item in holdouts), default=0.0),
            'holdout_residual_per_overlap_ms': max(
                (item['residual_per_overlap_ms'] for item in holdouts), default=0.0),
            'sampled_work_ranges': {key: [min(point['work'].get(key, 0.0) for point in points),
                                          max(point['work'].get(key, 0.0) for point in points)]
                                    for key in sorted({key for point in points for key in point['work']})},
            'sample_count': 10 * len(points), 'distinct_shapes': len(points),
            'uncertainty_kind': 'empirical_model_spread_not_confidence_interval',
            'diagnostics': {**design,
                            'leave_one_shape_out_max_relative_error': max(errors) if errors else None,
                            'leave_one_shape_out_mean_relative_error': float(np.mean(errors)) if errors else None,
                            'leave_largest_out_relative_error': largest_holdout['relative_error']
                                if largest_holdout else None,
                            'leave_one_shape_out_branch_max_relative_error': branch_max_errors,
                            'held_out_shapes': holdouts,
                            'concurrent_branch_measurements_required': True,
                            'branch_a_factor': central['branch_a']['factor'],
                            'branch_b_factor': central['branch_b']['factor'],
                            'whole_branch_factors_used_for_prediction': False,
                            'overlap_coefficient': central['overlap']['coefficient'],
                            'nonoverlap_tail_slowdown_applied': False,
                            'negative_dispatch_residual_samples': central['dispatch']['negative_residual_samples'],
                            'dispatch_raw_mean_ms': central['dispatch']['raw_mean_ms'],
                            'dispatch_offset_transfer': 'measured_fixed_overhead',
                            'full_target_executed': False}}


def _prediction_values(prediction):
    values = tuple(float(prediction[key]) for key in _TIME_KEYS)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Branch times must be finite and >= 0')
    mean, lower, upper = values
    if lower > mean or upper < mean:
        raise ValueError('Branch prediction bounds must contain the mean')
    return values


def predict_parallel_schedule(model, a, b, post=None, *, target_work=None, branch_components=None):
    a_mean, a_lower, a_upper = _prediction_values(a)
    b_mean, b_lower, b_upper = _prediction_values(b)
    post_mean, post_lower, post_upper = _prediction_values(
        post if post is not None else dict.fromkeys(_TIME_KEYS, 0.0))
    central = _at(model, a_mean, b_mean)
    fits = [model, *model['alternate_models']]
    lower_candidates = [_at(fit, a_lower, b_lower, endpoint='lower') for fit in fits]
    upper_candidates = [_at(fit, a_upper, b_upper, endpoint='upper') for fit in fits]
    margin = model['holdout_residual_per_overlap_ms'] * min(a_upper, b_upper)
    ideal_overlap = max(a_lower, b_lower)
    serial_sensitivity = a_upper + b_upper + model['dispatch']['empirical_scaled_max_ms']
    lower = max(0.0, min(lower_candidates + [central, ideal_overlap]) - margin)
    upper = max(upper_candidates + [central, serial_sensitivity]) + margin
    critical = max(a_mean, b_mean)
    balance = min(a_mean, b_mean) / critical if critical else 1.0
    ratios = {'branch_a': a_mean / model['sampled_max_branch_a_ms'],
              'branch_b': b_mean / model['sampled_max_branch_b_ms'],
              'critical_path': critical / model['sampled_max_critical_ms']}
    diagnostics = dict(model['diagnostics'])
    target = {} if target_work is None else {str(key): float(value) for key, value in target_work.items()}
    if any(not math.isfinite(value) or value < 0 for value in target.values()):
        raise ValueError('Target work must be finite and >= 0')
    ranges = model['sampled_work_ranges']
    missing = sorted(set(ranges) - set(target))
    outside = {key: {'target': value, 'sampled_range': ranges[key]}
               for key, value in target.items() if key in ranges
               and not ranges[key][0] <= value <= ranges[key][1]}
    diagnostics.update(target_branch_balance=balance,
                       sampled_branch_balance=[model['sampled_balance_min'], model['sampled_balance_max']],
                       branch_balance_extrapolation=not model['sampled_balance_min'] <= balance <= model['sampled_balance_max'],
                       extrapolation_ratios=ratios,
                       target_standalone_branch_a_ms=a_mean,
                       target_standalone_branch_b_ms=b_mean,
                       target_nonoverlap_tail_ms=abs(a_mean - b_mean),
                       target_overlap_penalty_ms=model['overlap']['coefficient'] * min(a_mean, b_mean),
                       target_work=target, sampled_work_ranges=ranges,
                       target_work_outside_sampled_ranges=outside,
                       missing_target_work_axes=missing,
                       target_work_extrapolation=target_work is None or bool(missing or outside),
                       composition_transfer_validated=False,
                       branch_components=branch_components,
                       conditional_transfer=True,
                       envelope_kind='empirical_and_execution_sensitivity_not_certified_bounds')
    return {'estimated_mean_ms': central + post_mean,
            'empirical_scaled_min_ms': lower + post_lower,
            'empirical_scaled_max_ms': upper + post_upper,
            'rate_kind': 'conditional_overlap_local_schedule_estimate',
            'model_kind': model['model_kind'], 'fit_diagnostics': diagnostics,
            'extrapolation_ratio': max(ratios.values()),
            'uncertainty_kind': model['uncertainty_kind'],
            'ideal_overlap_sensitivity_ms': ideal_overlap + post_lower,
            'serial_sensitivity_ms': serial_sensitivity + post_upper,
            'conditional_transfer': True,
            'branch_predictions': {'a': dict(a), 'b': dict(b),
                                   'post': dict(post) if post is not None else dict.fromkeys(_TIME_KEYS, 0.0)}}
