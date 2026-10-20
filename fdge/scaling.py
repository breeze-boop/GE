import math
import statistics


def _number(value, name, *, positive=False):
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return result


def _affine(points, field='mean_ms'):
    scale = max(point['units_per_sample'] for point in points)
    xs = [point['units_per_sample'] / scale for point in points]
    ys = [point[field] for point in points]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    variance = sum((x - x_mean) ** 2 for x in xs)
    candidates = [(max(0.0, y_mean), 0.0)]
    sum_x_squared = sum(x * x for x in xs)
    candidates.append((0.0, max(0.0, sum(x * y for x, y in zip(xs, ys)) / sum_x_squared)))
    if variance > 0:
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / variance
        intercept = y_mean - slope * x_mean
        if intercept >= 0 and slope >= 0:
            candidates.append((intercept, slope))
    intercept, normalized_slope = min(
        candidates,
        key=lambda pair: sum((y - pair[0] - pair[1] * x) ** 2 for x, y in zip(xs, ys)),
    )
    return {
        'intercept_ms': intercept,
        'slope_ms_per_unit': normalized_slope / scale,
        'normalization_units': scale,
        'fit_min_units': min(point['units_per_sample'] for point in points),
        'fit_max_units': scale,
    }


def _at(model, units):
    return max(0.0, model['intercept_ms'] + model['slope_ms_per_unit'] * units)


def _relative_error(actual, predicted):
    denominator = max(abs(actual), abs(predicted), 1e-12)
    return abs(actual - predicted) / denominator


def _squared_error(model, points):
    return sum((point['mean_ms'] - _at(model, point['units_per_sample'])) ** 2 for point in points)


def fit_measurements(points):
    points = list(points)
    sources = {point.get('source', 'unspecified') for point in points}
    if len(sources) > 1:
        raise ValueError('Fit each calibration source separately')
    grouped = {}
    for point in points:
        units = _number(point['units_per_sample'], 'units_per_sample', positive=True)
        samples = [_number(value, 'sample time') for value in point['samples_ms']]
        if not samples:
            raise ValueError('Each scale requires at least one measured sample')
        grouped.setdefault(units, []).extend(samples)
    if not grouped:
        raise ValueError('At least one measured scale is required')
    summary = []
    for units, samples in sorted(grouped.items()):
        summary.append({
            'units_per_sample': units,
            'mean_ms': statistics.mean(samples),
            'std_ms': statistics.stdev(samples) if len(samples) > 1 else 0.0,
            'empirical_min_ms': min(samples),
            'empirical_max_ms': max(samples),
            'sample_count': len(samples),
        })
    sample_min = summary[0]['units_per_sample']
    sample_max = summary[-1]['units_per_sample']
    common = {
        'sampled_min_units': sample_min,
        'sampled_max_units': sample_max,
        'distinct_scales': len(summary),
        'sample_count': sum(point['sample_count'] for point in summary),
        'points': summary,
        'calibration_source': next(iter(sources)),
        'fit_scope': 'all_distinct_scales_from_one_source',
        'uncertainty_kind': 'empirical_model_spread_not_confidence_interval',
    }
    if len(summary) == 1:
        return {
            **common,
            'model_kind': 'constant',
            'intercept_ms': summary[0]['mean_ms'],
            'slope_ms_per_unit': 0.0,
            'diagnostics': {
                'leave_largest_out_relative_error': None,
                'leave_one_scale_out_max_relative_error': None,
                'leave_one_scale_out': [],
                'positive_marginal_work_identified': False,
                'scaling_identification': 'one_distinct_scale',
            },
        }
    full = _affine(summary)
    models = {'all_scales': full}
    holdouts = []
    if len(summary) >= 3:
        for index, omitted in enumerate(summary):
            retained = summary[:index] + summary[index + 1:]
            model = _affine(retained)
            name = 'leave_largest_out' if index == len(summary) - 1 else f'leave_scale_{index}_out'
            models[name] = model
            predicted = _at(model, omitted['units_per_sample'])
            holdouts.append({
                'excluded_units': omitted['units_per_sample'],
                'actual_mean_ms': omitted['mean_ms'],
                'predicted_mean_ms': predicted,
                'relative_error': _relative_error(omitted['mean_ms'], predicted),
                'model': model,
            })
    slopes = [row['model']['slope_ms_per_unit'] for row in holdouts]
    noise_relative = max(
        point['std_ms'] / max(point['mean_ms'], 1e-12) for point in summary)
    lower = _affine(summary, 'empirical_min_ms')
    upper = _affine(summary, 'empirical_max_ms')
    residual = max(abs(point['mean_ms'] - _at(full, point['units_per_sample']))
                   for point in summary)
    return {
        **common,
        'model_kind': 'affine',
        **full,
        'alternate_models': models,
        'empirical_lower_model': lower,
        'empirical_upper_model': upper,
        'max_fit_residual_ms': residual,
        'diagnostics': {
            'leave_largest_out_relative_error': holdouts[-1]['relative_error'] if holdouts else None,
            'leave_one_scale_out_max_relative_error': max(
                (row['relative_error'] for row in holdouts), default=None),
            'leave_one_scale_out': holdouts,
            'leave_one_scale_out_slope_min_ms_per_unit': min(slopes) if slopes else None,
            'leave_one_scale_out_slope_max_ms_per_unit': max(slopes) if slopes else None,
            'holdout_predictor': 'all_retained_scales_nonnegative_affine',
            'max_sample_coefficient_of_variation': noise_relative,
            'positive_marginal_work_identified': full['slope_ms_per_unit'] > 0,
            'positive_marginal_work_stable_under_scale_omission': (
                full['slope_ms_per_unit'] > 0 and all(slope > 0 for slope in slopes) if slopes else None),
            'scaling_identification': 'positive_fitted_slope' if full['slope_ms_per_unit'] > 0 else 'zero_slope_boundary',
            'all_scale_rmse_ms': math.sqrt(_squared_error(full, summary) / len(summary)),
        },
    }


def predict_scaling(fit, units):
    target = _number(units, 'target units', positive=True)
    largest = fit['sampled_max_units']
    ratio = target / largest
    transfer = {
        'prediction_support': ('outside_measured_range' if target > largest else
                               'below_measured_range' if target < fit['sampled_min_units'] else
                               'within_measured_range'),
        'target_execution_validated': False,
        'fixed_cost_repeated': False,
        'calibration_source': fit['calibration_source'],
        'fit_scope': fit['fit_scope'],
        'leave_one_scale_out_target_min_ms': None,
        'leave_one_scale_out_target_max_ms': None,
    }
    if fit['model_kind'] == 'constant':
        if not math.isclose(target, largest, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError('Multiple work sizes required for scaling')
        point = fit['points'][0]
        mean = point['mean_ms']
        lower = point['empirical_min_ms']
        upper = point['empirical_max_ms']
    else:
        mean = _at(fit, target)
        omitted_predictions = [
            {'excluded_units': row['excluded_units'], 'estimated_mean_ms': _at(row['model'], target)}
            for row in fit['diagnostics']['leave_one_scale_out']
        ]
        omitted_values = [row['estimated_mean_ms'] for row in omitted_predictions]
        transfer.update({
            'leave_one_scale_out_target_predictions_ms': omitted_predictions,
            'leave_one_scale_out_target_min_ms': min(omitted_values) if omitted_values else None,
            'leave_one_scale_out_target_max_ms': max(omitted_values) if omitted_values else None,
            'leave_one_scale_out_target_max_relative_difference': max(
                (_relative_error(mean, value) for value in omitted_values), default=None),
            'transfer_assumption': 'same execution branch with affine marginal work beyond sampled scales',
        })
        alternatives = [_at(model, target) for model in fit['alternate_models'].values()]
        alternatives.extend((
            _at(fit['empirical_lower_model'], target),
            _at(fit['empirical_upper_model'], target),
            mean,
        ))
        residual_margin = fit['max_fit_residual_ms']
        lower = max(0.0, min(alternatives) - residual_margin)
        upper = max(alternatives) + residual_margin
    return {
        'estimated_mean_ms': mean,
        'empirical_scaled_min_ms': lower,
        'empirical_scaled_max_ms': upper,
        'extrapolation_ratio': ratio,
        'target_units': target,
        'model_kind': fit['model_kind'],
        'uncertainty_kind': fit['uncertainty_kind'],
        'prediction_support': transfer['prediction_support'],
        'fit_diagnostics': {**fit['diagnostics'], **transfer},
    }
