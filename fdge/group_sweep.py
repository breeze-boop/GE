from __future__ import annotations

import argparse
import sys

from .params import load_selected_profile


def _independent_policy_check(patterns, message, d):
    t = patterns.shape[1]
    mask = (1 << t) - 1
    encoded = [sum(int(bit) << j for j, bit in enumerate(row)) for row in patterns]
    window = sum(int(bit) << j for j, bit in enumerate(message[:t]))
    for offset in range(len(message) - t + 1):
        if any((window ^ pattern).bit_count() < d for pattern in encoded):
            return False
        if offset + t < len(message):
            window = ((window >> 1) | (int(message[offset + t]) << (t - 1))) & mask
    return True


def _print_consistency_warning(report):
    if report.get('run_status') == 'invalidated_by_source_change':
        changed = ', '.join(report.get('provenance_changed_files', []))
        print(f'Sources changed; rerun: {changed}',
              file=sys.stderr, flush=True)


def _print_module(row):
    value, status = row['mean_ms'], row['status']
    if value is None:
        cell = 'N/A (memory capacity)' if status == 'capacity_infeasible' else 'N/A (unvalidated)'
    elif status.startswith('estimated_') or '_plus_' in status:
        cell = f'≈{value:,.1f}'
    else:
        cell = f'{value:.3f}'
    label = 'JOIN-ISSUE' if row['module'] == 'JOIN_ISSUE' else row['module']
    print(f'  {label:<16}: {cell} ms', flush=True)


def _print_header():
    print('ms | mean; ≈ estimates; JOIN/OPEN core; fixture inputs', flush=True)
    print('GM/DEC shared across N; UKGEN = measured GPV + estimated F.', flush=True)
    print('Init warm/no fingerprints; P first context; V response; relation/key prep excluded.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='FDGE timings')
    pv_modes = parser.add_mutually_exclusive_group()
    pv_modes.add_argument('--pv-only', action='store_true',
                          help='print Init/P/V estimates')
    pv_modes.add_argument('--pv-calibrate', action='store_true',
                          help='calibrate P/V kernels and schedules')
    parser.add_argument('--estimate-open', action='store_true',
                        help='OPEN round calibration (untimed)')
    parser.add_argument('--repeats', type=int, default=None,
                        help='samples (default: 10)')
    parser.add_argument('--warmups', type=int, default=None,
                        help='warmups (default: 2)')
    args = parser.parse_args(argv)
    if args.pv_only:
        from .pv_estimate import run_pv_estimate
        run_pv_estimate()
        return
    if args.pv_calibrate:
        from .multiscale import run_pv_sweep
        run_pv_sweep()
        return
    if args.repeats is not None and args.repeats < 1:
        parser.error('--repeats must be at least 1')
    if args.warmups is not None and args.warmups < 0:
        parser.error('--warmups must be nonnegative')
    params, settings, _ = load_selected_profile()
    from .selected_experiment import run_selected_sweep

    active_N, printed = None, set()

    def population(N, depth=None):
        nonlocal active_N
        if active_N != N:
            active_N = N
            printed.clear()
            print(f'\nN = {N} | log2N = {N.bit_length() - 1 if depth is None else depth}', flush=True)

    def progress(event):
        phase = event['phase']
        if phase == 'shared_public_matrices':
            print('Calibration: public matrices', flush=True)
        elif phase == 'actual_full_GPV_core':
            mode = 'warmup' if event['warmup'] else 'sample'
            print(f"UKGEN core {event['trial']}/{event['trials']} ({mode})", flush=True)
        elif phase == 'scale_start':
            print(f"Calibration {event['index']}/{event['scales']}", flush=True)
        elif phase == 'fixed_work_start':
            print('Calibration: fixed work', flush=True)
        elif phase == 'relation_work':
            print(f"Calibration: relation N={event['N']}", flush=True)
        elif phase == 'parallel_schedule' and event['index'] in (1, event['shapes']):
            print(f"Calibration: {event['stage']} {event['index']}/{event['shapes']}", flush=True)
        elif phase == 'offline_work_start':
            print('Calibration: offline work', flush=True)
        elif phase == 'population_started':
            population(event['N'], event['depth'])
        elif phase == 'opening_proof_capacity':
            print('  OPEN proof: round estimate (memory limit)', flush=True)
        elif phase == 'actual_core_modules':
            population(event['N'])
            mode = 'warmup' if event['warmup'] else 'sample'
            print(f"  {mode} {event['index']}/{event['count']}", flush=True)
        elif phase == 'module_complete':
            population(event['N'])
            row = event['row']
            if row['module'] not in printed:
                _print_module(row)
                printed.add(row['module'])
        elif phase == 'full_OPEN_started' and event['trial'] in (1, event['trials']):
            print(f"  OPEN proof {event['trial']}/{event['trials']}", flush=True)
        elif phase == 'opening_proof_round_calibration':
            print('  OPEN proof calibration', flush=True)
        elif phase == 'protocol_step_simulation':
            print('  Estimates: Init, UKGEN, P, V', flush=True)
        elif phase == 'population_complete':
            pop = event['population']
            population(pop['N'], pop.get('depth'))
            for row in pop['rows']:
                if row['module'] not in printed:
                    _print_module(row)
                    printed.add(row['module'])

    _print_header()
    report = run_selected_sweep(params, settings,
                                repeats=10 if args.repeats is None else args.repeats,
                                warmups=2 if args.warmups is None else args.warmups,
                                estimate_open=args.estimate_open,
                                on_progress=progress)
    _print_consistency_warning(report)


if __name__ == '__main__':
    main()
