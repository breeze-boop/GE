
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import time
import traceback
import zipfile


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def private_pickle(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    path.chmod(0o600)


def source_files():
    root = Path(__file__).resolve().parent
    return [Path(__file__).resolve(), *sorted((root / 'lge_qsmf_repro').glob('*.py')),
            *sorted((root / 'lge_qsmf_repro').glob('*.c'))]


def source_hashes():
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files()}


def run(args):
    # Set before importing NumPy or any protocol module, including in spawned workers.
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['LATGE_PROOF_WORKERS'] = '1'
    from lge_qsmf_repro import scheme as s
    from lge_qsmf_repro.params import table_iii_experiment_params
    from lge_qsmf_repro.group_users_population import build_population
    from lge_qsmf_repro.group_users_timing import run_measurements

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = output / 'manifest.json'
    profile = table_iii_experiment_params()
    current_sources = source_hashes()
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        if manifest['source_before'] != current_sources:
            raise RuntimeError('Sources changed since this experiment began; refusing mixed-source resume')
        if manifest['configuration']['max_users'] != args.max_users:
            raise ValueError('Cannot change target population during resume')
        if manifest['configuration']['population_workers'] != args.workers:
            raise ValueError('Cannot change population worker count during resume')
        # Only load the local checkpoint produced by this experiment.
        with (output / 'setup-state.pkl').open('rb') as stream:
            state = pickle.load(stream)
        if state['pp'].params != profile:
            raise ValueError('Resumed public parameters do not match the fixed Table III profile')
        manifest['resume_events'].append(datetime.now(timezone.utc).isoformat())
    else:
        manifest = {
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'status': 'initializing', 'resume_events': [],
            'parameters': asdict(profile), 'source_before': current_sources,
            'configuration': {
                'max_users': args.max_users, 'group_sizes': [2**i for i in range(4, 11)
                                                          if 2**i <= args.max_users],
                'independent_repetitions_per_size': 1,
                'population_workers': args.workers,
                'population_proof_workers_per_process': 1,
                'timing_proof_workers': 8, 'blas_threads': 1,
                'recipient_selection': 'first preassigned candidate; never select by timing, hint, or outcome',
                'outer_retries': 0,
            },
            'runtime': {'python': sys.version, 'executable': sys.executable,
                        'platform': platform.platform(), 'machine': platform.machine(),
                        'cpu_count': os.cpu_count()},
            'method': [
                'Construct a real nested population through UKGEN, JOIN, ISSUE, acceptance and REGISTER.',
                'Assign member IDs before random draws; serialize registration in that order.',
                'Population throughput is excluded from algorithm latency measurements.',
                'Terminate population workers before timing; each size receives one fresh measurement per module.',
                'SETUPinit generates fresh public parameters; SETUPGM/OA generate fresh keys on the fixed pool pp. These outputs do not replace the population group.',
                'ISSUE tests an additional candidate against N existing users; OPEN uses the original N users.',
                'No trend smoothing, extrapolation, copied baseline rows or outcome-based selection.',
                'The paper reports 30-run means; this experiment initially reports one raw run per size.',
                'Exact supplementary distribution choices are the recorded implementation profile.',
            ],
        }
        project_root = Path(__file__).resolve().parent.parent
        manifest['papers'] = {name: hashlib.sha256((project_root / name).read_bytes()).hexdigest()
                              for name in ('Main.pdf', '2020-1183.pdf')}
        with zipfile.ZipFile(output / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in source_files():
                archive.write(path, path.relative_to(Path(__file__).resolve().parent))
        atomic_json(manifest_path, manifest)
        begin = time.perf_counter()
        try:
            pp = s.setup_init(profile.security_bits)
            if pp.params != profile:
                raise ValueError('Public SETUPinit returned an unexpected profile')
            gm, oa = s.setup_gm(pp), s.setup_oa(pp)
            state = {'pp': pp, 'gpk': s.FullGroupPublicKey(pp, gm.pk, oa.pk),
                     'gm_sk': gm.sk, 'oa_sk': oa.sk}
            private_pickle(output / 'setup-state.pkl', state)
        except BaseException as error:
            manifest.update(status='failed', failure_phase='population_setup',
                            error_type=type(error).__name__, error=str(error),
                            population_setup_seconds=time.perf_counter()-begin,
                            updated_at_utc=datetime.now(timezone.utc).isoformat(),
                            source_after=source_hashes())
            manifest['source_changed_during_run'] = [name for name in sorted(set(current_sources) | set(manifest['source_after']))
                                                    if manifest['source_after'].get(name) != current_sources.get(name)]
            (output / 'traceback.txt').write_text(traceback.format_exc())
            atomic_json(manifest_path, manifest)
            raise
        manifest['population_setup_seconds'] = time.perf_counter() - begin

    manifest['status'] = 'building_population'
    atomic_json(manifest_path, manifest)
    begin = time.perf_counter()
    try:
        pool = build_population(state['pp'], state['gpk'], state['gm_sk'],
                                output / 'population', max_users=args.max_users,
                                workers=args.workers)
        state.update({key: pool[key] for key in
                      ('entries', 'recipient_sk', 'recipient_pk', 'recipient_cert')})
        manifest['population_seconds_this_invocation'] = time.perf_counter() - begin
        manifest['population_count'] = len(state['entries'])
        manifest['status'] = 'timing'
        atomic_json(manifest_path, manifest)
        if source_hashes() != current_sources:
            raise RuntimeError('Source changed during population construction')
        # build_population joins/shuts down its executor before returning.
        os.environ['LATGE_PROOF_WORKERS'] = '8'
        report = run_measurements(output / 'measurements', state,
                                  group_sizes=tuple(manifest['configuration']['group_sizes']))
        manifest['status'] = 'measurements_completed'
        manifest['timing_report'] = 'measurements/timing_report.json'
        manifest['timing_report_status'] = report.get('status')
    except BaseException as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error))
        (output / 'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        manifest['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['source_after'] = source_hashes()
        manifest['source_changed_during_run'] = [name for name in sorted(set(current_sources) | set(manifest['source_after']))
                                               if manifest['source_after'].get(name) != current_sources.get(name)]
        if manifest['source_changed_during_run']:
            manifest['status'] = 'source_changed'
        atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--max-users', type=int, choices=[16, 32, 64, 128, 256, 512, 1024], default=1024)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        parser.error('--workers must be between 1 and 6 on this host')
    if not __debug__:
        parser.error('assertion checks must be enabled')
    result = run(args)
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve())}), flush=True)


if __name__ == '__main__':
    main()
