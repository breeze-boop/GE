from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

from . import scheme as s
from .decomposition import rdec
from .linalg import matrix_equal, vec_equal, vec_inf_norm
from .paper_zkp_backend import PaperZKPBackend
from .strict_layout import strict_paper_layout


SCHEMA = 'latge-group-users-timing-v1'
ALGORITHMS = ('SETUPinit', 'SETUPGM', 'SETUPOA', 'UKGEN', 'JOIN', 'ISSUE',
              'JOIN_ACCEPT', 'REGISTER', 'Gr', 'sampleR', 'ENC', 'P', 'V',
              'DEC', 'OPEN')
CHECKS = ('setup_parameters', 'recipient_certificate', 'prefix_certificates',
          'probe_certificate', 'join_acceptance', 'probe_registered',
          'encryption', 'proof', 'decryption', 'opening')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _get(state, name):
    return state[name] if isinstance(state, Mapping) else getattr(state, name)


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _frame_update(digest, value):
    digest.update(len(value).to_bytes(8, 'little'))
    digest.update(value)


def _certificate_bytes(cert):
    return (len(cert.tag).to_bytes(8, 'little') + bytes(cert.tag)
            + s._pack_vec(cert.r_i) + s._pack_vec(cert.v_i))


def _public_state_digest(pp, gpk, recipient_pk, recipient_cert):
    digest = hashlib.sha256(b'LatGE-group-users-timing-public-state-v1')
    values = [json.dumps(asdict(pp.params), sort_keys=True).encode(),
              s._pack_matrix(gpk.pk_gm.A), s._pack_matrix(gpk.pk_oa.A),
              s._pack_matrix(recipient_pk.A), _certificate_bytes(recipient_cert),
              pp.parcom.encode(), pp.ots_hash_seed, s._pack_vec(pp.u),
              s._pack_matrix(pp.B0), s._pack_matrix(pp.B1), s._pack_matrix(pp.B2)]
    values.extend(s._pack_matrix(matrix) for matrix in pp.A_tags)
    for value in values:
        _frame_update(digest, value)
    return digest.hexdigest()


def _prefix_digests(entries, sizes):
    wanted = set(sizes)
    result = {}
    digest = hashlib.sha256(b'LatGE-group-users-timing-registration-prefix-v1')
    for index, entry in enumerate(entries[:max(sizes)], 1):
        _frame_update(digest, s._pack_matrix(entry.pk.A))
        _frame_update(digest, _certificate_bytes(entry.cert))
        if index in wanted:
            result[index] = digest.hexdigest()
    return result


def _write_csv(path, measurements):
    fields = ['group_users', 'repeats', 'result', 'recipient_hint_multiplicity',
              'prefix_valid_certificates', 'prefix_valid_matching_hint_count']
    fields += [name + '_seconds' for name in ALGORITHMS]
    fields += [name + '_status' for name in ALGORITHMS]
    fields += ['check_' + name for name in CHECKS]
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(measurements, key=int):
            row = measurements[key]
            stages = {stage['algorithm']: stage for stage in row['stages']}
            record = {name: row.get(name) for name in fields[:6]}
            for name in ALGORITHMS:
                stage = stages.get(name, {})
                record[name + '_seconds'] = stage.get('seconds')
                record[name + '_status'] = stage.get('status')
            for name in CHECKS:
                record['check_' + name] = row.get('checks', {}).get(name)
            writer.writerow(record)
    temporary.replace(path)


def run_measurements(output_dir, state,
                     group_sizes=(16, 32, 64, 128, 256, 512, 1024)):


    sizes = tuple(group_sizes)
    if (not sizes or any(type(n) is not int or n < 1 for n in sizes)
            or len(set(sizes)) != len(sizes)):
        raise ValueError('group_sizes must contain distinct positive integers')
    pp, gpk, gm_sk, oa_sk, recipient_sk, recipient_pk, recipient_cert = (
        _get(state, name) for name in ('pp', 'gpk', 'gm_sk', 'oa_sk',
                                      'recipient_sk', 'recipient_pk', 'recipient_cert'))
    entries = list(_get(state, 'entries'))
    if len(entries) < max(sizes):
        raise ValueError('insufficient real registration entries')
    if not matrix_equal(entries[0].pk.A, recipient_pk.A):
        raise ValueError('the fixed recipient must be the first registered member')
    if _certificate_bytes(entries[0].cert) != _certificate_bytes(recipient_cert):
        raise ValueError('recipient certificate differs from its registration')
    if gpk.pp is not pp and gpk.pp != pp:
        raise ValueError('state public parameters and group disagree')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = _public_state_digest(pp, gpk, recipient_pk, recipient_cert)
    prefixes = _prefix_digests(entries, sizes)
    recipient_hint = rdec(recipient_pk.A[0][-1], (pp.params.q-1)//2)[0]
    opener = s.bind_open_context(gpk, oa_sk)
    workers = PaperZKPBackend()._streaming_options['workers']
    report = {
        'schema': SCHEMA, 'started_at_utc': _now(), 'status': 'running',
        'scope': 'one raw sequential observation per algorithm per group size; not 30-run averages',
        'repeats_per_size': 1, 'outer_retries': 0,
        'group_sizes': list(sizes), 'pool_entries': len(entries),
        'fixed_recipient_index': 0, 'public_state_sha256': identity,
        'parameters': asdict(pp.params),
        'packed_group_messages': strict_paper_layout(pp.params).packed_message_count,
        'runtime': {'python': platform.python_version(), 'platform': platform.platform(),
                    'proof_workers': workers,
                    'thread_environment': {name: os.environ.get(name) for name in
                        ('LATGE_PROOF_WORKERS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                         'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}},
        'timing_boundary': 'public algorithm call only; JSON/CSV, stdout, hashes, context binding, pool checks and proof disposal excluded',
        'setup_scope': 'SETUPinit is a fresh independent call; SETUPGM and SETUPOA use the existing pp; generated outputs do not replace the benchmark group',
        'registry_scope': 'ISSUE sees exactly prefix N; REGISTER appends its probe to a private list copy; OPEN sees prefix N without that probe',
        'source_validation': 'external caller owns source snapshot/hash verification, including resume',
        'randomness_counters': 'not instrumented; measured duration alone does not identify rejection-sampling counts',
        'measurements': {},
    }

    def save_summary():
        _atomic_json(output/'timing_report.json', report)
        _write_csv(output/'timings.csv', report['measurements'])

    for n in sizes:
        row_path = output/f'n-{n:04d}.json'
        if row_path.exists():
            prior = json.loads(row_path.read_text())
            if (prior.get('schema') != SCHEMA or prior.get('group_users') != n
                    or prior.get('public_state_sha256') != identity
                    or prior.get('registration_prefix_sha256') != prefixes[n]
                    or prior.get('repeats') != 1):
                raise ValueError(f'existing measurement N={n} belongs to different inputs')
            if prior.get('status') != 'completed':
                raise RuntimeError(f'N={n} has an incomplete measurement; no automatic retry or overwrite')
            report['measurements'][str(n)] = prior
            print(json.dumps({'event': 'resume_completed', 'group_users': n,
                              'result': prior.get('result')}), flush=True)
            save_summary()
            continue

        row = {
            'schema': SCHEMA, 'group_users': n, 'repeats': 1, 'outer_retries': 0,
            'status': 'running', 'started_at_utc': _now(),
            'public_state_sha256': identity, 'registration_prefix_sha256': prefixes[n],
            'stages': [], 'checks': {}, 'preparation_errors': [],
        }
        report['measurements'][str(n)] = row
        reg = s.FullRegistrationTable(list(entries[:n]))
        register_copy = s.FullRegistrationTable(list(entries[:n]))

        def save_row():
            _atomic_json(row_path, row)
            save_summary()

        def stage(name, operation=None, reason=None):
            measured = {'algorithm': name, 'status': 'running', 'seconds': None}
            row['stages'].append(measured)
            if operation is None:
                measured.update(status='skipped', reason=reason)
                save_row()
                print(json.dumps({'event': 'stage', 'group_users': n, **measured}), flush=True)
                return False, None
            save_row()
            print(json.dumps({'event': 'start', 'group_users': n, 'algorithm': name}), flush=True)
            started = time.perf_counter()
            try:
                value = operation()
            except Exception as error:
                elapsed = time.perf_counter()-started
                measured.update(status='failed', seconds=elapsed,
                                error_type=type(error).__name__, error=str(error))
                success, value = False, None
            else:
                elapsed = time.perf_counter()-started
                measured.update(status='completed', seconds=elapsed,
                                returned_none=value is None)
                if type(value) is bool:
                    measured['returned_boolean'] = value
                success = True
            save_row()
            print(json.dumps({'event': 'stage', 'group_users': n, **measured}), flush=True)
            return success, value

        def prepare(name, operation):
            try:
                return True, operation()
            except Exception as error:
                row['preparation_errors'].append({'operation': name,
                    'error_type': type(error).__name__, 'error': str(error)})
                save_row()
                return False, None

        valid_count, hint_count, valid_hint_count, pool_errors = 0, 0, 0, []
        for index, entry in enumerate(reg.entries):
            try:
                valid = bool(s.verify_certificate(gpk, entry.pk, entry.cert))
                matches_hint = rdec(entry.pk.A[0][-1], (pp.params.q-1)//2)[0] == recipient_hint
                valid_count += int(valid)
                hint_count += int(matches_hint)
                valid_hint_count += int(valid and matches_hint)
            except Exception as error:
                pool_errors.append({'index': index, 'error_type': type(error).__name__, 'error': str(error)})
        row.update(prefix_valid_certificates=valid_count,
                   recipient_hint_multiplicity=hint_count,
                   prefix_valid_matching_hint_count=valid_hint_count,
                   pool_check_errors=pool_errors)
        recipient_ok, recipient_valid = prepare('recipient_certificate',
            lambda: s.verify_certificate(gpk, recipient_pk, recipient_cert))
        row['checks'].update(recipient_certificate=bool(recipient_ok and recipient_valid),
                             prefix_certificates=valid_count == n and not pool_errors)
        save_row()

        setup_ok, fresh_pp = stage('SETUPinit', lambda: s.setup_init(pp.params.security_bits))
        row['checks']['setup_parameters'] = bool(setup_ok and fresh_pp is not None and fresh_pp.params == pp.params)
        del fresh_pp
        stage('SETUPGM', lambda: s.setup_gm(pp))
        stage('SETUPOA', lambda: s.setup_oa(pp))
        key_ok, probe_pair = stage('UKGEN', lambda: s.ukgen(pp))
        request = cert = acceptance = None
        if key_ok and probe_pair is not None:
            probe_pk, probe_sk = probe_pair
            join_ok, request = stage('JOIN', lambda: s.join(gpk, probe_pk, probe_sk))
        else:
            join_ok, request = stage('JOIN', reason='UKGEN produced no key pair')
        if join_ok and request is not None:
            bound_ok, issuer = prepare('bind_issue_context',
                lambda: s.bind_issue_context(gpk, reg, gm_sk, request))
            if bound_ok:
                issue_ok, cert = stage('ISSUE', lambda: s.issue(issuer, probe_pk))
            else:
                issue_ok, cert = stage('ISSUE', reason='issuer context could not be bound')
        else:
            issue_ok, cert = stage('ISSUE', reason='JOIN produced no request')
        if issue_ok and cert is not None:
            cert_ok, certificate_valid = prepare('probe_certificate',
                lambda: s.verify_certificate(gpk, probe_pk, cert))
            row['checks']['probe_certificate'] = bool(cert_ok and certificate_valid)
            row['probe_certificate_max_abs_v'] = vec_inf_norm(cert.v_i)
            accept_ok, acceptance = stage('JOIN_ACCEPT', lambda: s.join_accept(gpk, probe_pk, cert))
        else:
            row['checks']['probe_certificate'] = False
            accept_ok, acceptance = stage('JOIN_ACCEPT', reason='ISSUE produced no certificate')
        row['checks']['join_acceptance'] = bool(accept_ok and acceptance is not None)
        if accept_ok and acceptance is not None:
            registered, _ = stage('REGISTER', lambda: s.register_user(gpk, register_copy, acceptance))
            row['checks']['probe_registered'] = bool(registered and len(register_copy.entries) == n+1
                and matrix_equal(register_copy.entries[-1].pk.A, probe_pk.A))
        else:
            stage('REGISTER', reason='JOIN_ACCEPT did not accept the probe')
            row['checks']['probe_registered'] = False
        del probe_pair, request, cert, acceptance
        relation_ok, relation_key = stage('Gr', lambda: s.gr(pp))
        if relation_ok and relation_key is not None:
            sample_ok, sampled = stage('sampleR', lambda: s.sample_risis(*relation_key))
        else:
            sample_ok, sampled = stage('sampleR', reason='Gr produced no relation key')
        ct = coins = proof = None
        if sample_ok and sampled is not None:
            statement, message = sampled
            enc_ok, encrypted = stage('ENC', lambda: s.enc(gpk, recipient_pk, recipient_cert,
                                                          message, b'group-users-timing'))
        else:
            enc_ok, encrypted = stage('ENC', reason='sampleR produced no statement/message')
        row['checks']['encryption'] = bool(enc_ok and encrypted is not None)
        if enc_ok and encrypted is not None:
            ct, coins = encrypted
            bind_ok, bound_ct = prepare('bind_public_relation', lambda: s.bind_public_relation(ct, statement))
            if bind_ok:
                ct = bound_ct
                proof_ok, proof = stage('P', lambda: s.prove(gpk, recipient_pk, recipient_cert,
                                                            statement, message, ct, coins))
            else:
                proof_ok, proof = stage('P', reason='public relation could not be bound')
            if proof_ok and proof is not None:
                row['proof'] = {'bytes': len(proof.transcript),
                                'sha256': hashlib.sha256(proof.transcript).hexdigest(),
                                'statement_digest': proof.statement_digest.hex()}
                verified_ok, verified = stage('V', lambda: s.verify(gpk, ct, proof))
            else:
                verified_ok, verified = stage('V', reason='P produced no proof')
            row['checks']['proof'] = bool(verified_ok and verified)
            proof = None
            recovered_ok, recovered = stage('DEC', lambda: s.dec(recipient_sk, ct, ct.label))
            row['checks']['decryption'] = bool(recovered_ok and recovered is not None
                                              and vec_equal(recovered, message))
            opened_ok, opened = stage('OPEN', lambda: s.open_receiver(opener, reg, ct, ct.label))
            row['checks']['opening'] = bool(opened_ok and opened is not None
                                           and matrix_equal(opened.A, recipient_pk.A))
        else:
            for name in ('P', 'V', 'DEC', 'OPEN'):
                stage(name, reason='ENC produced no ciphertext')
            row['checks'].update(proof=False, decryption=False, opening=False)
        ct = coins = proof = encrypted = None
        row.update(status='completed', completed_at_utc=_now())
        row['result'] = ('passed' if all(row['checks'].get(name, False) for name in CHECKS)
                         and all(item['status'] == 'completed' for item in row['stages'])
                         else 'completed_with_failures')
        row['timed_seconds_sum'] = sum(item['seconds'] or 0.0 for item in row['stages'])
        save_row()
        print(json.dumps({'event': 'size_completed', 'group_users': n,
                          'result': row['result'], 'checks': row['checks']}), flush=True)

    report.update(status=('passed' if all(row['result'] == 'passed'
                                         for row in report['measurements'].values())
                          else 'completed_with_failures'), completed_at_utc=_now())
    save_summary()
    return report
