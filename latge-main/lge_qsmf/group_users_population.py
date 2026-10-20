from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import shutil
import stat
import tempfile
import time
import traceback

from . import scheme as s
from .decomposition import rdec
from .linalg import matrix_equal, vec_inf_norm
from .mp12_reference import key_instance_digest


class PopulationBuildError(RuntimeError):
    pass


_VERSION = 1
_WORKER = None


def _atomic(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name+'-', dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_pickle(path, value):
    _atomic(path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _save_json(path, value):
    _atomic(path, (json.dumps(value, indent=2, sort_keys=True)+'\n').encode())


def _private_bytes(path):
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077):
            raise PopulationBuildError(f'checkpoint is not a private owned file: {path}')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _load_pickle(path):
    return pickle.loads(_private_bytes(path))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _public_key_digest(pk):
    return hashlib.sha256(s._pack_matrix(pk.A)).hexdigest()


def _binding(pp, gpk, gm_sk):
    gpk.validate()
    if pp != gpk.pp:
        raise ValueError('population pp and gpk have different public contexts')
    s._secret_key_context(pp, gm_sk, pp.params.alpha, 3, 'population GM', gpk.pk_gm.A)
    digest = hashlib.sha256(b'LatGE-genuine-population-context-v1')
    pieces = [json.dumps(asdict(pp.params), sort_keys=True, separators=(',', ':')).encode(),
              pp.parcom.encode(), pp.ots_hash_seed,
              str((pp.s_full, pp.q1, pp.q2)).encode(),
              s._pack_matrix(gpk.pk_gm.A), s._pack_matrix(gpk.pk_oa.A),
              s._pack_matrix(pp.B0), s._pack_matrix(pp.B1), s._pack_matrix(pp.B2),
              s._pack_vec(pp.u), key_instance_digest(gm_sk.trapdoor)]
    pieces.extend(s._pack_matrix(matrix) for matrix in pp.A_tags)
    pieces.append(pickle.dumps(pp.trapdoor_distribution, protocol=5))
    for piece in pieces:
        digest.update(len(piece).to_bytes(8, 'little'))
        digest.update(piece)
    return digest.hexdigest()


def _initialize_worker(pp, gpk, gm_sk):
    global _WORKER
    os.environ['LATGE_PROOF_WORKERS'] = '1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    _WORKER = (pp, gpk, gm_sk)


def _worker_phase(index, directory, phase):
    directory = Path(directory)
    artifact = directory / f'{index:06d}-{phase}.pkl'
    marker = directory / f'{index:06d}-{phase}.json'
    if artifact.exists():
        saved = _load_pickle(artifact)
        if saved['index'] != index or saved['phase'] != phase:
            raise PopulationBuildError('saved enrollment stage has the wrong identity')
        return {'ok': True, 'index': index, 'phase': phase, 'path': str(artifact), 'resumed': True}
    if marker.exists():
        return {'ok': False, 'index': index, 'phase': phase,
                'error': 'prior stage has no durable result; refusing an outer retry',
                'prior_state': json.loads(_private_bytes(marker))}
    _save_json(marker, {'index': index, 'phase': phase, 'status': 'running'})
    timings = {}

    def timed(name, fn):
        start = time.perf_counter()
        try:
            return fn()
        finally:
            timings[name] = time.perf_counter()-start

    try:
        pp, gpk, gm_sk = _WORKER
        if phase == 'prepared':
            pk, sk = timed('UKGEN', lambda: s.ukgen(pp))
            request = timed('JOIN', lambda: s.join(gpk, pk, sk))
            proof = request.proof.pi_pki.transcript
            value = {'index': index, 'phase': phase, 'pk': pk, 'sk': sk,
                     'request': request, 'timings': timings,
                     'proof_sha256': hashlib.sha256(proof).hexdigest(),
                     'proof_bytes': len(proof), 'pk_sha256': _public_key_digest(pk)}
        elif phase == 'issued':
            prepared = _load_pickle(directory/f'{index:06d}-prepared.pkl')
            snapshot = _load_pickle(directory/'registry_snapshot.pkl')
            if not isinstance(snapshot, s.FullRegistrationTable):
                raise PopulationBuildError('issuance snapshot is not an actual registration table')
            issuer = s.bind_issue_context(gpk, snapshot, gm_sk, prepared['request'])
            cert = timed('ISSUE', lambda: s.issue(issuer, prepared['pk']))
            acceptance = timed('JOIN_ACCEPT', lambda: s.join_accept(gpk, prepared['pk'], cert))
            squared = sum(int(x)**2 for poly in cert.v_i for x in poly.centered())
            bound_squared = pp.params.beta**2 * 8 * pp.params.alpha * pp.params.n
            valid_infinity = s.verify_certificate(gpk, prepared['pk'], cert)
            checks = {'cert_l2': acceptance is not None,
                      'cert_l2_squared': squared, 'cert_l2_bound_squared': bound_squared,
                      'certificate_equation_and_infinity': valid_infinity,
                      'max_abs_v': vec_inf_norm(cert.v_i)}
            if acceptance is None:
                _save_pickle(directory/f'{index:06d}-rejected-certificate.pkl',
                             {'pk': prepared['pk'], 'cert': cert, 'checks': checks})
                raise PopulationBuildError(f'certificate rejected without retry: {checks}')
            h0 = rdec(prepared['pk'].A[0][-1], (pp.params.q-1)//2)[0]
            value = {'index': index, 'phase': phase, 'pk': prepared['pk'],
                     'cert': cert, 'acceptance': acceptance,
                     'recipient_sk': prepared['sk'] if index == 1 else None,
                     'log': {'index': index, 'status': 'accepted',
                             'timings': {**prepared['timings'], **timings},
                             'proof_sha256': prepared['proof_sha256'],
                             'proof_bytes': prepared['proof_bytes'],
                             'pk_sha256': prepared['pk_sha256'],
                             'registry_snapshot_count': len(snapshot.entries),
                             'h0': [int(x) for x in h0.centered()], **checks}}
        else:
            raise ValueError('unknown enrollment phase')
        _save_pickle(artifact, value)
        _save_json(marker, {'index': index, 'phase': phase, 'status': 'completed',
                            'timings': timings})
        return {'ok': True, 'index': index, 'phase': phase, 'path': str(artifact), 'resumed': False}
    except Exception as error:
        failure = {'ok': False, 'index': index, 'phase': phase,
                   'error_type': type(error).__name__, 'error': str(error),
                   'timings': timings, 'traceback': traceback.format_exc()}
        _save_json(marker, {**failure, 'status': 'failed'})
        return failure


def build_population(pp, gpk, gm_sk, output_dir, max_users=1024, workers=6):

    if type(max_users) is not int or max_users < 1:
        raise ValueError('max_users must be a positive integer')
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('workers must be an integer in 1..32')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not output.is_dir() or output.stat().st_uid != os.getuid():
        raise ValueError('population output must be an owned local directory')
    members = output/'members'; pending = output/'pending'
    members.mkdir(exist_ok=True, mode=0o700); pending.mkdir(exist_ok=True, mode=0o700)
    checkpoint_path = output/'checkpoint.pkl'
    binding = _binding(pp, gpk, gm_sk)
    resumed = checkpoint_path.exists()
    if resumed:
        checkpoint = _load_pickle(checkpoint_path)
        if checkpoint.get('version') != _VERSION or checkpoint.get('binding_hash') != binding:
            raise PopulationBuildError('checkpoint belongs to a different group or version')
        if checkpoint.get('failure'):
            raise PopulationBuildError('prior enrollment failed; refusing to retry or select members')
        if max_users < checkpoint['committed_count']:
            raise ValueError('requested size is below the already committed population')
    else:
        if any(members.iterdir()) or any(pending.iterdir()):
            raise PopulationBuildError('uncheckpointed enrollment artifacts require manual diagnosis')
        checkpoint = {'version': _VERSION, 'binding_hash': binding,
                      'committed_count': 0, 'members': [], 'active_batch': None}
        _save_pickle(checkpoint_path, checkpoint)
    reg = s.FullRegistrationTable()
    logs, recipient_sk, key_index = [], None, {}
    for item in checkpoint['members']:
        path = output/item['file']
        raw = _private_bytes(path)
        if hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise PopulationBuildError('member checkpoint digest mismatch')
        record = pickle.loads(raw)
        entry, index = record['entry'], record['index']
        if index != len(reg.entries)+1 or not isinstance(entry, s.FullRegistrationEntry):
            raise PopulationBuildError('member checkpoints are not a contiguous ordered population')
        if not s.verify_join_certificate(gpk, entry.pk, entry.cert):
            raise PopulationBuildError('restored member certificate is invalid')
        digest = _public_key_digest(entry.pk)
        if any(matrix_equal(old.A, entry.pk.A) for old in key_index.get(digest, ())):
            raise PopulationBuildError('duplicate public key in checkpoint')
        key_index.setdefault(digest, []).append(entry.pk)
        reg.entries.append(entry); logs.append(record['log'])
        if index == 1:
            recipient_sk = record['recipient_sk']
            if recipient_sk is None or not matrix_equal(recipient_sk.trapdoor.A, entry.pk.A):
                raise PopulationBuildError('fixed recipient key missing or mismatched')
    if len(reg.entries) != checkpoint['committed_count']:
        raise PopulationBuildError('checkpoint commit count disagrees with member files')

    def write_logs():
        _atomic(output/'membership.jsonl', ''.join(json.dumps(row, sort_keys=True)+'\n' for row in logs).encode())

    def progress(status, phase, **extra):
        _save_json(output/'progress.json', {'version': _VERSION, 'status': status,
            'phase': phase, 'requested_users': max_users, 'committed_users': len(reg.entries),
            'workers': workers, 'binding_hash': binding, 'resumed': resumed,
            'outer_retries': 0, 'active_batch': checkpoint['active_batch'], **extra})

    def collect(executor, indices, batch_dir, phase):
        progress('running', phase)
        futures = [executor.submit(_worker_phase, index, str(batch_dir), phase) for index in indices]
        results = [future.result() for future in futures]
        failures = [row for row in results if not row['ok']]
        if failures:
            _save_json(output/'failure.json', {'failures': failures})
            raise PopulationBuildError(f'enrollment stage failed: {failures}')
        return results

    write_logs()
    progress('running', 'restore' if resumed else 'start')
    executor = None
    try:
        if len(reg.entries) < max_users:
            executor = ProcessPoolExecutor(max_workers=min(workers, max_users),
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize_worker, initargs=(pp, gpk, gm_sk))
        while len(reg.entries) < max_users:
            batch = checkpoint['active_batch']
            if batch is None:
                start = len(reg.entries)+1
                indices = list(range(start, min(max_users+1, start+workers)))
                batch = {'start': start, 'indices': indices,
                         'snapshot_count': len(reg.entries), 'directory': f'pending/batch-{start:06d}'}
                batch_dir = output/batch['directory']
                batch_dir.mkdir(mode=0o700)
                _save_pickle(batch_dir/'registry_snapshot.pkl', reg)
                checkpoint['active_batch'] = batch
                _save_pickle(checkpoint_path, checkpoint)
            else:
                batch_dir = output/batch['directory']
                if max(batch['indices']) > max_users:
                    raise ValueError('cannot shrink an active enrollment batch')
            indices = [i for i in batch['indices'] if i > len(reg.entries)]
            prepared_paths = collect(executor, indices, batch_dir, 'prepared')
            reserved = {}
            for item in prepared_paths:
                prepared = _load_pickle(item['path'])
                pk, digest = prepared['pk'], _public_key_digest(prepared['pk'])
                if not matrix_equal(pk.A, prepared['request'].pk.A):
                    raise PopulationBuildError('prepared key differs from its actual JOIN request')
                for prior in (*key_index.get(digest, ()), *reserved.get(digest, ())):
                    if matrix_equal(prior.A, pk.A):
                        raise PopulationBuildError('duplicate user key before ISSUE; no replacement draw')
                reserved.setdefault(digest, []).append(pk)
            _save_json(batch_dir/'reservations.json', {'indices': indices,
                'registered_count_at_check': len(reg.entries),
                'pk_sha256': [_public_key_digest(_load_pickle(item['path'])['pk']) for item in prepared_paths]})
            issued_paths = collect(executor, indices, batch_dir, 'issued')
            for item in issued_paths:
                issued = _load_pickle(item['path'])
                index = issued['index']
                if index != len(reg.entries)+1:
                    raise PopulationBuildError('out-of-order registration attempt')
                started = time.perf_counter()
                s.register_user(gpk, reg, issued['acceptance'])
                log = issued['log']
                log['timings']['REGISTER'] = time.perf_counter()-started
                log['registration_index'] = index
                entry = reg.entries[-1]
                if not matrix_equal(entry.pk.A, issued['pk'].A):
                    raise PopulationBuildError('registration changed the accepted public key')
                if index == 1:
                    recipient_sk = issued['recipient_sk']
                record = {'index': index, 'entry': entry, 'log': log,
                          'recipient_sk': recipient_sk if index == 1 else None}
                relative = f'members/member-{index:06d}.pkl'
                _save_pickle(output/relative, record)
                checkpoint['members'].append({'index': index, 'file': relative,
                                              'sha256': _sha(output/relative)})
                checkpoint['committed_count'] = index
                _save_pickle(checkpoint_path, checkpoint)
                logs.append(log)
                key_index.setdefault(_public_key_digest(entry.pk), []).append(entry.pk)
                write_logs(); progress('running', 'registered')
                print(json.dumps({'event': 'member_registered', 'index': index,
                                  'max_users': max_users, 'timings': log['timings']}), flush=True)
            checkpoint['active_batch'] = None
            _save_pickle(checkpoint_path, checkpoint)
            shutil.rmtree(batch_dir)
        progress('completed', 'complete')
    except BaseException as error:
        status = 'failed' if isinstance(error, Exception) else 'interrupted'
        failure = {'error_type': type(error).__name__, 'error': str(error),
                   'traceback': traceback.format_exc()}
        if status == 'failed':
            checkpoint['failure'] = failure
        _save_pickle(checkpoint_path, checkpoint)
        progress(status, 'stopped', **failure)
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    first = reg.entries[0]
    return {'entries': reg.entries, 'recipient_sk': recipient_sk,
            'recipient_pk': first.pk, 'recipient_cert': first.cert,
            'membership_log': logs, 'completed_count': len(reg.entries),
            'checkpoint_path': str(checkpoint_path), 'resumed': resumed}
