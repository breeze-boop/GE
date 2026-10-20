from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import compress
import statistics
import time

import numpy as np

from . import paper_api as api
from .cost_model import estimate_modules
from .distributions import UniformBounded
from .fast_uniform import uniform_backend, uniform_field, uniform_source
from .gpv import decrypt_oa
from .group_sweep import _independent_policy_check
from .mathutils import (SystemUniformRNG, bits_from_ints_le, int_to_bits_le, rand_discrete_gaussian,
                        rand_mod_matrix, require_memory)
from .paper_measured import _RECIPIENT_COLUMN_BLOCK, blocked_recipient_keygen
from .params import PAPER_GROUP_DEPTHS, PAPER_SECURITY_BITS
from .provenance import changed_provenance_files, provenance
from .randomness import CryptographicRNG


def _rng_state(rng):
    return rng._counter


def _timed(function):
    started = time.perf_counter_ns()
    result = function()
    return result, (time.perf_counter_ns() - started) / 1e6


def _summary(samples):
    q1, q3 = np.percentile(samples, (25, 75))
    return {'mean_ms': statistics.mean(samples), 'median_ms': statistics.median(samples),
            'q1_ms': float(q1), 'q3_ms': float(q3), 'samples_ms': samples,
            'sample_count': len(samples), 'min_ms': min(samples), 'max_ms': max(samples),
            'std_ms': statistics.stdev(samples) if len(samples) > 1 else None}


def _opening_workspace(params):
    from .stern_opening import _rounds
    p=params
    width=p.m*p.depth*(2*p.beta).bit_length()+2*p.depth*(p.q//2).bit_length()+p.depth+1
    coordinates=2*width
    rounds=_rounds(PAPER_SECURITY_BITS)
    return rounds*(8*coordinates+256)+160*coordinates+32*1024**2+(16+4*p.depth)*p.N*p.nk


def _fixture_join_core(gpk, gm_sk, info, recipient_pk, recipient_sk, leaf, public_acc):
    api._check_join_key_inputs(gpk, recipient_pk, recipient_sk)
    with api.JOIN_CONTEXT(gpk, recipient_pk, info):
        manager=api.ISSUE(gm_sk)
        manager.receive(leaf.copy())
        certificate,=manager.finish()
    if not api.TVerify(public_acc,certificate.root,leaf,certificate.path):
        raise AssertionError('JOIN fixture certificate rejected')
    return certificate


def _integer_reference_hash(acc, left, right):
    acc._check_bits(left)
    acc._check_bits(right)
    bits = np.concatenate((left, right)).tolist()
    residues = np.fromiter(
        (sum(int(coefficient) for coefficient in compress(row, bits)) % acc.params.q
         for row in acc.A), dtype=np.int64, count=acc.params.n)
    return bits_from_ints_le(residues, acc.params.k)


@contextmanager
def _join_reference_arithmetic(*accumulators):
    saved = []
    try:
        for acc in accumulators:
            saved.append((acc, acc.__dict__.get('hash')))
            acc.hash = lambda left, right, instance=acc: _integer_reference_hash(instance, left, right)
        yield
    finally:
        for acc, original in reversed(saved):
            if original is None:
                del acc.hash
            else:
                acc.hash = original


def _recipient_checks(p, A, public, secret, replay_rng):
    if (secret.E1.shape != (p.m, p.m) or secret.E1.dtype.kind != 'i'
            or int(secret.E1.min()) < -p.beta or int(secret.E1.max()) > p.beta):
        raise AssertionError('Recipient shape/beta bound failed')
    for U in (public.U1, public.U2):
        if U.shape != (p.n, p.m) or int(U.min()) < 0 or int(U.max()) >= p.q:
            raise AssertionError('Actual recipient public key is not canonical')
    assert max(p.m * p.beta * (p.q-1), p.m * (p.q-1)**2) < 2**63
    rng = SystemUniformRNG()
    width = min(_RECIPIENT_COLUMN_BLOCK, p.m)
    for branch, U in enumerate((public.U1, public.U2)):
        probes = rng.integers(0, p.q, size=(p.m, 3))
        projected = np.zeros((p.m, 3), dtype=np.int64)
        for offset in range(0, p.m, width):
            stop = min(offset + width, p.m)
            block = rand_discrete_gaussian(replay_rng, (p.m, stop-offset),
                                           p.sigma, p.beta, p.max_memory_bytes)
            if int(block.min()) < -p.beta or int(block.max()) > p.beta:
                raise AssertionError('Recipient replay exceeds beta')
            if branch == 0 and not np.array_equal(secret.E1[:, offset:stop], block):
                raise AssertionError('Recipient E1/replay mismatch')

            projected += block @ probes[offset:stop]
        if not np.array_equal((A @ (projected % p.q)) % p.q, (U @ probes) % p.q):
            raise AssertionError(f'Independent recipient A*E{branch+1}=U{branch+1} projection failed')
    return {'U1_three_independent_projections_passed': True,
            'U2_three_independent_projections_passed': True,
            'projection_false_accept_bound_per_product': p.q**-3,
            'projection_false_accept_union_bound': 2*p.q**-3,
            'projection_randomness': type(rng).__name__, 'U2_shape_and_range_checked': True,
            'E2_product_independently_checked': True, 'F_binding_checked': False,
            'secret_replay_scope': 'same_sampler_replay'}


def _oa_checks(p, A, public, secret, replay_rng):
    if secret.E1.shape != (p.m, p.depth):
        raise AssertionError('OA secret has incorrect dimensions')
    assert p.m*p.beta*(p.q-1) < 2**63
    for branch, U in enumerate((public.U1, public.U2)):
        E = rand_discrete_gaussian(replay_rng, (p.m, p.depth), p.sigma, p.beta,
                                   p.max_memory_bytes)
        if U.shape != (p.n, p.depth) or int(U.min()) < 0 or int(U.max()) >= p.q:
            raise AssertionError('OA public key shape/range mismatch')
        if int(E.min()) < -p.beta or int(E.max()) > p.beta:
            raise AssertionError('OA replay exceeds beta')
        if branch == 0 and not np.array_equal(secret.E1, E):
            raise AssertionError('OA E1/replay mismatch')
        if not np.array_equal(U, (A @ E) % p.q):
            raise AssertionError(f'Independent OA A*E{branch+1}=U{branch+1} product failed')
    return {'U1_exact_product_checked': True, 'U2_exact_product_checked': True,
            'secret_replay_scope': 'same_sampler_replay'}


def _ciphertext_check(p, A, public, plaintext, ciphertext, coins):
    for i, (U, pair) in enumerate(zip((public.U1, public.U2), (ciphertext.c1, ciphertext.c2))):
        if pair.c1.shape != (p.m,) or pair.c2.shape != plaintext.shape:
            return False
        s, x, y = coins['s'][i], coins['x'][i], coins['y'][i]
        if (s.shape != (p.n,) or not np.all((s == 0) | (s == 1))
                or x.shape != (p.m,) or y.shape != plaintext.shape
                or np.any(x < -p.error_bound) or np.any(x > p.error_bound)
                or np.any(y < -p.error_bound) or np.any(y > p.error_bound)):
            return False

        first = (A.T @ coins['s'][i].astype(np.int64) + coins['x'][i]) % p.q
        second = (U.T @ coins['s'][i].astype(np.int64) + coins['y'][i]
                  + plaintext.astype(np.int64) * p.qhalf) % p.q
        if not np.array_equal(first, pair.c1) or not np.array_equal(second, pair.c2):
            return False
    return True


def _model_rows(model, gpv, forecast):
    modules = model['modules']
    def estimated(name, key):
        stages = forecast['stages_s'][forecast['point_case']]
        components = stages[name] if name == 'SETUP_Init' else {
            **{f'offline.{label}': value for label, value in stages[name + '_offline'].items()},
            **{f'online.{label}': value for label, value in stages[name + '_online'].items()},
        }
        return {'module': name, 'status': 'estimated_pipelined_protocol', 'sample_count': 0,
                'samples_ms': [], 'mean_ms': 1000 * forecast['point_s'][key],
                'components_s': components, 'point_case': forecast['point_case'],
                'timing_scope': (forecast['initialization_timing_scope'] if name == 'SETUP_Init'
                                  else 'first_context_attempt_with_compiled_relation_and_cached_key' if name == 'P'
                                  else 'prepared_context_response_verification'),
                'kernel_rate_source': 'effective_throughput_assumptions',
                'full_module_executed': False, 'timings_measured': False,
                'protocol_matrices_constructed': False, 'simulation_only': True}
    rows = [estimated('SETUP_Init', 'SETUP_Init')]
    rows[0]['generation_and_storage_ms'] = 1000 * forecast['point_s']['Init_generate']
    rows[0]['with_prepared_commitment_ms'] = 1000 * forecast['point_s']['Init_with_prepared_commitment']
    rows[0]['commitment_material_preparation_ms'] = 1000 * forecast['point_s']['Init_commitment_material_preparation']
    rows[0]['implementation_complete_ms'] = 1000 * forecast['point_s']['Init_implementation_complete']
    initialization = forecast['stages_s'][forecast['point_case']]['initialization_plan']
    rows[0]['material_component_work_s'] = initialization['material_plan']['component_work_s']
    rows[0]['commitment_backend'] = initialization['commitment_backend']
    rows[0]['target_kernel_calibrated'] = initialization['material_plan']['target_kernel_calibrated']
    rows.append({'module': 'UKGEN', 'status': 'measured_GPV_plus_estimated_F_CPU',
                 'mean_ms': gpv['mean_ms'] + modules['UKGEN_F']['estimated_mean_ms'],
                 'sample_count': 0, 'samples_ms': [], 'full_module_executed': False,
                 'measured_GPV_ms': gpv['mean_ms'], 'estimated_F_ms': modules['UKGEN_F']['estimated_mean_ms']})
    rows += [estimated('P', 'P_total'), estimated('V', 'V_total')]
    return rows


def run_selected_sweep(params, settings, *, repeats=10, warmups=2, estimate_open=False,
                       on_progress=None):
    params.validate()
    settings.validate(params.q)
    if ((params.error_bound, params.sigma, params.beta) != (1, 2.25, 11)
            or not isinstance(params.chi, UniformBounded) or params.chi.bound != 1
            or (settings.l1, settings.l2, settings.p, settings.sigma1, settings.auxiliary_rows,
                settings.auxiliary_hiding_bits, settings.gaussian_tail) != (222,222,2,12,222,128,14)):
        raise ValueError('Selected finite profile required')
    for name, value, lower in (('repeats', repeats, 1), ('warmups', warmups, 0)):
        if type(value) is not int or value < lower:
            raise ValueError(f'{name} must be an integer >= {lower}')
    from .metal_matrix import available as metal_available
    calibration_workspace = (20*64*params.n*params.m + 256*1024**2
                             if metal_available(params.q) else 512*1024**2)
    required = 8*params.m**2 + 64*params.n*params.m + 24*params.m*512 + calibration_workspace
    require_memory(required, params.max_memory_bytes, 'GPV/module/calibration workspace')
    start = provenance()
    rng = CryptographicRNG()
    report = {'schema_version': 1, 'mode': 'selected_live_measurements_and_estimates', 'run_status': 'running',
              'params': params.to_dict(), 'yang_settings': asdict(settings),
              'repeats': repeats, 'warmups': warmups, 'group_sizes': [1 << d for d in PAPER_GROUP_DEPTHS],
              'N_independent_shared_modules': ['SETUP_GM', 'DEC', 'UKGEN'],
              'measurement_performed': True, 'estimate_calibration_performed': True,
              'full_primitive_recalibration_performed': True,
              'kernel_rate_source': 'current_multiscale_measurements',
              'reference_source': None,
              'randomness_source': type(rng).__name__, 'seed': None,
              'Gaussian_sampler_status': 'unbounded_interval_CDF; crypto_rng; not_constant_time',
              'author_finite_parameters': False, 'security_validated': False,
              'Yang_soundness_conditions_hold': False, 'actual_registered_population': 0,
              'synthetic_registered_population': 'N_per_capacity', 'membership_input_source': 'explicit_synthetic_leaves_not_F_bound',
              'GM_OPEN_uniform_backend':uniform_backend(),
              'JOIN_arithmetic_backend':'scalar_python_binary_sum',
              'OPEN_timing_scope':'OA_key_check_and_integer_identity_recovery',
              'full_F_generated': False, 'proof_executed': False, 'sender_proof_executed': False,
              'opening_rounds_executed': False,
              'complete_opening_proof_generated': False, 'complete_FDGE_chain_executed': False,
              'Init_P_V_mode': 'pipelined_point_estimates',
              'PV_mode': 'Yang_CRH_SIS_construction_estimate',
              'protocol_calibration_matrices_generated': False,
              'conservative_measured_workspace_bytes': required, 'provenance_start': start,
              'population_results': []}
    def progress(phase, **values):
        if on_progress:
            on_progress({'phase': phase, **values})
    progress('shared_public_matrices')
    (Arec, Aoa), a_ms = _timed(lambda: (rand_mod_matrix(rng,params.n,params.m,params.q),
                                       rand_mod_matrix(rng,params.n,params.m,params.q)))
    gpv_samples, gpv_checks, gpv_excluded = [], [], []
    gpv_warmups, gpv_repeats = warmups, repeats
    for trial in range(gpv_warmups + gpv_repeats):
        is_warmup = trial < gpv_warmups
        progress('actual_full_GPV_core', trial=trial+1,
                 trials=gpv_warmups+gpv_repeats, warmup=is_warmup)
        if trial:
            del recipient_pk, recipient_sk
        replay_rng = deepcopy(rng)
        (recipient_pk, recipient_sk), elapsed = _timed(lambda: blocked_recipient_keygen(params,Arec,rng))
        gpv_checks.append(_recipient_checks(params,Arec,recipient_pk,recipient_sk,replay_rng))
        if _rng_state(replay_rng) != _rng_state(rng):
            raise AssertionError('Recipient RNG replay mismatch')
        (gpv_excluded if is_warmup else gpv_samples).append(elapsed)
    gpv = {**_summary(gpv_samples), 'warmup_samples_ms_excluded': gpv_excluded,
           'warmup_count_excluded': len(gpv_excluded), 'checks': gpv_checks, 'F_computed': False,
           'output_shapes': {'U1': [params.n,params.m], 'U2': [params.n,params.m], 'E1': [params.m,params.m]}}
    report['shared_measured'] = {'UKGEN_GPV_core': gpv, 'Arec_Aoa_preparation_ms': a_ms}
    from .multiscale import calibrate_multiscale
    calibration = calibrate_multiscale(params, settings, Arec, on_progress=on_progress)
    commitment_calibration = calibration['commitment_product_calibration']
    report['commitment_product_calibration'] = commitment_calibration
    shared_measurements = {}
    for depth in PAPER_GROUP_DEPTHS:
        p = replace(params, depth=depth)
        progress('population_started', N=p.N, depth=depth)
        calibration['kernels']['oa_matvec'] = calibration['fixed_work']['oa_by_depth'][depth]
        open_required=required+_opening_workspace(p)
        measure_full_open=not estimate_open and open_required<=p.max_memory_bytes
        open_reason=('explicit_estimate_open' if estimate_open else
                     'full_proof_exceeds_memory_budget' if not measure_full_open else None)
        if open_reason=='full_proof_exceeds_memory_budget':
            progress('opening_proof_capacity',N=p.N,required_bytes=open_required,
                     limit_bytes=p.max_memory_bytes)
        names = ('SETUP_GM','SETUP_OA','JOIN_ISSUE','ENC','DEC','OPEN')
        samples, excluded = {name: [] for name in names}, {name: [] for name in names}
        for name in ('SETUP_GM', 'DEC'):
            if name in shared_measurements:
                progress('module_complete', N=p.N, row={
                    'module': name, 'status': 'measured_shared_N_independent_workload',
                    **deepcopy(shared_measurements[name])})

        def publish_measurement(name, trial, elapsed=None):
            if trial != warmups + repeats - 1 or name in shared_measurements:
                return
            progress('module_complete', N=p.N, row={
                'module': name, 'status': 'measured_selected_parameters',
                **_summary([*samples[name], elapsed])})

        opening_calibration=None
        with api._experiment(p, rng=rng, proof_settings=settings) as runtime:
            pp = api.PaperPublicParameters(PAPER_SECURITY_BITS,p,None,Arec,Aoa,None,chi=p.chi)
            runtime.pp = pp
            for trial in range(warmups + repeats):
                is_warmup = trial < warmups
                progress('actual_core_modules', N=p.N, trial=trial+1, trials=warmups+repeats,
                         warmup=is_warmup, index=trial+1 if is_warmup else trial-warmups+1,
                         count=warmups if is_warmup else repeats)
                times = {}
                if 'SETUP_GM' in shared_measurements:
                    gm_pk, gm_sk = api.SETUPGM(pp)
                else:
                    (gm_pk, gm_sk), times['SETUP_GM'] = _timed(lambda: api.SETUPGM(pp))
                publish_measurement('SETUP_GM', trial, times.get('SETUP_GM'))
                replay_rng = deepcopy(rng)
                (oa_pk, oa_sk), times['SETUP_OA'] = _timed(lambda: api.SETUPOA(pp))
                _oa_checks(p,Aoa,oa_pk,oa_sk,replay_rng)
                if _rng_state(replay_rng) != _rng_state(rng):
                    raise AssertionError('OA key generation randomness replay diverged')
                publish_measurement('SETUP_OA', trial, times['SETUP_OA'])
                gpk,reg,gm=api.INITIALIZE(pp,gm_pk,oa_pk)

                with uniform_source(rng) as source:
                    residues=uniform_field(source,(p.N,p.n),p.q)
                leaves=[bits_from_ints_le(row,p.k).astype(np.uint8,copy=False) for row in residues]
                if (not all(np.any(leaf) for leaf in leaves)
                        or len({leaf.tobytes() for leaf in leaves})!=p.N):
                    raise AssertionError('fixture leaves must be distinct and nonzero')
                with api.JOIN_CONTEXT(gpk,recipient_pk,gm.info):
                    manager=api.ISSUE(gm_sk)
                    for leaf in leaves[:-1]:
                        manager.receive(leaf)
                    prepared=manager.finish()
                if len(prepared)!=p.N-1:
                    raise AssertionError('fixture population preparation failed')
                del prepared,manager,residues
                public_acc=api.TSetup(api._gm_matrix(pp,gm_pk))
                fixture_leaf=leaves[-1]
                def join_interaction():
                    return _fixture_join_core(gpk,gm_sk,gm.info,recipient_pk,recipient_sk,fixture_leaf,public_acc)
                with _join_reference_arithmetic(gm.accumulator,public_acc):
                    certificate,times['JOIN_ISSUE']=_timed(join_interaction)
                if len(gm.info.witnesses)!=p.N:
                    raise AssertionError('JOIN did not publish N paths')
                publish_measurement('JOIN_ISSUE', trial, times['JOIN_ISSUE'])
                pkR, skR = api.Gr(pp)
                patterns, plaintext = api.sampleR(pkR)
                identity = certificate.path.index
                id_bits = int_to_bits_le(identity,p.depth)
                encrypted,times['ENC']=_timed(lambda:api.ENC(
                    gm_pk,oa_pk,recipient_pk,certificate,gm.info,patterns,plaintext,b'selected-GPV-core'))
                if encrypted is None:
                    raise AssertionError('ENC rejected fixture certificate')
                publish_measurement('ENC', trial, times['ENC'])
                Psi,coins=encrypted
                rec_ct,oa_ct=Psi.c_rec,Psi.c_oa
                rec_coins={'s':coins.s_rec,'x':coins.x_rec,'y':coins.y_rec}
                oa_coins={'s':coins.s_oa,'x':coins.x_oa,'y':coins.y_oa}

                with api._public_inputs(pp, patterns):
                    if 'DEC' in shared_measurements:
                        message = api.DEC(recipient_sk,Psi,b'selected-GPV-core')
                    else:
                        message, times['DEC'] = _timed(lambda: api.DEC(recipient_sk,Psi,b'selected-GPV-core'))
                publish_measurement('DEC', trial, times.get('DEC'))
                recovered=decrypt_oa(p,oa_sk,oa_ct)
                opening_identity,times['OPEN']=_timed(lambda:api.OPEN_IDENTITY(
                    oa_sk,gm.info,reg,Psi,b'selected-GPV-core'))
                if opening_identity!=identity:
                    raise AssertionError('timed OPEN identity recovery failed')
                publish_measurement('OPEN', trial, times['OPEN'])
                if measure_full_open:
                    progress('full_OPEN_started',N=p.N,trial=trial+1,trials=warmups+repeats)
                    opened=api.OPEN(oa_sk,gm.info,reg,Psi,b'selected-GPV-core')
                    if opened[0]!=identity or not api.VERIFY_OPEN(gpk,gm.info,reg,Psi,b'selected-GPV-core',opened):
                        raise AssertionError('OPEN identity/proof check failed')
                    report['proof_executed']=True
                    report['opening_rounds_executed']=True
                    report['complete_opening_proof_generated']=True
                    del opened
                    progress('full_OPEN_complete',N=p.N,trial=trial+1,trials=warmups+repeats)
                else:
                    from .opening import OPEN_CONTEXT, _active_opening, _prepare_with_context
                    def prepare_open():
                        with OPEN_CONTEXT(gpk):
                            return _prepare_with_context(_active_opening.get(),oa_sk,gm.info,reg,Psi,b'selected-GPV-core')
                    opening_statement,opening_secret,prepared_identity=prepare_open()
                    if prepared_identity!=identity:
                        raise AssertionError('OPEN fixture identity mismatch')
                valid = (np.array_equal(gm_pk.pk,(api._gm_matrix(pp,gm_pk) @ gm_sk.sk) % p.q)
                         and pkR == (p.t,p.e) and skR == b''
                         and _independent_policy_check(patterns,plaintext,p.d)
                         and message is not None and np.array_equal(message,plaintext)
                         and np.array_equal(recovered,id_bits)
                         and _ciphertext_check(p,Arec,recipient_pk,plaintext,rec_ct,rec_coins)
                         and _ciphertext_check(p,Aoa,oa_pk,id_bits,oa_ct,oa_coins))
                if not valid:
                    raise AssertionError('Core correctness check failed')
                for name, elapsed in times.items():
                    (excluded if trial < warmups else samples)[name].append(elapsed)

                runtime.groups.pop(id(gpk))
            if not measure_full_open:
                from .stern_opening import calibrate_opening
                progress('opening_proof_round_calibration',N=p.N,reason=open_reason)
                opening_calibration=calibrate_opening(opening_statement,opening_secret,repeats=3)
        for name in ('SETUP_GM', 'DEC'):
            if name not in shared_measurements:
                shared_measurements[name] = {
                    **_summary(samples[name]),
                    'warmup_samples_ms_excluded': excluded[name],
                    'warmup_count_excluded': len(excluded[name]),
                    'measured_at_N': p.N,
                    'measurement_shared_across_N': True,
                    'N_work_relation': 'constant_for_fixed_non_population_parameters',
                }
        report['shared_measured'].update(deepcopy(shared_measurements))
        progress('protocol_step_simulation', N=p.N)
        model = estimate_modules(p,settings,calibration)
        from .multiscale import compose_matrix_products
        compose_matrix_products(model, calibration)
        leaf_kernels = calibration['fixed_work']['published_path']
        rows = [{'module': name, 'status': 'measured_selected_parameters',
                 **(_summary(samples[name]) if name not in shared_measurements else deepcopy(shared_measurements[name])),
                 **({'warmup_samples_ms_excluded': excluded[name],
                     'warmup_count_excluded': len(excluded[name])} if name not in shared_measurements else {})} for name in names]
        for row in rows:
            if row['module'] in shared_measurements:
                row.update(status='measured_shared_N_independent_workload')
            if row['module']=='JOIN_ISSUE':
                row.update(status='measured_interaction_core_with_fixture',
                           full_module_executed=False,module_preconditions_fully_validated=False,
                           synthetic_registered_population=p.N,
                           arithmetic_backend=report['JOIN_arithmetic_backend'])
            if row['module']=='ENC':
                row.update(status='measured_full_entry_with_membership_fixture',full_module_executed=True,
                           membership_input_source='explicit_synthetic_leaf_not_F_bound',
                           module_preconditions_fully_validated=False)
            if row['module']=='OPEN':
                row.update(status='measured_key_checked_identity_with_membership_fixture',
                           full_module_executed=False,identity_recovery_executed=True,
                           OA_key_consistency_checked_inside_timer=True,
                           module_preconditions_fully_validated=False,
                           complete_proof_generated=measure_full_open,
                           public_proof_verification_passed=measure_full_open,
                           Stern_timed=False,timing_scope=report['OPEN_timing_scope'])
        from .pv_estimate import estimate_population
        forecast = estimate_population(p, settings)
        estimates = _model_rows(model, gpv, forecast)
        from .memory_model import assess_pv_memory
        assessment = assess_pv_memory(p, settings)
        for row in estimates:
            row.setdefault('kernel_rate_source', report['kernel_rate_source'])
            row['estimate_model_evaluated_this_run'] = True
            if row['module'] in ('P', 'V'):
                row['resource_assessment'] = assessment
                row['supported_device_runtime_ms'] = None
        rows.extend(estimates)
        for row in estimates:
            progress('module_complete', N=p.N, row=row)

        d = model['dimensions']
        report['population_results'].append({'N': p.N, 'depth': depth, 'params': p.to_dict(), 'rows': rows,
            'N_work_counts': {'log2N': depth, 'OA_secret_entries_per_branch': p.m * depth,
                'single_member_path_levels': depth, 'registration_rows': p.N,
                'published_path_siblings': p.N * depth,
                'witness_coordinates': d['witness_coordinates'],
                'quadratic_constraints': d['quadratic_constraints']},
            'actual_registered_population': 0,'synthetic_registered_population':p.N,
            'opening_mode':'identity_measured_full_proof_untimed' if measure_full_open else 'identity_measured_proof_rounds_untimed',
            'opening_proof_round_calibration_untimed':opening_calibration,
            'opening_estimate_reason':open_reason,'aggregate_full_OPEN_array_plan_bytes':open_required,
            'checks_including_warmups_passed': warmups+repeats,
            'estimate_model_evaluated_this_run': True, 'estimate_calibration_performed': True,
            'full_primitive_recalibration_performed': True,
            'kernel_rate_source': report['kernel_rate_source'],
            'correctness_scope': {'GM_product_checked': True, 'OA_U1_product_checked': True,
                'OA_U2_product_checked': True, 'recipient_U1_U2_projections_checked': True,
                'policy_all_windows_checked': True, 'both_recipient_ciphertext_equations_checked': True,
                'both_OA_ciphertext_equations_checked': True, 'message_and_identity_recovered': True,
                'encryption_randomness_domains_checked': True,
                'published_fixture_membership_verified':True,
                'fixture_JOIN_ISSUE_interaction_executed':True,
                'full_JOIN_F_cache_initialization_executed':False,
                'full_ENC_entry_executed':True,'full_OPEN_entry_executed':measure_full_open,
                'OPEN_OA_key_consistency_checked_inside_timer':True,
                'complete_opening_proof_publicly_verified':measure_full_open,
                'membership_or_F_binding_checked': False},
            'kernel_calibration': calibration, 'cpu_model': model, 'pipelined_estimate': forecast,
            'PV_resource_assessment': assessment,
            'published_path_check_kernel_rates':leaf_kernels})
        progress('population_complete',N=p.N,population=report['population_results'][-1])
    end = provenance()
    report['provenance_end'] = end
    changed = changed_provenance_files(start, end)
    report['provenance_changed_files'] = changed
    unchanged = not changed
    report['source_and_references_unchanged'] = unchanged
    report['run_status'] = 'completed_selected_core_measurements_and_CPU_models' if unchanged else 'invalidated_by_source_change'
    return report
