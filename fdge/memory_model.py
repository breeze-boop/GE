from __future__ import annotations

from functools import lru_cache
import ctypes
import os
import platform
import subprocess


@lru_cache(maxsize=1)
def hardware_memory():
    def sysctl(name):
        result = subprocess.run(['sysctl', '-n', name], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    if platform.system() == 'Darwin':
        physical = int(sysctl('hw.memsize'))
        chip = sysctl('machdep.cpu.brand_string')
    else:
        physical = os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        chip = platform.processor()
    return {'physical_bytes': physical, 'chip': chip}


def assess_pv_memory(params, settings, hardware=None):
    from .relations import relation_resource_plan
    from .commitments import _lattice_storage_dtype
    from . import metal_matrix
    params.validate()
    settings.validate(params.q)
    host = dict(hardware_memory() if hardware is None else hardware)
    relation = relation_resource_plan(params)
    w, c, rows = (relation[key] for key in ('witness_coordinates', 'triples', 'rows'))
    l1, l2 = settings.l1, settings.l2
    itemsize = _lattice_storage_dtype(params.q).itemsize
    backend = metal_matrix._backend() if itemsize == 4 else None
    device_limit = backend.max_buffer_length if backend is not None else None
    gpu_resident = (backend.runtime.call(backend.device, 'recommendedMaxWorkingSetSize', metal_matrix._SIZE)
                    if backend is not None else None)
    shared = (backend is None or bool(backend.runtime.call(backend.device, 'hasUnifiedMemory', ctypes.c_bool)))
    host['CPU_GPU_unified_memory'] = shared
    h1, h2 = l1+w+l2, l1+c+l2
    rho = settings.auxiliary_rows * (params.q-1).bit_length() + 2*settings.auxiliary_hiding_bits
    matrices = {'F': 8*params.n*(2*params.n*params.m*params.k),
                'B1_top': itemsize*l1*(w+l2), 'B1_bottom': itemsize*w*l2,
                'B2_top': itemsize*l1*(c+l2), 'B2_bottom': itemsize*c*l2,
                'auxiliary': 8*settings.auxiliary_rows*(256+rho)}
    b1 = matrices['B1_top'] + matrices['B1_bottom']
    b2 = matrices['B2_top'] + matrices['B2_bottom']
    fixed = sum(matrices.values()) + relation['triples_storage_bytes'] + 8*rows
    ticket = 8*(5*w+4*c+8*l1+4*l2+rows+rho)
    payload = 8*(4*w+3*c+rows+6*l1+2*l2+rho)
    vector_live = {'P_offline': ticket+8*c,
                   'P_online': max(ticket+16*w+8*c, ticket+16*w+8*h1+24*h2),
                   'V_online': 2*payload+8*c+8*(l1+w)+8*(l1+c)}
    largest = max(matrices[name] for name in ('B1_top', 'B1_bottom', 'B2_top', 'B2_bottom'))
    budget = min(params.max_memory_bytes, settings.max_memory_bytes)
    reasons = []
    if fixed > budget:
        reasons.append('fixed_resident_arrays_exceed_configured_budget')
    if fixed > host['physical_bytes']:
        reasons.append('fixed_resident_arrays_exceed_physical_memory')
    if device_limit is not None and largest > device_limit:
        reasons.append('monolithic_commitment_buffer_exceeds_Metal_limit')
    feasible = not reasons
    phase_status = {}
    for name, live in vector_live.items():
        why = list(reasons)
        if fixed+live > budget and 'fixed_resident_arrays_exceed_configured_budget' not in why:
            why.append('live_arrays_exceed_configured_budget')
        if fixed+live > host['physical_bytes'] and 'fixed_resident_arrays_exceed_physical_memory' not in why:
            why.append('live_arrays_exceed_physical_memory')
        phase_status[name] = {'status': 'capacity_infeasible' if why else 'large_scale_transfer_unvalidated',
                             'runtime_estimate_ms': None, 'reasons': why,
                             'explicit_live_bytes_lower_account': fixed+live,
                             'vector_live_bytes_lower_account': live}
    feasible = all(not phase['reasons'] for phase in phase_status.values())
    reasons = sorted({reason for phase in phase_status.values() for reason in phase['reasons']})
    return {'N': params.N, 'hardware': host, 'configured_budget_bytes': budget,
            'commitment_storage_itemsize': itemsize, 'F_storage': 'resident_float64',
            'matrices_bytes': matrices, 'triples_bytes': relation['triples_storage_bytes'],
            'fixed_resident_bytes_lower_account': fixed,
            'largest_commitment_buffer_bytes': largest, 'Metal_max_buffer_bytes': device_limit,
            'Metal_recommended_working_set_bytes': gpu_resident,
            'offline_ticket_bytes': ticket, 'received_response_bytes': payload,
            'capacity_feasible': feasible, 'status': 'capacity_infeasible' if not feasible else 'large_scale_transfer_unvalidated',
            'reasons': reasons, 'phases': phase_status,
            'dense_matrix_stream_once_bytes': b1+b2+matrices['F'],
            'full_PV_out_of_core_backend_implemented': False, 'swap_or_disk_time_estimated': False,
            'full_target_executed': False}


def annotate_conditional_runtime(row, assessment):
    row['resource_assessment'] = assessment
    row['estimate_assumption'] = 'sufficient_resident_memory_no_swap_or_disk'
    row['supported_device_runtime_ms'] = {'P': None, 'V': None}
    row['full_target_executed'] = False
    return row
