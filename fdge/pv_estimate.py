from __future__ import annotations

from dataclasses import asdict, replace
from concurrent.futures import ThreadPoolExecutor
import ctypes
import math
import mmap
import os
import platform
import statistics
from threading import Barrier
import time

import numpy as np

from .cost_model import target_dimensions
from .params import PAPER_GROUP_DEPTHS, load_selected_profile


_RATE_CASES = {
    'fast': dict(
        HBM=1.2544e12, stream=.8e12, MAC=.4e12, rng=16e9,
        alias=1e9, gauss=.30e9, array=8e9, constraint=2e9,
        norm=1e9, sha=3e9, pcie=50e9, hostmem=200e9,
        launch=10e-6, control=5e-6, cpu_MAC=60e9, decode=800e9, pack=20e9,
    ),
    'conservative': dict(
        HBM=.896e12, stream=.3e12, MAC=.1e12, rng=8e9,
        alias=.25e9, gauss=.08e9, array=2e9, constraint=.5e9,
        norm=.25e9, sha=2e9, pcie=40e9, hostmem=120e9,
        launch=40e-6, control=20e-6, cpu_MAC=30e9, decode=200e9, pack=8e9,
    ),
}
_PACK_BLOCK = 65536
_PIPELINE_BUFFER_BYTES = 512 * 1024**2
_POINT_CASE = 'fast'
_RESPONSE_CHUNK_ELEMENTS = 1024**2
_INIT_DEVICE_CAPACITY_BYTES = 32_000_000_000
_INIT_DEVICE_PEAK_BANDWIDTH = 1.792e12
_INIT_MULTIPROCESSORS = 21760 // 128
_INIT_BASE_CLOCK_HZ = 2.01e9
_INIT_WARP_SIZE = 32
_INIT_SCAN_THREADS = 256
_INIT_RESULTS_PER_SM_CYCLE = {
    'add': 128, 'multiply_wide': 32, 'shift': 64, 'bitwise': 64,
    'compare': 64, 'shuffle': 32, 'population_count': 16,
    'ballot': 64, 'find_first': 16, 'barrier': 16,
}
_INIT_RANDOM_REFERENCE = {
    'algorithm': 'AES_128_CTR',
    'device_output_bytes_per_second': 1623e9 / 8,
    'source': 'AES_CTR_Speed_Record_publication_1623_Gbit_per_second',
    'target_device_measurement': False,
    'complete_field_generation_measurement': False,
}
_INIT_RUNTIME_REFERENCE = {
    'registration_source': 'bonsai_2026_decision_139_same_pod_API_measurements',
    'reported_buffer_bytes': 8.59e9,
    'register_range_s': (.26, .32),
    'unregister_range_s': (.11, .14),
    'registration_fixed_intercept_identified': False,
    'workspace_allocation': 'prewarmed_device_pool_checkout_and_return',
    'runtime_context_startup_included': False,
    'reference_hardware_and_driver_match_target': False,
    'target_runtime_calibrated': False,
}
_INIT_D2H_REFERENCE = {
    'source': 'AI_fundamentals_2026_nvbandwidth_v0_9_CE_D2H_report',
    'device_to_host_bytes_per_second': 56.77e9,
    'measurement_path': 'unidirectional_copy_engine_to_pinned_host_memory',
    'source_runtime_log_version': 13020,
    'source_intro_toolkit_version_disagrees_with_runtime_log': True,
    'independently_reproduced': False,
    'target_device_calibrated': False,
    'concurrent_generation_DMA_and_host_copy_measured': False,
}
_INIT_HOST_CALIBRATIONS = {}
_INIT_PAGE_CALIBRATION = None


def _pipeline_seconds(work, units, chunk):
    full, last = divmod(units, chunk)
    times = [value * chunk / units for value in work]
    completed = [sum(times[:i + 1]) + (full - 1) * max(times[:i + 1]) if full else 0.0
                 for i in range(len(times))]
    if last:
        previous = 0.0
        for i, duration in enumerate(work):
            completed[i] = max(completed[i], previous) + duration * last / units
            previous = completed[i]
    return completed[-1]


def _response_preparation(w, h1, h2, rho, response_bytes, rates):
    copied, checked = 0.0, 0.0
    check_work = 0.0
    vectors = ((w, 'array', 24), (h1, 'norm', 24), (h2, 'norm', 24), (rho, 'array', 16))
    for size, kind, traffic in vectors:
        for start in range(0, size, _RESPONSE_CHUNK_ELEMENTS):
            count = min(_RESPONSE_CHUNK_ELEMENTS, size - start)
            copied += 8 * count / rates['pcie']
            operation = max(count / rates[kind], traffic * count / rates['stream']) + rates['launch']
            check_work += operation
            checked = max(checked, copied) + operation
    z_bytes = 8 * (w + h1 + h2 + rho)
    copied += (response_bytes - z_bytes) / rates['pcie']
    return {
        'estimated_seconds': max(copied, checked),
        'transferred_bytes': response_bytes,
        'checked_coordinates': w + h1 + h2 + rho,
        'transfer_seconds': copied,
        'check_work_seconds': check_work,
        'serial_seconds': copied + check_work,
    }


def _scalar_frame_bytes(value):
    return 8 + len(value if isinstance(value, bytes) else str(value).encode())


def _array_frame_bytes(shape):
    return 8 + len(str(shape).encode() + b':i8:') + 8 * math.prod(shape)


def _estimate_context_prefix(dimensions, rates):
    rows, count = dimensions['relation_rows'], dimensions['quadratic_constraints']
    coordinates = rows + 3 * count
    index_bytes = 4 if dimensions['witness_coordinates'] <= (1 << 31) - 1 else 8
    source_bytes = 8 * rows + index_bytes * 3 * count
    hash_bytes = (len(b'Yang P1 FDGE statement context v1') + _scalar_frame_bytes(bytes(32))
                  + _array_frame_bytes((rows,)) + _array_frame_bytes((count, 3)))
    encoding = max(coordinates / rates['cpu_MAC'],
                   (source_bytes + 8 * coordinates) / rates['hostmem'])
    hashing = hash_bytes / rates['sha'] + hash_bytes / rates['hostmem'] + rates['control']
    elapsed = _pipeline_seconds([encoding, hashing], coordinates, 262144)
    return {
        'estimated_seconds': elapsed,
        'canonical_hash_bytes': hash_bytes,
        'encoded_coordinates': coordinates,
        'source_bytes': source_bytes,
        'encoding_seconds': encoding,
        'hash_seconds': hashing,
        'encoding_rate_source': 'shared_cpu_MAC_rate_proxy',
    }


def _scan_work(length):
    padded = 1 << (length - 1).bit_length() if length else 0
    levels = (padded.bit_length() - 1) if padded else 0
    return 2 * max(0, padded - 1), 2 * levels * min(_INIT_SCAN_THREADS, padded)


def _fused_initialization_work(count, q, packed, packed_bytes):
    warp = _INIT_WARP_SIZE
    warps = (count + warp - 1) // warp
    lanes = warp * warps
    rejection = ((1 << 32) % q) / (1 << 32)
    retry_work, retry_probability = 1.0, rejection
    while retry_probability:
        extra = -math.expm1(warp * math.log1p(-retry_probability))
        retry_work += extra
        if extra <= math.ulp(retry_work):
            break
        retry_probability *= rejection
    issued = lanes * retry_work
    random_bytes = 4 * issued
    operations = {key: 0.0 for key in _INIT_RESULTS_PER_SM_CYCLE}
    operations.update(multiply_wide=2 * issued, compare=issued, ballot=issued)
    operations['bitwise'] = issued
    blocks = (count + _PACK_BLOCK - 1) // _PACK_BLOCK if packed else 0
    output_bytes = packed_bytes(count) if packed else 8 * count
    bitmap_bytes = 4 * warps if packed else 0
    global_bytes = output_bytes
    shared_bytes = 0
    if packed:
        low_bits = q.bit_length() - 1
        operations['multiply_wide'] += 2 * lanes
        operations['shift'] += 5 * lanes
        operations['bitwise'] += 3 * lanes
        operations['add'] += 5 * lanes
        operations['compare'] += 3 * lanes
        operations['shuffle'] += 3 * lanes
        operations['ballot'] += lanes
        full, tail = divmod(count, _PACK_BLOCK)
        scan_nodes, barriers = _scan_work(_PACK_BLOCK // warp)
        tail_nodes, tail_barriers = _scan_work((tail + warp - 1) // warp)
        local_nodes = full * scan_nodes + tail_nodes
        local_barriers = full * barriers + tail_barriers
        prefix_nodes, prefix_barriers = _scan_work(blocks)
        exceptions = count * (q - (1 << low_bits)) / q
        operations['add'] += local_nodes + prefix_nodes + 2 * exceptions
        operations['population_count'] += warps
        operations['find_first'] += exceptions
        operations['bitwise'] += 2 * exceptions
        operations['barrier'] += 2 * local_barriers + prefix_barriers
        shared_bytes = 12 * (2 * local_nodes + prefix_nodes)
        global_bytes += 2 * bitmap_bytes + 12 * blocks
        workspace_bytes = 2 * (((low_bits * count + 7) // 8) + 2 * count
                               + bitmap_bytes + 8 * blocks)
        kernels = 3
    else:
        operations['add'] += 2 * lanes
        workspace_bytes, kernels = 16 * count, 1
    instruction_work = {
        key: value / (_INIT_MULTIPROCESSORS * _INIT_BASE_CLOCK_HZ * _INIT_RESULTS_PER_SM_CYCLE[key])
        for key, value in operations.items()
    }
    return {
        'random_bytes': random_bytes, 'warp_retry_factor': retry_work,
        'instruction_operations': operations, 'instruction_work_s': instruction_work,
        'global_bytes': global_bytes, 'shared_bytes': shared_bytes,
        'workspace_bytes': workspace_bytes, 'kernel_count': kernels,
        'output_bytes': output_bytes,
    }


def _pack_initialization_values(values, q):
    count = len(values)
    if not count:
        return {'q': q, 'count': 0, 'low': b'', 'exceptions': b''}
    low_bits = q.bit_length() - 1
    exceptions = np.flatnonzero(values >> low_bits).astype('<u2').tobytes()
    groups = (count + 7) // 8
    lanes = np.zeros(groups * 8, dtype=np.uint64)
    np.bitwise_and(values, (1 << low_bits) - 1, out=lanes[:count])
    lanes = lanes.reshape(groups, 8)
    words = np.zeros((groups, (low_bits + 7) // 8), dtype='<u8')
    for lane in range(8):
        column, shift = divmod(lane * low_bits, 64)
        words[:, column] |= lanes[:, lane] << shift
        if shift + low_bits > 64:
            words[:, column + 1] |= lanes[:, lane] >> (64 - shift)
    low = words.view(np.uint8).reshape(groups, -1)[:, :low_bits].tobytes()
    return {'q': q, 'count': count, 'low': low[:(count * low_bits + 7) // 8],
            'exceptions': exceptions}


def _copy_initialization_material(destination, source, executor, workers):
    if workers == 1:
        np.copyto(destination, source)
        return
    count = len(source)
    futures = [executor.submit(np.copyto,
                               destination[count * index // workers:count * (index + 1) // workers],
                               source[count * index // workers:count * (index + 1) // workers])
               for index in range(workers)]
    for future in futures:
        future.result()


class _PageLockedFinalWindow:
    def __init__(self, size):
        self.page_bytes = os.sysconf('SC_PAGE_SIZE')
        if not 0 < size <= _PIPELINE_BUFFER_BYTES // 2:
            raise ValueError('Final-memory window exceeds budget')
        self.size = size
        self.allocated_bytes = math.ceil(size / self.page_bytes) * self.page_bytes
        self.region = mmap.mmap(-1, self.allocated_bytes,
                                flags=mmap.MAP_PRIVATE | mmap.MAP_ANON,
                                prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.region))
        self.library = ctypes.CDLL(None, use_errno=True)
        for name in ('mlock', 'munlock'):
            operation = getattr(self.library, name)
            operation.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            operation.restype = ctypes.c_int
        self.locked = False

    def lock_pages(self):
        if self.locked:
            raise RuntimeError('Final-memory window is already locked')
        if self.library.mlock(self.address, self.allocated_bytes):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        self.locked = True

    def unlock_pages(self):
        if not self.locked:
            raise RuntimeError('Final-memory window is not locked')
        if self.library.munlock(self.address, self.allocated_bytes):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        self.locked = False

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        try:
            if self.locked:
                self.unlock_pages()
        finally:
            self.region.close()


def _calibrate_initialization_pages():
    global _INIT_PAGE_CALIBRATION
    if _INIT_PAGE_CALIBRATION is not None:
        return _INIT_PAGE_CALIBRATION
    if os.name != 'posix':
        _INIT_PAGE_CALIBRATION = {'status': 'OS_page_lock_measurement_unavailable', 'cases': [],
                                  'device_register_s': None, 'device_unregister_s': None,
                                  'device_completion_synchronization_s': None}
        return _INIT_PAGE_CALIBRATION
    warmups, repeats = 2, 10
    cases = []

    def trial(window, touch_first):
        start = time.perf_counter_ns()
        if touch_first:
            ctypes.memset(window.address, 165, window.size)
        touched = time.perf_counter_ns()
        window.lock_pages()
        locked = time.perf_counter_ns()
        window.unlock_pages()
        unlocked = time.perf_counter_ns()
        if (any(window.region[offset] != 165
                for offset in range(0, window.size, window.page_bytes))
                or window.region[window.size - 1] != 165):
            raise ArithmeticError('Page lock/unlock changed final-memory contents')
        return {'first_write_s': (touched - start) / 1e9 if touch_first else 0.0,
                'lock_s': (locked - touched) / 1e9,
                'unlock_s': (unlocked - locked) / 1e9,
                'cycle_s': (unlocked - touched) / 1e9,
                'full_host_preparation_s': (unlocked - start) / 1e9}

    try:
        for size in (4 * 1024**2, 16 * 1024**2, 64 * 1024**2, 256 * 1024**2):
            for mode in ('resident', 'fresh_writable_pages'):
                samples = []
                if mode == 'resident':
                    with _PageLockedFinalWindow(size) as window:
                        ctypes.memset(window.address, 165, size)
                        for _ in range(warmups):
                            trial(window, False)
                        for _ in range(repeats):
                            samples.append(trial(window, False))
                else:
                    for index in range(warmups + repeats):
                        with _PageLockedFinalWindow(size) as window:
                            sample = trial(window, True)
                            if index >= warmups:
                                samples.append(sample)
                cases.append({'window_bytes': size, 'mode': mode, 'samples': samples,
                              'first_write_s': statistics.median(row['first_write_s'] for row in samples),
                              'lock_s': statistics.median(row['lock_s'] for row in samples),
                              'unlock_s': statistics.median(row['unlock_s'] for row in samples),
                              'cycle_s': statistics.median(row['cycle_s'] for row in samples)})
    except OSError as error:
        _INIT_PAGE_CALIBRATION = {'status': 'OS_page_lock_measurement_failed', 'cases': cases,
                                  'error_errno': error.errno, 'device_register_s': None,
                                  'device_unregister_s': None,
                                  'device_completion_synchronization_s': None}
        return _INIT_PAGE_CALIBRATION
    _INIT_PAGE_CALIBRATION = {
        'status': 'OS_page_lock_only_measured', 'warmups': warmups, 'repeats': repeats,
        'page_bytes': os.sysconf('SC_PAGE_SIZE'), 'cases': cases,
        'source': 'local_mlock_munlock_on_anonymous_final_memory_windows',
        'device_register_s': None, 'device_unregister_s': None,
        'device_completion_synchronization_s': None,
        'OS_lock_is_not_device_registration_measurement': True,
        'driver_mapping_and_event_costs_measured': False,
    }
    return _INIT_PAGE_CALIBRATION


def _calibrate_initialization_host(q):
    from .fast_uniform import uniform_field, uniform_source, uniform_backend
    from .randomness import CryptographicRNG

    if q in _INIT_HOST_CALIBRATIONS:
        return _INIT_HOST_CALIBRATIONS[q]
    platform_name = f'{platform.system()}-{platform.release()}-{platform.machine()}'
    warmups, repeats = 2, 10
    low_bits = q.bit_length() - 1
    block_bytes = (low_bits * _PACK_BLOCK + 7) // 8 + 4
    blocks = math.ceil(64 * 1024**2 / block_bytes)
    coefficients = blocks * _PACK_BLOCK
    capacity = blocks * (block_bytes + 2 * _PACK_BLOCK)
    output = np.empty(capacity, dtype=np.uint8)
    output.fill(0)
    seed = bytes(range(32))
    cases = []

    def generate():
        cursor = 0
        with uniform_source(CryptographicRNG(seed)) as source:
            for _ in range(blocks):
                values = uniform_field(source, (_PACK_BLOCK,), q, dtype=np.uint32)
                block = _pack_initialization_values(values, q)
                header = (len(block['exceptions']) // 2).to_bytes(4, 'little')
                for part in (header, block['low'], block['exceptions']):
                    end = cursor + len(part)
                    output[cursor:end] = np.frombuffer(part, dtype=np.uint8)
                    cursor = end
        return cursor

    expected_length = generate()
    expected = output[:expected_length].copy()
    with (ThreadPoolExecutor(max_workers=2) as executor,
          ThreadPoolExecutor(max_workers=4) as copy_executor):
        for copy_bytes, copy_workers in ((size, workers) for size in (64 * 1024**2, 256 * 1024**2)
                                         for workers in (1, 2, 4)):
            source = np.arange(copy_bytes, dtype=np.uint8)
            destination = np.empty_like(source)
            destination.fill(0)

            def copy_once():
                _copy_initialization_material(destination, source, copy_executor, copy_workers)

            for _ in range(warmups):
                generate()
                copy_once()
            start = time.perf_counter_ns()
            generate()
            generation_pilot = time.perf_counter_ns() - start
            start = time.perf_counter_ns()
            for _ in range(repeats):
                copy_once()
            copy_pilot = (time.perf_counter_ns() - start) / repeats
            copies = max(1, round(generation_pilot / copy_pilot))
            payload_bytes = copy_bytes * copies

            def copy():
                for _ in range(copies):
                    copy_once()
                return payload_bytes

            def timed(operation, gate):
                gate.wait()
                started = time.perf_counter_ns()
                result = operation()
                return {'started_ns': started, 'finished_ns': time.perf_counter_ns(),
                        'result': result}

            def trial(mode):
                names = ('generation', 'copy') if mode == 'concurrent' else (mode,)
                gate = Barrier(len(names) + 1)
                futures = {name: executor.submit(timed, generate if name == 'generation' else copy, gate)
                           for name in names}
                gate.wait()
                record = {name: future.result() for name, future in futures.items()}
                if 'generation' in record:
                    if (record['generation']['result'] != expected_length
                            or not np.array_equal(output[:expected_length], expected)):
                        raise ArithmeticError('CPU sampling mismatch')
                if 'copy' in record and not np.array_equal(destination, source):
                    raise ArithmeticError('Host copy mismatch')
                first = min(value['started_ns'] for value in record.values())
                last = max(value['finished_ns'] for value in record.values())
                elapsed = (last - first) / 1e9
                overlap = (max(0, min(value['finished_ns'] for value in record.values())
                               - max(value['started_ns'] for value in record.values())) / 1e9
                           if len(record) == 2 else 0.0)
                return {'elapsed_s': elapsed, 'generation_coefficients_per_second':
                        coefficients / elapsed if 'generation' in record else None,
                        'copy_payload_bytes_per_second': payload_bytes / elapsed if 'copy' in record else None,
                        'overlap_fraction_of_joint_elapsed': overlap / elapsed,
                        'branches': record}

            modes = ('generation', 'copy', 'concurrent')
            for _ in range(warmups):
                observed = trial('concurrent')['branches']
                generation_ns = (observed['generation']['finished_ns']
                                 - observed['generation']['started_ns'])
                copy_ns = observed['copy']['finished_ns'] - observed['copy']['started_ns']
                copies = max(1, round(copies * generation_ns / copy_ns))
                payload_bytes = copy_bytes * copies
            for _ in range(warmups):
                for mode in modes:
                    trial(mode)
            samples = {mode: [] for mode in modes}
            for index in range(repeats):
                for mode in modes[index % len(modes):] + modes[:index % len(modes)]:
                    samples[mode].append(trial(mode))
            medians = {mode: statistics.median(row['elapsed_s'] for row in rows)
                       for mode, rows in samples.items()}
            cases.append({
                'copy_buffer_bytes': copy_bytes, 'copy_iterations': copies,
                'copy_workers': copy_workers,
                'copy_payload_bytes': payload_bytes, 'generation_coefficients': coefficients,
                'generation_output_bytes': expected_length,
                'generation_solo_coefficients_per_second': coefficients / medians['generation'],
                'copy_solo_payload_bytes_per_second': payload_bytes / medians['copy'],
                'generation_concurrent_coefficients_per_second': coefficients / medians['concurrent'],
                'copy_concurrent_payload_bytes_per_second': payload_bytes / medians['concurrent'],
                'generation_concurrent_active_coefficients_per_second': coefficients / statistics.median(
                    (row['branches']['generation']['finished_ns'] - row['branches']['generation']['started_ns']) / 1e9
                    for row in samples['concurrent']),
                'copy_concurrent_active_payload_bytes_per_second': payload_bytes / statistics.median(
                    (row['branches']['copy']['finished_ns'] - row['branches']['copy']['started_ns']) / 1e9
                    for row in samples['concurrent']),
                'median_joint_overlap_fraction': statistics.median(
                    row['overlap_fraction_of_joint_elapsed'] for row in samples['concurrent']),
                'samples': samples,
            })
            del source, destination
    largest = max(case['copy_buffer_bytes'] for case in cases)
    selected = max((case for case in cases if case['copy_buffer_bytes'] == largest),
                   key=lambda case: case['copy_solo_payload_bytes_per_second'])
    calibration = {
        'q': q, 'warmups': warmups, 'repeats': repeats, 'cases': cases,
        'selected_case': selected, 'selection': 'largest_buffer_best_median_copy_implementation',
        'source': 'local_CPU_generation_and_host_copy_solo_and_concurrent_wall_clock_measurements',
        'platform': platform_name, 'uniform_backend': uniform_backend(),
        'generation_backend': 'native_random_stream_unbiased_mapping_numpy_lossless_pack_final_store',
        'CPU_generation_and_copy_joint_throughput_measured': True,
        'pilot_workload_balancing_rounds': warmups,
        'DMA_contention_measured': False, 'target_device_calibration': False,
        'allocation_and_first_touch_in_timed_region': False,
        'GPU_rates_are_separate_uncalibrated_conditions': True,
    }
    _INIT_HOST_CALIBRATIONS[q] = calibration
    return calibration


def _initialization_material_schedule(groups, q, rates, packed_bytes, cpu_counts=None,
                                      *, _host_case=None, _runtime_costs=None):
    entries = sum(count for _, count, _ in groups)
    matrix_entries = sum(count for _, count, packed in groups if packed)
    packed_storage = sum(packed_bytes(count) for _, count, packed in groups if packed)
    retained_bytes = packed_storage + 8 * (entries - matrix_entries)
    uniform_accept = 1 - ((1 << 32) % q) / (1 << 32)
    transfer_bandwidth = min(rates['pcie'], rates['hostmem'] / 3)
    copy_bandwidth = rates['hostmem'] - transfer_bandwidth
    if copy_bandwidth <= 0:
        raise ArithmeticError('Insufficient host bandwidth')
    device_bandwidth = min(rates['stream'], _INIT_DEVICE_PEAK_BANDWIDTH)
    random_rate = _INIT_RANDOM_REFERENCE['device_output_bytes_per_second']
    cpu_counts = {} if cpu_counts is None else cpu_counts
    calibration = _calibrate_initialization_host(q)
    measured = calibration['selected_case'] if _host_case is None else _host_case
    cpu_entries = sum(cpu_counts.values())
    concurrent = 0 < cpu_entries < entries
    generation_rate = measured['generation_concurrent_coefficients_per_second'
                               if concurrent else 'generation_solo_coefficients_per_second']
    copy_payload_rate = min(copy_bandwidth / 2,
                            measured['copy_concurrent_payload_bytes_per_second'
                                     if concurrent else 'copy_solo_payload_bytes_per_second'])
    cpu_jobs = []
    cpu_generation_work = 0.0
    cpu_output_bytes, cpu_random_bytes = 0.0, 0.0
    for label, size, packed in groups:
        count = cpu_counts.get(label, 0)
        if not 0 <= count <= size or (count != size and count % _PACK_BLOCK):
            raise ValueError('CPU partitions must align to packing blocks')
        if not count:
            continue
        if not packed:
            raise ValueError('CPU calibration requires packed blocks')
        cpu_generation_work += count / generation_rate
        cpu_output_bytes += packed_bytes(count)
        cpu_random_bytes += 4 * count / uniform_accept
        cpu_jobs.append({'label': label, 'size': count})
    gpu_output_bytes = retained_bytes - cpu_output_bytes
    publications, cpu_clock = [], 0.0
    for job in cpu_jobs:
        cpu_clock += job['size'] / generation_rate
        publications.append((job['label'], 0, job['size'], cpu_clock, 'CPU'))

    encoded, transferred, gpu_work, stored, copy_clock = 0.0, 0.0, 0.0, 0.0, 0.0
    completion_clock = 0.0
    random_work, arithmetic_work, memory_work = 0.0, 0.0, 0.0
    packed_gpu_work = 0.0
    generated_random_bytes, device_global_bytes, device_shared_bytes = 0.0, 0.0, 0.0
    instruction_totals = dict.fromkeys(_INIT_RESULTS_PER_SM_CYCLE, 0.0)
    kernel_count = 0
    device_free = [0.0, 0.0]
    output_free = [0.0, 0.0]
    chunk_count, largest_tile, largest_workspace, largest_output = 0, 0, 0, 0
    first_publication, last_publication = {}, {}
    GPU_output_bytes_by_material = {label: 0.0 for label, _, _ in groups}
    buffer_budget = min(_PIPELINE_BUFFER_BYTES, _INIT_DEVICE_CAPACITY_BYTES)
    for label, size, packed in groups:
        bytes_per_coefficient = (2 * ((q.bit_length() - 1) / 8 + 2 + 1 / 8 + 8 / _PACK_BLOCK)
                                 if packed else 16)
        tile = max(_PACK_BLOCK, int(buffer_budget / bytes_per_coefficient)
                   // _PACK_BLOCK * _PACK_BLOCK)
        largest_tile = max(largest_tile, tile)
        for offset in range(cpu_counts.get(label, 0), size, tile):
            count = min(tile, size - offset)
            slot = chunk_count % 2
            work = _fused_initialization_work(count, q, packed, packed_bytes)
            chunk_bytes = work['output_bytes']
            GPU_output_bytes_by_material[label] += chunk_bytes
            random_time = work['random_bytes'] / random_rate
            arithmetic_time = sum(work['instruction_work_s'].values())
            memory_time = (work['global_bytes'] + work['shared_bytes']) / device_bandwidth
            random_work += random_time
            arithmetic_work += arithmetic_time
            memory_work += memory_time
            generated_random_bytes += work['random_bytes']
            device_global_bytes += work['global_bytes']
            device_shared_bytes += work['shared_bytes']
            kernel_count += work['kernel_count']
            for key, value in work['instruction_operations'].items():
                instruction_totals[key] += value
            device_work = random_time + arithmetic_time + memory_time + work['kernel_count'] * rates['launch']
            gpu_work += device_work
            if packed:
                packed_gpu_work += device_work
            encoded = max(encoded, device_free[slot]) + device_work
            transferred = max(transferred, encoded, output_free[slot]) + chunk_bytes / transfer_bandwidth
            device_free[slot] = transferred
            if _runtime_costs is not None:
                completion_clock = max(completion_clock, transferred) + _runtime_costs['per_chunk_completion_s']
            else:
                completion_clock = transferred
            copy_clock = max(copy_clock, completion_clock) + chunk_bytes / copy_payload_rate
            stored = copy_clock
            output_free[slot] = stored
            publications.append((label, offset, count, stored, 'GPU'))
            largest_workspace = max(largest_workspace, work['workspace_bytes'])
            largest_output = max(largest_output, work['workspace_bytes'] / 2)
            chunk_count += 1
    stored = max(stored, cpu_clock)
    if largest_workspace > buffer_budget:
        raise ArithmeticError('Tile exceeds device memory budget')
    entropy_bytes = 4 * entries / uniform_accept
    host_traffic = cpu_output_bytes + 3 * gpu_output_bytes
    completed = max(stored, host_traffic / rates['hostmem'])
    body_completed = completed
    if _runtime_costs is not None:
        completed += _runtime_costs['setup_s'] + _runtime_costs['teardown_s']
        publications = [(label, offset, count, ready + _runtime_costs['setup_s'], source)
                        for label, offset, count, ready, source in publications]
    launch_work = kernel_count * rates['launch']
    cpu_copy_work = gpu_output_bytes / copy_payload_rate
    for label, _, _, ready, _ in publications:
        first_publication[label] = min(first_publication.get(label, ready), ready)
        last_publication[label] = max(last_publication.get(label, ready), ready)
    return {
        'estimated_seconds': completed,
        'stages_s': {
            **({'buffer_runtime_setup': _runtime_costs['setup_s']} if _runtime_costs else {}),
            'device_generation_mapping_packing': gpu_work,
            'pipeline_completion_and_resource_waits': max(0.0, body_completed - gpu_work),
            **({'buffer_runtime_teardown': _runtime_costs['teardown_s']} if _runtime_costs else {}),
        },
        'runtime_lifecycle': ({**_runtime_costs,
                               'completion_call_count': chunk_count,
                               'completion_call_service_work_s': chunk_count * _runtime_costs['per_chunk_completion_s']}
                              if _runtime_costs is not None else None),
        'uniform_coefficients': entries,
        'entropy_bytes': entropy_bytes,
        'generated_random_bytes': generated_random_bytes + cpu_random_bytes,
        'GPU_generated_random_bytes': generated_random_bytes,
        'device_global_traffic_bytes': device_global_bytes,
        'device_shared_traffic_bytes': device_shared_bytes,
        'instruction_operations': instruction_totals,
        'retained_bytes': retained_bytes,
        'CPU_generated_coefficients': cpu_entries,
        'GPU_generated_coefficients': entries - cpu_entries,
        'CPU_coefficient_fraction': cpu_entries / entries if entries else 0.0,
        'CPU_generated_output_bytes': cpu_output_bytes,
        'GPU_generated_output_bytes': gpu_output_bytes,
        'GPU_output_bytes_by_material': GPU_output_bytes_by_material,
        'CPU_random_bytes': cpu_random_bytes,
        'CPU_partition_counts': dict(cpu_counts),
        'random_input_uploaded_bytes': 0,
        'public_material_downloaded_bytes': gpu_output_bytes,
        'transferred_bytes': gpu_output_bytes,
        'avoided_download_bytes': cpu_output_bytes,
        'avoided_host_copy_bytes': 2 * cpu_output_bytes,
        'host_traffic_bytes': host_traffic,
        'transfer_bandwidth_Bps': transfer_bandwidth,
        'host_copy_bandwidth_Bps': copy_bandwidth,
        'host_copy_payload_bytes_per_second': copy_payload_rate,
        'generation_order': [label for label, _, _ in groups],
        'tile_entries': largest_tile,
        'chunk_count': chunk_count,
        'kernel_count': kernel_count,
        'input_buffer_count': 0,
        'device_buffer_count': len(device_free),
        'output_buffer_count': len(output_free),
        'first_publication_s': first_publication,
        'last_publication_s': last_publication,
        'device_capacity_bytes': _INIT_DEVICE_CAPACITY_BYTES,
        'device_workspace_bytes': largest_workspace,
        'host_staging_bytes': 2 * largest_output,
        'required_host_memory_bytes': retained_bytes + 2 * largest_output,
        'device_peak_bandwidth_Bps': _INIT_DEVICE_PEAK_BANDWIDTH,
        'effective_device_bandwidth_Bps': device_bandwidth,
        'GPU_compute_stages_overlap': False,
        'compute_DMA_overlap_assumed': True,
        'serial_generation_seconds': random_work + arithmetic_work + memory_work + launch_work + cpu_generation_work,
        'component_work_is_additive_elapsed_time': False,
        'component_work_s': {
            'CPU_generation_including_mapping_packing_and_store': cpu_generation_work,
            'CPU_generation_scheduled_work': cpu_generation_work,
            'CPU_generation_and_host_copy_work_sum': cpu_generation_work + cpu_copy_work,
            'CPU_generation_and_host_copy_executor_floor': max(cpu_generation_work, cpu_copy_work),
            'GPU_random_stream_reference': random_work,
            'GPU_fused_integer_operations': arithmetic_work,
            'GPU_packed_and_scan_memory': memory_work,
            'GPU_kernel_launches': launch_work,
            'GPU_generation_scheduled_work': gpu_work,
            'GPU_packed_generation_scheduled_work': packed_gpu_work,
            'public_material_D2H': gpu_output_bytes / transfer_bandwidth,
            'host_material_copy': cpu_copy_work,
            'host_memory_bandwidth_floor': host_traffic / rates['hostmem'],
        },
        'rate_source': 'local_host_calibration_with_separate_uncalibrated_device_conditions',
        'CPU_model': {
            'arithmetic_rate_source': 'end_to_end_sampling_packing_final_store_measurements',
            'CPU_generation_and_copy_concurrency_calibrated': True,
            'CPU_generation_and_copy_profile': 'concurrent' if concurrent else 'solo',
            'CPU_generation_measured_coefficients_per_second': generation_rate,
            'copy_worker_measured_payload_bytes_per_second': copy_payload_rate,
            'copy_workers': measured['copy_workers'],
            'copy_worker_final_completion_s': copy_clock,
            'CPU_generator_final_completion_s': cpu_clock,
            'CPU_native_fused_backend_executed': False,
            'CPU_numpy_generation_backend_executed': True,
            'target_host_or_device_DMA_calibrated': False,
        },
        'host_calibration_scope': calibration['source'],
        'host_calibration_platform': calibration['platform'],
        'host_calibration_samples_per_mode': calibration['repeats'],
        'host_calibration_buffer_bytes': measured['copy_buffer_bytes'],
        'host_calibration_joint_overlap_fraction': measured['median_joint_overlap_fraction'],
        'random_stream_reference': dict(_INIT_RANDOM_REFERENCE),
        'instruction_model': {
            'multiprocessors': _INIT_MULTIPROCESSORS, 'clock_Hz': _INIT_BASE_CLOCK_HZ,
            'results_per_SM_cycle': dict(_INIT_RESULTS_PER_SM_CYCLE),
            'source_level_instruction_count_estimates': True,
            'peak_integer_issue_assumed': True,
            'register_spills_and_occupancy_calibrated': False,
            'allocation_and_first_touch_calibrated': False,
        },
        'rate_units': {'device_random_stream': 'random_stream_bytes_per_second',
                       'CPU_generation': 'accepted_and_packed_coefficients_per_second',
                       'CPU_copy': 'payload_bytes_per_second_including_both_read_and_write_time',
                       'pcie': 'bytes_per_second', 'hostmem': 'bytes_per_second',
                       'stream': 'bytes_per_second'},
        'rate_assumptions': {
            'device_random_stream': random_rate,
            **{key: rates[key] for key in ('pcie', 'hostmem', 'stream', 'launch')},
        },
        'device_generation_backend_executed': False,
        'target_kernel_calibrated': False,
        '_publication_events': publications,
    }


def _initialization_host_allocation(groups, q, rates, packed_bytes, measured, allow_cpu):
    baseline = _initialization_material_schedule(groups, q, rates, packed_bytes, _host_case=measured)
    entries = baseline['uniform_coefficients']
    packed_entries = sum(count for _, count, packed in groups if packed)
    packed_storage = sum(packed_bytes(count) for _, count, packed in groups if packed)
    cpu_full = packed_entries / measured['generation_concurrent_coefficients_per_second']
    joint_copy_rate = min(baseline['host_copy_bandwidth_Bps'] / 2,
                          measured['copy_concurrent_payload_bytes_per_second'])
    work = baseline['component_work_s']
    resource_lines = {
        'CPU_generator': (0.0, cpu_full),
        'host_copy_worker': (baseline['retained_bytes'] / joint_copy_rate,
                              -packed_storage / joint_copy_rate),
        'GPU_compute': (work['GPU_generation_scheduled_work'],
                         -work['GPU_packed_generation_scheduled_work']),
        'D2H': (work['public_material_D2H'],
                 -packed_storage / baseline['transfer_bandwidth_Bps']),
        'host_memory': (baseline['host_traffic_bytes'] / rates['hostmem'],
                        -2 * packed_storage / rates['hostmem']),
    }
    fractions = {0.0, 1.0}
    lines = list(resource_lines.values())
    for i, (offset, slope) in enumerate(lines):
        for other_offset, other_slope in lines[i + 1:]:
            if slope != other_slope:
                cut = (other_offset - offset) / (slope - other_slope)
                if 0 < cut < 1:
                    fractions.add(cut)
    fraction = min(fractions, key=lambda cut: max(offset + slope * cut for offset, slope in lines))
    candidates = [baseline]
    if allow_cpu and entries:
        seen = {tuple(0 for _ in groups)}
        for rounding in (math.floor, math.ceil):
            counts = {label: min(size, max(0, rounding(fraction * size / _PACK_BLOCK) * _PACK_BLOCK))
                      if packed else 0 for label, size, packed in groups}
            key = tuple(counts[label] for label, _, _ in groups)
            if key not in seen:
                seen.add(key)
                candidates.append(_initialization_material_schedule(
                    groups, q, rates, packed_bytes, counts, _host_case=measured))
    best = min(candidates, key=lambda plan: plan['estimated_seconds'])
    best['allocation_plan'] = {
        'resource_lines_s': resource_lines,
        'balanced_fraction_before_block_rounding': fraction,
        'evaluated_CPU_fractions': [plan['CPU_coefficient_fraction'] for plan in candidates],
        'evaluated_estimated_seconds': [plan['estimated_seconds'] for plan in candidates],
        'GPU_only_estimated_seconds': baseline['estimated_seconds'],
        'selection': 'min_time(GPU_only,balanced_blocks)',
        'CPU_reserved_for_publication_consumer': not allow_cpu,
        'host_calibration_is_same_hardware_as_conditional_device_profile': False,
    }
    return best


def _initialization_runtime_costs(materials, rates):
    reference = _INIT_RUNTIME_REFERENCE
    windows = materials['output_buffer_count'] if materials['chunk_count'] else 0
    page_calibration = _calibrate_initialization_pages()
    page_bytes = page_calibration.get('page_bytes', os.sysconf('SC_PAGE_SIZE'))
    window_bytes = (math.ceil(materials['host_staging_bytes'] / windows / page_bytes)
                    * page_bytes) if windows else 0
    fresh = [case for case in page_calibration['cases'] if case['mode'] == 'fresh_writable_pages']
    if windows and not fresh:
        raise RuntimeError('Missing buffer first-write calibration')
    page_case = max(fresh, key=lambda case: case['window_bytes']) if fresh else None
    first_write_s = (windows * window_bytes / page_case['window_bytes']
                     * page_case['first_write_s']) if windows else 0.0
    def api_cost(key):
        reported_time = statistics.mean(reference[key])
        slope = reported_time / reference['reported_buffer_bytes']
        return {'slope_s_per_byte': slope, 'intercept_s': None,
                'reported_time_point_s': reported_time,
                'fixed_call_floor_s': rates['control'],
                'per_window_s': max(rates['control'], slope * window_bytes) if windows else 0.0}
    register, unregister = api_cost('register_range_s'), api_cost('unregister_range_s')
    setup_control_calls = 4 * windows
    cleanup_control_calls = 3 * windows
    setup = (first_write_s + windows * register['per_window_s']
             + windows * rates['control']
             + setup_control_calls * rates['control'])
    teardown = (windows * unregister['per_window_s']
                + (rates['control'] if windows else 0.0)
                + cleanup_control_calls * rates['control'])
    return {
        'setup_s': setup, 'teardown_s': teardown,
        'first_write_staging_s': first_write_s,
        'first_write_source': 'local_first_writable_touch_largest_measured_window',
        'first_write_target_host_calibrated': False,
        'registration_s': windows * register['per_window_s'],
        'unregistration_s': windows * unregister['per_window_s'],
        'register_byte_cost_projection': register, 'unregister_byte_cost_projection': unregister,
        'registration_count': windows, 'unregistration_count': windows,
        'host_page_alignment_bytes': page_bytes,
        'registered_window_bytes': window_bytes,
        'registered_total_bytes': windows * window_bytes,
        'device_workspace_pool_checkout_s': windows * rates['control'],
        'device_workspace_pool_preallocated_and_resident': True,
        'device_context_initialization_excluded_after_warmup': True,
        'setup_control_calls': setup_control_calls,
        'cleanup_control_calls': cleanup_control_calls,
        'control_call_s': rates['control'],
        'control_counts': {'stream_create': windows, 'event_create': windows,
                           'host_allocation': windows, 'control_initialization': windows,
                           'stream_destroy': windows, 'event_destroy': windows, 'workspace_pool_return': windows},
        'per_chunk_completion_s': 4 * rates['control'],
        'final_synchronization_call_s': rates['control'] if windows else 0.0,
        'reference': dict(reference),
        'final_output_arena_condition': 'resident_reusable_output_storage_after_benchmark_warmup',
        'fresh_final_output_arena_allocation_included': False,
        'device_lifecycle_condition_estimated': True,
        'device_lifecycle_measured_on_target': False,
    }


def _initialization_material_plan(groups, q, rates, packed_bytes, *, _on_publication=None):
    calibration = _calibrate_initialization_host(q)
    largest = max(case['copy_buffer_bytes'] for case in calibration['cases'])
    cases = [case for case in calibration['cases'] if case['copy_buffer_bytes'] == largest]
    candidates = [_initialization_host_allocation(groups, q, rates, packed_bytes, measured,
                                                   _on_publication is None)
                  for measured in cases]
    best = min(candidates, key=lambda plan: plan['estimated_seconds'])
    best['allocation_plan']['evaluated_copy_workers'] = [case['copy_workers'] for case in cases]
    best['allocation_plan']['evaluated_copy_team_seconds'] = [plan['estimated_seconds'] for plan in candidates]
    best['allocation_plan']['copy_team_selection'] = 'lowest_simulated_time_from_measured_implementations'
    body_seconds = best['estimated_seconds']
    allocation_plan = best['allocation_plan']
    selected_case = next(case for case in cases if case['copy_workers'] == best['CPU_model']['copy_workers'])
    runtime_costs = _initialization_runtime_costs(best, rates)
    best = _initialization_material_schedule(groups, q, rates, packed_bytes,
                                            best['CPU_partition_counts'], _host_case=selected_case,
                                            _runtime_costs=runtime_costs)
    best['allocation_plan'] = allocation_plan
    best['allocation_plan']['selection_scope'] = 'material_flow_before_buffer_lifecycle_costs'
    best['material_flow_without_runtime_seconds'] = body_seconds
    best['runtime_increment_seconds'] = best['estimated_seconds'] - body_seconds
    best['required_host_memory_bytes'] = best['retained_bytes'] + runtime_costs['registered_total_bytes']
    events = best.pop('_publication_events')
    if _on_publication is not None:
        order = {label: index for index, (label, _, _) in enumerate(groups)}
        prefix_ready = dict.fromkeys(order, 0.0)
        for label, _, count, ready, _ in sorted(events, key=lambda event: (order[event[0]], event[1])):
            prefix_ready[label] = max(prefix_ready[label], ready)
            _on_publication(label, count, prefix_ready[label])
    return best


def _estimate_initialization(params, settings, dimensions, rates, coefficients, packed_bytes):
    l1, l2 = settings.l1, settings.l2
    w, c = dimensions['witness_coordinates'], dimensions['quadratic_constraints']
    rho = settings.auxiliary_rows * params.k + 2 * settings.auxiliary_hiding_bits
    auxiliary_shape = settings.auxiliary_rows, 256 + rho
    auxiliary_entries = math.prod(auxiliary_shape)
    public_A_entries = 2 * params.n * params.m
    matrix_entries = sum(coefficients.values())
    entries = matrix_entries + auxiliary_entries + public_A_entries
    digest_bytes = {'F': len(b'FDGE immutable F canonical int64 v1\0') + 24 + 8 * coefficients['F']}
    for label, width in (('B1', w), ('B2', c)):
        digest_bytes[label] = (len(b'Yang2019 commitment key v1')
                               + sum(_scalar_frame_bytes(value) for value in (params.q, l1, l2, width))
                               + _array_frame_bytes((l1, width + l2)) + _array_frame_bytes((width, l2)))
    outer_digest_bytes = (len(b'Yang P1 public parameters v1')
                          + _scalar_frame_bytes(str(asdict(settings)))
                          + 2 * _scalar_frame_bytes(bytes(32)) + _array_frame_bytes(auxiliary_shape)
                          + sum(_scalar_frame_bytes(value) for value in
                                (dimensions['relation_rows'], repr(dimensions['yang_sigma2']),
                                 repr(dimensions['yang_rejection_M']))))
    hash_bytes = sum(digest_bytes.values()) + outer_digest_bytes
    packed_storage = sum(packed_bytes(count) for count in coefficients.values())
    retained_bytes = packed_storage + 8 * (auxiliary_entries + public_A_entries)
    uniform_accept = 1 - ((1 << 32) % params.q) / (1 << 32)
    entropy_bytes = 4 * entries / uniform_accept
    phases = {'canonical_hash': hash_bytes / rates['sha']}
    workers = len(digest_bytes)
    decode_per_worker = rates['cpu_MAC'] / workers
    packed_per_coefficient = packed_storage / matrix_entries
    decoded_coefficients_per_second = min(decode_per_worker, rates['sha'] / 8)
    hash_memory_reserve = workers * (packed_per_coefficient + 16) * decoded_coefficients_per_second
    transfer_bandwidth = min(rates['pcie'], (rates['hostmem'] - hash_memory_reserve) / 3)
    if transfer_bandwidth <= 0:
        raise ArithmeticError('Insufficient publication bandwidth')
    phases['canonical_decode'] = matrix_entries / decode_per_worker
    hash_done = {label: (size - 8 * coefficients[label]) / rates['sha']
                 for label, size in digest_bytes.items()}
    decode_done = dict.fromkeys(digest_bytes, 0.0)
    hash_history = {label: [0.0, 0.0] for label in digest_bytes}
    hash_start = {}
    groups = [(label, coefficients[label], True)
              for label in sorted(coefficients, key=lambda key: digest_bytes[key], reverse=True)]
    groups.extend([('public_A', public_A_entries, False),
                   ('auxiliary', auxiliary_entries, False)])
    materials = _initialization_material_plan(groups, params.q, rates, packed_bytes)
    commitment_materials = _initialization_material_plan(
        [group for group in groups if group[0] in ('B1', 'B2', 'auxiliary')],
        params.q, rates, packed_bytes)
    with_prepared_commitment = _initialization_material_plan(
        [group for group in groups if group[0] in ('F', 'public_A')],
        params.q, rates, packed_bytes)
    def digest_publication(label, count, copied):
        if label not in hash_done:
            return
        hash_start.setdefault(label, copied)
        decode_done[label] = (max(copied, decode_done[label], hash_history[label].pop(0))
                              + count / decode_per_worker)
        hash_done[label] = max(decode_done[label], hash_done[label]) + 8 * count / rates['sha']
        hash_history[label].append(hash_done[label])

    implementation_materials = _initialization_material_plan(
        groups, params.q, {**rates, 'hostmem': rates['hostmem'] - hash_memory_reserve},
        packed_bytes, _on_publication=digest_publication)
    for label in hash_done:
        hash_done[label] += rates['control']
    copied = implementation_materials['estimated_seconds']
    transferred_bytes = implementation_materials['transferred_bytes']
    phases['transfer'] = transferred_bytes / transfer_bandwidth
    outer_start = max(hash_done['B1'], hash_done['B2'], copied)
    outer_finish = outer_start + outer_digest_bytes / rates['sha'] + rates['control']
    host_traffic = (implementation_materials['host_traffic_bytes'] + packed_storage + 16 * matrix_entries
                    + 8 * auxiliary_entries)
    parallel_finish = max(copied, *hash_done.values(), host_traffic / rates['hostmem'])
    final_finish = max(parallel_finish, outer_finish)
    implementation_stages = {
        'generation_transfer_and_independent_digests': parallel_finish,
        'dependent_public_digest': final_finish - parallel_finish,
    }
    phases['generation_encoding_packing'] = implementation_materials['serial_generation_seconds']
    return {
        'estimated_seconds': materials['estimated_seconds'],
        'generation_and_storage_seconds': materials['estimated_seconds'],
        'with_prepared_commitment_seconds': with_prepared_commitment['estimated_seconds'],
        'commitment_material_preparation_seconds': commitment_materials['estimated_seconds'],
        'implementation_complete_seconds': sum(implementation_stages.values()),
        'commitment_backend': 'selected_dense_generic_matrices',
        'concrete_commitment_parameters_from_paper': False,
        'target_generation_and_packing_backend_executed': False,
        'timing_scope': 'fresh_values; warm_storage/pool; lifecycle; no_fingerprints',
        'prepared_commitment_scope': 'fresh_F_Arec_Aoa_with_ppCOM_already_available',
        'stages_s': materials['stages_s'],
        'material_plan': materials,
        'D2H_reference': dict(_INIT_D2H_REFERENCE),
        'current_D2H_report_condition_applied': rates['pcie'] == _INIT_D2H_REFERENCE['device_to_host_bytes_per_second'],
        'commitment_material_plan': commitment_materials,
        'with_prepared_commitment_plan': with_prepared_commitment,
        'implementation_stages_s': implementation_stages,
        'implementation_material_plan': implementation_materials,
        'phase_work_s': phases,
        'uniform_coefficients': entries,
        'entropy_bytes': entropy_bytes,
        'canonical_hash_bytes': hash_bytes,
        'digest_bytes': {**digest_bytes, 'public_parameters': outer_digest_bytes},
        'independent_digest_workers': workers,
        'decode_pipeline_workers': workers,
        'generation_order': [label for label, _, _ in groups],
        'digest_start_s': hash_start,
        'digest_finish_s': hash_done,
        'dependent_digest_start_s': outer_start,
        'aggregate_sha_limit_Bps': workers * rates['sha'],
        'canonical_decode_rate_source': 'shared_cpu_MAC_rate_proxy',
        'decode_coefficients_per_second_per_worker': decode_per_worker,
        'hash_memory_reserve_Bps': hash_memory_reserve,
        'transfer_bandwidth_Bps': transfer_bandwidth,
        'retained_bytes': retained_bytes,
        'transferred_bytes': transferred_bytes,
        'host_traffic_bytes': host_traffic,
        'tile_entries': implementation_materials['tile_entries'],
    }


def estimate_population(params, settings):
    d = target_dimensions(params, settings)
    w, c, rows = (d[key] for key in
                  ('witness_coordinates', 'quadratic_constraints', 'relation_rows'))
    tail, depth = c - w, params.depth
    l1, l2 = settings.l1, settings.l2
    h1, h2 = l1 + w + l2, l1 + c + l2
    rho = settings.auxiliary_rows * params.k + 2 * settings.auxiliary_hiding_bits
    fields = rows + 4 * l1 + 2 * w + 2 * c
    hash_bytes = len(b'FDGE-aCommit-CRH-v1\0') + 8 + 8 + 32 + 5 * 8 + 4 * fields
    response_bytes = 8 * (fields + rho + w + h1 + h2)
    low_bits = params.q.bit_length() - 1
    exception_probability = (params.q - (1 << low_bits)) / params.q
    uniform_accept = 1 - ((1 << 32) % params.q) / (1 << 32)
    small_span = 2 * math.ceil(settings.gaussian_tail * settings.sigma1) + 1
    alias_accept = 1 - ((1 << 32) % small_span) / (1 << 32)
    sigma = d['yang_sigma2']
    probability = -math.expm1(-1 / sigma)
    centre = -sigma**2 * math.log1p(-probability)
    proposal_accept = (probability / (2 - probability)
                       * math.exp(-.5 * (centre / sigma)**2)
                       * math.sqrt(2 * math.pi) * sigma
                       * math.erf(settings.gaussian_tail / math.sqrt(2)))
    draws = h1 + h2
    entropy = {
        'uniform': 4 * w / uniform_accept,
        'small': draws * (8 + 4 / alias_accept),
        'large': 24 * draws / proposal_accept,
    }
    coefficients = {
        'B1': l1 * (w + l2) + w * l2,
        'B2': l1 * (c + l2) + c * l2,
        'F': d['F_entries'],
    }

    def packed_bytes(entries):
        return ((low_bits * entries + 7) // 8 + 2 * entries * exception_probability
                + 4 * ((entries + _PACK_BLOCK - 1) // _PACK_BLOCK))

    case_totals, case_stages = {}, {}
    for name, rates in _RATE_CASES.items():
        matrix_schedule = []

        def device(units, traffic, kind='array', calls=1):
            return max(units / rates[kind], traffic / rates['stream']) + calls * rates['launch']

        def pipe(entries, lanes, bandwidth):
            if not entries:
                return 0.0, 0.0
            tile = (_PIPELINE_BUFFER_BYTES
                    // ((low_bits + 16) * _PACK_BLOCK // 8 + 4) * _PACK_BLOCK)
            full, last = divmod(entries, tile)

            def times(count):
                return (packed_bytes(count) / bandwidth,
                        count * (lanes / rates['MAC'] + 1 / rates['decode']) + rates['launch'])

            copy, compute = times(tile)
            total = copy + compute + (full - 1) * max(copy, compute) if full else 0.0
            if last:
                last_copy, last_compute = times(last)
                total = (total + max(0.0, last_copy - compute) + last_compute
                         if full else last_copy + last_compute)
            busy = (entries * (lanes / rates['MAC'] + 1 / rates['decode'])
                    + (full + bool(last)) * rates['launch'])
            return total, busy

        def heterogeneous(width, lanes, role, hash_reserve=0, coupled_transfer_bytes=0, coupled_gpu_s=0):
            entries = ((l1 * (width + l2) + width * l2)
                       if role != 'F' else params.n * width)
            candidates = []
            for step in range(257):
                cut = min(width, max(0, round(width * step / 256 / _PACK_BLOCK) * _PACK_BLOCK))
                if step == 256:
                    cut = width
                cpu_entries = ((l1 + l2) * cut + (l1 * l2 if cut == width else 0)
                               if role != 'F' else params.n * cut)
                gpu_entries = entries - cpu_entries
                cpu_bytes, gpu_bytes = packed_bytes(cpu_entries), packed_bytes(gpu_entries)
                cpu_time = lanes * cpu_entries / rates['cpu_MAC']
                gpu_work = gpu_entries * (lanes / rates['MAC'] + 1 / rates['decode'])
                host_budget = rates['hostmem'] - hash_reserve
                resource = max(cpu_time, (gpu_bytes + coupled_transfer_bytes) / rates['pcie'],
                               gpu_work + coupled_gpu_s,
                               (cpu_bytes + 3 * (gpu_bytes + coupled_transfer_bytes)) / host_budget)
                cpu_bandwidth = cpu_bytes / resource if resource else 0.0
                coupled_bandwidth = coupled_transfer_bytes / resource if resource else 0.0
                gpu_bandwidth = min(rates['pcie'] - coupled_bandwidth,
                                    max(1.0, (host_budget - cpu_bandwidth - 3 * coupled_bandwidth) / 3))
                streamed, busy = pipe(gpu_entries, lanes, gpu_bandwidth)
                busy += coupled_gpu_s
                duration = max(resource, streamed)
                input_bytes = 4 * lanes * (cut + (l2 if role != 'F' and cut else 0))
                output_bytes = (4 * lanes * (cut + l1) if role != 'F' and cut
                                else 4 * lanes * params.n if cut else 0)
                auxiliary_bytes = input_bytes + output_bytes
                overhead = auxiliary_bytes / rates['pcie'] + auxiliary_bytes / rates['hostmem']
                candidates.append({
                    'total_s': duration + overhead,
                    'elapsed_s': duration,
                    'GPU_busy_s': busy,
                    'GPU_idle_s': duration - busy,
                    'CPU_fraction': cpu_entries / entries,
                    'CPU_cut': cut,
                    'packed_total_bytes': cpu_bytes + gpu_bytes,
                    'GPU_transfer_bytes': gpu_bytes,
                    'CPU_MACs': lanes * cpu_entries,
                    'GPU_MACs': lanes * gpu_entries,
                    'CPU_read_bytes': cpu_bytes,
                    'staged_GPU_DDR_bytes': 3 * gpu_bytes,
                    'host_budget_Bps': host_budget,
                    'assigned_CPU_Bps': cpu_bandwidth,
                    'assigned_GPU_stage_Bps': 3 * gpu_bandwidth,
                    'assigned_coupled_DDR_Bps': 3 * coupled_bandwidth,
                    'assigned_coupled_transfer_Bps': coupled_bandwidth,
                    'role': role,
                    'lanes': lanes,
                    'coupled_transfer_bytes': coupled_transfer_bytes,
                    'coupled_gpu_s': coupled_gpu_s,
                })
            best = min(candidates, key=lambda candidate: candidate['total_s'])
            if best['GPU_idle_s'] < -1e-9:
                raise ArithmeticError('Negative pipeline idle time')
            if (best['assigned_CPU_Bps'] + best['assigned_GPU_stage_Bps']
                    + best['assigned_coupled_DDR_Bps'] > best['host_budget_Bps'] + 1):
                raise ArithmeticError('Shared bandwidth budget exceeded')
            best['GPU_idle_s'] = max(0.0, best['GPU_idle_s'])
            matrix_schedule.append(best)
            return best['total_s']

        def commit(width, lanes):
            time = device(lanes * (l1 + width + l2), 12 * lanes * (l1 + width + l2))
            verifier_first = lanes == 1 and width == w
            time += heterogeneous(width, lanes, 'B1' if width == w else 'B2',
                                  4e9 if verifier_first else 0,
                                  4 * fields if verifier_first else 0,
                                  hash_format(True) + quadratic('response') if verifier_first else 0)
            return time + device(lanes * width, 24 * lanes * width)

        def field_product():
            return (device(d['F_columns'], 16 * d['F_columns'])
                    + heterogeneous(d['F_columns'], 1, 'F'))

        non_f_MAC = (2 * depth * params.n * params.nk + 4 * params.m * params.n
                     + params.e * (params.m - params.t + 1) * params.t)
        non_f_coordinates = d['F_columns'] + 6 * params.m + 2 * depth + d['policy_rows']

        def other_relation():
            return (max(non_f_MAC / rates['MAC'], 16 * non_f_MAC / rates['HBM'])
                    + device(non_f_coordinates, 16 * non_f_coordinates, calls=8 + 2 * depth))

        def quadratic(kind):
            traffic = {
                'square': 8 * w + 8 * c + 28 * tail,
                'linear': 16 * w + 8 * c + 52 * tail,
                'witness': 8 * w + 36 * tail,
                'response': 8 * w + 8 * c + 36 * tail,
            }[kind]
            return device(c, traffic, 'constraint')

        def hash_format(snapshot):
            return device(fields, (36 if snapshot else 20) * fields, calls=5)

        def hash_cpu():
            return hash_bytes / rates['sha'] + hash_bytes / rates['hostmem'] + rates['control']

        def auxiliary():
            entries = settings.auxiliary_rows * (256 + rho)
            return (max(entries / rates['MAC'], 8 * entries / rates['HBM'])
                    + device(256 + rho, 16 * (256 + rho), calls=3))

        offline = {
            'uniform_mask': entropy['uniform'] / rates['rng'] + device(w, 8 * w),
            'sigma1_Gaussians': entropy['small'] / rates['rng'] + device(draws, 8 * draws, 'alias', calls=2),
            'sigma2_Gaussians': entropy['large'] / rates['rng'] + device(draws, 8 * draws, 'gauss', calls=2),
            'square_constraints': quadratic('square'),
            'B1_pair': commit(w, 2),
            'B2_pair': commit(c, 2),
            'F_times_mask': field_product(),
            'other_linear_relation': other_relation() + device(w, 16 * w),
            'auxiliary_randomness': 4 * rho / rates['rng'] + device(rho, 8 * rho),
            'control': 4 * rates['control'],
        }
        small_B1 = (8 + 4 / alias_accept) * h1 / rates['rng'] + device(h1, 8 * h1, 'alias')
        small_B2 = offline['sigma1_Gaussians'] - small_B1
        large_B1 = 24 * h1 / proposal_accept / rates['rng'] + device(h1, 8 * h1, 'gauss')
        large_B2 = offline['sigma2_Gaussians'] - large_B1
        first_idle = matrix_schedule[0]['GPU_idle_s']
        second_idle = sum(record['GPU_idle_s'] for record in matrix_schedule[1:3])
        first_extra = small_B2 + large_B2 + offline['square_constraints']
        offline_sequential = sum(offline.values())
        offline = {
            'uniform_mask': offline['uniform_mask'],
            'B1_small_opening': small_B1,
            'B1_large_opening': large_B1,
            'B1_stream_with_B2_sampling_and_square': offline['B1_pair'] + max(0.0, first_extra - first_idle),
            'B2_F_stream_with_other_relation': (offline['B2_pair'] + offline['F_times_mask']
                                               + max(0.0, offline['other_linear_relation'] - second_idle)),
            'auxiliary_randomness': offline['auxiliary_randomness'],
            'control': offline['control'],
        }
        if sum(offline.values()) > offline_sequential + 1e-8:
            raise ArithmeticError('Pipeline increased sequential work')
        context_prefix = _estimate_context_prefix(d, rates)
        offline['first_context_prefix'] = context_prefix['estimated_seconds']
        private_input_bytes = 16 * params.n * params.m + 8 * (6 * params.m + 2 * depth + params.e * params.t)
        online = {
            'private_input_upload': private_input_bytes / rates['pcie'],
            'private_witness_build': device(w + d['F_columns'], 8 * w + 10 * d['F_columns'], calls=6),
            'witness_snapshot_domain': device(w, 24 * w, calls=2),
            'key_match_and_linear_check': device(d['F_columns'], 9 * d['F_columns']) + other_relation(),
            'witness_constraints': quadratic('witness'),
            'linear_mask_constraints': quadratic('linear'),
            'complete_commitment_bases': device(w + c, 24 * (w + c), calls=2),
            'payload_format_checks': hash_format(False),
            'payload_transfer': 4 * fields / rates['pcie'],
            'single_SHA256_chain': hash_cpu(),
            'auxiliary_commitment': auxiliary(),
            'z0_response': device(w, 40 * w),
            'opening_responses_rejection': device(h1 + h2, 104 * (h1 + h2), 'norm', calls=3) + 2 * rates['control'],
            'output_download_and_publication': response_bytes / rates['pcie'] + response_bytes / rates['hostmem'],
            'control': 4 * rates['control'],
        }
        preparation = _response_preparation(w, h1, h2, rho, response_bytes, rates)
        payload_pipeline = _pipeline_seconds(
            [hash_format(True), 4 * fields / rates['pcie'], hash_cpu()], fields, _RESPONSE_CHUNK_ELEMENTS)
        verifier = {
            'response_transfer_norm_pipeline': preparation['estimated_seconds'],
            'payload_B1_and_quadratic_pipeline': max(payload_pipeline, commit(w, 1)),
            'auxiliary_commitment_check': auxiliary() + device(settings.auxiliary_rows, 24 * settings.auxiliary_rows),
            'linear_equation': field_product() + other_relation() + device(rows, 24 * rows),
            'B1_equality': device(l1 + w, 24 * (l1 + w)),
            'B2_commitment_equality': commit(c, 1) + device(l1 + c, 24 * (l1 + c)),
            'control': 6 * rates['control'],
        }
        totals = {
            'P_offline': sum(offline.values()),
            'P_online': sum(online.values()),
            'V_offline': 0.0,
            'V_online': sum(verifier.values()),
        }
        totals['P_total'] = totals['P_offline'] + totals['P_online']
        totals['P_cached_context'] = totals['P_total'] - context_prefix['estimated_seconds']
        totals['V_total'] = totals['V_offline'] + totals['V_online']
        if not all(math.isfinite(value) and value >= 0 for value in totals.values()):
            raise ArithmeticError('Non-finite or negative estimate')
        entries = sum(coefficients.values())
        totals['shared_pack'] = max(entries / rates['pack'],
                                    (4 * entries + sum(packed_bytes(count) for count in coefficients.values()))
                                    / rates['hostmem'])
        initialization_rates = dict(rates)
        if name == _POINT_CASE:
            initialization_rates['pcie'] = _INIT_D2H_REFERENCE['device_to_host_bytes_per_second']
        initialization = _estimate_initialization(params, settings, d, initialization_rates, coefficients, packed_bytes)
        totals['SETUP_Init'] = initialization['estimated_seconds']
        totals['Init_generate'] = initialization['generation_and_storage_seconds']
        totals['Init_with_prepared_commitment'] = initialization['with_prepared_commitment_seconds']
        totals['Init_commitment_material_preparation'] = initialization['commitment_material_preparation_seconds']
        totals['Init_implementation_complete'] = initialization['implementation_complete_seconds']
        case_totals[name] = totals
        case_stages[name] = {
            'SETUP_Init': initialization['stages_s'],
            'initialization_plan': initialization,
            'first_context_plan': context_prefix,
            'P_offline': offline, 'P_online': online,
            'V_offline': {}, 'V_online': verifier,
            'verifier_preparation_plan': preparation,
            'verifier_payload_pipeline_s': payload_pipeline,
            'matrix_plan': matrix_schedule,
        }
    return {
        'N': params.N,
        'estimated': True,
        'timings_measured': False,
        'timing_scope': 'P:first_context; V:prepared_response; no_relation/key_prep',
        'initialization_timing_scope': case_stages[_POINT_CASE]['initialization_plan']['timing_scope'],
        'point_case': _POINT_CASE,
        'point_s': dict(case_totals[_POINT_CASE]),
        'cases_s': case_totals,
        'stages_s': case_stages,
        'dimensions': d,
        'hash_bytes_per_role': hash_bytes,
        'entropy_bytes': entropy,
        'proposal_acceptance': proposal_accept,
        'packed_matrix_expected_bytes': sum(packed_bytes(count) for count in coefficients.values()),
    }


def estimate_populations(params, settings):
    return [estimate_population(replace(params, depth=depth), settings)
            for depth in PAPER_GROUP_DEPTHS]


def print_estimates(rows):
    columns = ('Init_fresh', 'Init_given_ppCOM', 'P_first_context', 'V_response')
    print('s | estimates | Init: warm storage/workspace, no fingerprints', flush=True)
    print('P first context; V prepared; relation/key prep excluded', flush=True)
    print(f"{'N':<6}{'log2N':>7}" + ''.join(f'{name:>17}' for name in columns), flush=True)
    for row in rows:
        values = [f"{row['point_s'][key]:.6f}"
                  for key in ('SETUP_Init', 'Init_with_prepared_commitment', 'P_total', 'V_total')]
        print(f"{row['N']:<6}{row['dimensions']['depth']:>7}" + ''.join(f'{value:>17}' for value in values), flush=True)


def run_pv_estimate():
    params, settings, _ = load_selected_profile()
    rows = estimate_populations(params, settings)
    print_estimates(rows)
    return rows
