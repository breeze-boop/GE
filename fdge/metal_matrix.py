from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import lru_cache
import ctypes
import sys
import threading

import numpy as np


_Q = 524309
_BLOCK = 4096
_SHORT_ROWS_PER_SIMD = 4
_SHOUP_MIN_ENTRIES = 1048576
_SHOUP_MIN_ROWS = 32
_WORKSPACE_LIMIT = 64 * 1024 * 1024
_WORKSPACE_BUFFERS = 32
_KERNEL_PIPELINES = (
    ('field_product', 'pipeline'),
    ('field_product_vector_pair', 'pipeline_vector_pair'),
    ('binary_product', 'pipeline_binary'),
    ('field_product_short', 'pipeline_short'),
    ('field_product_short_pair', 'pipeline_short_pair'),
    ('prepare_shoup', 'pipeline_prepare_shoup'),
    ('field_product_shoup_short', 'pipeline_shoup_short'),
    ('field_product_shoup_short_pair', 'pipeline_shoup_short_pair'),
    ('commitment_tail', 'pipeline_commitment_tail'))
_POINTER = ctypes.c_void_p
_SIZE = ctypes.c_size_t
_INITIALIZE_LOCK = threading.RLock()
_SOURCE = '''
#include <metal_stdlib>
using namespace metal;

inline uint foldq(uint value) {
    int reduced = int(value & 524287u) - 21 * int(value >> 19);
    return uint(reduced < 0 ? reduced + 524309 : reduced);
}

inline uint mulq(uint left, uint right) {
    return foldq(left * right) + mulhi(left, right) * 352277u;
}

inline uint2 add_wide(uint2 left, uint2 right) {
    uint low = left.x + right.x;
    return uint2(low, left.y + right.y + uint(low < left.x));
}

inline uint2 mul_wide(uint left, uint right) {
    return uint2(left * right, mulhi(left, right));
}

inline uint fold_wide(uint2 value) {
    return foldq(foldq(value.x) + value.y * 352277u);
}

inline uint mulq_shoup(uint left, uint right, uint quotient) {
    return left * right - mulhi(left, quotient) * 524309u;
}

kernel void prepare_shoup(
    device uint *right [[buffer(0)]],
    constant ulong &count [[buffer(1)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    ulong index = ulong(group.x) * 256ul + tid;
    if (index < count)
        right[count + index] = uint((ulong(right[index]) << 32) / 524309ul);
}

kernel void field_product(
    device const uint *matrix [[buffer(0)]],
    device const uint *right [[buffer(1)]],
    device uint *partials [[buffer(2)]],
    constant ulong4 &shape [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    ulong row = ulong(group.x) * 8ul + ulong(tid >> 5);
    if (row >= shape.x) return;
    ulong block = ulong(group.y);
    ulong begin = block * 4096ul;
    ulong end = min(begin + 4096ul, shape.y);
    uint2 first = uint2(0u);
    for (ulong column = begin + ulong(tid & 31u);
         column < end; column += 32ul) {
        uint coefficient = matrix[row * shape.y + column];
        first = add_wide(first, mul_wide(coefficient, right[column]));
    }
    uint total = simd_sum(fold_wide(first));
    if ((tid & 31u) == 0u)
        partials[row * shape.z + block] = foldq(total);
}

kernel void field_product_vector_pair(
    device const uint *matrix [[buffer(0)]],
    device const uint *right [[buffer(1)]],
    device uint *partials [[buffer(2)]],
    constant ulong4 &shape [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    ulong row = ulong(group.x) * 8ul + ulong(tid >> 5);
    if (row >= shape.x) return;
    ulong block = ulong(group.y);
    ulong begin = block * 4096ul;
    ulong end = min(begin + 4096ul, shape.y);
    uint2 first[4] = {};
    uint2 second[4] = {};
    for (ulong column = begin + ulong(tid & 31u) * 4ul;
         column < end; column += 128ul) {
        if (column + 3ul < end) {
            uint4 coefficient = uint4(*reinterpret_cast<device const packed_uint4 *>(
                matrix + row * shape.y + column));
            uint4 first_pair = uint4(*reinterpret_cast<device const packed_uint4 *>(right + column * 2ul));
            uint4 second_pair = uint4(*reinterpret_cast<device const packed_uint4 *>(right + column * 2ul + 4ul));
            uint4 first_right = uint4(first_pair.x, first_pair.z, second_pair.x, second_pair.z);
            uint4 second_right = uint4(first_pair.y, first_pair.w, second_pair.y, second_pair.w);
            #pragma unroll
            for (uint component = 0u; component < 4u; ++component) {
                first[component] = add_wide(first[component], mul_wide(coefficient[component], first_right[component]));
                second[component] = add_wide(second[component], mul_wide(coefficient[component], second_right[component]));
            }
        } else {
            #pragma unroll
            for (uint component = 0u; component < 4u; ++component) {
                ulong tail = column + component;
                if (tail < end) {
                    uint coefficient = matrix[row * shape.y + tail];
                    first[component] = add_wide(first[component], mul_wide(coefficient, right[tail * 2ul]));
                    second[component] = add_wide(second[component], mul_wide(coefficient, right[tail * 2ul + 1ul]));
                }
            }
        }
    }
    uint2 first_sum = uint2(0u);
    uint2 second_sum = uint2(0u);
    #pragma unroll
    for (uint component = 0u; component < 4u; ++component) {
        first_sum = add_wide(first_sum, first[component]);
        second_sum = add_wide(second_sum, second[component]);
    }
    uint first_total = simd_sum(fold_wide(first_sum));
    uint second_total = simd_sum(fold_wide(second_sum));
    if ((tid & 31u) == 0u) {
        ulong offset = (row * shape.z + block) * 2ul;
        partials[offset] = foldq(first_total);
        partials[offset + 1ul] = foldq(second_total);
    }
}

kernel void binary_product(
    device const uint *matrix [[buffer(0)]],
    device const uint *right [[buffer(1)]],
    device uint *partials [[buffer(2)]],
    constant ulong4 &shape [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    ulong row = ulong(group.x) * 8ul + ulong(tid >> 5);
    if (row >= shape.x) return;
    ulong block = ulong(group.y);
    ulong begin = block * 4096ul;
    ulong end = min(begin + 4096ul, shape.y);
    uint total = 0u;
    for (ulong column = begin + ulong(tid & 31u);
         column < end; column += 32ul) {
        uint coefficient = matrix[row * shape.y + column];
        total += right[column] == 1u ? coefficient : 0u;
    }
    total = simd_sum(total);
    if ((tid & 31u) == 0u)
        partials[row * shape.z + block] = foldq(total);
}

template <uint LANES, bool SHOUP = false>
inline void short_field_product(
    device const uint *matrix,
    device const uint *right,
    device uint *partials,
    constant ulong4 &shape,
    uint3 group,
    uint tid) {
    ulong first_row = (ulong(group.x) * 8ul + ulong(tid >> 5)) * FDGE_SHORT_ROWS;
    if (first_row >= shape.x) return;
    uint first[FDGE_SHORT_ROWS] = {};
    uint second[FDGE_SHORT_ROWS] = {};
    for (ulong column = ulong(tid & 31u); column < shape.y; column += 32ul) {
        uint first_right = right[column * LANES];
        uint second_right = LANES == 2u ? right[column * LANES + 1ul] : 0u;
        uint first_quotient = SHOUP ? right[shape.y * LANES + column * LANES] : 0u;
        uint second_quotient = SHOUP && LANES == 2u ? right[shape.y * LANES + column * LANES + 1ul] : 0u;
        #pragma unroll
        for (uint offset = 0u; offset < FDGE_SHORT_ROWS; ++offset) {
            ulong row = first_row + offset;
            if (row < shape.x) {
                uint coefficient = matrix[row * shape.y + column];
                first[offset] += SHOUP ? mulq_shoup(coefficient, first_right, first_quotient)
                                       : mulq(coefficient, first_right);
                if (LANES == 2u)
                    second[offset] += SHOUP ? mulq_shoup(coefficient, second_right, second_quotient)
                                            : mulq(coefficient, second_right);
            }
        }
    }
    #pragma unroll
    for (uint offset = 0u; offset < FDGE_SHORT_ROWS; ++offset) {
        ulong row = first_row + offset;
        if (row < shape.x) {
            uint first_total = simd_sum(foldq(first[offset]));
            uint second_total = simd_sum(foldq(second[offset]));
            if ((tid & 31u) == 0u) {
                partials[row * LANES] = foldq(first_total);
                if (LANES == 2u)
                    partials[row * LANES + 1ul] = foldq(second_total);
            }
        }
    }
}

kernel void field_product_short(
    device const uint *matrix [[buffer(0)]],
    device const uint *right [[buffer(1)]],
    device uint *partials [[buffer(2)]],
    constant ulong4 &shape [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    short_field_product<1u>(matrix, right, partials, shape, group, tid);
}

kernel void field_product_short_pair(
    device const uint *matrix [[buffer(0)]],
    device const uint *right [[buffer(1)]],
    device uint *partials [[buffer(2)]],
    constant ulong4 &shape [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    short_field_product<2u>(matrix, right, partials, shape, group, tid);
}

kernel void field_product_shoup_short(
    device const uint *matrix [[buffer(0)]],
    device const uint *right [[buffer(1)]],
    device uint *partials [[buffer(2)]],
    constant ulong4 &shape [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    short_field_product<1u, true>(matrix, right, partials, shape, group, tid);
}

kernel void field_product_shoup_short_pair(
    device const uint *matrix [[buffer(0)]],
    device const uint *right [[buffer(1)]],
    device uint *partials [[buffer(2)]],
    constant ulong4 &shape [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    short_field_product<2u, true>(matrix, right, partials, shape, group, tid);
}

kernel void commitment_tail(
    device uint *product [[buffer(0)]],
    device const uint *randomness [[buffer(1)]],
    device const uint *message [[buffer(2)]],
    constant ulong &count [[buffer(3)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    ulong index = ulong(group.x) * 256ul + tid;
    if (index < count) {
        uint value = product[index] + randomness[index] + message[index];
        if (value >= 524309u) value -= 524309u;
        if (value >= 524309u) value -= 524309u;
        product[index] = value;
    }
}
'''


class _MetalSize(ctypes.Structure):
    _fields_ = [('width', _SIZE), ('height', _SIZE), ('depth', _SIZE)]


class _Runtime:
    def __init__(self):
        self.objc = ctypes.CDLL('/usr/lib/libobjc.A.dylib')
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.objc_getClass.restype = _POINTER
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = _POINTER
        self.foundation = ctypes.CDLL(
            '/System/Library/Frameworks/Foundation.framework/Foundation')
        self.metal = ctypes.CDLL(
            '/System/Library/Frameworks/Metal.framework/Metal')
        self.metal.MTLCreateSystemDefaultDevice.argtypes = []
        self.metal.MTLCreateSystemDefaultDevice.restype = _POINTER

    @lru_cache(maxsize=64)
    def function(self, restype, argtypes):
        return ctypes.CFUNCTYPE(restype, _POINTER, _POINTER, *argtypes)(
            ('objc_msgSend', self.objc))

    @lru_cache(maxsize=64)
    def selector(self, name):
        return self.objc.sel_registerName(name.encode())

    def call(self, receiver, name, restype=_POINTER, argtypes=(), args=()):
        return self.function(restype, argtypes)(
            receiver, self.selector(name), *args)

    def release(self, pointer):
        if pointer:
            self.call(pointer, 'release', None)

    @contextmanager
    def pool(self):
        pool = self.call(self.objc.objc_getClass(b'NSAutoreleasePool'), 'new')
        try:
            yield
        finally:
            self.call(pool, 'drain', None)

    def string(self, value):
        encoded = value.encode('utf-8')
        allocated = self.call(self.objc.objc_getClass(b'NSString'), 'alloc')
        result = self.call(allocated, 'initWithBytes:length:encoding:',
                           argtypes=(_POINTER, _SIZE, _SIZE),
                           args=(encoded, len(encoded), 4))
        if not result:
            raise RuntimeError('Metal source string allocation failed')
        return result

    def error(self, pointer):
        if not pointer:
            return 'no error description'
        description = self.call(pointer, 'localizedDescription')
        encoded = self.call(description, 'UTF8String', ctypes.c_char_p)
        return encoded.decode('utf-8', errors='replace') if encoded else 'unknown error'


class _Backend:
    def __init__(self, runtime, device):
        self.runtime, self.device = runtime, None
        self.queue = None
        for _, attribute in _KERNEL_PIPELINES:
            setattr(self, attribute, None)
        self.short_rows_per_simd = _SHORT_ROWS_PER_SIMD
        self.lock = threading.RLock()
        self._workspace = []
        self._workspace_bytes = 0
        self._workspace_limit = _WORKSPACE_LIMIT
        self.max_buffer_length = runtime.call(device, 'maxBufferLength', _SIZE)
        source = library = function = options = None
        try:
            source = runtime.string(f'#define FDGE_SHORT_ROWS {self.short_rows_per_simd}\n' + _SOURCE)
            options = runtime.call(runtime.objc.objc_getClass(b'MTLCompileOptions'), 'new')
            runtime.call(options, 'setFastMathEnabled:', None,
                         (ctypes.c_bool,), (False,))
            error = _POINTER()
            library = runtime.call(device, 'newLibraryWithSource:options:error:',
                                   argtypes=(_POINTER, _POINTER, ctypes.POINTER(_POINTER)),
                                   args=(source, options, ctypes.byref(error)))
            if not library:
                raise RuntimeError(f'Metal compilation failed: {runtime.error(error.value)}')
            for kernel_name, attribute in _KERNEL_PIPELINES:
                name = runtime.string(kernel_name)
                try:
                    function = runtime.call(library, 'newFunctionWithName:',
                                            argtypes=(_POINTER,), args=(name,))
                finally:
                    runtime.release(name)
                if not function:
                    raise RuntimeError(f'Metal {kernel_name} kernel was not found')
                pipeline = runtime.call(
                    device, 'newComputePipelineStateWithFunction:error:',
                    argtypes=(_POINTER, ctypes.POINTER(_POINTER)),
                    args=(function, ctypes.byref(error)))
                setattr(self, attribute, pipeline)
                if not pipeline:
                    raise RuntimeError(f'Metal pipeline creation failed: {runtime.error(error.value)}')
                if (runtime.call(pipeline, 'threadExecutionWidth', _SIZE) != 32
                        or runtime.call(pipeline, 'maxTotalThreadsPerThreadgroup', _SIZE) < 256):
                    raise RuntimeError('Metal requires SIMD=32 and group=256')
                runtime.release(function)
                function = None
            self.queue = runtime.call(device, 'newCommandQueue')
            if not self.queue:
                raise RuntimeError('Metal command queue allocation failed')
            self.device = device
        except BaseException:
            runtime.release(self.queue)
            self.queue = None
            for _, attribute in _KERNEL_PIPELINES:
                runtime.release(getattr(self, attribute))
                setattr(self, attribute, None)
            raise
        finally:
            for pointer in (options, function, library, source):
                runtime.release(pointer)

    def __del__(self):
        try:
            for buffer in self._workspace:
                buffer.close()
            self._workspace.clear()
            for pointer in (self.queue, *(getattr(self, attribute) for _, attribute in _KERNEL_PIPELINES), self.device):
                self.runtime.release(pointer)
        except Exception:
            pass

    def take_workspace(self, length):
        length = max(4, int(length))
        if not self._workspace_limit:
            return _Buffer(self, length)
        candidates = [(buffer.length, index) for index, buffer in enumerate(self._workspace)
                      if buffer.length >= length]
        if candidates:
            _, index = min(candidates)
            buffer = self._workspace.pop(index)
            self._workspace_bytes -= buffer.length
            return buffer
        capacity = (length + 4095) // 4096 * 4096 if length <= self._workspace_limit else length
        if capacity > self.max_buffer_length:
            capacity = length
        return _Buffer(self, capacity)

    def return_workspace(self, buffer):
        if not buffer.pointer:
            return
        if buffer.length > self._workspace_limit:
            buffer.close()
            return
        while self._workspace and (self._workspace_bytes + buffer.length > self._workspace_limit
                                   or len(self._workspace) >= _WORKSPACE_BUFFERS):
            discarded = self._workspace.pop(0)
            self._workspace_bytes -= discarded.length
            discarded.close()
        self._workspace.append(buffer)
        self._workspace_bytes += buffer.length


@lru_cache(maxsize=1)
def _backend_cached():
    if sys.platform != 'darwin':
        return None
    try:
        runtime = _Runtime()
    except OSError:
        return None
    with runtime.pool():
        device = runtime.metal.MTLCreateSystemDefaultDevice()
        if not device:
            return None
        try:
            return _Backend(runtime, device)
        except BaseException:
            runtime.release(device)
            raise


def _backend():
    with _INITIALIZE_LOCK:
        return _backend_cached()


@contextmanager
def _uncached_workspaces():
    backend = _backend()
    if backend is None:
        raise RuntimeError('Metal matrix acceleration is unavailable')
    with backend.lock:
        previous = backend._workspace_limit
        backend._workspace_limit = 0
        try:
            yield
        finally:
            backend._workspace_limit = previous


def available(q):
    return type(q) is int and q == _Q and _backend() is not None


class _Buffer:
    def __init__(self, backend, length):
        length = max(4, int(length))
        if length > backend.max_buffer_length:
            raise MemoryError(f'Metal buffer requires {length} bytes; limit is {backend.max_buffer_length}')
        self.backend, self.length, self.pointer = backend, length, None
        self.pointer = backend.runtime.call(
            backend.device, 'newBufferWithLength:options:',
            argtypes=(_SIZE, _SIZE), args=(length, 0))
        if not self.pointer:
            raise MemoryError(f'Metal could not allocate {length} bytes')
        self.address = backend.runtime.call(self.pointer, 'contents')
        if not self.address:
            self.close()
            raise RuntimeError('Metal shared buffer has no CPU address')

    def array(self, dtype, count, readonly=False):
        raw = (ctypes.c_ubyte * self.length).from_address(self.address)
        raw._metal_owner = self
        view = memoryview(raw)
        if readonly:
            view = view.toreadonly()
        return np.frombuffer(view, dtype=dtype, count=count)

    def close(self):
        if self.pointer:
            self.backend.runtime.release(self.pointer)
            self.pointer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _PreparedMatrix(np.ndarray):
    def __array_finalize__(self, source):
        self._metal_owner = getattr(source, '_metal_owner', None)


def prepare(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.dtype.kind not in 'iubfO':
        raise TypeError('Metal requires a canonical integer matrix')
    with np.nditer(values, flags=['external_loop', 'buffered', 'zerosize_ok', 'refs_ok'],
                   op_flags=['readonly'], buffersize=262144) as iterator:
        for chunk in iterator:
            if chunk.dtype.kind == 'O':
                if any(not isinstance(value, (int, np.integer, bool, np.bool_))
                       or not 0 <= value < _Q for value in chunk):
                    raise ValueError('Noncanonical Metal coefficients')
            elif (np.any(chunk < 0) or np.any(chunk >= _Q)
                  or (chunk.dtype.kind == 'f'
                      and (not np.all(np.isfinite(chunk))
                           or not np.all(chunk == np.floor(chunk))))):
                raise ValueError('Noncanonical Metal coefficients')
    backend = _backend()
    if backend is None:
        raise RuntimeError('Metal matrix acceleration is unavailable')
    with backend.runtime.pool():
        owner = _Buffer(backend, int(values.size) * 4)
        owner.columns = values.shape[1]
        destination = owner.array(np.uint32, values.size).reshape(values.shape)
        np.copyto(destination, values, casting='unsafe')
        del destination
        result = owner.array(np.uint32, values.size, readonly=True).reshape(values.shape).view(_PreparedMatrix)
        result._metal_owner = owner
        return result


def is_prepared(matrix):
    owner = getattr(matrix, '_metal_owner', None)
    if (not isinstance(matrix, _PreparedMatrix) or not isinstance(owner, _Buffer)
            or not owner.pointer or matrix.ndim != 2
            or matrix.dtype != np.dtype(np.uint32) or matrix.flags.writeable
            or matrix.shape[1] != owner.columns or not matrix.flags.c_contiguous):
        return False
    if not matrix.size:
        return True
    offset = int(matrix.ctypes.data) - owner.address
    row_bytes = owner.columns * 4
    return (matrix.strides == (row_bytes, 4) and offset >= 0
            and offset % row_bytes == 0 and offset + matrix.nbytes <= owner.length)


def _integer_operand(values):
    array = np.asarray(values)
    if array.dtype.kind == 'O':
        if any(not isinstance(item, (int, np.integer, bool, np.bool_))
               for item in array.flat):
            raise TypeError('Metal product requires integer right operands')
    elif array.dtype.kind not in 'iub':
        raise TypeError('Metal product requires integer right operands')
    return array


def _write_canonical(destination, array):
    if array.dtype.kind == 'O':
        np.remainder(array, _Q, out=destination, casting='unsafe')
    elif array.dtype.kind == 'u' and array.dtype.itemsize == 8:
        np.remainder(array, np.uint64(_Q), out=destination, casting='unsafe')
    else:
        np.remainder(array, _Q, out=destination, dtype=np.int64, casting='unsafe')


def _canonical(values):
    array = _integer_operand(values)
    result = np.empty(array.shape, dtype=np.uint32)
    _write_canonical(result, array)
    return result


def _product_spec(matrix, right):
    if not is_prepared(matrix):
        raise ValueError('Metal requires prepared matrix/full-row view')
    rows, inner = matrix.shape
    if isinstance(right, tuple):
        columns = tuple(_integer_operand(column) for column in right)
        is_vector = False
        if len(columns) not in (1, 2):
            raise ValueError('Metal requires 1 or 2 RHS')
        if any(column.ndim != 1 or len(column) != inner for column in columns):
            raise ValueError('Metal product dimensions do not match')
    else:
        right = _integer_operand(right)
        if right.ndim not in (1, 2) or right.shape[0] != inner:
            raise ValueError('Metal product dimensions do not match')
        is_vector = right.ndim == 1
        columns = (right,) if is_vector else tuple(right[:, lane] for lane in range(right.shape[1]))
    lanes = len(columns)
    if lanes not in (1, 2):
        raise ValueError('Metal requires 1 or 2 RHS')
    blocks = (inner + _BLOCK - 1) // _BLOCK
    if (rows + 7) // 8 > (1 << 32) - 1 or blocks > (1 << 32) - 1:
        raise MemoryError('Metal product grid exceeds supported dimensions')
    return matrix, columns, rows, inner, lanes, blocks, is_vector


def _read_product(output_buffer, rows, blocks, lanes, is_vector):
    partials = output_buffer.array(np.uint32, rows * blocks * lanes, readonly=True).reshape(rows, blocks, lanes)
    if blocks == 1:
        result = partials[:, 0, :].astype(np.int64)
    elif blocks <= np.iinfo(np.uint64).max // (_Q - 1):
        result = np.remainder(partials.sum(axis=1, dtype=np.uint64), _Q).astype(np.int64)
    else:
        result = np.zeros((rows, lanes), dtype=np.uint64)
        for start in range(0, blocks, 65536):
            result += partials[:, start:start + 65536, :].sum(axis=1, dtype=np.uint64)
            np.remainder(result, _Q, out=result)
        result = result.astype(np.int64)
    return result[:, 0] if is_vector else result


def _is_binary(vector):
    for start in range(0, len(vector), 65536):
        chunk = vector[start:start + 65536]
        if np.any(chunk < 0) or np.any(chunk > 1):
            return False
    return True


def _use_shoup(rows, inner, binary=False):
    return (not binary and inner <= 256 and rows >= _SHOUP_MIN_ROWS
            and rows * inner >= _SHOUP_MIN_ENTRIES)


def _encode_product(backend, encoder, spec, right_buffer, output_buffer, shoup, binary):
    runtime = backend.runtime
    matrix, _, rows, inner, lanes, blocks, _ = spec
    owner = matrix._metal_owner
    if shoup:
        runtime.call(encoder, 'setComputePipelineState:', None,
                     (_POINTER,), (backend.pipeline_prepare_shoup,))
        runtime.call(encoder, 'setBuffer:offset:atIndex:', None,
                     (_POINTER, _SIZE, _SIZE), (right_buffer.pointer, 0, 0))
        count = ctypes.c_uint64(inner * lanes)
        runtime.call(encoder, 'setBytes:length:atIndex:', None,
                     (_POINTER, _SIZE, _SIZE), (ctypes.byref(count), ctypes.sizeof(count), 1))
        runtime.call(encoder, 'dispatchThreadgroups:threadsPerThreadgroup:', None,
                     (_MetalSize, _MetalSize),
                     (_MetalSize((inner * lanes + 255) // 256, 1, 1), _MetalSize(256, 1, 1)))
    short = not binary and inner <= 256
    rows_per_group = 8 * backend.short_rows_per_simd if short else 8
    pipeline = (backend.pipeline_shoup_short if shoup and lanes == 1 else
                backend.pipeline_shoup_short_pair if shoup else
                backend.pipeline_binary if binary else
                backend.pipeline_short if short and lanes == 1 else
                backend.pipeline_short_pair if short else
                backend.pipeline if lanes == 1 else backend.pipeline_vector_pair)
    runtime.call(encoder, 'setComputePipelineState:', None,
                 (_POINTER,), (pipeline,))
    offset = int(matrix.ctypes.data) - owner.address
    for slot, pointer, position in ((0, owner.pointer, offset),
                                    (1, right_buffer.pointer, 0),
                                    (2, output_buffer.pointer, 0)):
        runtime.call(encoder, 'setBuffer:offset:atIndex:', None,
                     (_POINTER, _SIZE, _SIZE), (pointer, position, slot))
    shape = (ctypes.c_uint64 * 4)(rows, inner, blocks, lanes)
    runtime.call(encoder, 'setBytes:length:atIndex:', None,
                 (_POINTER, _SIZE, _SIZE), (shape, ctypes.sizeof(shape), 3))
    runtime.call(encoder, 'dispatchThreadgroups:threadsPerThreadgroup:', None,
                 (_MetalSize, _MetalSize),
                 (_MetalSize((rows + rows_per_group - 1) // rows_per_group, blocks, 1),
                  _MetalSize(256, 1, 1)))


def _encode_commitment_tail(backend, encoder, product, randomness, message, count):
    runtime = backend.runtime
    runtime.call(encoder, 'setComputePipelineState:', None,
                 (_POINTER,), (backend.pipeline_commitment_tail,))
    for slot, buffer in enumerate((product, randomness, message)):
        runtime.call(encoder, 'setBuffer:offset:atIndex:', None,
                     (_POINTER, _SIZE, _SIZE), (buffer.pointer, 0, slot))
    size = ctypes.c_uint64(count)
    runtime.call(encoder, 'setBytes:length:atIndex:', None,
                 (_POINTER, _SIZE, _SIZE), (ctypes.byref(size), ctypes.sizeof(size), 3))
    runtime.call(encoder, 'dispatchThreadgroups:threadsPerThreadgroup:', None,
                 (_MetalSize, _MetalSize),
                 (_MetalSize((count + 255) // 256, 1, 1), _MetalSize(256, 1, 1)))


def _can_fuse_commitment(top, bottom, q):
    return (type(q) is int and q == _Q and is_prepared(top) and is_prepared(bottom)
            and top._metal_owner.backend is bottom._metal_owner.backend
            and top.shape[0] > 0 and bottom.shape[0] > 0
            and 0 < bottom.shape[1] <= _BLOCK
            and top.shape[1] == sum(bottom.shape))


def _commitment_products(top, bottom, messages, randomness, q):
    if not _can_fuse_commitment(top, bottom, q):
        raise ValueError('Invalid fused commitment matrices')
    l1, width, l2 = top.shape[0], bottom.shape[0], bottom.shape[1]
    messages = tuple(_integer_operand(message) for message in messages)
    randomness = tuple(_integer_operand(value) for value in randomness)
    if (len(messages) not in (1, 2) or len(randomness) != len(messages)
            or any(message.shape != (width,) for message in messages)
            or any(value.shape != (l1 + width + l2,) for value in randomness)):
        raise ValueError('Fused operand shape mismatch')
    top_right = tuple(value[l1:] for value in randomness)
    bottom_right = tuple(value[-l2:] for value in randomness)
    return _matmul_many(((top, top_right), (bottom, bottom_right)), q,
                        _commitment=(messages, randomness))


def _matmul_many(jobs, q, *, binary=False, _commitment=None):
    if type(q) is not int or q != _Q:
        raise ValueError('Metal field products require q=524309')
    specs = [_product_spec(matrix, right) for matrix, right in jobs]
    if binary and any(spec[4] != 1 or not _is_binary(spec[1][0]) for spec in specs):
        raise ValueError('Metal requires one binary vector')
    results = [None] * len(specs)
    active = []
    backend = None
    for index, spec in enumerate(specs):
        matrix, columns, rows, inner, lanes, blocks, is_vector = spec
        if not rows or not inner:
            result = np.zeros((rows, lanes), dtype=np.int64)
            results[index] = result[:, 0] if is_vector else result
        else:
            owner_backend = matrix._metal_owner.backend
            if backend is not None and backend is not owner_backend:
                raise ValueError('Mixed Metal devices')
            backend = owner_backend
            active.append((index, spec))
    if not active:
        return results
    runtime = backend.runtime
    borrowed = []
    prepared = []
    command = encoder = None
    success = False
    submitted = completed = False
    with backend.lock, runtime.pool():
        try:
            for index, spec in active:
                matrix, columns, rows, inner, lanes, blocks, is_vector = spec
                shoup = _use_shoup(rows, inner, binary)
                if shoup and (inner * lanes + 255) // 256 > (1 << 32) - 1:
                    raise MemoryError('Shoup grid exceeds limit')
                right_buffer = backend.take_workspace(inner * lanes * (8 if shoup else 4))
                borrowed.append(right_buffer)
                destination = right_buffer.array(np.uint32, inner * lanes).reshape(inner, lanes)
                for lane, column in enumerate(columns):
                    _write_canonical(destination[:, lane], column)
                output_buffer = backend.take_workspace(rows * blocks * lanes * 4)
                borrowed.append(output_buffer)
                prepared.append((index, spec, right_buffer, output_buffer, shoup))
            if _commitment is not None:
                if (binary or len(prepared) != 2 or len(specs) != 2
                        or not _can_fuse_commitment(specs[0][0], specs[1][0], q)
                        or specs[1][5] != 1 or specs[0][4] != specs[1][4]):
                    raise ValueError('Incomplete fused commitment batch')
                messages, randomness = _commitment
                width, lanes = specs[1][2], specs[1][4]
                count = width * lanes
                if (count + 255) // 256 > (1 << 32) - 1:
                    raise MemoryError('Commitment tail grid exceeds limit')
                message_buffer = backend.take_workspace(count * 4)
                borrowed.append(message_buffer)
                destination = message_buffer.array(np.uint32, count).reshape(width, lanes)
                for lane, message in enumerate(messages):
                    _write_canonical(destination[:, lane], message)
            command = runtime.call(backend.queue, 'commandBuffer')
            if not command:
                raise RuntimeError('Metal command buffer allocation failed')
            encoder = runtime.call(command, 'computeCommandEncoder')
            if not encoder:
                raise RuntimeError('Metal compute encoder allocation failed')
            for index, spec, right_buffer, output_buffer, shoup in prepared:
                _encode_product(backend, encoder, spec, right_buffer, output_buffer, shoup, binary)
            runtime.call(encoder, 'endEncoding', None)
            encoder = None
            if _commitment is not None:
                encoder = runtime.call(command, 'computeCommandEncoder')
                if not encoder:
                    raise RuntimeError('Metal commitment tail encoder allocation failed')
                _encode_commitment_tail(backend, encoder, prepared[1][3], prepared[0][2],
                                        message_buffer, count)
                runtime.call(encoder, 'endEncoding', None)
                encoder = None
            runtime.call(command, 'commit', None)
            submitted = True
            runtime.call(command, 'waitUntilCompleted', None)
            completed = True
            if runtime.call(command, 'status', _SIZE) != 4:
                error = runtime.call(command, 'error')
                raise RuntimeError(f'Metal field product failed: {runtime.error(error)}')
            if _commitment is None:
                for index, spec, right_buffer, output_buffer, shoup in prepared:
                    matrix, columns, rows, inner, lanes, blocks, is_vector = spec
                    results[index] = _read_product(output_buffer, rows, blocks, lanes, is_vector)
            else:
                l1 = specs[0][2]
                top = _read_product(prepared[0][3], l1, specs[0][5], lanes, False)
                lower = prepared[1][3].array(np.uint32, count, readonly=True).reshape(width, lanes)
                results = tuple(np.empty(l1 + width, dtype=np.int64) for _ in range(lanes))
                for lane, (result, value) in enumerate(zip(results, randomness)):
                    upper = result[:l1]
                    np.remainder(value[:l1], q, out=upper)
                    np.add(upper, top[:, lane], out=upper)
                    np.subtract(upper, q, out=upper, where=upper >= q)
                    np.copyto(result[l1:], lower[:, lane], casting='safe')
            success = True
            return results
        finally:
            try:
                if encoder:
                    runtime.call(encoder, 'endEncoding', None)
            finally:
                try:
                    if submitted and not completed:
                        runtime.call(command, 'waitUntilCompleted', None)
                finally:
                    for buffer in borrowed:
                        if success:
                            backend.return_workspace(buffer)
                        else:
                            buffer.close()


def matmul_many(jobs, q):
    return _matmul_many(jobs, q)


def matmul(matrix, right, q):
    return matmul_many(((matrix, right),), q)[0]


def _stream_right_bytes(rows, columns):
    short_columns = min(columns, 256)
    return max(4 * columns, 8 * short_columns if _use_shoup(rows, short_columns) else 0)


def linear_product_plan(shape, q, *, max_workspace_bytes=_WORKSPACE_LIMIT, columns=None):
    if type(q) is not int or q != _Q:
        raise ValueError('Metal field products require q=524309')
    if (len(shape) != 2 or any(not isinstance(value, (int, np.integer))
                               or isinstance(value, (bool, np.bool_)) or value < 0 for value in shape)):
        raise ValueError('Invalid Metal matrix shape')
    if type(max_workspace_bytes) is not int or max_workspace_bytes < 1024 * 1024:
        raise ValueError('Metal workspace must be >= 1 MiB')
    if columns is not None and (type(columns) is not int or columns < 1):
        raise ValueError('Metal columns must be > 0')
    rows, width = map(int, shape)
    budget = min(max_workspace_bytes, _WORKSPACE_LIMIT)
    backend = _backend()
    if backend is None:
        raise RuntimeError('Metal matrix acceleration is unavailable')
    chunk_entries = min(65536, budget // 64)
    result_bytes = 8 * rows
    if result_bytes > budget:
        raise MemoryError('Metal output exceeds workspace')
    validation_bytes = 24 * chunk_entries + 6 * 4095 + result_bytes
    row_block = min(rows, 65536, backend.max_buffer_length // 4,
                    max(1, (budget - validation_bytes - 16) // 80))

    def slot_bytes(row_count, column_count):
        blocks = (column_count + _BLOCK - 1) // _BLOCK
        return (4 * row_count * column_count + _stream_right_bytes(row_count, column_count)
                + 4 * row_count * blocks)

    def active_bytes(row_count, column_count, slot_count):
        return validation_bytes + 64 * row_count + slot_count * slot_bytes(row_count, column_count)

    if not rows or not width:
        row_block = column_block = 0
        slots = 0
        active = result_bytes
    else:
        if not row_block:
            raise MemoryError('Metal device cannot hold a coefficient tile')
        upper = min(width, backend.max_buffer_length // (4 * row_block), backend.max_buffer_length // 4,
                    width if columns is None else columns)
        slots = 1 if upper == width and active_bytes(row_block, width, 1) <= budget else 2
        lower = 0
        while lower < upper:
            middle = (lower + upper + 1) // 2
            if active_bytes(row_block, middle, slots) <= budget:
                lower = middle
            else:
                upper = middle - 1
        column_block = lower
        if not column_block:
            raise MemoryError('Metal tile exceeds workspace')
        active = active_bytes(row_block, column_block, slots)
    return {'shape': [rows, width], 'row_block_size': row_block,
            'column_block_size': column_block, 'validation_chunk_entries': chunk_entries,
            'pipeline_slots': slots,
            'active_workspace_bytes': active, 'active_workspace_limit_bytes': budget,
            'cached_workspace_limit_bytes': _WORKSPACE_LIMIT,
            'peak_workspace_bound_bytes': active + _WORKSPACE_LIMIT,
            'result_bytes': result_bytes,
            'max_buffer_bytes': backend.max_buffer_length,
            'coefficient_storage': 'uint32', 'coefficient_upload_included': True,
            'full_matrix_gpu_allocation_required': False,
            'source_matrix_required': True,
            'memory_scope': 'workspace_only; excludes_source/OS/driver'}


def _copy_coefficient_tile(destination, values, chunk_entries):
    with np.nditer(values, flags=['external_loop', 'buffered', 'zerosize_ok'],
                   op_flags=['readonly'], buffersize=chunk_entries) as iterator:
        for chunk in iterator:
            if (np.any(chunk < 0) or np.any(chunk >= _Q)
                    or (chunk.dtype.kind == 'f'
                        and (not np.all(np.isfinite(chunk))
                             or not np.all(chunk == np.floor(chunk))))):
                raise ValueError('Noncanonical Metal coefficients')
    np.copyto(destination, values, casting='unsafe')


def _take_stream_workspace(backend, length):
    length = max(4, int(length))
    if not backend._workspace_limit:
        return _Buffer(backend, length)
    capacity = (length + 4095) // 4096 * 4096
    if capacity > backend.max_buffer_length:
        capacity = length
    candidates = [(buffer.length, index) for index, buffer in enumerate(backend._workspace)
                  if length <= buffer.length <= capacity]
    if candidates:
        _, index = min(candidates)
        buffer = backend._workspace.pop(index)
        backend._workspace_bytes -= buffer.length
        return buffer
    return _Buffer(backend, capacity)


class _StreamProduct:
    def __init__(self, backend, rows, columns):
        self.backend = backend
        self.command = None
        self.spec = None
        self.buffers = []
        try:
            for size in (4 * rows * columns, _stream_right_bytes(rows, columns),
                         4 * rows * ((columns + _BLOCK - 1) // _BLOCK)):
                self.buffers.append(_take_stream_workspace(backend, size))
        except BaseException:
            self.close()
            raise
        self.matrix_buffer, self.right_buffer, self.output_buffer = self.buffers

    def submit(self, matrix, right, binary, chunk_entries):
        if self.command:
            raise RuntimeError('Metal stream slot is still in use')
        rows, inner = matrix.shape
        if is_prepared(matrix):
            if matrix._metal_owner.backend is not self.backend:
                raise ValueError('Metal stream device mismatch')
        else:
            self.matrix_buffer.columns = inner
            destination = self.matrix_buffer.array(np.uint32, matrix.size).reshape(matrix.shape)
            _copy_coefficient_tile(destination, matrix, chunk_entries)
            del destination
            matrix = self.matrix_buffer.array(np.uint32, matrix.size, readonly=True).reshape(matrix.shape).view(_PreparedMatrix)
            matrix._metal_owner = self.matrix_buffer
        self.spec = _product_spec(matrix, right)
        if binary and not _is_binary(self.spec[1][0]):
            raise ValueError('Metal binary products require values in {0,1}')
        destination = self.right_buffer.array(np.uint32, inner)
        _write_canonical(destination, self.spec[1][0])
        shoup = _use_shoup(rows, inner, binary)
        runtime = self.backend.runtime
        encoder = None
        with runtime.pool():
            self.command = runtime.call(self.backend.queue, 'commandBuffer')
            if not self.command:
                raise RuntimeError('Metal command buffer allocation failed')
            runtime.call(self.command, 'retain')
            try:
                encoder = runtime.call(self.command, 'computeCommandEncoder')
                if not encoder:
                    raise RuntimeError('Metal compute encoder allocation failed')
                _encode_product(self.backend, encoder, self.spec, self.right_buffer,
                                self.output_buffer, shoup, binary)
                runtime.call(encoder, 'endEncoding', None)
                encoder = None
                runtime.call(self.command, 'commit', None)
            finally:
                if encoder:
                    runtime.call(encoder, 'endEncoding', None)

    def _wait(self):
        if not self.command:
            return 0
        runtime = self.backend.runtime
        status = runtime.call(self.command, 'status', _SIZE)
        if status in (1, 2, 3):
            runtime.call(self.command, 'waitUntilCompleted', None)
            status = runtime.call(self.command, 'status', _SIZE)
        if status not in (0, 4, 5):
            raise RuntimeError('Metal stream command did not finish')
        return status

    def _release_command(self):
        self.backend.runtime.release(self.command)
        self.command = None
        self.spec = None

    def finish(self):
        status = self._wait()
        try:
            if status != 4:
                error = self.backend.runtime.call(self.command, 'error')
                raise RuntimeError(f'Metal field product failed: {self.backend.runtime.error(error)}')
            _, _, rows, _, lanes, blocks, is_vector = self.spec
            return _read_product(self.output_buffer, rows, blocks, lanes, is_vector)
        finally:
            self._release_command()

    def close(self):
        self._wait()
        self._release_command()
        for buffer in self.buffers:
            self.backend.return_workspace(buffer)
        self.buffers.clear()


def linear_product(matrix, right, q, *, binary=None,
                   max_workspace_bytes=_WORKSPACE_LIMIT, columns=None, on_progress=None):
    if type(q) is not int or q != _Q:
        raise ValueError('Metal field products require q=524309')
    if (not isinstance(matrix, np.ndarray) or matrix.ndim != 2
            or matrix.dtype.kind not in 'iubf'):
        raise TypeError('Metal requires canonical matrix')
    right = _integer_operand(right)
    rows, width = matrix.shape
    if right.shape != (width,):
        raise ValueError('Metal RHS shape mismatch')
    if binary is not None and type(binary) is not bool:
        raise ValueError('Metal binary selection must be a bool or None')
    if binary is not False:
        actual_binary = _is_binary(right)
        if binary is True and not actual_binary:
            raise ValueError('Metal binary products require values in {0,1}')
        binary = actual_binary
    plan = linear_product_plan(matrix.shape, q, max_workspace_bytes=max_workspace_bytes,
                               columns=columns)
    if not rows or not width:
        return np.zeros(rows, dtype=np.int64)
    backend = _backend()
    blocks = (width + _BLOCK - 1) // _BLOCK
    right_itemsize = 8 if _use_shoup(rows, width, binary) else 4
    resident_working = right_itemsize * width + 4 * rows * blocks + 64 * rows + 2 * 4095
    if (is_prepared(matrix) and (columns is None or columns >= width)
            and resident_working <= plan['active_workspace_limit_bytes']
            and 4 * rows * blocks <= backend.max_buffer_length
            and right_itemsize * width <= backend.max_buffer_length):
        return _matmul_many(((matrix, right),), q, binary=binary)[0]
    result = np.zeros(rows, dtype=np.int64)
    with backend.lock, backend.runtime.pool(), ExitStack() as resources:
        slots = []
        for _ in range(plan['pipeline_slots']):
            slot = _StreamProduct(backend, plan['row_block_size'], plan['column_block_size'])
            resources.callback(slot.close)
            slots.append(slot)
        for row in range(0, rows, plan['row_block_size']):
            stop = min(rows, row + plan['row_block_size'])
            subtotal = np.zeros(stop - row, dtype=np.int64)
            for index, start in enumerate(range(0, width, plan['column_block_size'])):
                end = min(width, start + plan['column_block_size'])
                slot = slots[index % len(slots)]
                if slot.command:
                    subtotal += slot.finish()
                    np.subtract(subtotal, q, out=subtotal, where=subtotal >= q)
                slot.submit(matrix[row:stop, start:end], right[start:end], binary,
                            plan['validation_chunk_entries'])
            for slot in slots:
                if slot.command:
                    subtotal += slot.finish()
                    np.subtract(subtotal, q, out=subtotal, where=subtotal >= q)
            result[row:stop] = subtotal
            if on_progress:
                on_progress({'phase': 'apply_full_F', 'rows_completed': stop,
                             'total_rows': rows, 'entries_processed': stop * width,
                             'total_entries': rows * width})
    return result
