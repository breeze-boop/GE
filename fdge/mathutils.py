from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import os
import numpy as np


class SystemUniformRNG:
    def _random_bytes(self, size):
        return os.urandom(size)

    def integers(self, low, high=None, size=None, dtype=np.int64):
        if high is None:
            low, high = 0, low
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               for v in (low, high)):
            raise ValueError('uniform integer endpoints must be integers')
        low, high = int(low), int(high)
        dtype = np.dtype(dtype)
        if dtype.kind not in 'iu':
            raise ValueError('uniform integer dtype must be an integer dtype')
        limits = np.iinfo(dtype)
        span = high - low
        if (not 0 < span <= np.iinfo(np.int64).max or low < limits.min or high - 1 > limits.max
                or low < np.iinfo(np.int64).min or high - 1 > np.iinfo(np.int64).max):
            raise ValueError('Integer interval exceeds dtype/range')
        shape = () if size is None else (size,) if isinstance(size, (int, np.integer)) else tuple(size)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 0
               for v in shape):
            raise ValueError('Invalid uniform sample shape')
        count = math.prod(shape)
        result = np.empty(count, dtype=dtype)
        if span == 1:
            result.fill(low)
        else:
            bits = (span - 1).bit_length()
            word_bytes = 4 if bits <= 32 else 8
            raw_dtype = '<u4' if word_bytes == 4 else '<u8'
            mask = (1 << bits) - 1
            filled = 0
            while filled < count:
                batch = min(65536, max(64, count - filled if bits <= 32 else 2 * (count - filled)))
                words = np.frombuffer(self._random_bytes(batch * word_bytes), dtype=raw_dtype)
                if bits <= 32:
                    product = words.astype(np.uint64) * np.uint64(span)
                    accepted = (product >> 32)[(product & 0xffffffff) >= (1 << 32) % span]
                else:
                    candidates = words & mask
                    accepted = candidates[candidates < span]
                take = min(len(accepted), count - filled)
                result[filled:filled + take] = accepted[:take].astype(np.int64) + low
                filled += take
        result = result.reshape(shape)
        return result[()] if size is None else result


def _integer_array(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind in 'iub':
        return array
    if array.dtype.kind != 'O':
        raise TypeError(f'{name} requires integer values')
    result = np.empty(array.shape, dtype=object)
    for index, item in enumerate(array.flat):
        if not isinstance(item, (int, np.integer, bool, np.bool_)):
            raise TypeError(f'{name} requires integer values, including in object arrays')
        result.flat[index] = int(item)
    return result


def mod_q(x: np.ndarray, q: int) -> np.ndarray:
    if not isinstance(q, int) or not 1 < q <= np.iinfo(np.int64).max:
        raise ValueError("q must fit positive signed int64")
    value = _integer_array(x, 'mod_q')
    if value.dtype.kind != 'O' and value.dtype.itemsize < 8:
        value = value.astype(np.int64)
    return np.mod(value, q).astype(np.int64, copy=False)


def _max_abs_integer(array: np.ndarray) -> int:
    if not array.size:
        return 0
    return max(abs(int(array.min())), abs(int(array.max())))


_I64_MAX = int(np.iinfo(np.int64).max)
_FLOAT_EXACT_MAX = (1 << 53) - 1
_PRODUCT_TEMP_BYTES = 64 * 1024 * 1024


def _modular_product_plan(rows: int, inner: int, cols: int,
                          left_abs: int, right_abs: int, q: int) -> dict:
    left_abs, right_abs = int(left_abs), int(right_abs)
    tall_gemv = cols in (1, 2) and rows >= 1024 and 64 <= inner <= 512 and rows * inner >= 262144
    use_float = tall_gemv or (rows >= 4 and cols >= 4 and rows * inner * cols >= 262144)
    limit = _FLOAT_EXACT_MAX if use_float else _I64_MAX

    reduce_left = left_abs > q - 1 and (tall_gemv or left_abs * right_abs > limit)
    reduce_right = right_abs > q - 1 and (tall_gemv or left_abs * right_abs > limit)
    aa = min(left_abs, q - 1) if reduce_left else left_abs
    bb = min(right_abs, q - 1) if reduce_right else right_abs
    pair_bound = aa * bb
    if use_float and (aa > _FLOAT_EXACT_MAX or bb > _FLOAT_EXACT_MAX
                      or pair_bound * min(max(1, inner), 256) > _FLOAT_EXACT_MAX):
        use_float = False
        limit = _I64_MAX
    backend = "exact_float64" if use_float else "bounded_int64"
    if aa > _I64_MAX or bb > _I64_MAX or pair_bound > limit:
        backend = "python_integer"
    if backend == "python_integer":
        row_block, col_block, inner_block = 32, 128, 128
    else:
        col_block = min(max(1, cols), 512 if use_float else 128)
        row_block = min(max(1, rows), 256 if use_float else 262144 // col_block)
        memory_inner = max(1, _PRODUCT_TEMP_BYTES // (8 * (row_block + col_block)))
        safe_inner = limit // max(1, pair_bound)
        inner_block = min(max(1, inner), max(1, safe_inner), memory_inner)
    return {"backend": backend, "row_block": row_block, "col_block": col_block,
            "inner_block": inner_block, "reduce_left": reduce_left,
            "reduce_right": reduce_right, "absolute_product_bound": pair_bound,
            "accumulator_limit": limit, "operand_temp_bytes": _PRODUCT_TEMP_BYTES}


def _add_canonical_mod(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    if q <= _I64_MAX // 2 + 1:
        result = np.asarray(np.add(a, b), dtype=np.int64)
        if result.size >= 4096:
            np.subtract(result, q, out=result)
            result += (result >> 63) & q
        else:
            np.subtract(result, q, out=result, where=result >= q)
        return result
    gap = q - b
    wrap = a >= gap
    result = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.int64)
    np.subtract(a, gap, out=result, where=wrap)
    np.add(a, b, out=result, where=~wrap)
    return result


def _sub_canonical_mod(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    result = np.asarray(np.subtract(a, b), dtype=np.int64)
    if result.size >= 4096:
        result += (result >> 63) & q
    else:
        np.add(result, q, out=result, where=result < 0)
    return result


def _affine_canonical_mod(a, scalar, b, q, *, subtract=False):
    scalar = int(scalar) % q
    if scalar > q // 2:
        scalar -= q
    edge = scalar * (q - 1)
    lower, upper = min(0, edge), max(0, edge)
    if subtract:
        lower -= q - 1
    else:
        upper += q - 1
    if -_I64_MAX - 1 <= lower and upper <= _I64_MAX:
        result = np.asarray(np.multiply(a, scalar), dtype=np.int64)
        operation = np.subtract if subtract else np.add
        if np.shape(b) == result.shape:
            operation(result, b, out=result)
        else:
            result = np.asarray(operation(result, b), dtype=np.int64)
        np.remainder(result, q, out=result)
        return result
    scaled = scale_mod(a, scalar, q)
    operation = _sub_canonical_mod if subtract else _add_canonical_mod
    return operation(scaled, b, q)


def _affine_canonical_equal(left, a, scalar, b, q, *, subtract=False):
    if (type(q) is not int or not 1 < q <= _I64_MAX
            or not isinstance(scalar, (int, np.integer))
            or any(type(value) is not np.ndarray or value.ndim != 1
                   or value.dtype != np.dtype(np.int64) or not value.flags.c_contiguous
                   for value in (left, a, b))
            or not left.shape == a.shape == b.shape or a.size < 65536):
        return bool(np.array_equal(left, _affine_canonical_mod(a, scalar, b, q,
                                                              subtract=subtract)))
    scalar = int(scalar) % q
    if scalar > q // 2:
        scalar -= q
    edge = scalar * (q - 1)
    lower, upper = min(0, edge), max(0, edge)
    if subtract:
        lower -= q - 1
    else:
        upper += q - 1
    if lower < -_I64_MAX - 1 or upper > _I64_MAX:
        return bool(np.array_equal(left, _affine_canonical_mod(a, scalar, b, q,
                                                              subtract=subtract)))
    workspace = np.empty(65536, dtype=np.int64)
    operation = np.subtract if subtract else np.add
    for start in range(0, a.size, workspace.size):
        stop = min(start + workspace.size, a.size)
        output = workspace[:stop - start]
        np.multiply(a[start:stop], scalar, out=output)
        operation(output, b[start:stop], out=output)
        np.remainder(output, q, out=output)
        if not np.array_equal(left[start:stop], output):
            return False
    return True


def _modular_product_block(a: np.ndarray, b: np.ndarray, q: int, plan: dict) -> np.ndarray:
    if plan["reduce_left"]:
        a = np.mod(a, q)
    if plan["reduce_right"]:
        b = np.mod(b, q)
    dtype = {"exact_float64": np.float64, "bounded_int64": np.int64,
             "python_integer": object}[plan["backend"]]
    block = a.astype(dtype, copy=False) @ b.astype(dtype, copy=False)
    if plan["backend"] == "exact_float64":
        np.rint(block, out=block)
        block = block.astype(np.int64)
    np.remainder(block, q, out=block)
    return block.astype(np.int64, copy=False)


def matmul_mod(left: np.ndarray, right: np.ndarray, q: int) -> np.ndarray:
    return _matmul_mod_integer(left, right, q)


def _matmul_mod_canonical_left(left: np.ndarray, right: np.ndarray, q: int) -> np.ndarray:
    if (not isinstance(left, np.ndarray) or left.ndim != 2
            or left.dtype != np.dtype(np.int64) or left.flags.writeable
            or not left.flags.c_contiguous):
        raise ValueError('Matrix must be read-only contiguous int64')
    return _matmul_mod_integer(left, right, q, canonical_left=True)


def _prepared_product_plan(rows, inner, cols, q):
    if type(q) is not int or q <= 1 or (q-1)**2 > _FLOAT_EXACT_MAX:
        raise ValueError('Unsafe binary64 field product')
    return {'row_block': min(rows, 65536),
            'inner_block': min(inner, _FLOAT_EXACT_MAX//((q-1)**2)),
            'col_block': cols, 'backend': 'prepared_exact_float64',
            'accumulator_limit': _FLOAT_EXACT_MAX,
            'absolute_product_bound': (q-1)**2}


def _prepared_binary_product_plan(rows, inner, cols, q):
    plan = _prepared_product_plan(rows, inner, cols, q)
    plan['inner_block'] = min(inner, _FLOAT_EXACT_MAX // (q - 1),
                              max(1, _PRODUCT_TEMP_BYTES // (8 * cols)))
    plan['absolute_product_bound'] = q - 1
    plan['operand_temp_bytes'] = _PRODUCT_TEMP_BYTES
    return plan


def _matmul_mod_prepared_left(left, right, q):
    lanes = len(right) if isinstance(right, tuple) else (np.shape(right)[1] if np.ndim(right) == 2 else 1)
    workers = min(4 if lanes == 2 else 2, os.cpu_count() or 1)
    if (isinstance(left, np.ndarray) and left.ndim == 2 and left.shape[0] >= 65536
            and left.size >= 4194304 and workers > 1):
        slices = [slice(left.shape[0]*index//workers, left.shape[0]*(index+1)//workers)
                  for index in range(workers)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pieces = list(pool.map(lambda part: _matmul_mod_prepared_serial(left[part], right, q), slices))
        return np.concatenate(pieces, axis=0)
    return _matmul_mod_prepared_serial(left, right, q)


def _matmul_mod_prepared_serial(left, right, q):
    if (not isinstance(left, np.ndarray) or left.ndim != 2
            or left.dtype != np.dtype(np.float64) or left.flags.writeable
            or left.strides[1] != 8 or left.strides[0] < 8*left.shape[1]):
        raise ValueError('Matrix must be read-only contiguous float64')
    if isinstance(right, tuple):
        columns = tuple(_integer_array(column, 'prepared product') for column in right)
        is_vector = False
    else:
        right = _integer_array(right, 'prepared product')
        if right.ndim not in (1, 2):
            raise ValueError('prepared product requires a vector or matrix')
        is_vector = right.ndim == 1
        columns = (right,) if is_vector else tuple(right[:, index] for index in range(right.shape[1]))
    if not columns or any(column.ndim != 1 or len(column) != left.shape[1] for column in columns):
        raise ValueError('prepared product dimensions do not match')
    plan = _prepared_product_plan(left.shape[0], left.shape[1], len(columns), q)
    binary = left.shape[1] > plan['inner_block'] and all(
        not np.any(chunk < 0) and not np.any(chunk > 1)
        for column in columns
        for start in range(0, len(column), 65536)
        for chunk in (column[start:start + 65536],))
    if binary:
        plan = _prepared_binary_product_plan(left.shape[0], left.shape[1], len(columns), q)
    result = np.zeros((left.shape[0], len(columns)), dtype=np.int64)
    for row in range(0, left.shape[0], plan['row_block']):
        stop = min(left.shape[0], row+plan['row_block'])
        total = None
        for inner in range(0, left.shape[1], plan['inner_block']):
            end = min(left.shape[1], inner+plan['inner_block'])
            operand = np.empty((end-inner, len(columns)), dtype=np.float64)
            for index, column in enumerate(columns):
                if binary:
                    np.copyto(operand[:, index], column[inner:end], casting='unsafe')
                else:
                    operand[:, index] = mod_q(column[inner:end], q)
            matrix = left[row:stop, inner:end]
            if len(columns)==1:
                block = (matrix @ operand[:, 0]).astype(np.int64).reshape(-1, 1)
            elif len(columns)==2 and left.shape[0]<=512:
                values = [matrix @ np.ascontiguousarray(operand[:, index]) for index in range(2)]
                block = np.column_stack(values).astype(np.int64)
            else:
                block = (matrix @ operand).astype(np.int64)
            np.remainder(block, q, out=block)
            total = block if total is None else _add_canonical_mod(total, block, q)
        if total is not None:
            result[row:stop] = total
    return result[:, 0] if is_vector else result


def _matmul_mod_integer(left: np.ndarray, right: np.ndarray, q: int, *,
                        canonical_left: bool = False) -> np.ndarray:
    if not isinstance(q, int) or not 1 < q <= np.iinfo(np.int64).max:
        raise ValueError("q must fit positive signed int64")
    left, right = _integer_array(left, 'matmul_mod'), _integer_array(right, 'matmul_mod')
    if left.ndim not in (1, 2) or right.ndim not in (1, 2):
        raise ValueError("matmul_mod supports vectors and matrices")
    is_lvec, is_rvec = left.ndim == 1, right.ndim == 1
    a = left.reshape(1, -1) if is_lvec else left
    b = right.reshape(-1, 1) if is_rvec else right
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible product shapes")
    plan = _modular_product_plan(a.shape[0], a.shape[1], b.shape[1],
                                 q - 1 if canonical_left else _max_abs_integer(a),
                                 _max_abs_integer(b), q)
    rb, cb, kb = plan["row_block"], plan["col_block"], plan["inner_block"]
    if not plan["absolute_product_bound"]:
        product = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
    elif a.shape[0] <= rb and b.shape[1] <= cb and a.shape[1] <= kb:
        product = _modular_product_block(a, b, q, plan)
    else:
        product = np.zeros((a.shape[0], b.shape[1]), dtype=np.int64)
        for row in range(0, a.shape[0], rb):
            for col in range(0, b.shape[1], cb):
                total = None
                for inner in range(0, a.shape[1], kb):
                    block = _modular_product_block(a[row:row + rb, inner:inner + kb],
                                                   b[inner:inner + kb, col:col + cb], q, plan)
                    total = block if total is None else _add_canonical_mod(total, block, q)
                if total is not None:
                    product[row:row + rb, col:col + cb] = total
    if is_lvec and is_rvec:
        return product.reshape(())
    if is_lvec:
        return product[0]
    if is_rvec:
        return product[:, 0]
    return product


def add_mod(*arrays: np.ndarray, q: int) -> np.ndarray:
    if not arrays:
        raise ValueError("at least one summand is required")
    result = mod_q(np.asarray(arrays[0]), q)
    for array in arrays[1:]:
        result = _add_canonical_mod(result, mod_q(np.asarray(array), q), q)
    return result


def scale_mod(vector, scalar, q):
    if not isinstance(scalar, (int, np.integer)) or isinstance(scalar, (bool, np.bool_)):
        raise TypeError('modular scalar must be an integer')
    value = np.asarray(mod_q(vector, q))
    scalar = int(scalar) % q
    if scalar > q // 2:
        scalar -= q
    if (q - 1) * abs(scalar) <= _I64_MAX:
        np.multiply(value, scalar, out=value)
        np.remainder(value, q, out=value)
        return value
    return np.asarray((value.astype(object) * scalar) % q, dtype=np.int64)


def require_memory(required_bytes: int, max_memory_bytes: int, purpose: str) -> None:
    if required_bytes > max_memory_bytes:
        raise MemoryError(f"{purpose}: {required_bytes:,} B required, limit {max_memory_bytes:,} B")


def discrete_gaussian_tail_bound(sigma: float, beta: int) -> float:
    if not math.isfinite(sigma) or sigma <= 0 or beta < 0:
        raise ValueError("sigma must be positive and beta nonnegative")
    ratio_log = -math.pi * ((2 * beta + 3) / sigma) / sigma
    if ratio_log == 0.0:
        return 1.0
    log_bound = math.log(2) - math.pi * ((beta + 1) / sigma)**2 - math.log(-math.expm1(ratio_log))
    if log_bound >= 0:
        return 1.0
    return max(math.exp(log_bound), float(np.nextafter(0.0, 1.0)))


def discrete_gaussian_pmf(sigma: float, beta: int) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(sigma) or sigma <= 0 or not isinstance(beta, int) or beta < 0:
        raise ValueError('Require sigma > 0 and integer beta >= 0')
    support = np.arange(-beta, beta + 1, dtype=np.int64)
    weights = np.exp(-math.pi * (support.astype(np.float64) / sigma)**2)
    return support, weights / weights.sum()


def rand_discrete_gaussian(rng: np.random.Generator, shape: tuple[int, ...], sigma: float,
                          beta: int, max_memory_bytes: int = 512 * 1024 * 1024) -> np.ndarray:
    precise_sampler = getattr(rng, 'discrete_gaussian', None)
    if callable(precise_sampler):
        return precise_sampler(shape, sigma, beta, max_memory_bytes)
    if any(not isinstance(s, int) or s < 0 for s in shape):
        raise ValueError("invalid sample shape")
    choice = getattr(rng, 'choice', None)
    if not callable(choice):
        raise NotImplementedError('Gaussian sampling requires RNG.choice()')
    count = math.prod(shape)
    require_memory(8 * count + 32 * (2 * beta + 1) + 2 * 1024 * 1024,
                   max_memory_bytes, "discrete Gaussian sample and finite PMF")
    support, probabilities = discrete_gaussian_pmf(sigma, beta)
    result = np.empty(count, dtype=np.int64)

    for start in range(0, count, 65536):
        result[start:start + 65536] = choice(support, size=min(65536, count - start), p=probabilities)
    return result.reshape(shape)


def rand_mod_matrix(rng: np.random.Generator, rows: int, cols: int, q: int) -> np.ndarray:
    return rng.integers(0, q, size=(rows, cols), dtype=np.int64)


def rand_bits(rng: np.random.Generator, length: int) -> np.ndarray:
    return rng.integers(0, 2, size=length, dtype=np.uint8)


def bits_from_ints_le(values: np.ndarray, k: int) -> np.ndarray:
    vals = values.astype(np.int64, copy=False).reshape(-1)
    shifts = np.arange(k, dtype=np.int64)
    return (((vals[:, None] >> shifts) & 1).astype(np.uint8)).reshape(-1)


def int_to_bits_le(x: int, length: int) -> np.ndarray:
    return np.array([(x >> i) & 1 for i in range(length)], dtype=np.uint8)


def bits_to_int_le(bits: np.ndarray) -> int:
    out = 0
    for i, b in enumerate(bits.astype(np.uint8).tolist()):
        out |= int(b) << i
    return out


def digest_public_statement(*items: bytes | np.ndarray | str | int) -> str:
    h = hashlib.sha256()
    h.update(b"FDGE-ASIACRYPT2021-RELATION")
    for item in items:
        if isinstance(item, np.ndarray):
            arr = np.ascontiguousarray(item)
            h.update(str(arr.shape).encode())
            h.update(str(arr.dtype).encode())
            h.update(arr.tobytes())
        elif isinstance(item, bytes):
            h.update(item)
        elif isinstance(item, str):
            h.update(item.encode())
        elif isinstance(item, int):
            h.update(item.to_bytes(16, "little", signed=True))
        else:
            raise TypeError(type(item))
    return h.hexdigest()


def decrypt_bits_from_noisy_mod(values: np.ndarray, q: int) -> np.ndarray:
    v = np.mod(values, q)
    distance_zero = np.minimum(v, q - v)
    delta_one = np.abs(v - q // 2)
    distance_one = np.minimum(delta_one, q - delta_one)
    return (distance_one < distance_zero).astype(np.uint8)
