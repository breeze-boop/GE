from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading


class _GPVTemplate(ctypes.Structure):
    _fields_ = [('q', ctypes.c_uint64), ('digits', ctypes.c_uint32),
                ('columns', ctypes.c_int64*64), ('weights', ctypes.c_int64*64),
                ('denominators', ctypes.c_uint64*8), ('radii', ctypes.c_uint64*8),
                ('rho_low', ctypes.c_uint64*8), ('rho_high', ctypes.c_uint64*8),
                ('pi_lo', ctypes.c_uint64), ('pi_hi', ctypes.c_uint64)]


class _GPVState(ctypes.Structure):
    _fields_ = [('produced', ctypes.c_size_t), ('level', ctypes.c_int),
                ('pending', ctypes.c_int), ('center', ctypes.c_int64*8),
                ('candidate', ctypes.c_int64), ('draw', ctypes.c_uint64)]


class _RationalGaussianTemplate(ctypes.Structure):
    _fields_ = [('start', ctypes.c_int64), ('width', ctypes.c_uint64),
                ('numerator', ctypes.c_uint64*2), ('denominator', ctypes.c_uint64*2),
                ('pi_lo', ctypes.c_uint64*3), ('pi_hi', ctypes.c_uint64*3)]


class _GaussianEnvelope(ctypes.Structure):
    _fields_=[('variance',ctypes.c_uint64),('pi_lo',ctypes.c_uint64),
        ('pi_hi',ctypes.c_uint64),('bands',ctypes.c_uint64),
        ('start',ctypes.c_int64*256),('width',ctypes.c_uint64*256),
        ('lower',ctypes.c_uint64*256),('upper',ctypes.c_uint64*256),
        ('cumulative_lo',ctypes.c_uint64*256),('cumulative_hi',ctypes.c_uint64*256)]


class _EnvelopeState(ctypes.Structure):
    _fields_=[('stage',ctypes.c_int),('band',ctypes.c_int),
              ('candidate',ctypes.c_int64),('draw',ctypes.c_uint64)]


@lru_cache(maxsize=1)
def native_library():
    compiler = shutil.which('clang') or shutil.which('cc')
    if compiler is None:
        return None
    source = Path(__file__).with_suffix('.c')
    build = tempfile.TemporaryDirectory(prefix='latge-sampler-private-')
    output = Path(build.name)/('filter.dylib' if sys.platform == 'darwin' else 'filter.so')
    command = [compiler, '-O3', '-std=c11', '-fPIC',
               '-dynamiclib' if sys.platform == 'darwin' else '-shared',
               str(source), '-o', str(output)]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            build.cleanup()
            return None
        lib = ctypes.CDLL(str(output))
    except OSError:
        build.cleanup()
        return None
    lib._private_build = build
    lib._source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    lib.latge_rho_interval.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
                                      ctypes.c_uint, ctypes.POINTER(ctypes.c_uint64)]
    lib.latge_rho_interval.restype = ctypes.c_int
    lib.latge_rho_decide.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
                                    ctypes.c_uint, ctypes.c_uint64]
    lib.latge_rho_decide.restype = ctypes.c_int
    size_pointer = ctypes.POINTER(ctypes.c_size_t)
    lib.latge_table_gaussian_vector.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
        size_pointer, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_uint8, ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t, size_pointer]
    lib.latge_table_gaussian_vector.restype = ctypes.c_int
    lib.latge_gaussian_vector.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
        size_pointer, ctypes.c_int64, ctypes.c_uint64, ctypes.c_uint64,
        ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t, size_pointer, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint64)]
    lib.latge_gaussian_vector.restype = ctypes.c_int
    lib.latge_envelope_gaussian.argtypes=[ctypes.c_void_p,ctypes.c_size_t,
        size_pointer,ctypes.POINTER(_GaussianEnvelope),ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,size_pointer,ctypes.POINTER(_EnvelopeState)]
    lib.latge_envelope_gaussian.restype=ctypes.c_int
    lib.latge_rational_gaussian_vector.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
        size_pointer, ctypes.POINTER(_RationalGaussianTemplate), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t, size_pointer, ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_uint64)]
    lib.latge_rational_gaussian_vector.restype = ctypes.c_int
    lib.latge_gpv_vector.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
        size_pointer, ctypes.POINTER(_GPVTemplate), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(_GPVState)]
    lib.latge_gpv_vector.restype = ctypes.c_int
    return lib


def rho_interval(ylo, yhi, squares):
    if any(type(x) is not int for x in (ylo, yhi, squares)) or not (
            0 <= ylo <= yhi <= 1 << 57 and 0 <= squares <= 16):
        raise ValueError('native rho filter input outside exact bounds')
    lib = native_library()
    if lib is None:
        return None
    out = (ctypes.c_uint64*2)()
    if lib.latge_rho_interval(ylo, yhi, squares, out):
        raise ArithmeticError('native rho enclosure invariant failed')
    return int(out[0]), int(out[1]), 60


class SystemRandomBuffer:

    def __init__(self, block_bytes=16384):
        if type(block_bytes) is not int or not 64 <= block_bytes <= 1 << 20:
            raise ValueError('entropy buffer size outside bounded range')
        self.block_bytes = block_bytes
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._buffer, self._position = b'', 0

    def __getstate__(self):
        raise TypeError('a live entropy buffer cannot be copied or serialized')

    def __copy__(self):
        raise TypeError('a live entropy buffer cannot be copied or serialized')

    def __deepcopy__(self, memo):
        raise TypeError('a live entropy buffer cannot be copied or serialized')

    def randbelow(self, n):
        if type(n) is not int or n <= 0:
            raise ValueError('randbelow needs a positive integer')
        self._check_context()
        bits = (n-1).bit_length()
        if not bits:
            return 0
        count, mask = (bits+7)//8, (1 << bits)-1
        while True:
            if count > self.block_bytes:
                raw = os.urandom(count)
            else:
                if self._position + count > len(self._buffer):
                    self._refill()
                raw = self._buffer[self._position:self._position+count]
                self._position += count
            value = int.from_bytes(raw, 'little') & mask
            if value < n:
                return value

    def _check_context(self):
        pid, thread = os.getpid(), threading.get_ident()
        if pid != self._pid:
            self._pid, self._thread = pid, thread
            self._buffer, self._position = b'', 0
        if thread != self._thread:
            raise RuntimeError('entropy buffer must not be shared between threads')

    def _refill(self):
        self._buffer = os.urandom(self.block_bytes)
        self._position = 0


def table_gaussian_vector(support, cdf, count, rng):
    bits = (cdf[-1]-1).bit_length()
    draw_bytes, entry_bytes = (bits+7)//8, (cdf[-1].bit_length()+7)//8
    if draw_bytes > rng.block_bytes or not all(-(1 << 63) <= x < 1 << 63 for x in support):
        return None
    library = native_library()
    if library is None:
        return None
    table = b''.join(value.to_bytes(entry_bytes, 'little') for value in cdf)
    values = (ctypes.c_int64*len(support))(*support)
    output, produced = (ctypes.c_int64*count)(), ctypes.c_size_t()
    mask = (1 << (bits % 8))-1 if bits % 8 else 255
    rng._check_context()
    while produced.value < count:
        if not rng._buffer and draw_bytes:
            rng._refill()
        position = ctypes.c_size_t(rng._position)
        code = library.latge_table_gaussian_vector(rng._buffer, len(rng._buffer),
            ctypes.byref(position), table, entry_bytes, draw_bytes, mask,
            values, len(support), output, count, ctypes.byref(produced))
        rng._position = position.value
        if code == 0:
            rng._check_context()
            rng._refill()
        elif code != 1:
            raise ArithmeticError('native Gaussian table invariant failed')
    return list(output)


def integer_gaussian_vector(support, variance, count, rng):

    if (not isinstance(support, range) or support.step != 1 or not len(support)
            or support.start < -(1 << 31) or support.stop > 1 << 31
            or type(variance) is not int or not 0 < variance <= 1 << 56
            or type(count) is not int or count < 0 or type(rng) is not SystemRandomBuffer):
        raise ValueError('integer Gaussian vector outside its exact specialization')
    library = native_library()
    if library is None:
        return None
    from .gaussian_reference import _pi_fixed, _bernoulli_rho_refine
    pi_lo, pi_hi = _pi_fixed(60)
    output = (ctypes.c_int64*count)()
    produced, pending = ctypes.c_size_t(), ctypes.c_int()
    candidate, draw = ctypes.c_int64(), ctypes.c_uint64()
    rng._check_context()
    while produced.value < count:
        if not rng._buffer:
            rng._refill()
        position = ctypes.c_size_t(rng._position)
        code = library.latge_gaussian_vector(rng._buffer, len(rng._buffer),
            ctypes.byref(position), support.start, len(support), variance,
            pi_lo, pi_hi, output, count, ctypes.byref(produced),
            ctypes.byref(pending), ctypes.byref(candidate), ctypes.byref(draw))
        rng._position = position.value
        if code < 0:
            raise ArithmeticError('native Gaussian vector invariant failed')
        if code == 0:
            rng._check_context()
            rng._refill()
        elif code == 2:
            if _bernoulli_rho_refine(candidate.value**2, variance, draw.value, 48, rng.randbelow):
                output[produced.value] = candidate.value
                produced.value += 1
        elif code != 1:
            raise ArithmeticError('unknown native Gaussian vector status')
    return list(output)


def rational_gaussian_vector(support, variance, count, rng):

    if (not isinstance(support, range) or support.step != 1 or not len(support)
            or support.start < -(1 << 31) or support.stop > 1 << 31
            or type(count) is not int or count < 0 or type(rng) is not SystemRandomBuffer):
        raise ValueError('rational Gaussian vector outside its exact specialization')
    numerator, denominator = variance.numerator, variance.denominator
    max_square = max(abs(support.start), abs(support.stop-1))**2
    if not 0 < numerator < 1 << 120 or max_square*denominator >= 1 << 126:
        return None
    library = native_library()
    if library is None:
        return None
    from .gaussian_reference import _pi_fixed, _bernoulli_rho_refine
    geometry = _RationalGaussianTemplate(start=support.start, width=len(support))
    pi_lo, pi_hi = _pi_fixed(128)
    mask = (1 << 64)-1
    for array, value in ((geometry.numerator, numerator), (geometry.denominator, denominator),
                         (geometry.pi_lo, pi_lo), (geometry.pi_hi, pi_hi)):
        for i in range(len(array)):
            array[i] = (value >> (64*i)) & mask
    output = (ctypes.c_int64*count)()
    produced, pending = ctypes.c_size_t(), ctypes.c_int()
    candidate, draw = ctypes.c_int64(), ctypes.c_uint64()
    rng._check_context()
    while produced.value < count:
        if not rng._buffer:
            rng._refill()
        position = ctypes.c_size_t(rng._position)
        code = library.latge_rational_gaussian_vector(rng._buffer, len(rng._buffer),
            ctypes.byref(position), ctypes.byref(geometry), output, count,
            ctypes.byref(produced), ctypes.byref(pending), ctypes.byref(candidate), ctypes.byref(draw))
        rng._position = position.value
        if code < 0:
            raise ArithmeticError('native rational Gaussian vector invariant failed')
        if code == 0:
            rng._check_context()
            rng._refill()
        elif code == 2:
            if _bernoulli_rho_refine(candidate.value**2*denominator, numerator,
                                     draw.value, 48, rng.randbelow):
                output[produced.value] = candidate.value
                produced.value += 1
        elif code != 1:
            raise ArithmeticError('unknown native rational Gaussian vector status')
    return list(output)


def _gadget_template(sampler):

    digits = sampler.digits
    if not 1 <= digits <= 8 or not 2 <= sampler.q < 1 << 63:
        return None
    geometry = _GPVTemplate(q=sampler.q, digits=digits)
    bounds = [sampler.q-1]+[0]*(digits-1)
    for i in reversed(range(digits)):
        weights, denominator = sampler.projections[i]
        radius, width_num, width_den = sampler.sampling_templates[i]
        square = denominator*denominator
        if (not 0 < denominator < 1 << 63 or not 0 <= radius < (1 << 31)-1
                or square % width_den or (radius+2)*denominator >= 1 << 63):
            return None
        rho_denominator = square//width_den*width_num
        if not 0 < rho_denominator < 1 << 120:
            return None
        if any(abs(value) >= 1 << 63 for _, value in weights):
            return None
        numerator_bound = sum(bounds[j]*abs(weight) for j, weight in weights)
        if numerator_bound >= 1 << 126:
            return None
        coefficient_bound = (numerator_bound+denominator-1)//denominator+radius+1
        for j, value in enumerate(sampler.columns[i]):
            bounds[j] += coefficient_bound*abs(value)
            if abs(value) >= 1 << 63 or bounds[j] >= 1 << 61:
                return None
            geometry.columns[8*i+j] = value
        for j, value in weights:
            geometry.weights[8*i+j] = value
        geometry.denominators[i], geometry.radii[i] = denominator, radius
        geometry.rho_low[i] = rho_denominator & ((1 << 64)-1)
        geometry.rho_high[i] = rho_denominator >> 64
    from .gaussian_reference import _pi_fixed
    geometry.pi_lo, geometry.pi_hi = _pi_fixed(60)
    return geometry


def gadget_gaussian_vector(sampler, targets, rng):
    if type(rng) is not SystemRandomBuffer:
        raise ValueError('native gadget sampling requires its call-local entropy buffer')
    geometry = _gadget_template(sampler)
    if geometry is None:
        return None
    library = native_library()
    if library is None:
        return None
    if not targets:
        return []
    from .gaussian_reference import _bernoulli_rho_refine
    digits, count = sampler.digits, len(targets)
    targets_array = (ctypes.c_int64*count)(*(target % sampler.q for target in targets))
    output = (ctypes.c_int64*(count*digits))()
    state = _GPVState(level=digits)
    rng._check_context()
    while state.produced < count:
        if not rng._buffer:
            rng._refill()
        position = ctypes.c_size_t(rng._position)
        code = library.latge_gpv_vector(rng._buffer, len(rng._buffer),
            ctypes.byref(position), ctypes.byref(geometry), targets_array, count,
            output, ctypes.byref(state))
        rng._position = position.value
        if code == 0:
            rng._check_context()
            rng._refill()
        elif code == 2:
            level = state.level
            weights, denominator = sampler.projections[level]
            center = sum(int(state.center[j])*weight for j, weight in weights)
            remainder = center % denominator
            nearest = min(remainder, denominator-remainder)
            displacement = state.candidate*denominator-center
            numerator = displacement*displacement-nearest*nearest
            rho_denominator = int(geometry.rho_low[level])+(int(geometry.rho_high[level]) << 64)
            accepted = _bernoulli_rho_refine(numerator, rho_denominator,
                                            state.draw, 48, rng.randbelow)
            state.pending = 3 if accepted else 0
        elif code != 1:
            raise ArithmeticError('native GPV sampling invariant failed')
    values = [tuple(output[i*digits+j] for j in range(digits)) for i in range(count)]
    powers = tuple(sampler.base**i for i in range(digits))
    for target, value in zip(targets, values):
        if sum(a*b for a, b in zip(powers, value)) % sampler.q != target % sampler.q:
            raise ArithmeticError('gadget sampler returned an invalid coset element')
    return values


@lru_cache(maxsize=8)
def gaussian_envelope(width):
    from fractions import Fraction
    from .gaussian_reference import _pi_fixed,_rho_dyadic_interval
    if type(width) is not int or width<1 or width*width>1<<56 or 6*width>=1<<31:
        raise ValueError('Gaussian width outside native envelope range')
    lo,hi=_pi_fixed(60);g=_GaussianEnvelope(variance=width*width,pi_lo=lo,pi_hi=hi)
    first=-6*width;length=12*width+1;g.bands=min(256,length);total=0
    def height(x,upper):
        low,high,bits=_rho_dyadic_interval(Fraction(x*x,width*width))
        value=high if upper else low
        if bits>=48:
            shift=bits-48
            return (value+(1<<shift)-1)>>shift if upper else value>>shift
        return value<<(48-bits)
    for i in range(g.bands):
        a=first+length*i//g.bands;b=first+length*(i+1)//g.bands
        closest=0 if a<=0<b else min(abs(a),abs(b-1))
        farthest=max(abs(a),abs(b-1))
        g.start[i]=a;g.width[i]=b-a
        g.upper[i]=height(closest,True);g.lower[i]=height(farthest,False)
        total+=(b-a)*g.upper[i]
        g.cumulative_lo[i]=total&((1<<64)-1);g.cumulative_hi[i]=total>>64
    return g


def envelope_gaussian_array(width,count,rng):

    import numpy as np
    from .gaussian_reference import _bernoulli_rho_refine
    library=native_library()
    if library is None:raise RuntimeError('native Gaussian sampler unavailable')
    g=gaussian_envelope(width);out=np.empty(count,dtype=np.int64)
    produced=ctypes.c_size_t();state=_EnvelopeState();rng._check_context()
    while produced.value<count:
        if not rng._buffer:rng._refill()
        position=ctypes.c_size_t(rng._position)
        code=library.latge_envelope_gaussian(rng._buffer,len(rng._buffer),ctypes.byref(position),
            ctypes.byref(g),out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),count,
            ctypes.byref(produced),ctypes.byref(state))
        rng._position=position.value
        if code==0:rng._refill()
        elif code==2:
            if _bernoulli_rho_refine(state.candidate**2,width*width,state.draw,48,rng.randbelow):
                out[produced.value]=state.candidate;produced.value+=1
        elif code!=1:raise ArithmeticError('Gaussian envelope invariant failed')
    return out
