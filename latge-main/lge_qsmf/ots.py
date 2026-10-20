from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
import threading
from typing import List, Sequence, Tuple

from .ring import Poly, Ring, is_prime_u64


def _integer_root(value: int, degree: int) -> int:
    low, high = 0, 1 << ((value.bit_length() + degree - 1) // degree)
    while low + 1 < high:
        middle = (low + high) // 2
        if middle ** degree <= value:
            low = middle
        else:
            high = middle
    return high if high ** degree <= value else low


@dataclass(frozen=True)
class OTSParams:
    n: int = 128
    m: int | None = None
    p: int = 70368744177679

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or isinstance(self.n, bool) or not 4 <= self.n <= 256 or self.n & (self.n - 1):
            raise ValueError("OTS n must be a power of two in [4, 256] for SHA-256 truncation")
        expected_m = self.n.bit_length() - 1
        if self.m is None:
            object.__setattr__(self, "m", expected_m)
        if type(self.m) is not int or self.m != expected_m:
            raise ValueError("TCC 2008 OTS requires m=ceil(log2(n))")
        if type(self.p) is not int or not 2 < self.p < 1 << 63 or not is_prime_u64(self.p):
            raise ValueError("Main.pdf OTS requires a prime modulus below 2^63")
        if 2 * self.signature_bound >= self.p:
            raise ValueError("OTS modulus is too small for the original no-wrap bound")

    @property
    def mixture_limit(self) -> int:
        return (self.n.bit_length() - 1) ** 2

    @property
    def signature_bound(self) -> int:
        scale = 10 * self.n * self.mixture_limit
        return _integer_root(scale ** self.m * self.p, self.m)

    def secret_bounds(self, component: int) -> tuple[int, int]:
        if not 1 <= component <= self.mixture_limit:
            raise ValueError("invalid mixture component")
        first_scale = 5 * component
        second_scale = first_scale * self.n
        return (_integer_root(first_scale ** self.m * self.p, self.m),
                _integer_root(second_scale ** self.m * self.p, self.m))


DEFAULT_OTS_PARAMS = OTSParams()
_USED_SIGNING_KEYS: set[bytes] = set()
_SIGNING_LOCK = threading.Lock()


@dataclass(frozen=True)
class OTSKeypair:
    sk_bytes: bytes
    vk_bytes: bytes


def _shake(data: bytes, outlen: int) -> bytes:
    return hashlib.shake_256(data).digest(outlen)


def _pack_u32(x: int) -> bytes:
    return int(x).to_bytes(4, "little")


def _ring_for_params(params: OTSParams) -> Ring:
    return Ring(params.n, params.p)


def _poly_step(ring: Ring) -> int:
    return ring.n * ((ring.q.bit_length() + 7) // 8)


def _polyvec_inf_norm(v: Sequence[Poly]) -> int:
    return max((max(abs(int(c)) for c in p.centered()) for p in v), default=0)


def _sample_bounded_poly(ring: Ring, bound: int, seed: bytes, nonce: int) -> Poly:
    if not 0 <= bound <= (ring.q - 1) // 2:
        raise ValueError("invalid centered sampling bound")
    span = 2 * bound + 1
    width = max(1, ((span - 1).bit_length() + 7) // 8)
    limit = (1 << (8 * width)) // span * span
    out, counter = [], 0
    while len(out) < ring.n:
        stream = _shake(seed + _pack_u32(nonce) + _pack_u32(counter), width * ring.n)
        counter += 1
        for pos in range(0, len(stream), width):
            value = int.from_bytes(stream[pos:pos + width], "little")
            if value < limit:
                out.append(value % span - bound)
                if len(out) == ring.n:
                    break
    return Poly.from_ints(ring, out)


def _sample_uniform_poly(ring: Ring, seed: bytes, nonce: int) -> Poly:
    return _sample_bounded_poly(ring, (ring.q - 1) // 2, seed, nonce)


def _inner(row: Sequence[Poly], vec: Sequence[Poly]) -> Poly:
    if not row or len(row) != len(vec):
        raise ValueError("dimension mismatch")
    from .linalg import mat_vec_mul
    return mat_vec_mul([row], vec)[0]


def _header(magic: bytes, params: OTSParams) -> bytes:
    return magic + _pack_u32(params.n) + _pack_u32(params.m) + params.p.to_bytes(8, "little")


def _parse_header(raw: bytes, magic: bytes) -> OTSParams:
    if len(raw) < 22 or raw[:6] != magic:
        raise ValueError("invalid or unsupported OTS encoding")
    return OTSParams(n=int.from_bytes(raw[6:10], "little"),
                     m=int.from_bytes(raw[10:14], "little"),
                     p=int.from_bytes(raw[14:22], "little"))


def _decode_polyvec(ring: Ring, raw: bytes, count: int) -> List[Poly]:
    step = _poly_step(ring)
    if len(raw) != count * step:
        raise ValueError("bad OTS polynomial-vector length")
    return [Poly.from_bytes(ring, raw[i * step:(i + 1) * step]) for i in range(count)]


def _parse_sk(raw: bytes) -> Tuple[OTSParams, bytes, bytes]:
    params = _parse_header(raw, b"LM08SK")
    if len(raw) != 86:
        raise ValueError("bad OTS secret-key length")
    return params, raw[22:54], raw[54:86]


@lru_cache(maxsize=256)
def _parse_vk(raw: bytes, params: OTSParams, public_seed: bytes | None) -> Tuple[bytes, Poly, Poly]:
    if public_seed is not None:
        if not isinstance(public_seed, bytes) or len(public_seed) != 32:
            raise ValueError("shared OTS hash seed must be 32 bytes")
        values = _decode_polyvec(_ring_for_params(params), raw, 2)
        return public_seed, values[0], values[1]
    if _parse_header(raw, b"LM08VK") != params:
        raise ValueError("OTS key does not match the public parameter profile")
    values = _decode_polyvec(_ring_for_params(params), raw[54:], 2)
    return raw[22:54], values[0], values[1]


def _vk_digest(raw: bytes, params: OTSParams, public_seed: bytes | None) -> bytes:
    context = _header(b"LM08CT", params)
    context += b"individual" if public_seed is None else b"shared" + public_seed
    return hashlib.sha256(context + raw).digest()


def _parse_sig(raw: bytes) -> Tuple[OTSParams, bytes, List[Poly]]:
    params = _parse_header(raw, b"LM08SG")
    values = _decode_polyvec(_ring_for_params(params), raw[54:], params.m)
    return params, raw[22:54], values


def _serialize_sig(params: OTSParams, signature: Sequence[Poly], vk_digest: bytes) -> bytes:
    if len(signature) != params.m or len(vk_digest) != 32:
        raise ValueError("invalid OTS signature shape")
    if any(p.ring != _ring_for_params(params) for p in signature):
        raise ValueError("OTS signature ring mismatch")
    return _header(b"LM08SG", params) + vk_digest + b"".join(p.to_bytes() for p in signature)


def _mixture_component(params: OTSParams, seed: bytes) -> int:
    limit = params.mixture_limit
    bits = _shake(seed + b"LM08-MIXTURE", (limit + 7) // 8)
    for index in range(limit):
        if (bits[index // 8] >> (7 - index % 8)) & 1:
            return index + 1
    return limit


def _expand_secret(params: OTSParams, seed: bytes) -> Tuple[List[Poly], List[Poly]]:
    ring = _ring_for_params(params)
    bound_k, bound_l = params.secret_bounds(_mixture_component(params, seed))
    return ([_sample_bounded_poly(ring, bound_k, seed + b"LM08-K", i) for i in range(params.m)],
            [_sample_bounded_poly(ring, bound_l, seed + b"LM08-L", i) for i in range(params.m)])


@lru_cache(maxsize=256)
def _expand_public_row(params: OTSParams, seed: bytes) -> List[Poly]:
    return [_sample_uniform_poly(_ring_for_params(params), seed + b"LM08-H", i) for i in range(params.m)]


def hash_digest_to_message_poly(digest: bytes, params: OTSParams = DEFAULT_OTS_PARAMS) -> Poly:
    if len(digest) != 32:
        raise ValueError("message encoding requires a SHA-256 digest")
    return Poly.from_ints(_ring_for_params(params),
                         [(digest[i // 8] >> (7 - i % 8)) & 1 for i in range(params.n)])


def ots_gen(params: OTSParams = DEFAULT_OTS_PARAMS, seed: bytes | None = None, *, public_seed: bytes | None = None) -> OTSKeypair:
    if seed is None:
        seed = os.urandom(32)
    if not isinstance(seed, bytes) or len(seed) < 32:
        raise ValueError("OTS generation requires at least 32 bytes of seed material")
    seed_sk = _shake(seed + b"LM08-SK", 32)
    if public_seed is not None and (not isinstance(public_seed, bytes) or len(public_seed) != 32):
        raise ValueError("shared OTS hash seed must be 32 bytes")
    seed_h = _shake(seed + b"LM08-H", 32) if public_seed is None else public_seed
    k, l = _expand_secret(params, seed_sk)
    row = _expand_public_row(params, seed_h)
    body = _inner(row, k).to_bytes() + _inner(row, l).to_bytes()
    vk = _header(b"LM08VK", params) + seed_h + body if public_seed is None else body
    sk = _header(b"LM08SK", params) + seed_sk + _vk_digest(vk, params, public_seed)
    return OTSKeypair(sk, vk)


def ots_sign(sk_bytes: bytes, msg: bytes, enforce_one_time: bool = True) -> bytes:
    if not enforce_one_time:
        raise ValueError("one-time key reuse cannot be disabled")
    params, seed, vk_digest = _parse_sk(sk_bytes)
    key_id = hashlib.sha256(seed).digest()
    with _SIGNING_LOCK:
        if key_id in _USED_SIGNING_KEYS:
            raise ValueError("OTS secret key has already been used")
        k, l = _expand_secret(params, seed)
        message = hash_digest_to_message_poly(hashlib.sha256(msg).digest(), params)
        signature = [a * message + b for a, b in zip(k, l)]
        if _polyvec_inf_norm(signature) > params.signature_bound:
            raise ValueError("OTS signature exceeds the original norm bound")
        encoded = _serialize_sig(params, signature, vk_digest)
        _USED_SIGNING_KEYS.add(key_id)
        return encoded


def _verify_signature(vk_bytes: bytes, sig: bytes, msg: bytes, params: OTSParams, public_seed: bytes | None) -> bool:
    seed_h, hash_k, hash_l = _parse_vk(vk_bytes, params, public_seed)
    sig_params, digest, signature = _parse_sig(sig)
    if sig_params != params or digest != _vk_digest(vk_bytes, params, public_seed):
        return False
    if _polyvec_inf_norm(signature) > params.signature_bound:
        return False
    message = hash_digest_to_message_poly(hashlib.sha256(msg).digest(), params)
    return _inner(_expand_public_row(params, seed_h), signature) == hash_k * message + hash_l


def ots_verify(vk_bytes: bytes, sig: bytes, msg: bytes, *, params: OTSParams = DEFAULT_OTS_PARAMS, public_seed: bytes | None = None) -> bool:
    try:
        return _verify_signature(bytes(vk_bytes), bytes(sig), bytes(msg), params, public_seed)
    except (ValueError, TypeError, OverflowError):
        return False


