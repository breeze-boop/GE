from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributions import BoundedErrorDistribution, require_error_distribution, sample_bounded_errors
from .mathutils import (add_mod, decrypt_bits_from_noisy_mod, matmul_mod,
                        rand_bits, rand_discrete_gaussian, require_memory)
from .params import FDGEParams


def _immutable_public_matrix(value):
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype.kind not in 'iub':
        raise ValueError('Public keys must be integer matrices')

    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class UserPublicKey:
    U1: np.ndarray
    U2: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'U1', _immutable_public_matrix(self.U1))
        object.__setattr__(self, 'U2', _immutable_public_matrix(self.U2))

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (self.U1, self.U2)

    def as_paper(self) -> tuple[np.ndarray, np.ndarray]:
        return self.U1, self.U2


@dataclass
class UserSecretKey:
    E1: np.ndarray

    def as_paper(self) -> np.ndarray:
        return self.E1


@dataclass(frozen=True)
class OAPublicKey:
    U1: np.ndarray
    U2: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'U1', _immutable_public_matrix(self.U1))
        object.__setattr__(self, 'U2', _immutable_public_matrix(self.U2))

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (self.U1, self.U2)

    def as_paper(self) -> tuple[np.ndarray, np.ndarray]:
        return self.U1, self.U2


@dataclass
class OASecretKey:
    E1: np.ndarray

    def as_paper(self) -> np.ndarray:
        return self.E1


@dataclass
class GPVCipherPair:
    c1: np.ndarray
    c2: np.ndarray

    def as_paper(self) -> tuple[np.ndarray, np.ndarray]:
        return self.c1, self.c2


@dataclass
class RecipientCiphertext:
    c1: GPVCipherPair
    c2: GPVCipherPair

    def as_paper(self):
        return self.c1.as_paper(), self.c2.as_paper()


@dataclass
class OACiphertext:
    c1: GPVCipherPair
    c2: GPVCipherPair

    def as_paper(self):
        return self.c1.as_paper(), self.c2.as_paper()


@dataclass
class EncryptionCoins:
    s_rec: list[np.ndarray]
    x_rec: list[np.ndarray]
    y_rec: list[np.ndarray]
    s_oa: list[np.ndarray]
    x_oa: list[np.ndarray]
    y_oa: list[np.ndarray]

    @property
    def randrec(self):
        return tuple((self.s_rec[i], self.x_rec[i], self.y_rec[i]) for i in range(2))

    @property
    def randoa(self):
        return tuple((self.s_oa[i], self.x_oa[i], self.y_oa[i]) for i in range(2))

    def as_paper(self):
        return self.randrec, self.randoa


def _integer_array(value, shape, name):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer array of shape {shape}") from exc
    if array.shape != shape or array.dtype.kind not in 'iubO':
        raise ValueError(f"{name} must be an integer array of shape {shape}")
    if array.dtype.kind == 'O' and not all(
            isinstance(x, (int, np.integer, bool, np.bool_)) for x in array.flat):
        raise ValueError(f"{name} must contain only integers")
    return array


def validate_decryption_inputs(params: FDGEParams, sk, ct, *, plaintext_length: int):
    secret = _integer_array(getattr(sk, 'E1', None), (params.m, plaintext_length), 'E1')
    first = getattr(ct, 'c1', None)
    vectors = []
    for name, length in (('c1', params.m), ('c2', plaintext_length)):
        vector = _integer_array(getattr(first, name, None), (length,), f'first branch {name}')
        if np.any(vector < 0) or np.any(vector >= params.q):
            raise ValueError(f'first branch {name} must contain canonical Z_q entries')
        vectors.append(vector.astype(np.int64, copy=False))
    return secret, vectors[0], vectors[1]


def keygen_oa(params: FDGEParams, Aoa: np.ndarray, rng: np.random.Generator) -> tuple[OAPublicKey, OASecretKey]:
    params.validate()
    if Aoa.shape != (params.n, params.m):
        raise ValueError("Aoa must have shape (n,m)")
    _keygen_memory_preflight(params, params.depth)
    E1 = rand_discrete_gaussian(rng, (params.m, params.depth), params.resolved_sigma(),
                               params.resolved_beta(), params.max_memory_bytes)
    U1 = matmul_mod(Aoa, E1, params.q)
    E2 = rand_discrete_gaussian(rng, (params.m, params.depth), params.resolved_sigma(),
                               params.resolved_beta(), params.max_memory_bytes)
    U2 = matmul_mod(Aoa, E2, params.q)
    return OAPublicKey(U1=U1, U2=U2), OASecretKey(E1=E1)


def _keygen_memory_preflight(params: FDGEParams, columns: int) -> None:
    needed = (16 * params.m * columns + 32 * params.n * columns
              + 8 * params.n * params.m + 32 * (2 * params.resolved_beta() + 1)
              + 2 * 1024 * 1024)
    require_memory(needed, params.max_memory_bytes, "dense GPV key generation peak")


def _encryption_product(matrix: np.ndarray, vector: np.ndarray, q: int) -> np.ndarray:
    result = np.empty(matrix.shape[1], dtype=np.int64)
    for column in range(matrix.shape[1]):
        result[column] = matmul_mod(matrix[:, column], vector, q)
    return result


def encrypt_recipient(params: FDGEParams, Arec: np.ndarray, pk: UserPublicKey, message: np.ndarray, rng: np.random.Generator, *, chi: BoundedErrorDistribution | None = None) -> tuple[RecipientCiphertext, dict[str, list[np.ndarray]]]:
    params.validate()
    chi = require_error_distribution(params.chi if chi is None else chi, params.error_bound)
    if (message.shape != (params.m,) or message.dtype.kind not in 'iub'
            or not np.all((message == 0) | (message == 1))):
        raise ValueError(f"message must be {params.m} bits")
    if Arec.shape != (params.n, params.m) or pk.U1.shape != Arec.shape or pk.U2.shape != Arec.shape:
        raise ValueError("recipient public matrices must have shape (n,m)")
    pairs: list[GPVCipherPair] = []
    s_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for U in (pk.U1, pk.U2):
        s = rand_bits(rng, params.n).astype(np.int64)
        x = sample_bounded_errors(chi, rng, params.m, params.error_bound)
        y = sample_bounded_errors(chi, rng, params.m, params.error_bound)
        c1 = add_mod(_encryption_product(Arec, s, params.q), x, q=params.q)
        c2 = add_mod(_encryption_product(U, s, params.q), y,
                     message.astype(np.int64) * params.qhalf, q=params.q)
        pairs.append(GPVCipherPair(c1=c1, c2=c2))
        s_list.append(s.astype(np.uint8))
        x_list.append(x)
        y_list.append(y)
    return RecipientCiphertext(c1=pairs[0], c2=pairs[1]), {"s": s_list, "x": x_list, "y": y_list}


def decrypt_recipient(params: FDGEParams, sk: UserSecretKey, ct: RecipientCiphertext) -> np.ndarray:
    params.validate()
    secret, c1, c2 = validate_decryption_inputs(params, sk, ct, plaintext_length=params.m)
    product = np.empty(params.m, dtype=np.int64)

    for offset in range(0, params.m, 512):
        stop = min(offset + 512, params.m)
        product[offset:stop] = matmul_mod(secret[:, offset:stop].T, c1, params.q)
    noisy = c2 - product
    return decrypt_bits_from_noisy_mod(noisy, params.q)


def encrypt_oa(params: FDGEParams, Aoa: np.ndarray, pk: OAPublicKey, id_bits: np.ndarray, rng: np.random.Generator, *, chi: BoundedErrorDistribution | None = None) -> tuple[OACiphertext, dict[str, list[np.ndarray]]]:
    params.validate()
    chi = require_error_distribution(params.chi if chi is None else chi, params.error_bound)
    if (id_bits.shape != (params.depth,) or id_bits.dtype.kind not in 'iub'
            or not np.all((id_bits == 0) | (id_bits == 1))):
        raise ValueError(f"id_bits must be {params.depth} bits")
    if Aoa.shape != (params.n, params.m) or pk.U1.shape != (params.n, params.depth) or pk.U2.shape != pk.U1.shape:
        raise ValueError("OA public matrices have incompatible dimensions")
    pairs: list[GPVCipherPair] = []
    s_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for U in (pk.U1, pk.U2):
        s = rand_bits(rng, params.n).astype(np.int64)
        x = sample_bounded_errors(chi, rng, params.m, params.error_bound)
        y = sample_bounded_errors(chi, rng, params.depth, params.error_bound)
        c1 = add_mod(_encryption_product(Aoa, s, params.q), x, q=params.q)
        c2 = add_mod(_encryption_product(U, s, params.q), y,
                     id_bits.astype(np.int64) * params.qhalf, q=params.q)
        pairs.append(GPVCipherPair(c1=c1, c2=c2))
        s_list.append(s.astype(np.uint8))
        x_list.append(x)
        y_list.append(y)
    return OACiphertext(c1=pairs[0], c2=pairs[1]), {"s": s_list, "x": x_list, "y": y_list}


def decrypt_oa(params: FDGEParams, sk: OASecretKey, ct: OACiphertext) -> np.ndarray:
    params.validate()
    secret, c1, c2 = validate_decryption_inputs(params, sk, ct, plaintext_length=params.depth)

    product = []
    for column in secret.T:
        value = 0
        for coefficient, ciphertext_entry in zip(column, c1):
            value = (value + int(coefficient) * int(ciphertext_entry)) % params.q
        product.append(value)
    noisy = c2 - np.asarray(product, dtype=np.int64)
    return decrypt_bits_from_noisy_mod(noisy, params.q)
