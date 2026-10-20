from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mathutils import add_mod, bits_from_ints_le, matmul_mod, rand_mod_matrix
from .params import FDGEParams


@dataclass
class AccumulatorWitness:
    index: int
    siblings_bottom_up: list[np.ndarray]


class LNWXAccumulator:
    def __init__(self, params: FDGEParams, rng: np.random.Generator):
        self.params = params
        self.A = rand_mod_matrix(rng, params.n, 2 * params.nk, params.q)
        self.A0 = self.A[:, : params.nk]
        self.A1 = self.A[:, params.nk :]
        self.N = params.N
        self.zero_leaf = np.zeros(params.nk, dtype=np.uint8)

        self.nodes: list[np.ndarray | None] = [None] * (2 * self.N)
        for i in range(1, 2 * self.N):
            self.nodes[i] = self.zero_leaf.copy()

    def hash(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        self._check_bits(left)
        self._check_bits(right)
        v = add_mod(matmul_mod(self.A0, left, self.params.q), matmul_mod(self.A1, right, self.params.q), q=self.params.q)
        return bits_from_ints_le(v, self.params.k)

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return self.hash(left, right)

    def root(self) -> np.ndarray:
        return self.nodes[1].copy()

    def leaf(self, index: int) -> np.ndarray:
        return self.nodes[self._leaf_position(index)].copy()

    def update(self, index: int, value: np.ndarray) -> None:
        self._check_index(index)
        self._check_bits(value)
        pos = self._leaf_position(index)
        self.nodes[pos] = value.astype(np.uint8, copy=True)
        pos //= 2
        while pos >= 1:
            self.nodes[pos] = self.hash(self.nodes[2 * pos], self.nodes[2 * pos + 1])
            pos //= 2

    def witness(self, index: int) -> AccumulatorWitness:
        self._check_index(index)
        pos = self._leaf_position(index)
        siblings: list[np.ndarray] = []
        while pos > 1:
            sib = pos ^ 1
            siblings.append(self.nodes[sib].copy())
            pos //= 2
        return AccumulatorWitness(index=index, siblings_bottom_up=siblings)

    def verify(self, root: np.ndarray, leaf: np.ndarray, wit: AccumulatorWitness) -> bool:
        if not isinstance(wit, AccumulatorWitness):
            return False
        try:
            self._check_index(wit.index)
            self._check_bits(root)
            self._check_bits(leaf)
            if not isinstance(wit.siblings_bottom_up, (list, tuple)):
                return False
            if len(wit.siblings_bottom_up) != self.params.depth:
                return False
            for sibling in wit.siblings_bottom_up:
                self._check_bits(sibling)
        except (ValueError, AttributeError, TypeError):
            return False
        pos = self._leaf_position(wit.index)
        node = leaf.astype(np.uint8, copy=True)
        for sibling in wit.siblings_bottom_up:
            if pos % 2 == 0:
                node = self.hash(node, sibling)
            else:
                node = self.hash(sibling, node)
            pos //= 2
        return bool(np.array_equal(node, root))

    def _leaf_position(self, index: int) -> int:
        self._check_index(index)
        position = 1
        for bit in range(self.params.depth):
            position = 2 * position + ((int(index) >> bit) & 1)
        return position

    def _check_index(self, index: int) -> None:
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not (0 <= index < self.N):
            raise ValueError(f"index {index} outside [0,{self.N})")

    def _check_bits(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray) or value.shape != (self.params.nk,):
            raise ValueError(f"accumulator node must have shape {(self.params.nk,)}")
        if value.dtype.kind not in 'biu':
            raise ValueError("accumulator nodes must have integer or boolean dtype")
        if not np.all((value == 0) | (value == 1)):
            raise ValueError("accumulator nodes must be binary before conversion")
