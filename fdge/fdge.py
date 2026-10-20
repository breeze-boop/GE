from __future__ import annotations

from dataclasses import dataclass

from .gpv import OACiphertext, RecipientCiphertext


@dataclass
class FDGECiphertext:
    c_rec: RecipientCiphertext
    c_oa: OACiphertext
    epoch: int
    label: bytes
    context_id: str
