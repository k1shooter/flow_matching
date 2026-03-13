# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""Slot-based edit operations for fixed-length note-tuple tensors.

EditFlow training targets are derived on index-aligned padded slots, so sampling
uses the same slot semantics to avoid sequence-shift mismatch:
- delete(pos): set slot `pos` to PAD (all attributes = 0)
- insert(pos): fill an empty slot `pos` (pitch==0) with a non-PAD tuple
- substitute(pos): replace an active slot `pos` (pitch>0) with a non-PAD tuple
"""

from __future__ import annotations

from typing import Dict

from data.note_tuple import ATTR_NAMES
from torch import Tensor


def sanitize_pitch_gating_(x: Dict[str, Tensor]) -> None:
    """Enforce note-tuple structure: pitch==0 => all other attrs must be 0."""
    inactive = x["pitch"] == 0
    for attr in ATTR_NAMES:
        if attr == "pitch":
            continue
        x[attr][inactive] = 0


def delete_slot_(x: Dict[str, Tensor], pos: int) -> None:
    for attr in ATTR_NAMES:
        x[attr][pos] = 0


def substitute_slot_(x: Dict[str, Tensor], pos: int, token: Dict[str, int]) -> None:
    for attr in ATTR_NAMES:
        x[attr][pos] = int(token[attr])
    sanitize_pitch_gating_(x)


def insert_slot_(x: Dict[str, Tensor], pos: int, token: Dict[str, int]) -> bool:
    if int(x["pitch"][pos].item()) != 0:
        return False
    substitute_slot_(x=x, pos=pos, token=token)
    return True
