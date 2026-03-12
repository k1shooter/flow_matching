# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

"""Placeholder edit operators for future insertion/deletion style flows.

This example uses fixed-length note-slot tuples where PAD (token 0 across all
attributes) denotes an empty slot. Future edit-flow work can reuse this
representation to model insertion/deletion as slot activation/deactivation.
"""

from __future__ import annotations

from typing import Dict

from torch import Tensor


def insert_operation_stub(x: Dict[str, Tensor]) -> Dict[str, Tensor]:
    raise NotImplementedError("Edit-flow insert operations are not implemented yet.")


def delete_operation_stub(x: Dict[str, Tensor]) -> Dict[str, Tensor]:
    raise NotImplementedError("Edit-flow delete operations are not implemented yet.")
