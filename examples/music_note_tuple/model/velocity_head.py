# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict

from data.note_tuple import ATTR_NAMES
from torch import nn, Tensor


class VelocityHead(nn.Module):
    """Per-attribute log-hazard head for CTMC velocity parameterization.

    For each attribute `a`, this head predicts `log_lambda_a(i)` per sequence
    position `i`. The corresponding non-negative hazard is recovered by
    `lambda_a(i) = softplus(log_lambda_a(i))`.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.lambda_heads = nn.ModuleDict(
            {attr: nn.Linear(d_model, 1) for attr in ATTR_NAMES}
        )

    def forward(self, hidden: Tensor) -> Dict[str, Tensor]:
        return {
            attr: head(hidden).squeeze(-1) for attr, head in self.lambda_heads.items()
        }
