# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from typing import Dict

import torch

from data.note_tuple import ATTR_NAMES
from torch import nn, Tensor


class TimestepEmbedder(nn.Module):
    def __init__(self, output_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    @staticmethod
    def timestep_embedding(time: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=time.device)
            / half
        )
        args = time[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, time: Tensor) -> Tensor:
        return self.mlp(self.timestep_embedding(time, self.frequency_embedding_size))


class Transformer(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], config) -> None:
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.d_model = int(config.d_model)
        self.max_seq_len = int(config.max_seq_len)
        self.enable_edit_flow = bool(getattr(config, "enable_edit_flow", False))

        self.attribute_embeddings = nn.ModuleDict(
            {attr: nn.Embedding(vocab_sizes[attr], self.d_model) for attr in ATTR_NAMES}
        )

        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.time_embedding = TimestepEmbedder(output_dim=self.d_model)
        self.input_dropout = nn.Dropout(float(config.dropout))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(config.n_heads),
            dim_feedforward=int(config.ff_mult) * self.d_model,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(config.n_layers),
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(self.d_model)

        self.output_heads = nn.ModuleDict(
            {attr: nn.Linear(self.d_model, vocab_sizes[attr]) for attr in ATTR_NAMES}
        )

        if self.enable_edit_flow:
            self.edit_heads = nn.ModuleDict(
                {
                    "insert": nn.Linear(self.d_model, 1),
                    "delete": nn.Linear(self.d_model, 1),
                    "substitute": nn.Linear(self.d_model, 1),
                }
            )

    def _encode(
        self,
        x_t: Dict[str, Tensor],
        time: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        _, sequence_length = x_t["pitch"].shape
        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds max_seq_len={self.max_seq_len}."
            )

        hidden: Tensor = 0
        for attr in ATTR_NAMES:
            hidden = hidden + self.attribute_embeddings[attr](x_t[attr])

        positions = torch.arange(sequence_length, device=time.device)
        hidden = hidden + self.position_embedding(positions)[None, :, :]

        hidden = hidden + self.time_embedding(time)[:, None, :]
        hidden = self.input_dropout(hidden)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        hidden = self.final_norm(hidden)
        return hidden

    def forward(
        self,
        x_t: Dict[str, Tensor],
        time: Tensor,
        attention_mask: Tensor | None = None,
        return_edit_logits: bool = False,
    ):
        hidden = self._encode(x_t=x_t, time=time, attention_mask=attention_mask)
        logits = {attr: head(hidden) for attr, head in self.output_heads.items()}

        if return_edit_logits:
            if not self.enable_edit_flow:
                raise RuntimeError(
                    "return_edit_logits=True but model was built with enable_edit_flow=False"
                )
            edit_logits = {
                name: head(hidden).squeeze(-1) for name, head in self.edit_heads.items()
            }
            return logits, edit_logits

        return logits
