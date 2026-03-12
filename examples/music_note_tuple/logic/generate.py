# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, List

import torch

from data.note_tuple import ATTR_NAMES

from torch import nn, Tensor

from .edit_flow import generate_samples_edit_flow
from .flow import MixtureDiscreteNoteTuplePath, SourceDistribution


def _sample_categorical_from_logits(logits: Tensor, temperature: float) -> Tensor:
    scaled_logits = logits.float() / max(temperature, 1e-6)
    probs = torch.softmax(scaled_logits, dim=-1)
    samples = torch.multinomial(
        probs.reshape(-1, probs.shape[-1]),
        num_samples=1,
    ).reshape(*probs.shape[:-1])
    return samples.long()


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    path: MixtureDiscreteNoteTuplePath,
    source_distribution: SourceDistribution,
    sample_batch_size: int,
    sequence_length: int,
    sampling_steps: int,
    device: torch.device,
    time_epsilon: float = 1e-3,
    temperature: float = 1.0,
    rho_cap: float = 0.95,
    final_full_resample: bool = True,
    return_intermediates: bool = False,
) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
    x_t = source_distribution.sample(
        batch_size=sample_batch_size,
        sequence_length=sequence_length,
        device=device,
    )

    t_final = 1.0 - time_epsilon
    base_h = t_final / max(1, sampling_steps)
    t = 0.0

    intermediates: List[Dict[str, Tensor]] = []

    while t < t_final - 1e-12:
        h = min(base_h, t_final - t)

        t_scalar = torch.full((sample_batch_size,), t, device=device)

        # Adaptive step splitting near t=1 to keep Bernoulli update probabilities stable.
        while True:
            max_rho = path.max_rho(t=t_scalar, h=h).max().item()
            if max_rho <= rho_cap or h <= 1e-5:
                break
            h *= 0.5

        logits = model(x_t=x_t, time=t_scalar)

        for attr in ATTR_NAMES:
            rho = path.rho(attr=attr, t=t_scalar, h=h).clamp(min=0.0, max=1.0)
            update_mask = torch.rand_like(x_t[attr], dtype=torch.float32) < rho[:, None]
            sampled_values = _sample_categorical_from_logits(
                logits=logits[attr], temperature=temperature
            )
            x_t[attr] = torch.where(update_mask, sampled_values, x_t[attr])

        t = t + h

        if return_intermediates:
            intermediates.append({k: v.clone() for k, v in x_t.items()})

    if final_full_resample:
        logits = model(
            x_t=x_t,
            time=torch.full((sample_batch_size,), t_final, device=device),
        )
        for attr in ATTR_NAMES:
            x_t[attr] = _sample_categorical_from_logits(
                logits=logits[attr], temperature=temperature
            )

    if return_intermediates:
        return intermediates

    return x_t


@torch.no_grad()
def generate_samples_edit(
    model: nn.Module,
    vocab_sizes: Dict[str, int],
    sample_batch_size: int,
    sequence_length: int,
    sampling_steps: int,
    device: torch.device,
    sampler: str,
    tau_leaping_step_size: float,
    max_events_per_step: int,
    time_epsilon: float = 1e-3,
    temperature: float = 1.0,
    rate_scale: float = 1.0,
    init_mode: str = "empty",
) -> Dict[str, Tensor]:
    return generate_samples_edit_flow(
        model=model,
        vocab_sizes=vocab_sizes,
        sample_batch_size=sample_batch_size,
        max_length=sequence_length,
        sampling_steps=sampling_steps,
        device=device,
        sampler=sampler,
        tau_leaping_step_size=tau_leaping_step_size,
        time_epsilon=time_epsilon,
        temperature=temperature,
        rate_scale=rate_scale,
        max_events_per_step=max_events_per_step,
        init_mode=init_mode,
    )
