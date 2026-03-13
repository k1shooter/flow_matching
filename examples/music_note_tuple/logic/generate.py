# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn.functional as F

from data.note_tuple import ATTR_NAMES
from samplers.ctmc_sampler import ctmc_step_note_tuple

from torch import nn, Tensor

from .edit_flow import generate_samples_edit_flow
from .flow import MixtureDiscreteNoteTuplePath, SourceDistribution

LOGGER = logging.getLogger(__name__)
NON_PITCH_ATTRS = [attr for attr in ATTR_NAMES if attr != "pitch"]


def _sample_categorical_from_logits(
    logits: Tensor,
    temperature: float,
    disallow_zero_token: bool = False,
) -> Tensor:
    scaled_logits = logits.float() / max(temperature, 1e-6)
    if disallow_zero_token and scaled_logits.shape[-1] > 1:
        scaled_logits = scaled_logits.clone()
        scaled_logits[..., 0] = torch.finfo(scaled_logits.dtype).min
    probs = torch.softmax(scaled_logits, dim=-1)
    samples = torch.multinomial(
        probs.reshape(-1, probs.shape[-1]),
        num_samples=1,
    ).reshape(*probs.shape[:-1])
    return samples.long()


def _apply_pitch_gating_(x_t: Dict[str, Tensor]) -> None:
    inactive = x_t["pitch"] == 0
    for attr in NON_PITCH_ATTRS:
        x_t[attr][inactive] = 0


def _invalid_tuple_rate(x_t: Dict[str, Tensor]) -> Tensor:
    active = x_t["pitch"] > 0
    invalid = active & (
        (x_t["track"] == 0)
        | (x_t["bar"] == 0)
        | (x_t["pos"] == 0)
        | (x_t["dur"] == 0)
        | (x_t["vel"] == 0)
    )
    return invalid.float().mean()


@torch.no_grad()
def _generate_samples_posterior(
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
    _ = rho_cap  # kept for backward-compatible function signature

    x_t = source_distribution.sample(
        batch_size=sample_batch_size,
        sequence_length=sequence_length,
        device=device,
    )
    _apply_pitch_gating_(x_t)

    t_final = 1.0 - time_epsilon
    base_h = t_final / max(1, sampling_steps)
    t = 0.0

    intermediates: List[Dict[str, Tensor]] = []

    while t < t_final - 1e-12:
        h = min(base_h, t_final - t)

        t_scalar = torch.full((sample_batch_size,), t, device=device)
        attention_mask = (x_t["pitch"] != 0).long()
        logits = model(x_t=x_t, time=t_scalar, attention_mask=attention_mask)

        rho_pitch = path.update_prob(attr="pitch", t=t_scalar, h=h)
        pitch_update_mask = (
            torch.rand_like(x_t["pitch"], dtype=torch.float32) < rho_pitch[:, None]
        )
        sampled_pitch = _sample_categorical_from_logits(
            logits=logits["pitch"],
            temperature=temperature,
            disallow_zero_token=False,
        )
        x_t["pitch"] = torch.where(pitch_update_mask, sampled_pitch, x_t["pitch"])

        # pitch=0 => entire slot is PAD
        _apply_pitch_gating_(x_t)
        active = x_t["pitch"] > 0

        for attr in NON_PITCH_ATTRS:
            rho = path.update_prob(attr=attr, t=t_scalar, h=h)
            attr_update_mask = (
                torch.rand_like(x_t[attr], dtype=torch.float32) < rho[:, None]
            )
            attr_update_mask = attr_update_mask & active
            sampled_values = _sample_categorical_from_logits(
                logits=logits[attr],
                temperature=temperature,
                disallow_zero_token=True,
            )
            x_t[attr] = torch.where(attr_update_mask, sampled_values, x_t[attr])
            x_t[attr][~active] = 0

        t = t + h

        if return_intermediates:
            intermediates.append({k: v.clone() for k, v in x_t.items()})

    if final_full_resample:
        t_tensor = torch.full((sample_batch_size,), t_final, device=device)
        attention_mask = (x_t["pitch"] != 0).long()
        logits = model(
            x_t=x_t,
            time=t_tensor,
            attention_mask=attention_mask,
        )
        x_t["pitch"] = _sample_categorical_from_logits(
            logits=logits["pitch"],
            temperature=temperature,
            disallow_zero_token=False,
        )
        _apply_pitch_gating_(x_t)

        active = x_t["pitch"] > 0
        for attr in NON_PITCH_ATTRS:
            sampled_attr = _sample_categorical_from_logits(
                logits=logits[attr],
                temperature=temperature,
                disallow_zero_token=True,
            )
            x_t[attr] = torch.where(active, sampled_attr, torch.zeros_like(x_t[attr]))

    _apply_pitch_gating_(x_t)
    invalid_rate = _invalid_tuple_rate(x_t=x_t).item()
    note_density = (x_t["pitch"] > 0).float().mean().item()
    bar_min = int(x_t["bar"].min().item())
    bar_max = int(x_t["bar"].max().item())
    pos_min = int(x_t["pos"].min().item())
    pos_max = int(x_t["pos"].max().item())
    LOGGER.info(
        "DFM sampling stats: note_density=%.6f invalid_tuple_rate=%.6f "
        "bar_range=[%d,%d] pos_range=[%d,%d]",
        note_density,
        invalid_rate,
        bar_min,
        bar_max,
        pos_min,
        pos_max,
    )

    if return_intermediates:
        return intermediates

    return x_t


@torch.no_grad()
def _generate_samples_velocity(
    model: nn.Module,
    source_distribution: SourceDistribution,
    sample_batch_size: int,
    sequence_length: int,
    sampling_steps: int,
    device: torch.device,
    time_epsilon: float = 1e-3,
    velocity_eps: float = 1e-12,
    corrector_weight: float = 0.0,
) -> Dict[str, Tensor]:
    x_t = source_distribution.sample(
        batch_size=sample_batch_size,
        sequence_length=sequence_length,
        device=device,
    )
    _apply_pitch_gating_(x_t)

    t_final = 1.0 - time_epsilon
    base_h = t_final / max(1, sampling_steps)
    t = 0.0
    lambda_hist_total = torch.zeros(10, device=device)
    changed_fracs = []

    while t < t_final - 1e-12:
        h = min(base_h, t_final - t)
        t_scalar = torch.full((sample_batch_size,), t, device=device)
        attention_mask = (x_t["pitch"] != 0).long()
        velocity_output = model(
            x_t=x_t,
            time=t_scalar,
            attention_mask=attention_mask,
            return_velocity=True,
        )

        step_lambda = torch.cat(
            [
                F.softplus(velocity_output["log_lambda"][attr].reshape(-1).float())
                for attr in ATTR_NAMES
            ],
            dim=0,
        )
        max_hist = float(step_lambda.max().item()) if step_lambda.numel() else 1.0
        lambda_hist_total += torch.histc(
            step_lambda,
            bins=10,
            min=0.0,
            max=max(1e-6, max_hist),
        )

        x_t, step_stats = ctmc_step_note_tuple(
            x_t=x_t,
            token_logits=velocity_output["token_logits"],
            log_lambda=velocity_output["log_lambda"],
            h=h,
            eps=velocity_eps,
            corrector_weight=corrector_weight,
        )

        attr_changed = [
            float(step_stats[f"{attr}_changed_frac"])
            for attr in ATTR_NAMES
            if f"{attr}_changed_frac" in step_stats
        ]
        if attr_changed:
            changed_fracs.append(sum(attr_changed) / len(attr_changed))

        t = t + h

    _apply_pitch_gating_(x_t)
    invalid_rate = _invalid_tuple_rate(x_t=x_t).item()
    note_density = (x_t["pitch"] > 0).float().mean().item()
    changed_mean = float(sum(changed_fracs) / max(1, len(changed_fracs)))

    LOGGER.info(
        "Velocity CTMC sampling stats: note_density=%.6f invalid_tuple_rate=%.6f "
        "changed_frac_mean=%.6f lambda_hist=%s",
        note_density,
        invalid_rate,
        changed_mean,
        ",".join(f"{float(v):.2f}" for v in lambda_hist_total.tolist()),
    )
    return x_t


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
    parameterization: str = "posterior",
    velocity_eps: float = 1e-12,
    velocity_corrector_weight: float = 0.0,
) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
    if parameterization == "velocity":
        if return_intermediates:
            raise ValueError(
                "return_intermediates=True is not supported for velocity sampling."
            )
        return _generate_samples_velocity(
            model=model,
            source_distribution=source_distribution,
            sample_batch_size=sample_batch_size,
            sequence_length=sequence_length,
            sampling_steps=sampling_steps,
            device=device,
            time_epsilon=time_epsilon,
            velocity_eps=velocity_eps,
            corrector_weight=velocity_corrector_weight,
        )

    return _generate_samples_posterior(
        model=model,
        path=path,
        source_distribution=source_distribution,
        sample_batch_size=sample_batch_size,
        sequence_length=sequence_length,
        sampling_steps=sampling_steps,
        device=device,
        time_epsilon=time_epsilon,
        temperature=temperature,
        rho_cap=rho_cap,
        final_full_resample=final_full_resample,
        return_intermediates=return_intermediates,
    )


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
