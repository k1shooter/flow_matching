# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, Optional

import torch

from data.note_tuple import ATTR_NAMES

from omegaconf.dictconfig import DictConfig
from torch import nn, Tensor
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from utils.logging import TrainLogger

from .flow import BaseDiscretePath, SourceDistribution
from .state import TrainState


def _get_lr(lr: float, step: int, warmup: int, n_iters: int, eta_min_ratio: float):
    if step < warmup:
        return lr * (step / warmup)

    eta_min = eta_min_ratio * lr
    cosine_decay = 0.5 * (1 + math.cos(math.pi * (step - warmup) / (n_iters - warmup)))
    return eta_min + (lr - eta_min) * cosine_decay


def optimization_step(
    state: TrainState,
    scaler: GradScaler,
    loss: Tensor,
    optim_params: DictConfig,
    logger: TrainLogger,
) -> None:
    scaler.scale(loss).backward()
    scaler.unscale_(state.optimizer)

    lr = _get_lr(
        lr=optim_params.lr,
        step=state.step,
        warmup=optim_params.warmup,
        n_iters=optim_params.n_iters,
        eta_min_ratio=optim_params.eta_min_ratio,
    )

    for group in state.optimizer.param_groups:
        group["lr"] = lr

    if state.step % optim_params.log_lr_every == 0:
        logger.log_lr(value=lr, step=state.step)

    if optim_params.grad_clip >= 0:
        torch.nn.utils.clip_grad_norm_(
            state.model.parameters(), max_norm=optim_params.grad_clip
        )

    scaler.step(state.optimizer)
    scaler.update()
    state.optimizer.zero_grad(set_to_none=True)


def _extract_target(
    batch: Dict[str, Tensor], device: torch.device
) -> Dict[str, Tensor]:
    return {
        attr: batch[attr].to(device=device, dtype=torch.long) for attr in ATTR_NAMES
    }


def step(
    state: TrainState,
    loss_fn: nn.Module,
    path: BaseDiscretePath,
    scaler: GradScaler,
    iterator: DataLoader,
    device: torch.device,
    source_distribution: SourceDistribution,
    logger: TrainLogger,
    training: bool,
    use_edit_flow: bool,
    parameterization: str = "posterior",
    optim_params: Optional[DictConfig] = None,
    time_epsilon: float = 0.0,
) -> Tensor:
    assert (training and (optim_params is not None)) or (not training)

    if training:
        state.train()
    else:
        state.eval()

    batch = next(iterator)
    x_1 = _extract_target(batch=batch, device=device)

    with torch.no_grad():
        x_0 = source_distribution.sample_like(x_1)
        t = torch.rand(x_1["pitch"].shape[0], device=device) * (1.0 - time_epsilon)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        inactive = path_sample.x_t["pitch"] == 0
        for attr in ATTR_NAMES:
            if attr == "pitch":
                continue
            path_sample.x_t[attr][inactive] = 0
        # Keep train-time masking consistent with sampling-time masking.
        attention_mask = (path_sample.x_t["pitch"] != 0).long()

    ctx = nullcontext() if training else torch.no_grad()
    autocast_enabled = device.type == "cuda"

    with ctx:
        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            if use_edit_flow:
                logits, edit_logits = state.model(
                    x_t=path_sample.x_t,
                    time=path_sample.t,
                    attention_mask=attention_mask,
                    return_edit_logits=True,
                )
                loss = loss_fn(
                    logits=logits,
                    edit_logits=edit_logits,
                    x_t=path_sample.x_t,
                    target=x_1,
                )
                if (
                    hasattr(loss_fn, "last_stats")
                    and state.step % logger.cfg.logging.log_freq == 0
                ):
                    stats = getattr(loss_fn, "last_stats", {})
                    stage = "Train" if training else "Eval"
                    logger.info(
                        f"[{state.step}] {stage} Edit Targets: "
                        f"insert={stats.get('insert_mean', 0.0):.4f}, "
                        f"delete={stats.get('delete_mean', 0.0):.4f}, "
                        f"substitute={stats.get('substitute_mean', 0.0):.4f}, "
                        f"active_t={stats.get('activet_mean', 0.0):.4f}, "
                        f"active_1={stats.get('active1_mean', 0.0):.4f}"
                    )
            elif parameterization == "velocity":
                velocity_output = state.model(
                    x_t=path_sample.x_t,
                    time=path_sample.t,
                    attention_mask=attention_mask,
                    return_velocity=True,
                )
                loss = loss_fn(
                    velocity_output=velocity_output,
                    x_t=path_sample.x_t,
                    x_0=path_sample.x_0,
                    x_1=x_1,
                    t=path_sample.t,
                )
                if (
                    hasattr(loss_fn, "last_stats")
                    and state.step % logger.cfg.logging.log_freq == 0
                ):
                    stats = getattr(loss_fn, "last_stats", {})
                    stage = "Train" if training else "Eval"
                    logger.info(
                        f"[{state.step}] {stage} Velocity Stats: "
                        f"lambda_mean={stats.get('lambda_mean', 0.0):.4f}, "
                        f"lambda_max={stats.get('lambda_max', 0.0):.4f}, "
                        f"target_change={stats.get('target_change_mean', 0.0):.4f}, "
                        f"lambda_hist={stats.get('lambda_hist', '')}"
                    )
            else:
                logits = state.model(
                    x_t=path_sample.x_t,
                    time=path_sample.t,
                    attention_mask=attention_mask,
                )
                loss = loss_fn(logits=logits, target=x_1)

    if training:
        optimization_step(
            state=state,
            loss=loss,
            scaler=scaler,
            optim_params=optim_params,
            logger=logger,
        )

    return loss.detach()
