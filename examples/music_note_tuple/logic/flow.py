# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from data.note_tuple import ATTR_NAMES
from losses import VelocityMatchingLoss
from schedules import build_kappa_scheduler

from torch import nn, Tensor

from .edit_flow import EditFlowLoss


@dataclass
class MultiAttributePathSample:
    x_1: Dict[str, Tensor]
    x_0: Dict[str, Tensor]
    t: Tensor
    x_t: Dict[str, Tensor]


class AttributeScheduler(ABC):
    @abstractmethod
    def kappa(self, t: Tensor) -> Tensor:
        ...

    @abstractmethod
    def dkappa(self, t: Tensor) -> Tensor:
        ...

    def kappa_dot(self, t: Tensor) -> Tensor:
        return self.dkappa(t=t)


class PolynomialAttributeScheduler(AttributeScheduler):
    def __init__(self, exponent: float, scale: float = 1.0) -> None:
        if exponent <= 0:
            raise ValueError(f"Scheduler exponent must be > 0. Got {exponent}.")
        if scale <= 0:
            raise ValueError(f"Scheduler scale must be > 0. Got {scale}.")

        self.exponent = float(exponent)
        self.scale = float(scale)

    def kappa(self, t: Tensor) -> Tensor:
        kappa = self.scale * torch.pow(t, self.exponent)
        return torch.clamp(kappa, min=0.0, max=1.0)

    def dkappa(self, t: Tensor) -> Tensor:
        # Clamp t away from 0 to avoid undefined gradients when exponent < 1.
        t_safe = torch.clamp(t, min=1e-8)
        dkappa = self.scale * self.exponent * torch.pow(t_safe, self.exponent - 1.0)
        return torch.clamp(dkappa, min=0.0)


class GenericAttributeScheduler(AttributeScheduler):
    def __init__(self, schedule: str, exponent: float, scale: float) -> None:
        self.scheduler = build_kappa_scheduler(
            schedule=schedule,
            exponent=exponent,
            scale=scale,
        )

    def kappa(self, t: Tensor) -> Tensor:
        return self.scheduler.kappa(t=t)

    def dkappa(self, t: Tensor) -> Tensor:
        return self.scheduler.kappa_dot(t=t)


class BaseDiscretePath(ABC):
    @abstractmethod
    def sample(
        self,
        t: Tensor,
        x_0: Dict[str, Tensor],
        x_1: Dict[str, Tensor],
    ) -> MultiAttributePathSample:
        ...

    @abstractmethod
    def rho(self, attr: str, t: Tensor, h: float) -> Tensor:
        ...

    def update_prob(self, attr: str, t: Tensor, h: float) -> Tensor:
        """Return per-step update probability for [t, t+h]."""
        return self.rho(attr=attr, t=t, h=h)


class MixtureDiscreteNoteTuplePath(BaseDiscretePath):
    def __init__(self, schedulers: Dict[str, AttributeScheduler]) -> None:
        missing = [attr for attr in ATTR_NAMES if attr not in schedulers]
        if missing:
            raise ValueError(f"Missing schedulers for attributes: {missing}")
        self.schedulers = schedulers

    def sample(
        self,
        t: Tensor,
        x_0: Dict[str, Tensor],
        x_1: Dict[str, Tensor],
    ) -> MultiAttributePathSample:
        x_t: Dict[str, Tensor] = {}

        for attr in ATTR_NAMES:
            kappa = (
                self.schedulers[attr]
                .kappa(t=t)
                .to(device=x_1[attr].device, dtype=torch.float32)
            )
            kappa = kappa[:, None].expand_as(x_1[attr])
            keep_target_mask = torch.rand_like(x_1[attr], dtype=torch.float32) < kappa
            x_t[attr] = torch.where(keep_target_mask, x_1[attr], x_0[attr])

        return MultiAttributePathSample(x_1=x_1, x_0=x_0, t=t, x_t=x_t)

    def update_prob(self, attr: str, t: Tensor, h: float) -> Tensor:
        """Exact jump probability for the mixture path over [t, t+h].

        For attribute `a` with scheduler :math:`\\kappa_a(t)`, the exact update
        probability is:

        .. math::

            \\rho_{\\text{exact}}(t, h)
            = 1 - \\frac{1 - \\kappa_a(t+h)}{1 - \\kappa_a(t)}
            = \\frac{\\kappa_a(t+h) - \\kappa_a(t)}{1 - \\kappa_a(t)}.
        """
        t_next = torch.clamp(t + h, min=0.0, max=1.0)
        kappa_t = self.schedulers[attr].kappa(t=t)
        kappa_t_next = self.schedulers[attr].kappa(t=t_next)

        denom = torch.clamp(1.0 - kappa_t, min=1e-8)
        rho = (kappa_t_next - kappa_t) / denom
        return torch.clamp(rho, min=0.0, max=1.0)

    def rho(self, attr: str, t: Tensor, h: float) -> Tensor:
        # Backward-compatible alias.
        return self.update_prob(attr=attr, t=t, h=h)

    def max_rho(self, t: Tensor, h: float) -> Tensor:
        max_rho = torch.zeros_like(t)
        for attr in ATTR_NAMES:
            max_rho = torch.maximum(max_rho, self.update_prob(attr=attr, t=t, h=h))
        return max_rho


class FutureGeneralDiscretePath(BaseDiscretePath):
    """Hook for future custom paths (General Discrete Paths, ICLR 2025)."""

    def sample(
        self,
        t: Tensor,
        x_0: Dict[str, Tensor],
        x_1: Dict[str, Tensor],
    ) -> MultiAttributePathSample:
        raise NotImplementedError(
            "Custom path hooks are reserved for future extension."
        )

    def rho(self, attr: str, t: Tensor, h: float) -> Tensor:
        raise NotImplementedError(
            "Custom path hooks are reserved for future extension."
        )


class SourceDistribution(ABC):
    @abstractmethod
    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Dict[str, Tensor]:
        ...

    @abstractmethod
    def sample_like(self, tensor_like: Dict[str, Tensor]) -> Dict[str, Tensor]:
        ...


class UniformSourceDistribution(SourceDistribution):
    def __init__(self, vocab_sizes: Dict[str, int], include_pad: bool = True):
        self.vocab_sizes = vocab_sizes
        self.include_pad = bool(include_pad)

    @property
    def masked(self) -> bool:
        return False

    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Dict[str, Tensor]:
        out = {}
        for attr in ATTR_NAMES:
            low = 0 if self.include_pad else 1
            out[attr] = torch.randint(
                low=low,
                high=self.vocab_sizes[attr],
                size=(batch_size, sequence_length),
                device=device,
                dtype=torch.long,
            )
        return out

    def sample_like(self, tensor_like: Dict[str, Tensor]) -> Dict[str, Tensor]:
        out = {}
        for attr in ATTR_NAMES:
            low = 0 if self.include_pad else 1
            out[attr] = torch.randint(
                low=low,
                high=self.vocab_sizes[attr],
                size=tensor_like[attr].shape,
                device=tensor_like[attr].device,
                dtype=torch.long,
            )
        return out


class PadHeavyGatedSourceDistribution(SourceDistribution):
    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        pitch_pad_prob: float,
        include_pad: bool = True,
    ) -> None:
        _ = include_pad  # kept for API compatibility
        if pitch_pad_prob < 0.0 or pitch_pad_prob > 1.0:
            raise ValueError(f"pitch_pad_prob must be in [0,1], got {pitch_pad_prob}.")
        self.vocab_sizes = vocab_sizes
        self.pitch_pad_prob = float(pitch_pad_prob)

    def _sample_shape(
        self, shape: tuple[int, int], device: torch.device
    ) -> Dict[str, Tensor]:
        active = torch.rand(shape, device=device) >= self.pitch_pad_prob
        out = {
            attr: torch.zeros(shape, device=device, dtype=torch.long)
            for attr in ATTR_NAMES
        }

        pitch_vocab = int(self.vocab_sizes["pitch"])
        if pitch_vocab > 1:
            pitch_vals = torch.randint(
                low=1,
                high=pitch_vocab,
                size=shape,
                device=device,
                dtype=torch.long,
            )
            out["pitch"][active] = pitch_vals[active]

        for attr in ATTR_NAMES:
            if attr == "pitch":
                continue
            vocab = int(self.vocab_sizes[attr])
            if vocab <= 1:
                continue
            vals = torch.randint(
                low=1,
                high=vocab,
                size=shape,
                device=device,
                dtype=torch.long,
            )
            out[attr][active] = vals[active]

        return out

    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Dict[str, Tensor]:
        return self._sample_shape(shape=(batch_size, sequence_length), device=device)

    def sample_like(self, tensor_like: Dict[str, Tensor]) -> Dict[str, Tensor]:
        shape = tuple(tensor_like["pitch"].shape)
        return self._sample_shape(shape=shape, device=tensor_like["pitch"].device)


class MaskedSourceDistribution(SourceDistribution):
    """All-PAD source distribution (pitch=0 and all other attrs=0)."""

    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Dict[str, Tensor]:
        shape = (batch_size, sequence_length)
        return {
            attr: torch.zeros(shape, device=device, dtype=torch.long)
            for attr in ATTR_NAMES
        }

    def sample_like(self, tensor_like: Dict[str, Tensor]) -> Dict[str, Tensor]:
        shape = tuple(tensor_like["pitch"].shape)
        return {
            attr: torch.zeros(
                shape, device=tensor_like["pitch"].device, dtype=torch.long
            )
            for attr in ATTR_NAMES
        }


class MultiAttributeCrossEntropyLoss(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], ignore_pad: bool = False) -> None:
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.ignore_pad = ignore_pad

    def forward(self, logits: Dict[str, Tensor], target: Dict[str, Tensor]) -> Tensor:
        total = torch.zeros((), device=next(iter(logits.values())).device)
        active_mask = target["pitch"].reshape(-1) > 0

        for attr in ATTR_NAMES:
            attr_logits = logits[attr].reshape(-1, self.vocab_sizes[attr])
            attr_target = target[attr].reshape(-1)
            loss_values = F.cross_entropy(attr_logits, attr_target, reduction="none")

            if attr == "pitch":
                if self.ignore_pad:
                    valid = attr_target != 0
                    if valid.any():
                        loss = loss_values[valid].mean()
                    else:
                        loss = torch.zeros((), device=loss_values.device)
                else:
                    # Pitch learns PAD/non-PAD occupancy on all slots.
                    loss = loss_values.mean()
            else:
                # Non-pitch attributes are meaningful only on active (pitch>0) slots.
                valid = active_mask
                if self.ignore_pad:
                    valid = valid & (attr_target != 0)
                if valid.any():
                    loss = loss_values[valid].mean()
                else:
                    loss = torch.zeros((), device=loss_values.device)

            total = total + loss

        return total


def _get_scheduler_config_for_attribute(
    flow_cfg, attr: str
) -> Tuple[str, float, float]:
    default_schedule = str(getattr(flow_cfg, "kappa_schedule", "power"))
    if "kappa" not in flow_cfg or attr not in flow_cfg.kappa:
        return default_schedule, 1.0, 1.0

    attr_cfg = flow_cfg.kappa[attr]
    schedule = str(getattr(attr_cfg, "schedule", default_schedule))
    exponent = float(getattr(attr_cfg, "exponent", 1.0))
    scale = float(getattr(attr_cfg, "scale", 1.0))
    return schedule, exponent, scale


def get_path(flow_cfg) -> BaseDiscretePath:
    path_type = str(getattr(flow_cfg, "path_type", "mixture"))

    if path_type == "mixture":
        schedulers = {}
        for attr in ATTR_NAMES:
            schedule, exponent, scale = _get_scheduler_config_for_attribute(
                flow_cfg, attr=attr
            )
            schedulers[attr] = GenericAttributeScheduler(
                schedule=schedule, exponent=exponent, scale=scale
            )
        return MixtureDiscreteNoteTuplePath(schedulers=schedulers)

    if path_type in {"general_discrete", "custom"}:
        return FutureGeneralDiscretePath()

    raise ValueError(f"Unsupported path_type: {path_type}")


def get_source_distribution(
    source_distribution: str,
    vocab_sizes: Dict[str, int],
    include_pad: bool,
    pitch_pad_prob: float | None = None,
) -> SourceDistribution:
    if source_distribution == "uniform":
        return UniformSourceDistribution(
            vocab_sizes=vocab_sizes, include_pad=include_pad
        )

    if source_distribution == "pad_heavy":
        if pitch_pad_prob is None:
            raise ValueError(
                "source_distribution='pad_heavy' requires flow.source_pitch_pad_prob."
            )
        return PadHeavyGatedSourceDistribution(
            vocab_sizes=vocab_sizes,
            pitch_pad_prob=float(pitch_pad_prob),
            include_pad=include_pad,
        )

    if source_distribution == "masked":
        return MaskedSourceDistribution()

    raise ValueError(
        f"Unsupported source distribution: {source_distribution}. "
        "Supported: uniform, pad_heavy, masked."
    )


def get_loss_function(
    loss_function: str,
    vocab_sizes: Dict[str, int],
    path: BaseDiscretePath | None = None,
    parameterization: str = "posterior",
    use_edit_flow: bool = False,
    edit_flow_cfg=None,
    velocity_cfg=None,
) -> nn.Module:
    if loss_function != "cross_entropy":
        raise ValueError(
            f"Only cross_entropy loss is currently supported. Got {loss_function}."
        )

    if use_edit_flow:
        return EditFlowLoss(vocab_sizes=vocab_sizes, cfg=edit_flow_cfg)

    if parameterization == "velocity":
        if not isinstance(path, MixtureDiscreteNoteTuplePath):
            raise ValueError(
                "Velocity parameterization currently requires path_type='mixture'."
            )
        loss_type = str(getattr(velocity_cfg, "loss_type", "bregman"))
        eps = float(getattr(velocity_cfg, "eps", 1e-8))
        return VelocityMatchingLoss(
            vocab_sizes=vocab_sizes,
            path=path,
            loss_type=loss_type,
            eps=eps,
        )

    return MultiAttributeCrossEntropyLoss(vocab_sizes=vocab_sizes)
