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

    def rho(self, attr: str, t: Tensor, h: float) -> Tensor:
        kappa_t = self.schedulers[attr].kappa(t=t)
        dkappa_t = self.schedulers[attr].dkappa(t=t)

        denom = torch.clamp(1.0 - kappa_t, min=1e-8)
        return h * dkappa_t / denom

    def max_rho(self, t: Tensor, h: float) -> Tensor:
        max_rho = torch.zeros_like(t)
        for attr in ATTR_NAMES:
            max_rho = torch.maximum(max_rho, self.rho(attr=attr, t=t, h=h))
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


class MultiAttributeCrossEntropyLoss(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], ignore_pad: bool = False) -> None:
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.ignore_pad = ignore_pad

    def forward(self, logits: Dict[str, Tensor], target: Dict[str, Tensor]) -> Tensor:
        total = torch.zeros((), device=next(iter(logits.values())).device)

        for attr in ATTR_NAMES:
            attr_logits = logits[attr].reshape(-1, self.vocab_sizes[attr])
            attr_target = target[attr].reshape(-1)
            loss_values = F.cross_entropy(attr_logits, attr_target, reduction="none")

            if self.ignore_pad:
                valid = (attr_target != 0).float()
                denom = valid.sum().clamp_min(1.0)
                loss = (loss_values * valid).sum() / denom
            else:
                loss = loss_values.mean()

            total = total + loss

        return total


def _get_scheduler_config_for_attribute(flow_cfg, attr: str) -> Tuple[float, float]:
    if "kappa" not in flow_cfg or attr not in flow_cfg.kappa:
        return 1.0, 1.0

    attr_cfg = flow_cfg.kappa[attr]
    exponent = float(getattr(attr_cfg, "exponent", 1.0))
    scale = float(getattr(attr_cfg, "scale", 1.0))
    return exponent, scale


def get_path(flow_cfg) -> BaseDiscretePath:
    path_type = str(getattr(flow_cfg, "path_type", "mixture"))

    if path_type == "mixture":
        schedulers = {}
        for attr in ATTR_NAMES:
            exponent, scale = _get_scheduler_config_for_attribute(flow_cfg, attr=attr)
            schedulers[attr] = PolynomialAttributeScheduler(
                exponent=exponent, scale=scale
            )
        return MixtureDiscreteNoteTuplePath(schedulers=schedulers)

    if path_type in {"general_discrete", "custom"}:
        return FutureGeneralDiscretePath()

    raise ValueError(f"Unsupported path_type: {path_type}")


def get_source_distribution(
    source_distribution: str,
    vocab_sizes: Dict[str, int],
    include_pad: bool,
) -> SourceDistribution:
    if source_distribution != "uniform":
        raise ValueError(
            f"Only uniform source distribution is currently supported. Got {source_distribution}."
        )

    return UniformSourceDistribution(vocab_sizes=vocab_sizes, include_pad=include_pad)


def get_loss_function(
    loss_function: str,
    vocab_sizes: Dict[str, int],
    use_edit_flow: bool = False,
    edit_flow_cfg=None,
) -> nn.Module:
    if loss_function != "cross_entropy":
        raise ValueError(
            f"Only cross_entropy loss is currently supported. Got {loss_function}."
        )

    if use_edit_flow:
        return EditFlowLoss(vocab_sizes=vocab_sizes, cfg=edit_flow_cfg)

    return MultiAttributeCrossEntropyLoss(vocab_sizes=vocab_sizes)
