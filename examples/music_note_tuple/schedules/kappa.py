from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseKappaScheduler(ABC):
    @abstractmethod
    def kappa(self, t: Tensor) -> Tensor:
        ...

    @abstractmethod
    def kappa_dot(self, t: Tensor) -> Tensor:
        ...


class PowerKappaScheduler(BaseKappaScheduler):
    def __init__(self, exponent: float = 1.0, scale: float = 1.0) -> None:
        if exponent <= 0:
            raise ValueError(f"exponent must be > 0, got {exponent}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        self.exponent = float(exponent)
        self.scale = float(scale)

    def kappa(self, t: Tensor) -> Tensor:
        return torch.clamp(self.scale * torch.pow(t, self.exponent), 0.0, 1.0)

    def kappa_dot(self, t: Tensor) -> Tensor:
        t_safe = torch.clamp(t, min=1e-8)
        return torch.clamp(
            self.scale * self.exponent * torch.pow(t_safe, self.exponent - 1.0),
            min=0.0,
        )


class SmootherStepKappaScheduler(BaseKappaScheduler):
    """Polynomial scheduler with zero derivative at both endpoints.

    Uses the classic smootherstep polynomial:
        kappa(t) = 6 t^5 - 15 t^4 + 10 t^3
    """

    def kappa(self, t: Tensor) -> Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        return t**3 * (10.0 + t * (-15.0 + 6.0 * t))

    def kappa_dot(self, t: Tensor) -> Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        return torch.clamp(30.0 * t**2 * (1.0 - t) ** 2, min=0.0)


class CosineKappaScheduler(BaseKappaScheduler):
    """Cosine schedule kappa(t)=sin^2(pi t / 2)."""

    def kappa(self, t: Tensor) -> Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        return torch.sin(0.5 * math.pi * t) ** 2

    def kappa_dot(self, t: Tensor) -> Tensor:
        t = torch.clamp(t, 0.0, 1.0)
        return torch.clamp(0.5 * math.pi * torch.sin(math.pi * t), min=0.0)


def build_kappa_scheduler(
    schedule: str,
    exponent: float = 1.0,
    scale: float = 1.0,
) -> BaseKappaScheduler:
    name = str(schedule).lower()
    if name in {"power", "polynomial", "poly"}:
        return PowerKappaScheduler(exponent=exponent, scale=scale)
    if name in {"smoothstep", "smootherstep", "poly_endpoint"}:
        return SmootherStepKappaScheduler()
    if name in {"cosine", "cos"}:
        return CosineKappaScheduler()
    raise ValueError(f"Unsupported kappa schedule: {schedule}")
