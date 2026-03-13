from __future__ import annotations

from typing import Dict, TYPE_CHECKING

import torch
import torch.nn.functional as F

from data.note_tuple import ATTR_NAMES

from samplers.ctmc_sampler import rates_from_velocity_parameterization
from torch import nn, Tensor

if TYPE_CHECKING:
    from logic.flow import MixtureDiscreteNoteTuplePath


def _scheduler_kappa(scheduler, t: Tensor) -> Tensor:
    if hasattr(scheduler, "kappa"):
        return scheduler.kappa(t=t)
    raise AttributeError("Scheduler does not define kappa(t).")


def _scheduler_kappa_dot(scheduler, t: Tensor) -> Tensor:
    if hasattr(scheduler, "kappa_dot"):
        return scheduler.kappa_dot(t=t)
    if hasattr(scheduler, "dkappa"):
        return scheduler.dkappa(t=t)
    raise AttributeError("Scheduler does not define kappa_dot/dkappa.")


class VelocityMatchingLoss(nn.Module):
    """Generator-matching loss for velocity-parameterized discrete FM.

    Given state `x_t`, model outputs:
      - `log_lambda` per position (hazard, via softplus)
      - `token_logits` (off-diagonal jump distribution after masking current token)

    For mixture-path supervision:
      if x_t == x_0:
          lambda* = kappa_dot(t)/(1-kappa(t))
          r*(v) is a delta at x_1
      else:
          r* = 0.
    """

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        path: "MixtureDiscreteNoteTuplePath",
        loss_type: str = "bregman",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.path = path
        self.loss_type = str(loss_type)
        self.eps = float(eps)
        self.last_stats: Dict[str, float] = {}

    def _target_lambda(self, attr: str, t: Tensor) -> Tensor:
        scheduler = self.path.schedulers[attr]
        k_t = _scheduler_kappa(scheduler, t=t)
        k_dot_t = _scheduler_kappa_dot(scheduler, t=t)
        return k_dot_t / torch.clamp(1.0 - k_t, min=1e-8)

    def forward(
        self,
        velocity_output: Dict[str, Dict[str, Tensor]],
        x_t: Dict[str, Tensor],
        x_0: Dict[str, Tensor],
        x_1: Dict[str, Tensor],
        t: Tensor,
    ) -> Tensor:
        token_logits = velocity_output["token_logits"]
        log_lambda = velocity_output["log_lambda"]

        total_loss = torch.zeros((), device=t.device)
        lambda_values_all = []
        change_target_all = []

        for attr in ATTR_NAMES:
            rates, lambda_values, probs = rates_from_velocity_parameterization(
                token_logits=token_logits[attr],
                log_lambda=log_lambda[attr],
                current_tokens=x_t[attr],
                disallow_zero_token=(attr != "pitch"),
                eps=self.eps,
                corrector_weight=0.0,
            )
            _ = probs

            target_lambda = self._target_lambda(attr=attr, t=t)[:, None].expand_as(
                x_t[attr]
            )
            source_mask = x_t[attr] == x_0[attr]
            jump_target = x_1[attr]
            offdiag_target = source_mask & (jump_target != x_t[attr])

            if self.loss_type == "factorized":
                masked_logits = token_logits[attr].float().clone()
                min_value = torch.finfo(masked_logits.dtype).min
                masked_logits.scatter_(-1, x_t[attr].unsqueeze(-1), min_value)
                if attr != "pitch" and masked_logits.shape[-1] > 1:
                    masked_logits[..., 0] = min_value

                if bool(offdiag_target.any()):
                    ce = F.cross_entropy(
                        masked_logits[offdiag_target],
                        jump_target[offdiag_target],
                    )
                    pred_log_lambda = torch.log(
                        lambda_values[offdiag_target] + self.eps
                    )
                    tgt_log_lambda = torch.log(target_lambda[offdiag_target] + self.eps)
                    reg = F.smooth_l1_loss(pred_log_lambda, tgt_log_lambda)
                else:
                    ce = torch.zeros((), device=t.device)
                    reg = torch.zeros((), device=t.device)

                inactive = ~offdiag_target
                inactive_penalty = (
                    (lambda_values[inactive] ** 2).mean()
                    if bool(inactive.any())
                    else torch.zeros((), device=t.device)
                )
                loss_attr = ce + reg + inactive_penalty
            else:
                target_rates = torch.zeros_like(rates)
                target_rates.scatter_(
                    -1,
                    jump_target.unsqueeze(-1),
                    target_lambda.unsqueeze(-1),
                )
                target_rates = target_rates * offdiag_target.unsqueeze(-1).float()
                target_rates.scatter_(-1, x_t[attr].unsqueeze(-1), 0.0)

                # Bregman-style generator matching:
                # L = sum_v [ r(v) - r*(v) log(r(v)+eps) ] over off-diagonal v.
                loss_attr = (
                    (rates - target_rates * torch.log(rates + self.eps))
                    .sum(dim=-1)
                    .mean()
                )

            total_loss = total_loss + loss_attr
            lambda_values_all.append(lambda_values.reshape(-1))
            change_target_all.append(offdiag_target.float().mean())

        lambda_flat = torch.cat(lambda_values_all, dim=0)
        max_hist = float(lambda_flat.max().item()) if lambda_flat.numel() else 1.0
        hist = torch.histc(lambda_flat, bins=10, min=0.0, max=max(1e-6, max_hist))
        self.last_stats = {
            "lambda_mean": float(lambda_flat.mean().item()),
            "lambda_max": float(lambda_flat.max().item()),
            "target_change_mean": float(torch.stack(change_target_all).mean().item()),
            "lambda_hist": ",".join(f"{float(v):.2f}" for v in hist.tolist()),
        }
        return total_loss
