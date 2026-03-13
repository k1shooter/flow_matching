from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from data.note_tuple import ATTR_NAMES
from torch import Tensor

NON_PITCH_ATTRS = [attr for attr in ATTR_NAMES if attr != "pitch"]


def _masked_token_distribution(
    logits: Tensor,
    current_tokens: Tensor,
    disallow_zero_token: bool = False,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor]:
    """Build off-diagonal token distribution and valid-row indicator.

    Args:
        logits (Tensor): Unnormalized transition logits with shape ``(B, L, V)``.
        current_tokens (Tensor): Current categorical state with shape ``(B, L)``.
        disallow_zero_token (bool): If ``True``, token ``0`` is masked out.
        eps (float): Numerical stability constant for normalization checks.

    Returns:
        Tuple[Tensor, Tensor]: A tuple ``(probs, valid_rows)`` where ``probs`` has
            shape ``(B, L, V)`` and ``valid_rows`` has shape ``(B, L)``.
    """
    scaled_logits = logits.float()
    min_value = torch.finfo(scaled_logits.dtype).min
    vocab_size = int(scaled_logits.shape[-1])

    one_hot_current = F.one_hot(current_tokens, num_classes=vocab_size).bool()
    allow_mask = ~one_hot_current
    if disallow_zero_token and vocab_size > 1:
        zero_block = torch.zeros_like(allow_mask)
        zero_block[..., 0] = True
        allow_mask = allow_mask & (~zero_block)

    masked_logits = scaled_logits.masked_fill(~allow_mask, min_value)
    probs = torch.softmax(masked_logits, dim=-1)
    probs = probs * allow_mask.float()

    denom = probs.sum(dim=-1, keepdim=True)
    valid_rows = denom.squeeze(-1) > eps
    probs = torch.where(
        valid_rows.unsqueeze(-1),
        probs / denom.clamp_min(eps),
        torch.zeros_like(probs),
    )

    if bool(valid_rows.any()):
        row_sum = probs.sum(dim=-1)[valid_rows]
        if row_sum.numel() > 0:
            assert torch.all(torch.isfinite(row_sum)), "Non-finite transition probs"
            assert torch.allclose(
                row_sum,
                torch.ones_like(row_sum),
                atol=1e-4,
            ), "Transition probabilities must sum to 1."

    return probs, valid_rows


def _uniform_offdiag_distribution(
    current_tokens: Tensor,
    vocab_size: int,
    disallow_zero_token: bool = False,
    eps: float = 1e-12,
) -> Tensor:
    probs = torch.ones(
        (*current_tokens.shape, vocab_size),
        device=current_tokens.device,
        dtype=torch.float32,
    )
    one_hot_current = F.one_hot(current_tokens, num_classes=vocab_size).float()
    probs = probs * (1.0 - one_hot_current)
    if disallow_zero_token and vocab_size > 1:
        zero_mask = torch.ones_like(probs)
        zero_mask[..., 0] = 0.0
        probs = probs * zero_mask
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return probs / denom


def rates_from_velocity_parameterization(
    token_logits: Tensor,
    log_lambda: Tensor,
    current_tokens: Tensor,
    disallow_zero_token: bool = False,
    eps: float = 1e-12,
    corrector_weight: float = 0.0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert velocity head outputs to CTMC rates.

    The model predicts:
      - `log_lambda` (B, L), mapped to `lambda = softplus(log_lambda) >= 0`
      - `token_logits` (B, L, V), mapped to `pi(v)` over v != current token

    Rates are:
      r(v) = lambda * pi(v), for v != current token.
    """
    lambda_values = F.softplus(log_lambda.float())
    assert torch.all(lambda_values >= 0), "Hazards must be non-negative."
    assert torch.isfinite(lambda_values).all(), "Hazards contain NaN/Inf."

    probs, valid_rows = _masked_token_distribution(
        logits=token_logits,
        current_tokens=current_tokens,
        disallow_zero_token=disallow_zero_token,
        eps=eps,
    )

    weight = float(max(0.0, min(1.0, corrector_weight)))
    if weight > 0.0:
        uniform = _uniform_offdiag_distribution(
            current_tokens=current_tokens,
            vocab_size=token_logits.shape[-1],
            disallow_zero_token=disallow_zero_token,
            eps=eps,
        )
        probs = (1.0 - weight) * probs + weight * uniform
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

    lambda_values = torch.where(
        valid_rows, lambda_values, torch.zeros_like(lambda_values)
    )
    rates = lambda_values.unsqueeze(-1) * probs
    current_one_hot = F.one_hot(
        current_tokens, num_classes=token_logits.shape[-1]
    ).float()
    rates = rates * (1.0 - current_one_hot)

    assert torch.isfinite(rates).all(), "Rates contain NaN/Inf."
    return rates, lambda_values, probs


def ctmc_step_attribute(
    current_tokens: Tensor,
    token_logits: Tensor,
    log_lambda: Tensor,
    h: float,
    disallow_zero_token: bool = False,
    active_mask: Tensor | None = None,
    eps: float = 1e-12,
    corrector_weight: float = 0.0,
) -> Tuple[Tensor, Dict[str, float], Tensor]:
    """Always-valid first-order CTMC step for one attribute.

    For hazard `lambda` and off-diagonal distribution `pi(v)`:
      - stay prob: p_stay = exp(-h * lambda)
      - change prob: 1 - p_stay
      - conditioned on change: sample `v ~ pi`
    """
    rates, lambda_values, probs = rates_from_velocity_parameterization(
        token_logits=token_logits,
        log_lambda=log_lambda,
        current_tokens=current_tokens,
        disallow_zero_token=disallow_zero_token,
        eps=eps,
        corrector_weight=corrector_weight,
    )
    _ = rates

    p_stay = torch.exp(-float(h) * lambda_values)
    p_stay = torch.clamp(p_stay, min=0.0, max=1.0)

    random_draw = torch.rand_like(p_stay)
    change_mask = random_draw >= p_stay
    if active_mask is not None:
        change_mask = change_mask & active_mask

    sampled = torch.multinomial(
        probs.reshape(-1, probs.shape[-1]),
        num_samples=1,
    ).reshape_as(current_tokens)
    next_tokens = torch.where(change_mask, sampled, current_tokens)

    lambda_flat = lambda_values.reshape(-1)
    max_hist = float(lambda_flat.max().item()) if lambda_flat.numel() else 1.0
    hist = torch.histc(
        lambda_flat,
        bins=10,
        min=0.0,
        max=max(1e-6, max_hist),
    )
    stats = {
        "changed_frac": float(change_mask.float().mean().item()),
        "lambda_mean": float(lambda_values.mean().item()),
        "lambda_max": float(lambda_values.max().item()),
        "lambda_hist": ",".join(f"{float(v):.2f}" for v in hist.tolist()),
    }
    return next_tokens, stats, lambda_values


def _apply_pitch_gating_(x_t: Dict[str, Tensor]) -> None:
    inactive = x_t["pitch"] == 0
    for attr in NON_PITCH_ATTRS:
        x_t[attr][inactive] = 0


def ctmc_step_note_tuple(
    x_t: Dict[str, Tensor],
    token_logits: Dict[str, Tensor],
    log_lambda: Dict[str, Tensor],
    h: float,
    eps: float = 1e-12,
    corrector_weight: float = 0.0,
) -> Tuple[Dict[str, Tensor], Dict[str, float]]:
    next_state = {attr: x_t[attr].clone() for attr in ATTR_NAMES}
    stats: Dict[str, float] = {}

    next_pitch, pitch_stats, _ = ctmc_step_attribute(
        current_tokens=next_state["pitch"],
        token_logits=token_logits["pitch"],
        log_lambda=log_lambda["pitch"],
        h=h,
        disallow_zero_token=False,
        eps=eps,
        corrector_weight=corrector_weight,
    )
    next_state["pitch"] = next_pitch
    stats.update({f"pitch_{k}": v for k, v in pitch_stats.items()})

    _apply_pitch_gating_(next_state)
    active = next_state["pitch"] > 0

    for attr in NON_PITCH_ATTRS:
        next_attr, attr_stats, _ = ctmc_step_attribute(
            current_tokens=next_state[attr],
            token_logits=token_logits[attr],
            log_lambda=log_lambda[attr],
            h=h,
            disallow_zero_token=True,
            active_mask=active,
            eps=eps,
            corrector_weight=corrector_weight,
        )
        next_state[attr] = next_attr
        next_state[attr][~active] = 0
        stats.update({f"{attr}_{k}": v for k, v in attr_stats.items()})

    _apply_pitch_gating_(next_state)
    return next_state, stats
