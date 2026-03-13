# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from data.note_tuple import ATTR_NAMES
from torch import nn, Tensor

from .edit_ops import (
    delete_slot_,
    insert_slot_,
    sanitize_pitch_gating_,
    substitute_slot_,
)


@dataclass
class EditTargets:
    insert: Tensor
    delete: Tensor
    substitute: Tensor
    attr_update_mask: Tensor


def _active_mask(x: Dict[str, Tensor]) -> Tensor:
    return x["pitch"] != 0


def _equal_mask(x_a: Dict[str, Tensor], x_b: Dict[str, Tensor]) -> Tensor:
    equal = torch.ones_like(x_a["pitch"], dtype=torch.bool)
    for attr in ATTR_NAMES:
        equal = equal & (x_a[attr] == x_b[attr])
    return equal


def derive_edit_targets(x_t: Dict[str, Tensor], x_1: Dict[str, Tensor]) -> EditTargets:
    active_t = _active_mask(x_t)
    active_1 = _active_mask(x_1)
    equal = _equal_mask(x_t, x_1)

    delete = active_t & (~active_1)
    insert = (~active_t) & active_1
    substitute = active_t & active_1 & (~equal)
    attr_update_mask = insert | substitute

    return EditTargets(
        insert=insert.float(),
        delete=delete.float(),
        substitute=substitute.float(),
        attr_update_mask=attr_update_mask,
    )


class EditFlowLoss(nn.Module):
    """Loss for edit-based training with slot-aligned edit targets."""

    def __init__(self, vocab_sizes: Dict[str, int], cfg) -> None:
        super().__init__()
        self.vocab_sizes = vocab_sizes
        self.lambda_attr = float(getattr(cfg, "lambda_attr", 1.0))
        self.lambda_insert = float(getattr(cfg, "lambda_insert", 1.0))
        self.lambda_delete = float(getattr(cfg, "lambda_delete", 1.0))
        self.lambda_substitute = float(getattr(cfg, "lambda_substitute", 1.0))

    def _attribute_ce(
        self,
        logits: Dict[str, Tensor],
        target: Dict[str, Tensor],
        mask: Tensor,
    ) -> Tensor:
        if mask.sum() == 0:
            return torch.zeros((), device=next(iter(logits.values())).device)

        total = torch.zeros((), device=mask.device)
        mask_flat = mask.reshape(-1)
        denom = mask_flat.sum().clamp_min(1.0)

        for attr in ATTR_NAMES:
            attr_logits = logits[attr].reshape(-1, self.vocab_sizes[attr])
            attr_target = target[attr].reshape(-1)
            losses = F.cross_entropy(attr_logits, attr_target, reduction="none")
            total = total + (losses * mask_flat).sum() / denom

        return total

    def forward(
        self,
        logits: Dict[str, Tensor],
        edit_logits: Dict[str, Tensor],
        x_t: Dict[str, Tensor],
        target: Dict[str, Tensor],
    ) -> Tensor:
        targets = derive_edit_targets(x_t=x_t, x_1=target)

        insert_loss = F.binary_cross_entropy_with_logits(
            edit_logits["insert"], targets.insert
        )
        delete_loss = F.binary_cross_entropy_with_logits(
            edit_logits["delete"], targets.delete
        )
        substitute_loss = F.binary_cross_entropy_with_logits(
            edit_logits["substitute"], targets.substitute
        )
        attr_loss = self._attribute_ce(
            logits=logits,
            target=target,
            mask=targets.attr_update_mask,
        )

        return (
            self.lambda_attr * attr_loss
            + self.lambda_insert * insert_loss
            + self.lambda_delete * delete_loss
            + self.lambda_substitute * substitute_loss
        )


def _sample_index_from_logits(
    logits: Tensor,
    temperature: float,
    disallow_zero: bool,
) -> int:
    scaled = logits.float() / max(temperature, 1e-6)
    if disallow_zero and scaled.shape[-1] > 1:
        scaled = scaled.clone()
        scaled[0] = torch.finfo(scaled.dtype).min
    probs = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _sample_tuple_from_logits(
    attr_logits: Dict[str, Tensor],
    position_index: int,
    temperature: float,
    nonzero_only: bool = True,
) -> Dict[str, int]:
    token: Dict[str, int] = {}
    for attr in ATTR_NAMES:
        logits = attr_logits[attr][position_index]
        token[attr] = _sample_index_from_logits(
            logits=logits,
            temperature=temperature,
            disallow_zero=nonzero_only,
        )
    return token


def _sample_uniform_nonzero_tuple(vocab_sizes: Dict[str, int]) -> Dict[str, int]:
    token: Dict[str, int] = {}
    for attr in ATTR_NAMES:
        token[attr] = int(torch.randint(1, vocab_sizes[attr], (1,)).item())
    return token


def _new_empty_state(max_length: int, device: torch.device) -> Dict[str, Tensor]:
    return {
        attr: torch.zeros(max_length, dtype=torch.long, device=device)
        for attr in ATTR_NAMES
    }


def _collect_gillespie_events(
    state: Dict[str, Tensor],
    op_logits: Dict[str, Tensor],
    rate_scale: float,
) -> list[tuple[str, int, float]]:
    events: list[tuple[str, int, float]] = []
    active = state["pitch"] != 0

    insert_rates = torch.sigmoid(op_logits["insert"]) * rate_scale
    delete_rates = torch.sigmoid(op_logits["delete"]) * rate_scale
    substitute_rates = torch.sigmoid(op_logits["substitute"]) * rate_scale

    for pos in range(state["pitch"].shape[0]):
        if bool(active[pos].item()):
            delete_rate = float(delete_rates[pos].item())
            substitute_rate = float(substitute_rates[pos].item())
            if delete_rate > 1e-8:
                events.append(("delete", pos, delete_rate))
            if substitute_rate > 1e-8:
                events.append(("substitute", pos, substitute_rate))
        else:
            insert_rate = float(insert_rates[pos].item())
            if insert_rate > 1e-8:
                events.append(("insert", pos, insert_rate))

    return events


def _sample_event(events: Sequence[tuple[str, int, float]]) -> tuple[str, int, float]:
    rates = torch.tensor([event[2] for event in events], dtype=torch.float32)
    probs = rates / rates.sum()
    idx = int(torch.multinomial(probs, num_samples=1).item())
    return events[idx]


def _apply_tau_leaping_step(
    state: Dict[str, Tensor],
    attr_logits: Dict[str, Tensor],
    op_logits: Dict[str, Tensor],
    h: float,
    temperature: float,
    rate_scale: float,
    max_events_per_step: int,
) -> Dict[str, Tensor]:
    out = {attr: tensor.clone() for attr, tensor in state.items()}
    sanitize_pitch_gating_(out)

    active = out["pitch"] != 0

    insert_prob = torch.clamp(
        torch.sigmoid(op_logits["insert"]) * h * rate_scale, min=0.0, max=1.0
    )
    delete_prob = torch.clamp(
        torch.sigmoid(op_logits["delete"]) * h * rate_scale, min=0.0, max=1.0
    )
    substitute_prob = torch.clamp(
        torch.sigmoid(op_logits["substitute"]) * h * rate_scale, min=0.0, max=1.0
    )

    events: list[tuple[str, int]] = []
    length = out["pitch"].shape[0]
    for pos in range(length):
        if bool(active[pos].item()):
            if bool(
                (torch.rand((), device=out["pitch"].device) < delete_prob[pos]).item()
            ):
                events.append(("delete", pos))
            if bool(
                (
                    torch.rand((), device=out["pitch"].device) < substitute_prob[pos]
                ).item()
            ):
                events.append(("substitute", pos))
        else:
            if bool(
                (torch.rand((), device=out["pitch"].device) < insert_prob[pos]).item()
            ):
                events.append(("insert", pos))

    if len(events) > max_events_per_step:
        perm = torch.randperm(len(events), device=out["pitch"].device)[
            :max_events_per_step
        ]
        events = [events[int(i.item())] for i in perm]

    # Deterministic operation order for same-step collisions.
    for op_name in ("delete", "substitute", "insert"):
        for event_name, pos in events:
            if event_name != op_name:
                continue
            if op_name == "delete":
                if int(out["pitch"][pos].item()) > 0:
                    delete_slot_(out, pos)
            elif op_name == "substitute":
                if int(out["pitch"][pos].item()) > 0:
                    token = _sample_tuple_from_logits(
                        attr_logits=attr_logits,
                        position_index=pos,
                        temperature=temperature,
                        nonzero_only=True,
                    )
                    substitute_slot_(out, pos, token)
            else:
                if int(out["pitch"][pos].item()) == 0:
                    token = _sample_tuple_from_logits(
                        attr_logits=attr_logits,
                        position_index=pos,
                        temperature=temperature,
                        nonzero_only=True,
                    )
                    insert_slot_(out, pos, token)

    sanitize_pitch_gating_(out)
    return out


def _apply_gillespie_step(
    state: Dict[str, Tensor],
    attr_logits: Dict[str, Tensor],
    op_logits: Dict[str, Tensor],
    h: float,
    temperature: float,
    rate_scale: float,
    max_events_per_step: int,
) -> Dict[str, Tensor]:
    out = {attr: tensor.clone() for attr, tensor in state.items()}
    sanitize_pitch_gating_(out)

    local_t = 0.0
    n_events = 0

    while local_t < h and n_events < max_events_per_step:
        events = _collect_gillespie_events(
            state=out,
            op_logits=op_logits,
            rate_scale=rate_scale,
        )
        if len(events) == 0:
            break

        total_rate = sum(event[2] for event in events)
        if total_rate <= 1e-8:
            break

        dt = float(torch.distributions.Exponential(total_rate).sample().item())
        if local_t + dt > h:
            break

        local_t += dt
        op_type, pos, _ = _sample_event(events)

        if op_type == "delete":
            if int(out["pitch"][pos].item()) > 0:
                delete_slot_(out, pos)
        elif op_type == "substitute":
            if int(out["pitch"][pos].item()) > 0:
                token = _sample_tuple_from_logits(
                    attr_logits=attr_logits,
                    position_index=pos,
                    temperature=temperature,
                    nonzero_only=True,
                )
                substitute_slot_(out, pos, token)
        elif op_type == "insert":
            if int(out["pitch"][pos].item()) == 0:
                token = _sample_tuple_from_logits(
                    attr_logits=attr_logits,
                    position_index=pos,
                    temperature=temperature,
                    nonzero_only=True,
                )
                insert_slot_(out, pos, token)

        n_events += 1

    sanitize_pitch_gating_(out)
    return out


@torch.no_grad()
def generate_samples_edit_flow(
    model: nn.Module,
    vocab_sizes: Dict[str, int],
    sample_batch_size: int,
    max_length: int,
    sampling_steps: int,
    device: torch.device,
    sampler: str = "tau_leaping",
    tau_leaping_step_size: float | None = None,
    time_epsilon: float = 1e-3,
    temperature: float = 1.0,
    rate_scale: float = 1.0,
    max_events_per_step: int = 64,
    init_mode: str = "empty",
) -> Dict[str, Tensor]:
    if sampler not in {"tau_leaping", "gillespie"}:
        raise ValueError(f"Unsupported edit-flow sampler: {sampler}")

    x_t = {
        attr: torch.zeros(
            sample_batch_size,
            max_length,
            dtype=torch.long,
            device=device,
        )
        for attr in ATTR_NAMES
    }

    for b in range(sample_batch_size):
        if init_mode == "empty":
            continue
        if init_mode != "uniform":
            raise ValueError(f"Unsupported init_mode: {init_mode}")

        init_len = int(torch.randint(0, max_length + 1, (1,)).item())
        if init_len == 0:
            continue
        positions = torch.randperm(max_length, device=device)[:init_len]
        state = _new_empty_state(max_length=max_length, device=device)
        for pos in positions.tolist():
            token = _sample_uniform_nonzero_tuple(vocab_sizes=vocab_sizes)
            insert_slot_(state, int(pos), token)
        for attr in ATTR_NAMES:
            x_t[attr][b] = state[attr]

    t_final = 1.0 - time_epsilon
    if tau_leaping_step_size is not None and tau_leaping_step_size > 0:
        base_h = float(tau_leaping_step_size)
    else:
        base_h = t_final / max(1, sampling_steps)

    t = 0.0
    while t < t_final - 1e-12:
        h = min(base_h, t_final - t)
        t_tensor = torch.full((sample_batch_size,), t, device=device)
        attention_mask = (x_t["pitch"] != 0).long()

        try:
            attr_logits_batch, edit_logits_batch = model(
                x_t=x_t,
                time=t_tensor,
                attention_mask=attention_mask,
                return_edit_logits=True,
            )
        except TypeError:
            attr_logits_batch, edit_logits_batch = model(
                x_t=x_t,
                time=t_tensor,
                return_edit_logits=True,
            )

        updated_batch = {attr: x_t[attr].clone() for attr in ATTR_NAMES}

        for b in range(sample_batch_size):
            state = {attr: x_t[attr][b] for attr in ATTR_NAMES}
            attr_logits = {attr: attr_logits_batch[attr][b] for attr in ATTR_NAMES}
            op_logits = {
                "insert": edit_logits_batch["insert"][b],
                "delete": edit_logits_batch["delete"][b],
                "substitute": edit_logits_batch["substitute"][b],
            }

            if sampler == "tau_leaping":
                state_next = _apply_tau_leaping_step(
                    state=state,
                    attr_logits=attr_logits,
                    op_logits=op_logits,
                    h=h,
                    temperature=temperature,
                    rate_scale=rate_scale,
                    max_events_per_step=max_events_per_step,
                )
            else:
                state_next = _apply_gillespie_step(
                    state=state,
                    attr_logits=attr_logits,
                    op_logits=op_logits,
                    h=h,
                    temperature=temperature,
                    rate_scale=rate_scale,
                    max_events_per_step=max_events_per_step,
                )

            for attr in ATTR_NAMES:
                updated_batch[attr][b] = state_next[attr]

        x_t = updated_batch
        sanitize_pitch_gating_(x_t)
        t += h

    out = {attr: value.clone() for attr, value in x_t.items()}
    out["length"] = (out["pitch"] > 0).sum(dim=1)
    return out
