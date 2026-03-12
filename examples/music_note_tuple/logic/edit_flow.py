# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from data.note_tuple import ATTR_NAMES

from torch import nn, Tensor


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
    """Loss for edit-based training.

    - Operation loss: BCE over insert/delete/substitute event indicators.
    - Attribute loss: CE over factorized tuple attributes on insert/substitute sites.
    """

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
            # Keep gradient flow stable even when no edit op is selected.
            fallback = torch.zeros((), device=next(iter(logits.values())).device)
            for attr in ATTR_NAMES:
                attr_logits = logits[attr].reshape(-1, self.vocab_sizes[attr])
                attr_target = target[attr].reshape(-1)
                fallback = fallback + F.cross_entropy(
                    attr_logits, attr_target, reduction="mean"
                )
            return fallback

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

        total = (
            self.lambda_attr * attr_loss
            + self.lambda_insert * insert_loss
            + self.lambda_delete * delete_loss
            + self.lambda_substitute * substitute_loss
        )
        return total


def _sample_tuple_from_logits(
    attr_logits: Dict[str, Tensor],
    position_index: int,
    temperature: float,
) -> Dict[str, int]:
    token = {}
    for attr in ATTR_NAMES:
        logits = attr_logits[attr][position_index] / max(temperature, 1e-6)
        probs = torch.softmax(logits.float(), dim=-1)
        token[attr] = int(torch.multinomial(probs, num_samples=1).item())
    return token


def padded_to_variable_sequence(x: Dict[str, Tensor]) -> List[Dict[str, int]]:
    seq: List[Dict[str, int]] = []
    for i in range(x["pitch"].shape[0]):
        if int(x["pitch"][i].item()) == 0:
            continue
        seq.append({attr: int(x[attr][i].item()) for attr in ATTR_NAMES})
    return seq


def variable_sequence_to_padded(
    sequence: Sequence[Dict[str, int]],
    max_length: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    out = {
        attr: torch.zeros(max_length, dtype=torch.long, device=device)
        for attr in ATTR_NAMES
    }
    for i, token in enumerate(sequence[:max_length]):
        for attr in ATTR_NAMES:
            out[attr][i] = int(token[attr])
    return out


def batch_variable_sequences_to_padded(
    sequences: Sequence[Sequence[Dict[str, int]]],
    max_length: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    out = {
        attr: torch.zeros(len(sequences), max_length, dtype=torch.long, device=device)
        for attr in ATTR_NAMES
    }

    for b, seq in enumerate(sequences):
        for i, token in enumerate(seq[:max_length]):
            for attr in ATTR_NAMES:
                out[attr][b, i] = int(token[attr])

    return out


def _sanitize_sequence(
    sequence: List[Dict[str, int]], max_length: int
) -> List[Dict[str, int]]:
    clean = []
    for token in sequence:
        if int(token["pitch"]) == 0:
            continue
        clean.append(token)
        if len(clean) >= max_length:
            break
    return clean


def _apply_tau_leaping_step(
    sequence: List[Dict[str, int]],
    attr_logits: Dict[str, Tensor],
    op_logits: Dict[str, Tensor],
    h: float,
    max_length: int,
    temperature: float,
    rate_scale: float,
    max_events_per_step: int,
) -> List[Dict[str, int]]:
    seq = list(sequence)
    seq_len = len(seq)

    insert_prob = torch.clamp(
        torch.sigmoid(op_logits["insert"]) * h * rate_scale, 0.0, 1.0
    )
    delete_prob = torch.clamp(
        torch.sigmoid(op_logits["delete"]) * h * rate_scale, 0.0, 1.0
    )
    substitute_prob = torch.clamp(
        torch.sigmoid(op_logits["substitute"]) * h * rate_scale, 0.0, 1.0
    )

    delete_events = []
    for pos in range(min(seq_len, delete_prob.shape[0])):
        if torch.rand(()) < delete_prob[pos]:
            delete_events.append(pos)

    substitute_events = []
    for pos in range(min(seq_len, substitute_prob.shape[0])):
        if torch.rand(()) < substitute_prob[pos]:
            substitute_events.append(pos)

    insert_events = []
    for pos in range(min(seq_len + 1, insert_prob.shape[0])):
        if torch.rand(()) < insert_prob[pos]:
            insert_events.append(pos)

    events_budget = max_events_per_step
    if len(delete_events) > events_budget:
        delete_events = delete_events[:events_budget]
    events_budget -= len(delete_events)

    if len(substitute_events) > events_budget:
        substitute_events = substitute_events[:events_budget]
    events_budget -= len(substitute_events)

    if len(insert_events) > events_budget:
        insert_events = insert_events[:events_budget]

    for idx in sorted(delete_events, reverse=True):
        if 0 <= idx < len(seq):
            seq.pop(idx)

    for idx in substitute_events:
        if 0 <= idx < len(seq):
            new_token = _sample_tuple_from_logits(
                attr_logits=attr_logits,
                position_index=idx,
                temperature=temperature,
            )
            seq[idx] = new_token

    offset = 0
    for idx in sorted(insert_events):
        if len(seq) >= max_length:
            break
        new_token = _sample_tuple_from_logits(
            attr_logits=attr_logits,
            position_index=min(idx, attr_logits["pitch"].shape[0] - 1),
            temperature=temperature,
        )
        pos = min(idx + offset, len(seq))
        seq.insert(pos, new_token)
        offset += 1

    return _sanitize_sequence(seq, max_length=max_length)


def _collect_gillespie_events(
    sequence: List[Dict[str, int]],
    op_logits: Dict[str, Tensor],
    max_length: int,
    rate_scale: float,
) -> List[tuple[str, int, float]]:
    seq_len = len(sequence)
    events: List[tuple[str, int, float]] = []

    insert_rates = torch.sigmoid(op_logits["insert"]) * rate_scale
    delete_rates = torch.sigmoid(op_logits["delete"]) * rate_scale
    substitute_rates = torch.sigmoid(op_logits["substitute"]) * rate_scale

    for pos in range(min(seq_len + 1, insert_rates.shape[0])):
        if seq_len < max_length:
            rate = float(insert_rates[pos].item())
            if rate > 1e-8:
                events.append(("insert", pos, rate))

    for pos in range(min(seq_len, delete_rates.shape[0])):
        rate = float(delete_rates[pos].item())
        if rate > 1e-8:
            events.append(("delete", pos, rate))

    for pos in range(min(seq_len, substitute_rates.shape[0])):
        rate = float(substitute_rates[pos].item())
        if rate > 1e-8:
            events.append(("substitute", pos, rate))

    return events


def _sample_event(events: Sequence[tuple[str, int, float]]) -> tuple[str, int, float]:
    rates = torch.tensor([e[2] for e in events], dtype=torch.float32)
    probs = rates / rates.sum()
    idx = int(torch.multinomial(probs, num_samples=1).item())
    return events[idx]


def _apply_gillespie_step(
    sequence: List[Dict[str, int]],
    attr_logits: Dict[str, Tensor],
    op_logits: Dict[str, Tensor],
    h: float,
    max_length: int,
    temperature: float,
    rate_scale: float,
    max_events_per_step: int,
) -> List[Dict[str, int]]:
    seq = list(sequence)
    local_t = 0.0
    n_events = 0

    while local_t < h and n_events < max_events_per_step:
        events = _collect_gillespie_events(
            sequence=seq,
            op_logits=op_logits,
            max_length=max_length,
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
            if 0 <= pos < len(seq):
                seq.pop(pos)
        elif op_type == "substitute":
            if 0 <= pos < len(seq):
                seq[pos] = _sample_tuple_from_logits(
                    attr_logits=attr_logits,
                    position_index=pos,
                    temperature=temperature,
                )
        elif op_type == "insert":
            if len(seq) < max_length:
                insert_pos = min(pos, len(seq))
                seq.insert(
                    insert_pos,
                    _sample_tuple_from_logits(
                        attr_logits=attr_logits,
                        position_index=min(pos, attr_logits["pitch"].shape[0] - 1),
                        temperature=temperature,
                    ),
                )

        n_events += 1

    return _sanitize_sequence(seq, max_length=max_length)


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

    sequences: List[List[Dict[str, int]]] = []

    for _ in range(sample_batch_size):
        if init_mode == "empty":
            sequences.append([])
        elif init_mode == "uniform":
            init_len = int(torch.randint(0, max_length + 1, (1,)).item())
            seq = []
            for _ in range(init_len):
                token = {
                    attr: int(torch.randint(1, vocab_sizes[attr], (1,)).item())
                    for attr in ATTR_NAMES
                }
                seq.append(token)
            sequences.append(seq)
        else:
            raise ValueError(f"Unsupported init_mode: {init_mode}")

    t_final = 1.0 - time_epsilon
    if tau_leaping_step_size is not None and tau_leaping_step_size > 0:
        base_h = float(tau_leaping_step_size)
    else:
        base_h = t_final / max(1, sampling_steps)

    t = 0.0

    while t < t_final - 1e-12:
        h = min(base_h, t_final - t)

        padded = batch_variable_sequences_to_padded(
            sequences=sequences,
            max_length=max_length,
            device=device,
        )
        t_tensor = torch.full((sample_batch_size,), t, device=device)

        attr_logits_batch, edit_logits_batch = model(
            x_t=padded,
            time=t_tensor,
            return_edit_logits=True,
        )

        new_sequences = []
        for b in range(sample_batch_size):
            attr_logits = {attr: attr_logits_batch[attr][b] for attr in ATTR_NAMES}
            op_logits = {
                "insert": edit_logits_batch["insert"][b],
                "delete": edit_logits_batch["delete"][b],
                "substitute": edit_logits_batch["substitute"][b],
            }

            if sampler == "tau_leaping":
                updated = _apply_tau_leaping_step(
                    sequence=sequences[b],
                    attr_logits=attr_logits,
                    op_logits=op_logits,
                    h=h,
                    max_length=max_length,
                    temperature=temperature,
                    rate_scale=rate_scale,
                    max_events_per_step=max_events_per_step,
                )
            else:
                updated = _apply_gillespie_step(
                    sequence=sequences[b],
                    attr_logits=attr_logits,
                    op_logits=op_logits,
                    h=h,
                    max_length=max_length,
                    temperature=temperature,
                    rate_scale=rate_scale,
                    max_events_per_step=max_events_per_step,
                )

            new_sequences.append(updated)

        sequences = new_sequences
        t += h

    out = batch_variable_sequences_to_padded(
        sequences=sequences,
        max_length=max_length,
        device=device,
    )
    out["length"] = torch.tensor([len(seq) for seq in sequences], device=device)
    return out
