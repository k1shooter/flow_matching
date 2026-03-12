from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from data.note_tuple import ATTR_NAMES, NoteTupleMetadata, quantize_velocity_bin


def note_tuple_to_pianoroll(
    note_tuple: Dict[str, torch.Tensor],
    metadata: NoteTupleMetadata,
    value_mode: str = "binary",
) -> np.ndarray:
    total_steps = metadata.segment_bars * metadata.steps_per_bar
    roll = np.zeros((total_steps, 128), dtype=np.float32)

    for i in range(note_tuple["pitch"].shape[0]):
        pitch = int(note_tuple["pitch"][i].item())
        bar = int(note_tuple["bar"][i].item())
        pos = int(note_tuple["pos"][i].item())
        dur = int(note_tuple["dur"][i].item())
        vel = int(note_tuple["vel"][i].item())

        if pitch <= 0 or bar <= 0 or pos <= 0 or dur <= 0:
            continue

        start = (bar - 1) * metadata.steps_per_bar + (pos - 1)
        end = min(start + dur, total_steps)
        if start < 0 or start >= end:
            continue

        pitch_idx = max(0, min(127, pitch - 1))
        if value_mode == "binary":
            roll[start:end, pitch_idx] = 1.0
        else:
            roll[start:end, pitch_idx] = max(roll[start:end, pitch_idx].max(), vel)

    return roll


def pianoroll_to_note_tuple(
    pianoroll: np.ndarray,
    metadata: NoteTupleMetadata,
    max_notes: int,
    default_track: int = 1,
) -> Dict[str, torch.Tensor]:
    total_steps, n_pitches = pianoroll.shape
    if n_pitches != 128:
        raise ValueError(f"Expected pianoroll with 128 pitches, got {n_pitches}")

    notes = []

    for pitch in range(128):
        active = pianoroll[:, pitch] > 0
        start = None
        for t in range(total_steps):
            if active[t] and start is None:
                start = t
            if (not active[t] or t == total_steps - 1) and start is not None:
                end = t if not active[t] else t + 1
                dur = max(1, end - start)
                bar = start // metadata.steps_per_bar
                pos = start % metadata.steps_per_bar
                if bar < metadata.segment_bars:
                    vel_value = float(np.max(pianoroll[start:end, pitch]))
                    if vel_value <= 0:
                        vel_token = 1
                    elif vel_value <= 1.0:
                        vel_token = quantize_velocity_bin(
                            velocity=max(1, int(round(vel_value * 127.0))),
                            vel_bins=metadata.vel_bins,
                        )
                    else:
                        vel_token = quantize_velocity_bin(
                            velocity=max(1, int(round(vel_value))),
                            vel_bins=metadata.vel_bins,
                        )

                    notes.append(
                        (
                            default_track,
                            bar + 1,
                            pos + 1,
                            pitch + 1,
                            min(dur, metadata.dur_max_steps),
                            vel_token,
                        )
                    )
                start = None

    notes.sort(key=lambda x: (x[1], x[2], x[0], x[3], x[4], x[5]))
    notes = notes[:max_notes]

    if len(notes) < max_notes:
        notes.extend([(0, 0, 0, 0, 0, 0)] * (max_notes - len(notes)))

    arr = np.asarray(notes, dtype=np.int64)
    sample = {
        "track": torch.from_numpy(arr[:, 0]),
        "bar": torch.from_numpy(arr[:, 1]),
        "pos": torch.from_numpy(arr[:, 2]),
        "pitch": torch.from_numpy(arr[:, 3]),
        "dur": torch.from_numpy(arr[:, 4]),
        "vel": torch.from_numpy(arr[:, 5]),
    }
    sample["attention_mask"] = (sample["pitch"] != 0).long()
    return sample


def split_note_tuple_by_measures(
    note_tuple: Dict[str, torch.Tensor],
    metadata: NoteTupleMetadata,
    measure_bars: int,
) -> list[Dict[str, torch.Tensor]]:
    if measure_bars <= 0:
        raise ValueError("measure_bars must be positive")

    out = []
    n_windows = metadata.segment_bars // measure_bars

    for window_idx in range(n_windows):
        start_bar = window_idx * measure_bars
        end_bar = start_bar + measure_bars

        tokens = []
        for i in range(note_tuple["pitch"].shape[0]):
            bar = int(note_tuple["bar"][i].item())
            if bar <= 0:
                continue
            zero_bar = bar - 1
            if start_bar <= zero_bar < end_bar:
                token = {attr: int(note_tuple[attr][i].item()) for attr in ATTR_NAMES}
                token["bar"] = (zero_bar - start_bar) + 1
                tokens.append(token)

        tokens.sort(key=lambda x: (x["bar"], x["pos"], x["track"], x["pitch"]))
        max_notes = note_tuple["pitch"].shape[0]
        if len(tokens) > max_notes:
            tokens = tokens[:max_notes]

        arr = np.zeros((max_notes, len(ATTR_NAMES)), dtype=np.int64)
        for i, tok in enumerate(tokens):
            for j, attr in enumerate(ATTR_NAMES):
                arr[i, j] = tok[attr]

        sample = {
            attr: torch.from_numpy(arr[:, j]) for j, attr in enumerate(ATTR_NAMES)
        }
        sample["attention_mask"] = (sample["pitch"] != 0).long()
        out.append(sample)

    return out
