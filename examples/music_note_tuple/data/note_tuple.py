# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import pretty_midi as pretty_midi_types

ATTR_NAMES: Tuple[str, ...] = ("track", "bar", "pos", "pitch", "dur", "vel")


@dataclass(frozen=True)
class NoteTupleMetadata:
    beats_per_bar: int
    steps_per_beat: int
    segment_bars: int
    dur_max_steps: int
    vel_bins: int
    num_tracks: int
    max_notes_per_segment: int
    default_tempo: float = 120.0

    @property
    def steps_per_bar(self) -> int:
        return self.beats_per_bar * self.steps_per_beat

    @property
    def step_duration_seconds(self) -> float:
        return 60.0 / self.default_tempo / self.steps_per_beat


def get_note_tuple_vocab_sizes(metadata: NoteTupleMetadata) -> Dict[str, int]:
    return {
        "track": metadata.num_tracks + 1,
        "bar": metadata.segment_bars + 1,
        "pos": metadata.steps_per_bar + 1,
        "pitch": 129,
        "dur": metadata.dur_max_steps + 1,
        "vel": metadata.vel_bins + 1,
    }


def quantize_time_to_bar_pos(
    time_seconds: float,
    step_duration_seconds: float,
    steps_per_bar: int,
    segment_bars: int,
) -> Tuple[int, int] | None:
    if time_seconds < 0:
        return None

    total_steps = int(np.round(time_seconds / step_duration_seconds))
    bar = total_steps // steps_per_bar
    pos = total_steps % steps_per_bar

    if bar < 0 or bar >= segment_bars:
        return None

    return bar, pos


def dequantize_bar_pos_to_time(
    bar: int,
    pos: int,
    step_duration_seconds: float,
    steps_per_bar: int,
) -> float:
    total_steps = bar * steps_per_bar + pos
    return float(total_steps * step_duration_seconds)


def quantize_duration_steps(
    duration_seconds: float,
    step_duration_seconds: float,
    dur_max_steps: int,
) -> int:
    steps = int(np.round(duration_seconds / step_duration_seconds))
    return int(np.clip(steps, 1, dur_max_steps))


def quantize_velocity_bin(velocity: int, vel_bins: int) -> int:
    velocity = int(np.clip(velocity, 1, 127))
    ratio = velocity / 128.0
    vel_bin = int(np.floor(ratio * vel_bins)) + 1
    return int(np.clip(vel_bin, 1, vel_bins))


def velocity_bin_to_midi_velocity(velocity_bin: int, vel_bins: int) -> int:
    if velocity_bin <= 0:
        return 0

    bin_width = 127.0 / float(vel_bins)
    velocity = int(np.round((velocity_bin - 0.5) * bin_width))
    return int(np.clip(velocity, 1, 127))


def get_tempo_from_midi(midi_object, default_tempo: float) -> float:
    tempo_times, tempos = midi_object.get_tempo_changes()
    if tempos.size == 0:
        return float(default_tempo)

    return float(tempos[0])


def midi_file_to_segment_tuples(
    midi_path: Path,
    metadata: NoteTupleMetadata,
    segment_stride_bars: int,
) -> List[Dict[str, torch.Tensor]]:
    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError(
            "pretty_midi is required. Install it with `pip install pretty_midi`."
        ) from exc

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    tempo = get_tempo_from_midi(midi, default_tempo=metadata.default_tempo)

    beat_duration = 60.0 / tempo
    step_duration = beat_duration / metadata.steps_per_beat
    bar_duration = beat_duration * metadata.beats_per_bar

    segment_duration = metadata.segment_bars * bar_duration
    stride_duration = max(1, segment_stride_bars) * bar_duration

    end_time = max(0.0, midi.get_end_time())

    if segment_duration <= 0:
        return []

    n_segments = max(
        1, int(np.ceil(max(0.0, end_time - segment_duration) / stride_duration)) + 1
    )

    segments: List[Dict[str, torch.Tensor]] = []

    # Deterministic instrument->track assignment with overflow bucket.
    track_mapping: Dict[Tuple[int, bool], int] = {}

    for segment_idx in range(n_segments):
        segment_start = segment_idx * stride_duration
        segment_end = segment_start + segment_duration

        tuples: List[Tuple[int, int, int, int, int, int]] = []

        for instrument in midi.instruments:
            key = (instrument.program, bool(instrument.is_drum))
            if key not in track_mapping:
                if len(track_mapping) < metadata.num_tracks:
                    track_mapping[key] = len(track_mapping) + 1
                else:
                    track_mapping[key] = metadata.num_tracks

            track_id = track_mapping[key]

            for note in instrument.notes:
                if note.start < segment_start or note.start >= segment_end:
                    continue

                rel_start = note.start - segment_start
                bar_pos = quantize_time_to_bar_pos(
                    time_seconds=rel_start,
                    step_duration_seconds=step_duration,
                    steps_per_bar=metadata.steps_per_bar,
                    segment_bars=metadata.segment_bars,
                )
                if bar_pos is None:
                    continue

                bar, pos = bar_pos

                duration_steps = quantize_duration_steps(
                    duration_seconds=max(note.end - note.start, step_duration),
                    step_duration_seconds=step_duration,
                    dur_max_steps=metadata.dur_max_steps,
                )
                velocity_bin = quantize_velocity_bin(
                    velocity=note.velocity,
                    vel_bins=metadata.vel_bins,
                )

                tuples.append(
                    (
                        track_id,
                        bar + 1,
                        pos + 1,
                        int(np.clip(note.pitch, 0, 127)) + 1,
                        duration_steps,
                        velocity_bin,
                    )
                )

        tuples.sort(key=lambda x: (x[1], x[2], x[0], x[3], x[4], x[5]))
        tuples = tuples[: metadata.max_notes_per_segment]

        pad_len = metadata.max_notes_per_segment - len(tuples)
        if pad_len > 0:
            tuples.extend([(0, 0, 0, 0, 0, 0)] * pad_len)

        np_tuples = np.asarray(tuples, dtype=np.int64)
        sample = {
            "track": torch.from_numpy(np_tuples[:, 0]),
            "bar": torch.from_numpy(np_tuples[:, 1]),
            "pos": torch.from_numpy(np_tuples[:, 2]),
            "pitch": torch.from_numpy(np_tuples[:, 3]),
            "dur": torch.from_numpy(np_tuples[:, 4]),
            "vel": torch.from_numpy(np_tuples[:, 5]),
        }
        sample["attention_mask"] = (sample["pitch"] != 0).long()
        segments.append(sample)

    return segments


def note_tuple_batch_to_pretty_midi(
    sample: Dict[str, torch.Tensor],
    metadata: NoteTupleMetadata,
    track_programs: Sequence[int],
) -> "pretty_midi_types.PrettyMIDI":
    try:
        import pretty_midi
    except ImportError as exc:
        raise ImportError(
            "pretty_midi is required. Install it with `pip install pretty_midi`."
        ) from exc

    midi = pretty_midi.PrettyMIDI(initial_tempo=metadata.default_tempo)

    instruments = {}
    for track_id in range(1, metadata.num_tracks + 1):
        program = (
            int(track_programs[track_id - 1])
            if track_id - 1 < len(track_programs)
            else 0
        )
        instruments[track_id] = pretty_midi.Instrument(
            program=int(np.clip(program, 0, 127)),
            is_drum=False,
            name=f"track_{track_id}",
        )

    step_duration = metadata.step_duration_seconds

    track = sample["track"].cpu().numpy()
    bar = sample["bar"].cpu().numpy()
    pos = sample["pos"].cpu().numpy()
    pitch = sample["pitch"].cpu().numpy()
    dur = sample["dur"].cpu().numpy()
    vel = sample["vel"].cpu().numpy()

    for i in range(track.shape[0]):
        if (
            track[i] <= 0
            or pitch[i] <= 0
            or dur[i] <= 0
            or bar[i] <= 0
            or pos[i] <= 0
            or vel[i] <= 0
        ):
            continue

        track_id = int(np.clip(track[i], 1, metadata.num_tracks))
        note_pitch = int(np.clip(pitch[i] - 1, 0, 127))
        note_velocity = velocity_bin_to_midi_velocity(int(vel[i]), metadata.vel_bins)

        start = dequantize_bar_pos_to_time(
            bar=int(bar[i] - 1),
            pos=int(pos[i] - 1),
            step_duration_seconds=step_duration,
            steps_per_bar=metadata.steps_per_bar,
        )
        end = start + int(dur[i]) * step_duration

        if start < 0 or end <= start:
            continue

        instruments[track_id].notes.append(
            pretty_midi.Note(
                velocity=note_velocity,
                pitch=note_pitch,
                start=float(start),
                end=float(end),
            )
        )

    midi.instruments = [inst for inst in instruments.values() if inst.notes]
    return midi
