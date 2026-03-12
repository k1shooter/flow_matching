from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from data.note_tuple import (
    ATTR_NAMES,
    midi_file_to_segment_tuples,
    note_tuple_batch_to_pretty_midi,
    NoteTupleMetadata,
)
from utils.rendering import maybe_render_midi_to_wav as render_midi_to_wav


SUPPORTED_MIDI_SUFFIXES = (".mid", ".midi")


@dataclass
class SegmentRecord:
    song_id: str
    segment_id: str
    note_tuple: Dict[str, torch.Tensor]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def scan_midi_files(root: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        files = [
            p
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_MIDI_SUFFIXES
        ]
    else:
        files = [
            p
            for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_MIDI_SUFFIXES
        ]

    return sorted(files)


def midi_dir_to_segments(
    midi_dir: Path,
    metadata: NoteTupleMetadata,
    segment_stride_bars: int,
    recursive: bool = True,
) -> List[SegmentRecord]:
    midi_files = scan_midi_files(midi_dir, recursive=recursive)
    segments: List[SegmentRecord] = []

    for midi_file in midi_files:
        song_id = midi_file.stem
        tuple_segments = midi_file_to_segment_tuples(
            midi_path=midi_file,
            metadata=metadata,
            segment_stride_bars=segment_stride_bars,
        )
        for i, seg in enumerate(tuple_segments):
            segments.append(
                SegmentRecord(
                    song_id=song_id,
                    segment_id=f"{song_id}_seg_{i:04d}",
                    note_tuple=seg,
                )
            )

    return segments


def sample_dict_to_midi(
    sample: Dict[str, torch.Tensor],
    metadata: NoteTupleMetadata,
    track_programs: Sequence[int],
):
    return note_tuple_batch_to_pretty_midi(
        sample=sample,
        metadata=metadata,
        track_programs=track_programs,
    )


def generate_samples_to_midi_dir(
    ckpt_path: str,
    out_dir: Path,
    seed: int,
    num_samples: int,
    batch_size: int,
    device: torch.device,
) -> List[Path]:
    from data.data import get_note_tuple_metadata, get_vocab_sizes
    from logic import flow, generate
    from logic.flow import MixtureDiscreteNoteTuplePath
    from utils import checkpointing

    set_global_seed(seed)

    cfg = checkpointing.load_cfg_from_path(ckpt_path)
    model = checkpointing.load_model_from_path(ckpt_path, device=device)
    model.eval()

    metadata = get_note_tuple_metadata(config=cfg)
    vocab_sizes = get_vocab_sizes(config=cfg)

    path = flow.get_path(flow_cfg=cfg.flow)
    source_distribution = flow.get_source_distribution(
        source_distribution=cfg.flow.source_distribution,
        vocab_sizes=vocab_sizes,
        include_pad=cfg.data.source_include_pad,
    )

    out_dir = ensure_dir(out_dir)
    saved_paths: List[Path] = []
    remaining = int(num_samples)
    saved = 0

    use_edit_flow = bool(getattr(cfg.flow, "use_edit_flow", False))

    while remaining > 0:
        current_bs = min(batch_size, remaining)

        if use_edit_flow:
            samples = generate.generate_samples_edit(
                model=model,
                vocab_sizes=vocab_sizes,
                sample_batch_size=current_bs,
                sequence_length=cfg.data.max_notes_per_segment,
                sampling_steps=int(cfg.flow.sampling_steps),
                device=device,
                sampler=str(cfg.flow.edit_flow.sampler),
                tau_leaping_step_size=float(cfg.flow.edit_flow.tau_leaping_step_size),
                max_events_per_step=int(cfg.flow.edit_flow.max_events_per_step),
                time_epsilon=float(cfg.flow.time_epsilon),
                temperature=float(cfg.flow.temperature),
                rate_scale=float(cfg.flow.edit_flow.rate_scale),
                init_mode=str(cfg.flow.edit_flow.init_mode),
            )
        else:
            if not isinstance(path, MixtureDiscreteNoteTuplePath):
                raise RuntimeError(
                    "Expected mixture path for non-edit-flow generation."
                )
            samples = generate.generate_samples(
                model=model,
                path=path,
                source_distribution=source_distribution,
                sample_batch_size=current_bs,
                sequence_length=cfg.data.max_notes_per_segment,
                sampling_steps=int(cfg.flow.sampling_steps),
                device=device,
                time_epsilon=float(cfg.flow.time_epsilon),
                temperature=float(cfg.flow.temperature),
                rho_cap=float(cfg.flow.rho_cap),
                final_full_resample=bool(cfg.flow.final_resample),
            )

        for i in range(current_bs):
            sample = {attr: samples[attr][i].detach().cpu() for attr in ATTR_NAMES}
            midi = sample_dict_to_midi(
                sample=sample,
                metadata=metadata,
                track_programs=cfg.data.track_programs,
            )
            file_path = out_dir / f"gen_{saved:05d}.mid"
            midi.write(str(file_path))
            saved_paths.append(file_path)
            saved += 1

        remaining -= current_bs

    return saved_paths


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_markdown_table(
    path: Path, headers: Sequence[str], rows: Sequence[Sequence[object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def maybe_render_midi_to_wav(
    midi_path: Path, wav_path: Path, soundfont_path: str | None
) -> bool:
    return render_midi_to_wav(
        midi_path=midi_path,
        wav_path=wav_path,
        soundfont_path=soundfont_path,
        sample_rate=44_100,
    )
