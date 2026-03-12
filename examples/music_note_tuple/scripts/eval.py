# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.note_tuple import NoteTupleMetadata
from eval.common import (
    ensure_dir,
    generate_samples_to_midi_dir,
    midi_dir_to_segments,
    save_json,
    set_global_seed,
)
from eval.d3pia_pop909 import (
    analyze_d3pia_subjective_results,
    create_d3pia_subjective_package,
    evaluate_d3pia_objective,
    load_key_labels,
    write_d3pia_report,
)
from eval.wholesonggen_iclr24 import (
    analyze_wholesong_subjective_results,
    create_wholesong_subjective_package,
    evaluate_wholesong_objective,
    write_wholesong_report,
)


def _metadata_from_cfg_or_args(args: argparse.Namespace) -> NoteTupleMetadata:
    if args.ckpt is not None:
        from utils import checkpointing

        cfg = checkpointing.load_cfg_from_path(args.ckpt)
        return NoteTupleMetadata(
            beats_per_bar=int(cfg.data.beats_per_bar),
            steps_per_beat=int(cfg.data.steps_per_beat),
            segment_bars=int(cfg.data.segment_bars),
            dur_max_steps=int(cfg.data.dur_max_steps),
            vel_bins=int(cfg.data.vel_bins),
            num_tracks=int(cfg.data.num_tracks),
            max_notes_per_segment=int(cfg.data.max_notes_per_segment),
            default_tempo=float(cfg.data.default_tempo),
        )

    return NoteTupleMetadata(
        beats_per_bar=args.beats_per_bar,
        steps_per_beat=args.steps_per_beat,
        segment_bars=args.segment_bars,
        dur_max_steps=args.dur_max_steps,
        vel_bins=args.vel_bins,
        num_tracks=args.num_tracks,
        max_notes_per_segment=args.max_notes_per_segment,
        default_tempo=args.default_tempo,
    )


def _resolve_generated_midi_dir(
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> Path:
    if args.generated_dir is not None:
        return Path(args.generated_dir)

    if args.ckpt is None:
        raise ValueError("Either --generated_dir or --ckpt must be provided.")

    gen_dir = out_dir / "generated_midis"
    generate_samples_to_midi_dir(
        ckpt_path=args.ckpt,
        out_dir=gen_dir,
        seed=args.seed,
        num_samples=args.num_generate_samples,
        batch_size=args.generate_batch_size,
        device=device,
    )
    return gen_dir


def run_d3pia(args: argparse.Namespace, out_dir: Path, device: torch.device) -> dict:
    metadata = _metadata_from_cfg_or_args(args)
    generated_dir = _resolve_generated_midi_dir(
        args=args, out_dir=out_dir, device=device
    )

    notes = []
    if args.reference_dir is None:
        reference_dir = generated_dir
        notes.append(
            "reference_dir was not provided; generated set is used as reference (protocol comparability risk)."
        )
    else:
        reference_dir = Path(args.reference_dir)
    generated_segments = midi_dir_to_segments(
        midi_dir=generated_dir,
        metadata=metadata,
        segment_stride_bars=metadata.segment_bars,
        recursive=True,
    )
    reference_segments = midi_dir_to_segments(
        midi_dir=reference_dir,
        metadata=metadata,
        segment_stride_bars=metadata.segment_bars,
        recursive=True,
    )

    key_labels = load_key_labels(args.key_labels_csv)

    report = evaluate_d3pia_objective(
        generated_segments=generated_segments,
        reference_segments=reference_segments,
        metadata=metadata,
        key_labels=key_labels,
        beats_per_chord=args.beats_per_chord,
        chord_encoder_ckpt=args.chord_encoder_ckpt,
        device=device,
    )

    report_dir = ensure_dir(out_dir / "d3pia_pop909")
    write_d3pia_report(
        report=report, out_dir=report_dir, table_name="table_d3pia_pop909"
    )

    subjective = {}
    if args.create_subjective_package:
        subjective["package"] = create_d3pia_subjective_package(
            generated_midi_dir=generated_dir,
            reference_midi_dir=reference_dir,
            out_dir=report_dir,
            seed=args.seed,
            num_excerpts=args.subjective_num_excerpts,
            render_audio=args.render_audio,
            soundfont_path=args.soundfont_path,
        )

    if args.subjective_csv is not None:
        subjective["analysis"] = analyze_d3pia_subjective_results(
            survey_csv_path=Path(args.subjective_csv),
            out_dir=report_dir,
        )

    return {
        "protocol": "d3pia_pop909",
        "generated_dir": str(generated_dir),
        "reference_dir": str(reference_dir),
        "objective": report.summary,
        "notes": report.notes + notes,
        "subjective": subjective,
    }


def run_wholesong(
    args: argparse.Namespace, out_dir: Path, device: torch.device
) -> dict:
    base_metadata = _metadata_from_cfg_or_args(args)
    full_song_bars = args.full_song_bars

    metadata = NoteTupleMetadata(
        beats_per_bar=base_metadata.beats_per_bar,
        steps_per_beat=base_metadata.steps_per_beat,
        segment_bars=full_song_bars,
        dur_max_steps=base_metadata.dur_max_steps,
        vel_bins=base_metadata.vel_bins,
        num_tracks=base_metadata.num_tracks,
        max_notes_per_segment=base_metadata.max_notes_per_segment,
        default_tempo=base_metadata.default_tempo,
    )

    generated_dir = _resolve_generated_midi_dir(
        args=args, out_dir=out_dir, device=device
    )

    notes = []
    if args.train_dir is None:
        train_dir = generated_dir
        notes.append(
            "train_dir was not provided; generated set is used as DoC bank (protocol comparability risk)."
        )
    else:
        train_dir = Path(args.train_dir)

    generated_segments = midi_dir_to_segments(
        midi_dir=generated_dir,
        metadata=metadata,
        segment_stride_bars=metadata.segment_bars,
        recursive=True,
    )
    train_segments = midi_dir_to_segments(
        midi_dir=train_dir,
        metadata=metadata,
        segment_stride_bars=metadata.segment_bars,
        recursive=True,
    )

    report = evaluate_wholesong_objective(
        generated_segments=generated_segments,
        train_segments=train_segments,
        metadata=metadata,
        phrase_bars=args.phrase_bars,
        latent_dim=args.latent_dim,
        latent_encoder_ckpt_dir=args.latent_encoder_ckpt_dir,
        doc_measure_bars=args.doc_measure_bars,
        doc_remove_all_rest=args.doc_remove_all_rest,
    )

    report_dir = ensure_dir(out_dir / "wholesonggen_iclr24")
    write_wholesong_report(
        report=report,
        out_dir=report_dir,
        table_name="table_wholesonggen_iclr24",
    )

    subjective = {}
    if args.create_subjective_package:
        subjective["package"] = create_wholesong_subjective_package(
            generated_midi_dir=generated_dir,
            out_dir=report_dir,
            seed=args.seed,
            samples_per_group=args.samples_per_group,
            short_term_bars=args.short_term_bars,
            full_song_bars=args.full_song_bars,
            render_audio=args.render_audio,
            soundfont_path=args.soundfont_path,
        )

    if args.subjective_csv is not None:
        subjective["analysis"] = analyze_wholesong_subjective_results(
            survey_csv_path=Path(args.subjective_csv),
            out_dir=report_dir,
        )

    return {
        "protocol": "wholesonggen_iclr24",
        "generated_dir": str(generated_dir),
        "train_dir": str(train_dir),
        "objective": report.summary,
        "notes": report.notes + notes,
        "subjective": subjective,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate note-tuple symbolic music generators."
    )
    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        choices=["d3pia_pop909", "wholesonggen_iclr24"],
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--generated_dir", type=str, default=None)
    parser.add_argument("--reference_dir", type=str, default=None)
    parser.add_argument("--train_dir", type=str, default=None)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_generate_samples", type=int, default=128)
    parser.add_argument("--generate_batch_size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--create_subjective_package", action="store_true")
    parser.add_argument("--subjective_csv", type=str, default=None)
    parser.add_argument("--render_audio", action="store_true")
    parser.add_argument("--soundfont_path", type=str, default=None)

    # D3PIA options
    parser.add_argument("--key_labels_csv", type=str, default=None)
    parser.add_argument("--chord_encoder_ckpt", type=str, default=None)
    parser.add_argument("--beats_per_chord", type=int, default=1)
    parser.add_argument("--subjective_num_excerpts", type=int, default=10)

    # WholeSongGen options
    parser.add_argument("--phrase_bars", type=int, default=8)
    parser.add_argument("--short_term_bars", type=int, default=8)
    parser.add_argument("--full_song_bars", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--latent_encoder_ckpt_dir", type=str, default=None)
    parser.add_argument("--doc_measure_bars", type=int, default=2)
    parser.add_argument("--doc_remove_all_rest", action="store_true")
    parser.add_argument("--samples_per_group", type=int, default=3)

    # Fallback metadata if ckpt is not provided.
    parser.add_argument("--beats_per_bar", type=int, default=4)
    parser.add_argument("--steps_per_beat", type=int, default=4)
    parser.add_argument("--segment_bars", type=int, default=8)
    parser.add_argument("--dur_max_steps", type=int, default=32)
    parser.add_argument("--vel_bins", type=int, default=8)
    parser.add_argument("--num_tracks", type=int, default=8)
    parser.add_argument("--max_notes_per_segment", type=int, default=512)
    parser.add_argument("--default_tempo", type=float, default=120.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    out_dir = ensure_dir(Path(args.out_dir))

    if args.protocol == "d3pia_pop909":
        result = run_d3pia(args=args, out_dir=out_dir, device=device)
    else:
        result = run_wholesong(args=args, out_dir=out_dir, device=device)

    save_json(out_dir / "eval_result.json", result)
    print(f"Evaluation finished. Results saved under: {out_dir}")


if __name__ == "__main__":
    main()
