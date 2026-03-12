# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.data import get_note_tuple_metadata, get_vocab_sizes
from data.note_tuple import ATTR_NAMES, note_tuple_batch_to_pretty_midi
from logic import flow, generate
from logic.flow import MixtureDiscreteNoteTuplePath
from utils import checkpointing
from utils.rendering import (
    maybe_render_midi_to_musicxml,
    maybe_render_midi_to_wav,
    maybe_render_musicxml_to_png,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample note tuples and save MIDI files."
    )
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--rho_cap", type=float, default=None)
    parser.add_argument("--time_epsilon", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_final_resample", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--disable_musicxml", action="store_true")
    parser.add_argument("--disable_wav", action="store_true")
    parser.add_argument("--disable_score_png", action="store_true")
    parser.add_argument("--score_png_scale", type=int, default=45)
    parser.add_argument("--soundfont_path", type=str, default=None)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    cfg = checkpointing.load_cfg_from_path(args.work_dir)
    model = checkpointing.load_model_from_path(args.work_dir, device=device)
    model.eval()

    vocab_sizes = get_vocab_sizes(config=cfg)
    path = flow.get_path(flow_cfg=cfg.flow)

    source_distribution = flow.get_source_distribution(
        source_distribution=cfg.flow.source_distribution,
        vocab_sizes=vocab_sizes,
        include_pad=cfg.data.source_include_pad,
    )

    sampling_steps = (
        int(args.sampling_steps)
        if args.sampling_steps is not None
        else int(cfg.flow.sampling_steps)
    )
    temperature = (
        float(args.temperature)
        if args.temperature is not None
        else float(cfg.flow.temperature)
    )
    rho_cap = (
        float(args.rho_cap) if args.rho_cap is not None else float(cfg.flow.rho_cap)
    )
    time_epsilon = (
        float(args.time_epsilon)
        if args.time_epsilon is not None
        else float(cfg.flow.time_epsilon)
    )
    final_full_resample = (
        bool(cfg.flow.final_resample) and not args.disable_final_resample
    )

    metadata = get_note_tuple_metadata(config=cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_dir = output_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    musicxml_dir = output_dir / "musicxml"
    wav_dir = output_dir / "wav"
    score_png_dir = output_dir / "score_png"

    remaining = int(args.num_samples)
    saved = 0
    saved_musicxml = 0
    saved_wav = 0
    saved_score_png = 0

    use_edit_flow = bool(getattr(cfg.flow, "use_edit_flow", False))
    export_musicxml = not bool(args.disable_musicxml)
    export_wav = not bool(args.disable_wav)
    export_score_png = not bool(args.disable_score_png)
    if export_score_png:
        export_musicxml = True

    while remaining > 0:
        batch_size = min(int(args.batch_size), remaining)

        if use_edit_flow:
            samples = generate.generate_samples_edit(
                model=model,
                vocab_sizes=vocab_sizes,
                sample_batch_size=batch_size,
                sequence_length=cfg.data.max_notes_per_segment,
                sampling_steps=sampling_steps,
                device=device,
                sampler=str(cfg.flow.edit_flow.sampler),
                tau_leaping_step_size=float(cfg.flow.edit_flow.tau_leaping_step_size),
                max_events_per_step=int(cfg.flow.edit_flow.max_events_per_step),
                time_epsilon=time_epsilon,
                temperature=temperature,
                rate_scale=float(cfg.flow.edit_flow.rate_scale),
                init_mode=str(cfg.flow.edit_flow.init_mode),
            )
        else:
            if not isinstance(path, MixtureDiscreteNoteTuplePath):
                raise NotImplementedError("Only mixture path sampling is implemented.")
            samples = generate.generate_samples(
                model=model,
                path=path,
                source_distribution=source_distribution,
                sample_batch_size=batch_size,
                sequence_length=cfg.data.max_notes_per_segment,
                sampling_steps=sampling_steps,
                device=device,
                time_epsilon=time_epsilon,
                temperature=temperature,
                rho_cap=rho_cap,
                final_full_resample=final_full_resample,
            )

        for i in range(batch_size):
            sample = {attr: samples[attr][i] for attr in ATTR_NAMES}
            midi = note_tuple_batch_to_pretty_midi(
                sample=sample,
                metadata=metadata,
                track_programs=cfg.data.track_programs,
            )

            out_path = midi_dir / f"sample_{saved:05d}.mid"
            midi.write(str(out_path))

            if export_musicxml:
                xml_path = musicxml_dir / f"sample_{saved:05d}.musicxml"
                musicxml_ok = maybe_render_midi_to_musicxml(out_path, xml_path)
                if musicxml_ok:
                    saved_musicxml += 1
                    if export_score_png:
                        score_png_path = score_png_dir / f"sample_{saved:05d}.png"
                        if maybe_render_musicxml_to_png(
                            musicxml_path=xml_path,
                            png_path=score_png_path,
                            scale=int(args.score_png_scale),
                        ):
                            saved_score_png += 1

            if export_wav:
                wav_path = wav_dir / f"sample_{saved:05d}.wav"
                if maybe_render_midi_to_wav(
                    midi_path=out_path,
                    wav_path=wav_path,
                    soundfont_path=args.soundfont_path,
                ):
                    saved_wav += 1

            saved += 1

        remaining -= batch_size

    print(f"Saved {saved} MIDI files to {midi_dir}")
    if export_musicxml:
        print(f"Saved {saved_musicxml} MusicXML scores to {musicxml_dir}")
        if saved_musicxml < saved:
            print(
                "MusicXML export skipped for some samples. "
                "Install `music21` if not already installed."
            )
    if export_score_png:
        print(f"Saved {saved_score_png} score PNG files to {score_png_dir}")
        if saved_score_png < saved:
            print(
                "Score PNG export skipped for some samples. "
                "Install `verovio` + `cairosvg`, or install headless `musescore3`."
            )
    if export_wav:
        print(f"Saved {saved_wav} WAV files to {wav_dir}")
        if saved_wav < saved:
            print(
                "WAV export skipped for some samples. "
                "Install `pyfluidsynth` and a SoundFont (.sf2) for reliable rendering."
            )


if __name__ == "__main__":
    main(parse_args())
