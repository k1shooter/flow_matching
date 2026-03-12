# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist

from data import data
from data.note_tuple import (
    ATTR_NAMES,
    note_tuple_batch_to_pretty_midi,
    NoteTupleMetadata,
)
from logic import flow, generate, training
from logic.flow import (
    BaseDiscretePath,
    MixtureDiscreteNoteTuplePath,
    SourceDistribution,
)
from logic.state import TrainState
from model import Transformer
from omegaconf import OmegaConf, open_dict
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import checkpointing, logging
from utils.rendering import (
    maybe_render_midi_to_musicxml,
    maybe_render_midi_to_wav,
    maybe_render_musicxml_to_png,
)


def _dataset_samples_to_segments(dataset, split_name: str):
    from eval.common import SegmentRecord

    samples = getattr(dataset, "samples", None)
    if samples is None:
        return []

    segments = []
    for i, sample in enumerate(samples):
        segments.append(
            SegmentRecord(
                song_id=f"{split_name}_song_{i // 1000:04d}",
                segment_id=f"{split_name}_seg_{i:06d}",
                note_tuple=sample,
            )
        )
    return segments


def _evaluate_training_preview(
    cfg: OmegaConf,
    state: TrainState,
    metadata: NoteTupleMetadata,
    step_dir: Path,
    device: torch.device,
    logger: logging.TrainLogger,
    default_seed: int,
) -> None:
    preview_cfg = getattr(cfg.training, "preview", None)
    if preview_cfg is None:
        return

    preview_eval_cfg = getattr(preview_cfg, "eval", None)
    if preview_eval_cfg is None:
        return
    if not bool(getattr(preview_eval_cfg, "enabled", False)):
        return

    generated_midi_dir = step_dir / "midi"
    if not generated_midi_dir.exists():
        return

    protocol = str(getattr(preview_eval_cfg, "protocol", cfg.eval.eval_protocol))
    eval_seed_raw = getattr(preview_eval_cfg, "seed", None)
    eval_seed = int(eval_seed_raw) if eval_seed_raw is not None else int(default_seed)

    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(eval_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(eval_seed)

        from eval.common import midi_dir_to_segments, save_json

        generated_segments = midi_dir_to_segments(
            midi_dir=generated_midi_dir,
            metadata=metadata,
            segment_stride_bars=metadata.segment_bars,
            recursive=True,
        )
        if len(generated_segments) == 0:
            logger.warning(
                f"No generated segments for preview evaluation at step {state.step}."
            )
            return

        if protocol == "d3pia_pop909":
            from eval.d3pia_pop909 import (
                evaluate_d3pia_objective,
                load_key_labels,
                write_d3pia_report,
            )

            reference_segments = _dataset_samples_to_segments(
                dataset=state.data_state.test.dataset,
                split_name="valid",
            )
            notes: List[str] = []
            if len(reference_segments) == 0:
                reference_segments = generated_segments
                notes.append(
                    "Validation dataset segments were unavailable; generated samples were used as reference."
                )

            key_labels = load_key_labels(
                getattr(
                    preview_eval_cfg,
                    "key_labels_csv",
                    getattr(cfg.eval.d3pia, "key_labels_csv", None),
                )
            )
            report = evaluate_d3pia_objective(
                generated_segments=generated_segments,
                reference_segments=reference_segments,
                metadata=metadata,
                key_labels=key_labels,
                beats_per_chord=int(
                    getattr(
                        preview_eval_cfg,
                        "beats_per_chord",
                        getattr(cfg.eval.d3pia, "beats_per_chord", 1),
                    )
                ),
                chord_encoder_ckpt=getattr(
                    preview_eval_cfg,
                    "chord_encoder_ckpt",
                    getattr(cfg.eval.d3pia, "chord_encoder_ckpt", None),
                ),
                device=device,
            )

            report_out_dir = step_dir / "eval" / "d3pia_pop909"
            table_name = str(
                getattr(cfg.eval.d3pia, "report_table_name", "table_d3pia_pop909")
            )
            write_d3pia_report(
                report=report,
                out_dir=report_out_dir,
                table_name=table_name,
            )
            save_json(
                report_out_dir / "preview_eval_result.json",
                {
                    "protocol": protocol,
                    "step": state.step,
                    "summary": report.summary,
                    "notes": report.notes + notes,
                    "generated_midi_dir": str(generated_midi_dir),
                },
            )
            logger.info(
                "Saved preview evaluation report "
                f"(protocol={protocol}) to {report_out_dir}"
            )
            return

        if protocol == "wholesonggen_iclr24":
            from eval.wholesonggen_iclr24 import (
                evaluate_wholesong_objective,
                write_wholesong_report,
            )

            train_segments = _dataset_samples_to_segments(
                dataset=state.data_state.train.dataset,
                split_name="train",
            )
            if len(train_segments) == 0:
                logger.warning(
                    "Training dataset segments are unavailable; "
                    "skipping WholeSong preview evaluation."
                )
                return

            report = evaluate_wholesong_objective(
                generated_segments=generated_segments,
                train_segments=train_segments,
                metadata=metadata,
                phrase_bars=int(
                    getattr(
                        preview_eval_cfg,
                        "phrase_bars",
                        getattr(cfg.eval.wholesonggen, "phrase_bars", 8),
                    )
                ),
                latent_dim=int(
                    getattr(
                        preview_eval_cfg,
                        "latent_dim",
                        getattr(cfg.eval.wholesonggen, "latent_dim", 64),
                    )
                ),
                latent_encoder_ckpt_dir=getattr(
                    preview_eval_cfg,
                    "latent_encoder_ckpt_dir",
                    getattr(cfg.eval.wholesonggen, "latent_encoder_ckpt_dir", None),
                ),
                doc_measure_bars=int(
                    getattr(
                        preview_eval_cfg,
                        "doc_measure_bars",
                        getattr(cfg.eval.wholesonggen, "doc_measure_bars", 2),
                    )
                ),
                doc_remove_all_rest=bool(
                    getattr(
                        preview_eval_cfg,
                        "doc_remove_all_rest",
                        getattr(cfg.eval.wholesonggen, "doc_remove_all_rest", True),
                    )
                ),
            )

            report_out_dir = step_dir / "eval" / "wholesonggen_iclr24"
            table_name = str(
                getattr(
                    cfg.eval.wholesonggen,
                    "report_table_name",
                    "table_wholesonggen_iclr24",
                )
            )
            write_wholesong_report(
                report=report,
                out_dir=report_out_dir,
                table_name=table_name,
            )
            save_json(
                report_out_dir / "preview_eval_result.json",
                {
                    "protocol": protocol,
                    "step": state.step,
                    "summary": report.summary,
                    "notes": report.notes,
                    "generated_midi_dir": str(generated_midi_dir),
                },
            )
            logger.info(
                "Saved preview evaluation report "
                f"(protocol={protocol}) to {report_out_dir}"
            )
            return

        logger.warning(f"Unsupported training.preview.eval.protocol: {protocol}")


def _save_preview_artifacts(
    sample_idx: int,
    step_dir: Path,
    sample: dict[str, torch.Tensor],
    metadata: NoteTupleMetadata,
    track_programs: list[int],
    save_musicxml: bool,
    save_score_png: bool,
    save_wav: bool,
    score_png_scale: int,
    soundfont_path: str | None,
) -> tuple[bool, bool, bool]:
    midi_dir = step_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    midi_path = midi_dir / f"sample_{sample_idx:05d}.mid"

    midi = note_tuple_batch_to_pretty_midi(
        sample=sample,
        metadata=metadata,
        track_programs=track_programs,
    )
    midi.write(str(midi_path))

    saved_musicxml = False
    saved_score_png = False
    if save_musicxml:
        musicxml_dir = step_dir / "musicxml"
        xml_path = musicxml_dir / f"sample_{sample_idx:05d}.musicxml"
        saved_musicxml = maybe_render_midi_to_musicxml(
            midi_path=midi_path,
            musicxml_path=xml_path,
        )
        if saved_musicxml and save_score_png:
            score_png_dir = step_dir / "score_png"
            png_path = score_png_dir / f"sample_{sample_idx:05d}.png"
            saved_score_png = maybe_render_musicxml_to_png(
                musicxml_path=xml_path,
                png_path=png_path,
                scale=int(score_png_scale),
            )

    saved_wav = False
    if save_wav:
        wav_dir = step_dir / "wav"
        wav_path = wav_dir / f"sample_{sample_idx:05d}.wav"
        saved_wav = maybe_render_midi_to_wav(
            midi_path=midi_path,
            wav_path=wav_path,
            soundfont_path=soundfont_path,
        )

    return saved_musicxml, saved_score_png, saved_wav


def _resolve_preview_track_programs(
    cfg: OmegaConf,
    preview_cfg: OmegaConf,
    metadata: NoteTupleMetadata,
) -> list[int]:
    override_programs = getattr(preview_cfg, "track_programs_override", None)
    if override_programs is not None:
        programs = [int(v) for v in override_programs]
    elif bool(getattr(preview_cfg, "piano_only", False)):
        programs = [0 for _ in range(int(metadata.num_tracks))]
    else:
        programs = [int(v) for v in cfg.data.track_programs]

    if len(programs) < int(metadata.num_tracks):
        programs.extend([0] * (int(metadata.num_tracks) - len(programs)))
    return programs[: int(metadata.num_tracks)]


@torch.no_grad()
def _maybe_generate_training_preview(
    rank: int,
    cfg: OmegaConf,
    state: TrainState,
    model: torch.nn.Module,
    path: BaseDiscretePath,
    source_distribution: SourceDistribution,
    vocab_sizes: dict[str, int],
    metadata: NoteTupleMetadata,
    work_dirs: checkpointing.WorkDirectory,
    device: torch.device,
    logger: logging.TrainLogger,
    use_edit_flow: bool,
    force_run: bool = False,
    preview_step_override: int | None = None,
    random_source_only: bool = False,
) -> None:
    preview_cfg = getattr(cfg.training, "preview", None)
    if preview_cfg is None:
        return
    if not bool(getattr(preview_cfg, "enabled", False)):
        return
    if rank != 0:
        return

    step_value = (
        int(preview_step_override)
        if preview_step_override is not None
        else int(state.step)
    )

    every_steps = int(getattr(preview_cfg, "every_steps", cfg.training.eval_freq))
    if (not force_run) and (every_steps <= 0 or step_value % every_steps != 0):
        return

    n_samples = int(getattr(preview_cfg, "num_samples", 2))
    if n_samples <= 0:
        return

    batch_size = max(1, int(getattr(preview_cfg, "batch_size", n_samples)))
    sampling_steps = int(
        getattr(preview_cfg, "sampling_steps", int(cfg.flow.sampling_steps))
    )
    save_musicxml = bool(getattr(preview_cfg, "save_musicxml", True))
    save_score_png = bool(getattr(preview_cfg, "save_score_png", True))
    save_wav = bool(getattr(preview_cfg, "save_wav", True))
    if save_score_png:
        save_musicxml = True
    score_png_scale = int(getattr(preview_cfg, "score_png_scale", 45))
    soundfont_path = getattr(preview_cfg, "soundfont_path", None)
    time_epsilon = float(
        getattr(preview_cfg, "time_epsilon", float(cfg.flow.time_epsilon))
    )
    temperature = float(
        getattr(preview_cfg, "temperature", float(cfg.flow.temperature))
    )
    rho_cap = float(getattr(preview_cfg, "rho_cap", float(cfg.flow.rho_cap)))
    final_full_resample = bool(
        getattr(preview_cfg, "final_resample", bool(cfg.flow.final_resample))
    )
    preview_seed_raw = getattr(preview_cfg, "seed", None)
    if preview_seed_raw is None:
        preview_seed = int(cfg.training.seed + step_value)
    else:
        preview_seed = int(preview_seed_raw)

    step_dir = work_dirs.samples / f"step_{step_value:08d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    preview_track_programs = _resolve_preview_track_programs(
        cfg=cfg,
        preview_cfg=preview_cfg,
        metadata=metadata,
    )

    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(preview_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(preview_seed)

        was_training = model.training
        model.eval()
        try:
            saved = 0
            saved_musicxml = 0
            saved_score_png = 0
            saved_wav = 0
            remaining = n_samples
            while remaining > 0:
                current_bs = min(batch_size, remaining)
                if random_source_only:
                    samples = source_distribution.sample(
                        batch_size=current_bs,
                        sequence_length=cfg.data.max_notes_per_segment,
                        device=device,
                    )
                elif use_edit_flow:
                    samples = generate.generate_samples_edit(
                        model=model,
                        vocab_sizes=vocab_sizes,
                        sample_batch_size=current_bs,
                        sequence_length=cfg.data.max_notes_per_segment,
                        sampling_steps=sampling_steps,
                        device=device,
                        sampler=str(cfg.flow.edit_flow.sampler),
                        tau_leaping_step_size=float(
                            cfg.flow.edit_flow.tau_leaping_step_size
                        ),
                        max_events_per_step=int(cfg.flow.edit_flow.max_events_per_step),
                        time_epsilon=time_epsilon,
                        temperature=temperature,
                        rate_scale=float(cfg.flow.edit_flow.rate_scale),
                        init_mode=str(cfg.flow.edit_flow.init_mode),
                    )
                else:
                    if not isinstance(path, MixtureDiscreteNoteTuplePath):
                        raise RuntimeError(
                            "Expected mixture path for non-edit-flow preview generation."
                        )
                    samples = generate.generate_samples(
                        model=model,
                        path=path,
                        source_distribution=source_distribution,
                        sample_batch_size=current_bs,
                        sequence_length=cfg.data.max_notes_per_segment,
                        sampling_steps=sampling_steps,
                        device=device,
                        time_epsilon=time_epsilon,
                        temperature=temperature,
                        rho_cap=rho_cap,
                        final_full_resample=final_full_resample,
                    )

                for i in range(current_bs):
                    sample = {
                        attr: samples[attr][i].detach().cpu() for attr in ATTR_NAMES
                    }
                    xml_ok, png_ok, wav_ok = _save_preview_artifacts(
                        sample_idx=saved,
                        step_dir=step_dir,
                        sample=sample,
                        metadata=metadata,
                        track_programs=preview_track_programs,
                        save_musicxml=save_musicxml,
                        save_score_png=save_score_png,
                        save_wav=save_wav,
                        score_png_scale=score_png_scale,
                        soundfont_path=soundfont_path,
                    )
                    saved += 1
                    if xml_ok:
                        saved_musicxml += 1
                    if png_ok:
                        saved_score_png += 1
                    if wav_ok:
                        saved_wav += 1

                remaining -= current_bs
        finally:
            model.train(was_training)

    logger.info(
        "Saved training preview samples at step "
        f"{step_value}: midi={saved}, musicxml={saved_musicxml}, "
        f"score_png={saved_score_png}, wav={saved_wav} -> {step_dir}"
    )
    if save_musicxml and saved_musicxml < saved:
        logger.warning(
            "Some MusicXML previews were not exported. Install `music21` to enable score export."
        )
    if save_wav and saved_wav < saved:
        logger.warning(
            "Some WAV previews were not exported. Install `pyfluidsynth` and provide a SoundFont path."
        )
    if save_score_png and saved_score_png < saved:
        logger.warning(
            "Some score PNG previews were not exported. "
            "Install `verovio` + `cairosvg`, or install headless `musescore3`."
        )

    try:
        _evaluate_training_preview(
            cfg=cfg,
            state=state,
            metadata=metadata,
            step_dir=step_dir,
            device=device,
            logger=logger,
            default_seed=preview_seed + 100_003,
        )
    except Exception as exc:
        logger.warning(
            "Preview evaluation failed at step "
            f"{state.step}; training will continue. Reason: {exc}"
        )


def run_train(rank: int, cfg: OmegaConf) -> None:
    torch.manual_seed(cfg.training.seed + rank)

    work_dirs = checkpointing.get_work_dirs(work_dir=cfg.work_dir, rank=rank)
    logger = logging.TrainLogger(log_dir=work_dirs.root, rank=rank, cfg=cfg)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    logger.log_devices(device=device, logger=logger)

    use_edit_flow = bool(getattr(cfg.flow, "use_edit_flow", False))
    with open_dict(cfg):
        cfg.model.enable_edit_flow = use_edit_flow

    vocab_sizes = data.get_vocab_sizes(config=cfg)
    source_distribution = flow.get_source_distribution(
        source_distribution=cfg.flow.source_distribution,
        vocab_sizes=vocab_sizes,
        include_pad=cfg.data.source_include_pad,
    )

    model = Transformer(config=cfg.model, vocab_sizes=vocab_sizes).to(device)
    num_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of parameters in the model: {num_parameters}")

    if device.type == "cuda":
        model = DDP(model, device_ids=[rank], static_graph=not use_edit_flow)
    else:
        model = DDP(model)

    logger.info(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, cfg.optim.beta2),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
        fused=cfg.optim.fused,
    )

    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")
    data_state = data.get_data_state(config=cfg)

    state = TrainState(model=model, optimizer=optimizer, step=1, data_state=data_state)
    state.restore_checkpoint(ckpt_dir=work_dirs.checkpoint, device=device, rank=rank)

    train_iter, eval_iter = data.get_data_loaders(config=cfg, data_state=data_state)

    if cfg.model.compile:
        state.compile_model()
        torch.set_float32_matmul_precision("high")

    path = flow.get_path(flow_cfg=cfg.flow)
    loss_fn = flow.get_loss_function(
        loss_function=cfg.flow.loss_function,
        vocab_sizes=vocab_sizes,
        use_edit_flow=use_edit_flow,
        edit_flow_cfg=cfg.flow.edit_flow,
    )
    preview_metadata = data.get_note_tuple_metadata(config=cfg)
    sampling_model = (
        state.model.module if hasattr(state.model, "module") else state.model
    )

    num_train_steps = cfg.optim.n_iters
    logger.info(f"Starting training loop at step {state.step}.")

    preview_cfg = getattr(cfg.training, "preview", None)
    if (
        preview_cfg is not None
        and bool(getattr(preview_cfg, "enabled", False))
        and bool(getattr(preview_cfg, "sample_at_start", True))
        and state.step <= 1
    ):
        _maybe_generate_training_preview(
            rank=rank,
            cfg=cfg,
            state=state,
            model=sampling_model,
            path=path,
            source_distribution=source_distribution,
            vocab_sizes=vocab_sizes,
            metadata=preview_metadata,
            work_dirs=work_dirs,
            device=device,
            logger=logger,
            use_edit_flow=use_edit_flow,
            force_run=True,
            preview_step_override=0,
            random_source_only=bool(
                getattr(preview_cfg, "random_source_at_start", True)
            ),
        )

    train_loss_values = []

    while state.step <= num_train_steps:
        loss = training.step(
            loss_fn=loss_fn,
            path=path,
            state=state,
            scaler=scaler,
            iterator=train_iter,
            optim_params=cfg.optim,
            device=device,
            source_distribution=source_distribution,
            logger=logger,
            training=True,
            use_edit_flow=use_edit_flow,
            time_epsilon=float(cfg.flow.time_epsilon),
        )
        train_loss_values.append(loss)

        if state.step % cfg.logging.log_freq == 0:
            agg_train_loss = torch.tensor(train_loss_values, device=device).mean()
            dist.all_reduce(agg_train_loss, dist.ReduceOp.AVG)
            logger.log_metric(
                value=agg_train_loss.item(),
                name="Loss",
                stage="Train",
                step=state.step,
            )
            train_loss_values = []

        if state.step % cfg.training.snapshot == 0:
            logger.info("Saving checkpoint...", step=state.step)
            state.save_checkpoint(ckpt_dir=work_dirs.checkpoint, rank=rank)

        if state.step % cfg.training.eval_freq == 0:
            logger.info("Evaluating loss...", step=state.step)
            eval_loss = training.step(
                state=state,
                loss_fn=loss_fn,
                path=path,
                scaler=scaler,
                iterator=eval_iter,
                device=device,
                source_distribution=source_distribution,
                logger=logger,
                training=False,
                use_edit_flow=use_edit_flow,
                time_epsilon=float(cfg.flow.time_epsilon),
            )
            dist.all_reduce(eval_loss, dist.ReduceOp.AVG)
            logger.log_metric(
                value=eval_loss.item(),
                name="Loss",
                stage="Evaluation",
                step=state.step,
            )

        _maybe_generate_training_preview(
            rank=rank,
            cfg=cfg,
            state=state,
            model=sampling_model,
            path=path,
            source_distribution=source_distribution,
            vocab_sizes=vocab_sizes,
            metadata=preview_metadata,
            work_dirs=work_dirs,
            device=device,
            logger=logger,
            use_edit_flow=use_edit_flow,
        )

        state.step = state.step + 1

    state.step = num_train_steps
    logger.info("Saving final checkpoint...", step=state.step)
    state.save_checkpoint(ckpt_dir=work_dirs.checkpoint, rank=rank)

    logger.finish()


def setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    timeout = datetime.timedelta(minutes=30)
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout)


def cleanup() -> None:
    dist.destroy_process_group()


def run_mp_training(rank: int, world_size: int, cfg: OmegaConf, port: int) -> None:
    try:
        setup(rank=rank, world_size=world_size, port=port)
        run_train(rank=rank, cfg=cfg)
    finally:
        cleanup()
