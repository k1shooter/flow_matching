from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from data.note_tuple import NoteTupleMetadata

from .common import (
    maybe_render_midi_to_wav,
    read_csv_rows,
    save_csv,
    save_json,
    save_markdown_table,
    scan_midi_files,
    SegmentRecord,
    set_global_seed,
)
from .d3pia_pop909 import extract_chords_rule_based
from .pianoroll import note_tuple_to_pianoroll, split_note_tuple_by_measures
from .stats import mean_and_ci, within_subject_ci


PHRASE_TYPES = ["A", "B", "C", "D"]


@dataclass
class WholeSongReport:
    summary: Dict[str, float]
    rows: List[Dict[str, object]]
    notes: List[str]


def _downsample_vector(x: np.ndarray, target_dim: int) -> np.ndarray:
    if x.size == target_dim:
        return x.astype(np.float32)
    if x.size == 0:
        return np.zeros(target_dim, dtype=np.float32)

    src_idx = np.linspace(0.0, 1.0, num=x.size)
    dst_idx = np.linspace(0.0, 1.0, num=target_dim)
    return np.interp(dst_idx, src_idx, x).astype(np.float32)


def _extract_pitch_contour(roll: np.ndarray) -> np.ndarray:
    steps, pitches = np.where(roll > 0)
    contour = np.zeros(roll.shape[0], dtype=np.float32)
    if steps.size == 0:
        return contour

    for t in range(roll.shape[0]):
        active = np.where(roll[t] > 0)[0]
        if active.size > 0:
            contour[t] = float(active.mean() / 127.0)
    return contour


def _extract_rhythm_feature(roll: np.ndarray) -> np.ndarray:
    onset = np.zeros(roll.shape[0], dtype=np.float32)
    active = roll > 0
    onset[0] = 1.0 if np.any(active[0]) else 0.0
    for i in range(1, roll.shape[0]):
        onset[i] = 1.0 if np.any(active[i] & (~active[i - 1])) else 0.0
    return onset


def _extract_texture_feature(
    roll: np.ndarray, metadata: NoteTupleMetadata
) -> np.ndarray:
    bar_vectors = []
    for bar in range(metadata.segment_bars):
        start = bar * metadata.steps_per_bar
        end = start + metadata.steps_per_bar
        sub = roll[start:end]

        note_density = float(np.mean(sub > 0))
        active_voices = float(np.mean(np.sum(sub > 0, axis=1))) / 8.0
        rhythmic_density = float(np.mean(np.sum(sub > 0, axis=1) > 0))
        bar_vectors.extend([note_density, active_voices, rhythmic_density])

    return np.asarray(bar_vectors, dtype=np.float32)


def _extract_chord_feature(
    note_tuple: Dict[str, torch.Tensor], metadata: NoteTupleMetadata
) -> np.ndarray:
    chords = extract_chords_rule_based(note_tuple, metadata=metadata, beats_per_chord=1)
    vec = np.zeros(24, dtype=np.float32)  # 12 major + 12 minor bucket

    name_to_pc = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
    }

    for chord in chords:
        if chord == "N" or ":" not in chord:
            continue
        root, quality = chord.split(":")
        if root not in name_to_pc:
            continue
        root_idx = name_to_pc[root]
        if quality == "min":
            vec[12 + root_idx] += 1.0
        else:
            vec[root_idx] += 1.0

    if vec.sum() > 0:
        vec = vec / vec.sum()

    return vec


def _feature_map_for_phrase(
    note_tuple: Dict[str, torch.Tensor],
    metadata: NoteTupleMetadata,
) -> Dict[str, np.ndarray]:
    roll = note_tuple_to_pianoroll(
        note_tuple=note_tuple, metadata=metadata, value_mode="binary"
    )

    pitch = _downsample_vector(_extract_pitch_contour(roll), 64)
    rhythm = _downsample_vector(_extract_rhythm_feature(roll), 64)
    chord = _downsample_vector(
        _extract_chord_feature(note_tuple, metadata=metadata), 64
    )
    texture = _downsample_vector(_extract_texture_feature(roll, metadata=metadata), 64)

    return {
        "p": pitch,
        "r": rhythm,
        "chd": chord,
        "txt": texture,
    }


def _load_projection(
    theta: str, latent_dim: int, ckpt_dir: str | None
) -> Tuple[np.ndarray, np.ndarray, bool]:
    input_dim = 64
    if ckpt_dir is not None:
        ckpt_path = Path(ckpt_dir) / f"{theta}.npz"
        if ckpt_path.exists():
            data = np.load(ckpt_path)
            if "w" in data and "b" in data:
                return data["w"], data["b"], True

    seed = {
        "p": 7,
        "r": 13,
        "chd": 17,
        "txt": 19,
    }[theta]
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.2, size=(input_dim, latent_dim)).astype(np.float32)
    b = np.zeros(latent_dim, dtype=np.float32)
    return w, b, False


def _latent_encode(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = np.tanh(x @ w + b)
    norm = np.linalg.norm(z)
    if norm > 0:
        z = z / norm
    return z.astype(np.float32)


def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _build_phrase_records(
    song_segments: Sequence[SegmentRecord],
    metadata: NoteTupleMetadata,
    phrase_bars: int,
) -> Dict[str, List[Tuple[str, Dict[str, np.ndarray]]]]:
    songs: Dict[str, List[Tuple[str, Dict[str, np.ndarray]]]] = {}

    for record in song_segments:
        phrase_samples = split_note_tuple_by_measures(
            note_tuple=record.note_tuple,
            metadata=metadata,
            measure_bars=phrase_bars,
        )
        song_entries = []
        for i, phrase_sample in enumerate(phrase_samples):
            phrase_type = PHRASE_TYPES[i % len(PHRASE_TYPES)]
            feats = _feature_map_for_phrase(
                note_tuple=phrase_sample,
                metadata=NoteTupleMetadata(
                    beats_per_bar=metadata.beats_per_bar,
                    steps_per_beat=metadata.steps_per_beat,
                    segment_bars=phrase_bars,
                    dur_max_steps=metadata.dur_max_steps,
                    vel_bins=metadata.vel_bins,
                    num_tracks=metadata.num_tracks,
                    max_notes_per_segment=metadata.max_notes_per_segment,
                    default_tempo=metadata.default_tempo,
                ),
            )
            song_entries.append((phrase_type, feats))
        songs[record.song_id] = song_entries

    return songs


def compute_ils(
    song_segments: Sequence[SegmentRecord],
    metadata: NoteTupleMetadata,
    phrase_bars: int,
    latent_dim: int,
    latent_encoder_ckpt_dir: str | None,
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    notes = []
    songs = _build_phrase_records(
        song_segments=song_segments,
        metadata=metadata,
        phrase_bars=phrase_bars,
    )

    projectors = {}
    pretrained_flags = {}
    for theta in ["p", "r", "chd", "txt"]:
        w, b, is_pretrained = _load_projection(
            theta=theta,
            latent_dim=latent_dim,
            ckpt_dir=latent_encoder_ckpt_dir,
        )
        projectors[theta] = (w, b)
        pretrained_flags[theta] = is_pretrained

    if not all(pretrained_flags.values()):
        notes.append(
            "Some latent encoders were not found; deterministic fallback projections are used (ILS comparability risk)."
        )

    theta_scores: Dict[str, List[float]] = {k: [] for k in ["p", "r", "chd", "txt"]}

    for _, phrases in songs.items():
        for theta in theta_scores:
            type_to_embeddings: Dict[str, List[np.ndarray]] = {}
            w, b = projectors[theta]
            for phrase_type, features in phrases:
                z = _latent_encode(features[theta], w=w, b=b)
                type_to_embeddings.setdefault(phrase_type, []).append(z)

            local_scores = []
            for _, emb_list in type_to_embeddings.items():
                if len(emb_list) < 2:
                    continue
                for i in range(len(emb_list)):
                    for j in range(i + 1, len(emb_list)):
                        local_scores.append(_cosine_np(emb_list[i], emb_list[j]))

            if len(local_scores) > 0:
                theta_scores[theta].append(float(np.mean(local_scores)))

    summary = {}
    for theta, vals in theta_scores.items():
        stats = mean_and_ci(vals)
        summary[theta] = {
            "mean": stats["mean"],
            "std": stats["std"],
            "n": stats["n"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
        }

    return summary, notes


def _segment_to_doc_feature(
    note_tuple: Dict[str, torch.Tensor],
    metadata: NoteTupleMetadata,
) -> Dict[str, np.ndarray]:
    roll = note_tuple_to_pianoroll(
        note_tuple=note_tuple, metadata=metadata, value_mode="binary"
    )
    pitch_class_hist = np.zeros(12, dtype=np.float32)
    pitch_indices = np.where(roll > 0)[1]
    if pitch_indices.size > 0:
        for p in pitch_indices:
            pitch_class_hist[p % 12] += 1.0
        pitch_class_hist = pitch_class_hist / max(1.0, pitch_class_hist.sum())

    rhythm = _extract_rhythm_feature(roll).astype(np.float32)
    if rhythm.size > 0:
        rhythm = rhythm / max(1.0, rhythm.sum())

    return {
        "pitch_hist": pitch_class_hist,
        "rhythm": _downsample_vector(rhythm, 64),
        "pitch_latent": _downsample_vector(_extract_pitch_contour(roll), 64),
        "rhythm_latent": _downsample_vector(_extract_rhythm_feature(roll), 64),
        "is_rest": float(np.sum(roll)) == 0.0,
    }


def _transpose_pitch_hist(hist: np.ndarray, semitone: int) -> np.ndarray:
    return np.roll(hist, semitone).astype(np.float32)


def build_doc_bank(
    train_segments: Sequence[SegmentRecord],
    metadata: NoteTupleMetadata,
    measure_bars: int,
    remove_all_rest: bool,
) -> List[Dict[str, np.ndarray]]:
    bank = []
    for record in train_segments:
        windows = split_note_tuple_by_measures(
            note_tuple=record.note_tuple,
            metadata=metadata,
            measure_bars=measure_bars,
        )
        for w in windows:
            feat = _segment_to_doc_feature(
                note_tuple=w,
                metadata=NoteTupleMetadata(
                    beats_per_bar=metadata.beats_per_bar,
                    steps_per_beat=metadata.steps_per_beat,
                    segment_bars=measure_bars,
                    dur_max_steps=metadata.dur_max_steps,
                    vel_bins=metadata.vel_bins,
                    num_tracks=metadata.num_tracks,
                    max_notes_per_segment=metadata.max_notes_per_segment,
                    default_tempo=metadata.default_tempo,
                ),
            )
            if remove_all_rest and feat["is_rest"]:
                continue

            for semitone in range(12):
                bank.append(
                    {
                        "pitch_hist": _transpose_pitch_hist(
                            feat["pitch_hist"], semitone
                        ),
                        "rhythm": feat["rhythm"],
                        "pitch_latent": _transpose_pitch_hist(
                            feat["pitch_latent"][:12], semitone
                        ),
                        "rhythm_latent": feat["rhythm_latent"],
                    }
                )

    return bank


def _simlt_p(query: Dict[str, np.ndarray], ref: Dict[str, np.ndarray]) -> float:
    return _cosine_np(query["pitch_hist"], ref["pitch_hist"])


def _simlt_r(query: Dict[str, np.ndarray], ref: Dict[str, np.ndarray]) -> float:
    return _cosine_np(query["rhythm"], ref["rhythm"])


def _simrb(query: Dict[str, np.ndarray], ref: Dict[str, np.ndarray]) -> float:
    q_pc = query["pitch_hist"] > 0
    r_pc = ref["pitch_hist"] > 0
    inter_pc = np.logical_and(q_pc, r_pc).sum()
    union_pc = np.logical_or(q_pc, r_pc).sum()
    jacc_pc = float(inter_pc / union_pc) if union_pc > 0 else 0.0

    q_r = query["rhythm"] > 0
    r_r = ref["rhythm"] > 0
    inter_r = np.logical_and(q_r, r_r).sum()
    union_r = np.logical_or(q_r, r_r).sum()
    jacc_r = float(inter_r / union_r) if union_r > 0 else 0.0

    return 0.5 * (jacc_pc + jacc_r)


def compute_doc(
    query_segments: Sequence[SegmentRecord],
    metadata: NoteTupleMetadata,
    measure_bars: int,
    bank: Sequence[Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, float]]:
    if len(bank) == 0:
        return {
            "simlt_p": mean_and_ci([]),
            "simlt_r": mean_and_ci([]),
            "simrb": mean_and_ci([]),
        }

    all_scores = {
        "simlt_p": [],
        "simlt_r": [],
        "simrb": [],
    }

    sub_metadata = NoteTupleMetadata(
        beats_per_bar=metadata.beats_per_bar,
        steps_per_beat=metadata.steps_per_beat,
        segment_bars=measure_bars,
        dur_max_steps=metadata.dur_max_steps,
        vel_bins=metadata.vel_bins,
        num_tracks=metadata.num_tracks,
        max_notes_per_segment=metadata.max_notes_per_segment,
        default_tempo=metadata.default_tempo,
    )

    for record in query_segments:
        windows = split_note_tuple_by_measures(
            note_tuple=record.note_tuple,
            metadata=metadata,
            measure_bars=measure_bars,
        )
        for w in windows:
            query_feat = _segment_to_doc_feature(w, metadata=sub_metadata)
            max_p = max(_simlt_p(query_feat, ref) for ref in bank)
            max_r = max(_simlt_r(query_feat, ref) for ref in bank)
            max_rb = max(_simrb(query_feat, ref) for ref in bank)

            all_scores["simlt_p"].append(max_p)
            all_scores["simlt_r"].append(max_r)
            all_scores["simrb"].append(max_rb)

    return {k: mean_and_ci(v) for k, v in all_scores.items()}


def evaluate_wholesong_objective(
    generated_segments: Sequence[SegmentRecord],
    train_segments: Sequence[SegmentRecord],
    metadata: NoteTupleMetadata,
    phrase_bars: int,
    latent_dim: int,
    latent_encoder_ckpt_dir: str | None,
    doc_measure_bars: int,
    doc_remove_all_rest: bool,
) -> WholeSongReport:
    notes = []

    ils_summary, ils_notes = compute_ils(
        song_segments=generated_segments,
        metadata=metadata,
        phrase_bars=phrase_bars,
        latent_dim=latent_dim,
        latent_encoder_ckpt_dir=latent_encoder_ckpt_dir,
    )
    notes.extend(ils_notes)

    bank = build_doc_bank(
        train_segments=train_segments,
        metadata=metadata,
        measure_bars=doc_measure_bars,
        remove_all_rest=doc_remove_all_rest,
    )

    doc_summary = compute_doc(
        query_segments=generated_segments,
        metadata=metadata,
        measure_bars=doc_measure_bars,
        bank=bank,
    )

    summary = {
        "ILS_p_mean": ils_summary["p"]["mean"],
        "ILS_r_mean": ils_summary["r"]["mean"],
        "ILS_chd_mean": ils_summary["chd"]["mean"],
        "ILS_txt_mean": ils_summary["txt"]["mean"],
        "DoC_simlt_p_mean": doc_summary["simlt_p"]["mean"],
        "DoC_simlt_r_mean": doc_summary["simlt_r"]["mean"],
        "DoC_simrb_mean": doc_summary["simrb"]["mean"],
        "n_generated_segments": len(generated_segments),
        "n_train_segments": len(train_segments),
    }

    rows = []
    for k, v in summary.items():
        rows.append({"metric": k, "value": float(v)})

    return WholeSongReport(summary=summary, rows=rows, notes=notes)


def write_wholesong_report(
    report: WholeSongReport, out_dir: Path, table_name: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(out_dir / "objective_summary.json", report.summary)
    save_json(out_dir / "objective_notes.json", {"notes": report.notes})
    save_csv(out_dir / "objective_metrics.csv", report.rows)

    headers = ["Metric", "Value"]
    rows = [[r["metric"], f"{r['value']:.6f}"] for r in report.rows]
    save_markdown_table(out_dir / f"{table_name}.md", headers=headers, rows=rows)


def create_wholesong_subjective_package(
    generated_midi_dir: Path,
    out_dir: Path,
    seed: int,
    samples_per_group: int,
    short_term_bars: int,
    full_song_bars: int,
    render_audio: bool = False,
    soundfont_path: str | None = None,
) -> Dict[str, object]:
    set_global_seed(seed)

    midi_files = scan_midi_files(generated_midi_dir)
    if len(midi_files) == 0:
        raise ValueError(f"No MIDI files in {generated_midi_dir}")

    groups = []
    for context in ["short_term", "whole_song"]:
        for role in ["lead_sheet", "accompaniment"]:
            for idx in range(3):
                groups.append(
                    {
                        "group": f"{context}_{role}_{idx}",
                        "context": context,
                        "role": role,
                    }
                )

    random.shuffle(groups)

    pkg_dir = out_dir / "subjective_package"
    midi_dir = pkg_dir / "midi"
    audio_dir = pkg_dir / "audio"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)
    if render_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    hidden = []
    q_rows = []

    item_idx = 0
    for g in groups:
        selected = random.sample(midi_files, min(samples_per_group, len(midi_files)))
        for midi_path in selected:
            item_id = f"item_{item_idx:04d}"
            target = midi_dir / f"{item_id}.mid"
            target.write_bytes(midi_path.read_bytes())

            if render_audio:
                maybe_render_midi_to_wav(
                    midi_path=target,
                    wav_path=audio_dir / f"{item_id}.wav",
                    soundfont_path=soundfont_path,
                )

            manifest.append(
                {
                    "item_id": item_id,
                    "group": g["group"],
                    "context": g["context"],
                    "role": g["role"],
                    "path": str(target),
                }
            )
            hidden.append({"item_id": item_id, "source": str(midi_path)})

            q_row = {
                "participant_id": "",
                "item_id": item_id,
                "group": g["group"],
                "context": g["context"],
                "role": g["role"],
                "creativity": "",
                "naturalness": "",
                "musicality": "",
                "boundary_clarity": "",
                "phrase_similarity": "",
                "comment": "",
            }
            q_rows.append(q_row)
            item_idx += 1

    random.shuffle(manifest)

    save_csv(pkg_dir / "manifest.csv", manifest)
    save_csv(pkg_dir / "hidden_key.csv", hidden)
    save_csv(pkg_dir / "questionnaire_template.csv", q_rows)

    html = """<!doctype html>
<html>
<head><meta charset=\"utf-8\"><title>WholeSongGen Subjective Survey</title></head>
<body>
<h1>WholeSongGen Subjective Survey Template</h1>
<p>Two-part survey:</p>
<ul>
<li>Short-term (8 measures): Creativity, Naturalness, Musicality</li>
<li>Whole-song (32 measures): Naturalness, Musicality, Boundary Clarity, Phrase Similarity</li>
</ul>
</body>
</html>
"""
    (pkg_dir / "survey_template.html").write_text(html, encoding="utf-8")

    return {
        "package_dir": str(pkg_dir),
        "num_items": len(manifest),
        "short_term_bars": short_term_bars,
        "full_song_bars": full_song_bars,
    }


def analyze_wholesong_subjective_results(
    survey_csv_path: Path,
    out_dir: Path,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    rows = read_csv_rows(survey_csv_path)

    metrics = [
        "creativity",
        "naturalness",
        "musicality",
        "boundary_clarity",
        "phrase_similarity",
    ]

    overall = {}
    for metric in metrics:
        metric_rows = []
        for row in rows:
            if metric not in row or row[metric] == "":
                continue
            metric_rows.append(
                {
                    "participant": row.get("participant_id", "unknown"),
                    "condition": row.get("group", row.get("context", "default")),
                    "score": float(row[metric]),
                }
            )

        overall[metric] = within_subject_ci(
            records=metric_rows,
            subject_key="participant",
            condition_key="condition",
            value_key="score",
            confidence=0.95,
        )

    save_json(out_dir / "subjective_within_subject_ci.json", overall)

    flat = []
    for metric, cond_stats in overall.items():
        for condition, stats in cond_stats.items():
            flat.append(
                {
                    "metric": metric,
                    "condition": condition,
                    "mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "std": stats["std"],
                    "n": stats["n"],
                }
            )

    save_csv(out_dir / "subjective_within_subject_ci.csv", flat)

    headers = ["Metric", "Condition", "Mean", "95% CI"]
    rows_md = [
        [
            r["metric"],
            r["condition"],
            f"{r['mean']:.3f}",
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
        ]
        for r in flat
    ]
    save_markdown_table(out_dir / "subjective_within_subject_ci.md", headers, rows_md)

    return overall
