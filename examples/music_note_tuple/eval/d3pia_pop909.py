from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from data.note_tuple import NoteTupleMetadata

from torch import nn

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
from .stats import mean_and_ci

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAME_TO_PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "F": 5,
    "E#": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
}

CHORD_QUALITIES = ["maj", "min", "dim", "aug", "sus2", "sus4", "7", "maj7", "min7"]
CHORD_TEMPLATES = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
}


@dataclass
class D3PIAReport:
    summary: Dict[str, float]
    rows: List[Dict[str, object]]
    notes: List[str]


class ChordEmbeddingEncoder(nn.Module):
    def __init__(self, num_chords: int = 128, dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(num_chords, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, chord_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(chord_ids)
        _, h = self.gru(x)
        return h[-1]


def _parse_key_to_pitch_classes(key_name: str) -> set[int]:
    key_name = key_name.strip()
    if not key_name:
        return {0, 2, 4, 5, 7, 9, 11}

    parts = key_name.replace(" ", "").replace("_", ":").split(":")
    tonic = parts[0].upper()
    mode = parts[1].lower() if len(parts) > 1 else "maj"

    pc = NOTE_NAME_TO_PC.get(tonic, 0)
    if mode.startswith("min"):
        intervals = [0, 2, 3, 5, 7, 8, 10]
    else:
        intervals = [0, 2, 4, 5, 7, 9, 11]

    return {(pc + x) % 12 for x in intervals}


def load_key_labels(path: str | None) -> Dict[str, str]:
    if path is None:
        return {}

    rows = read_csv_rows(Path(path))
    labels = {}
    for row in rows:
        keys = list(row.keys())
        if "song_id" in row:
            song_id = row["song_id"]
        else:
            song_id = row[keys[0]]

        if "key" in row:
            key_name = row["key"]
        else:
            key_name = row[keys[1]] if len(keys) > 1 else "C:maj"

        labels[song_id] = key_name

    return labels


def _collect_note_events(
    note_tuple: Dict[str, torch.Tensor], metadata: NoteTupleMetadata
):
    events = []
    for i in range(note_tuple["pitch"].shape[0]):
        pitch = int(note_tuple["pitch"][i].item())
        bar = int(note_tuple["bar"][i].item())
        pos = int(note_tuple["pos"][i].item())
        dur = int(note_tuple["dur"][i].item())

        if pitch <= 0 or bar <= 0 or pos <= 0 or dur <= 0:
            continue

        start = (bar - 1) * metadata.steps_per_bar + (pos - 1)
        end = min(start + dur, metadata.segment_bars * metadata.steps_per_bar)
        if end <= start:
            continue

        events.append((start, end, pitch - 1))

    return events


def compute_ook_ratio(
    note_tuple: Dict[str, torch.Tensor],
    metadata: NoteTupleMetadata,
    key_name: str,
) -> float:
    key_pitch_classes = _parse_key_to_pitch_classes(key_name)

    total = 0
    oook = 0

    for _, _, pitch in _collect_note_events(note_tuple=note_tuple, metadata=metadata):
        total += 1
        if (pitch % 12) not in key_pitch_classes:
            oook += 1

    if total == 0:
        return 0.0

    return float(oook / total)


def _infer_chord_from_pitch_classes(pitch_classes: set[int]) -> str:
    if len(pitch_classes) == 0:
        return "N"

    best_root = 0
    best_quality = "maj"
    best_score = -1e9

    for root in range(12):
        for quality, template in CHORD_TEMPLATES.items():
            target = {(root + x) % 12 for x in template}
            overlap = len(target.intersection(pitch_classes))
            extra = len(pitch_classes - target)
            score = overlap - 0.25 * extra
            if score > best_score:
                best_score = score
                best_root = root
                best_quality = quality

    return f"{PITCH_CLASS_NAMES[best_root]}:{best_quality}"


def extract_chords_rule_based(
    note_tuple: Dict[str, torch.Tensor],
    metadata: NoteTupleMetadata,
    beats_per_chord: int = 1,
) -> List[str]:
    total_steps = metadata.segment_bars * metadata.steps_per_bar
    total_beats = total_steps // metadata.steps_per_beat

    beat_pitch_classes = [set() for _ in range(total_beats)]

    for start, end, pitch in _collect_note_events(
        note_tuple=note_tuple, metadata=metadata
    ):
        beat_start = start // metadata.steps_per_beat
        beat_end = max(beat_start + 1, (end - 1) // metadata.steps_per_beat + 1)
        for beat in range(beat_start, min(total_beats, beat_end)):
            beat_pitch_classes[beat].add(pitch % 12)

    chords = []
    for group_start in range(0, total_beats, beats_per_chord):
        pcs = set()
        for beat in range(group_start, min(total_beats, group_start + beats_per_chord)):
            pcs.update(beat_pitch_classes[beat])
        chords.append(_infer_chord_from_pitch_classes(pcs))

    return chords


def _chord_to_id(chord: str) -> int:
    if chord == "N":
        return 0
    try:
        root_name, quality = chord.split(":")
        root = NOTE_NAME_TO_PC[root_name.upper()]
        q_idx = CHORD_QUALITIES.index(quality)
        return 1 + q_idx * 12 + root
    except Exception:
        return 0


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    return float(F.cosine_similarity(a[None], b[None]).item())


def _load_chord_encoder(chord_encoder_ckpt: str | None, device: torch.device):
    model = ChordEmbeddingEncoder(num_chords=1 + len(CHORD_QUALITIES) * 12).to(device)
    if chord_encoder_ckpt is not None and Path(chord_encoder_ckpt).exists():
        state = torch.load(chord_encoder_ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return model.eval(), True

    return model.eval(), False


def compute_chord_similarity(
    generated_chords: Sequence[str],
    reference_chords: Sequence[str],
    beats_per_bar: int,
    beats_per_chord: int,
    encoder: ChordEmbeddingEncoder,
    device: torch.device,
) -> float:
    if len(generated_chords) == 0 or len(reference_chords) == 0:
        return 0.0

    chord_per_2bar = max(1, (2 * beats_per_bar) // beats_per_chord)

    g_ids = [_chord_to_id(c) for c in generated_chords]
    r_ids = [_chord_to_id(c) for c in reference_chords]

    n_windows = min(len(g_ids), len(r_ids)) // chord_per_2bar
    if n_windows == 0:
        return 0.0

    sims = []
    for w in range(n_windows):
        gs = torch.tensor(
            g_ids[w * chord_per_2bar : (w + 1) * chord_per_2bar],
            device=device,
            dtype=torch.long,
        )[None]
        rs = torch.tensor(
            r_ids[w * chord_per_2bar : (w + 1) * chord_per_2bar],
            device=device,
            dtype=torch.long,
        )[None]

        g_emb = encoder(gs)[0]
        r_emb = encoder(rs)[0]
        sims.append(_cosine_similarity(g_emb, r_emb))

    return float(np.mean(sims)) if sims else 0.0


def _bar_onset_vectors(
    note_tuple: Dict[str, torch.Tensor], metadata: NoteTupleMetadata
) -> np.ndarray:
    bars = metadata.segment_bars
    steps_per_bar = metadata.steps_per_bar
    vec = np.zeros((bars, steps_per_bar), dtype=np.float32)

    for i in range(note_tuple["pitch"].shape[0]):
        pitch = int(note_tuple["pitch"][i].item())
        bar = int(note_tuple["bar"][i].item())
        pos = int(note_tuple["pos"][i].item())
        if pitch <= 0 or bar <= 0 or pos <= 0:
            continue
        vec[bar - 1, pos - 1] = 1.0

    return vec


def compute_grooving_similarity(
    note_tuple: Dict[str, torch.Tensor], metadata: NoteTupleMetadata
) -> float:
    vec = _bar_onset_vectors(note_tuple=note_tuple, metadata=metadata)
    if vec.shape[0] <= 1:
        return 0.0

    sims = []
    for i in range(vec.shape[0]):
        for j in range(i + 1, vec.shape[0]):
            sims.append(1.0 - float(np.mean(np.abs(vec[i] - vec[j]))))

    if len(sims) == 0:
        return 0.0
    return float(np.mean(sims))


def evaluate_d3pia_objective(
    generated_segments: Sequence[SegmentRecord],
    reference_segments: Sequence[SegmentRecord],
    metadata: NoteTupleMetadata,
    key_labels: Dict[str, str],
    beats_per_chord: int,
    chord_encoder_ckpt: str | None,
    device: torch.device,
) -> D3PIAReport:
    n = min(len(generated_segments), len(reference_segments))
    rows = []
    notes = []

    encoder, used_pretrained_encoder = _load_chord_encoder(
        chord_encoder_ckpt=chord_encoder_ckpt,
        device=device,
    )
    if not used_pretrained_encoder:
        notes.append(
            "Chord encoder checkpoint not found; using randomly initialized fallback encoder (CS comparability risk)."
        )

    for i in range(n):
        g = generated_segments[i]
        r = reference_segments[i]

        key_name = key_labels.get(r.song_id, "C:maj")

        g_ook = compute_ook_ratio(g.note_tuple, metadata=metadata, key_name=key_name)

        g_chords = extract_chords_rule_based(
            g.note_tuple,
            metadata=metadata,
            beats_per_chord=beats_per_chord,
        )
        r_chords = extract_chords_rule_based(
            r.note_tuple,
            metadata=metadata,
            beats_per_chord=beats_per_chord,
        )

        m = min(len(g_chords), len(r_chords))
        ca = (
            float(
                np.mean([1.0 if g_chords[k] == r_chords[k] else 0.0 for k in range(m)])
            )
            if m > 0
            else 0.0
        )

        cs = compute_chord_similarity(
            generated_chords=g_chords,
            reference_chords=r_chords,
            beats_per_bar=metadata.beats_per_bar,
            beats_per_chord=beats_per_chord,
            encoder=encoder,
            device=device,
        )

        gs_gen = compute_grooving_similarity(g.note_tuple, metadata=metadata)
        gs_ref = compute_grooving_similarity(r.note_tuple, metadata=metadata)
        gs_alignment = 1.0 - abs(gs_gen - gs_ref)

        rows.append(
            {
                "segment_id": r.segment_id,
                "song_id": r.song_id,
                "OOK": g_ook,
                "CA": ca,
                "CS": cs,
                "GS_gen": gs_gen,
                "GS_ref": gs_ref,
                "GS_alignment": gs_alignment,
            }
        )

    summary = {
        "n_segments": n,
        "OOK_mean": float(np.mean([r["OOK"] for r in rows])) if rows else 0.0,
        "CA_mean": float(np.mean([r["CA"] for r in rows])) if rows else 0.0,
        "CS_mean": float(np.mean([r["CS"] for r in rows])) if rows else 0.0,
        "GS_gen_mean": float(np.mean([r["GS_gen"] for r in rows])) if rows else 0.0,
        "GS_ref_mean": float(np.mean([r["GS_ref"] for r in rows])) if rows else 0.0,
        "GS_alignment_mean": float(np.mean([r["GS_alignment"] for r in rows]))
        if rows
        else 0.0,
    }

    return D3PIAReport(summary=summary, rows=rows, notes=notes)


def write_d3pia_report(report: D3PIAReport, out_dir: Path, table_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(out_dir / "objective_summary.json", report.summary)
    save_json(out_dir / "objective_notes.json", {"notes": report.notes})
    save_csv(out_dir / "objective_segments.csv", report.rows)

    headers = ["Metric", "Value", "Direction"]
    rows = [
        ["OOK", f"{report.summary['OOK_mean']:.6f}", "down"],
        ["CA", f"{report.summary['CA_mean']:.6f}", "up"],
        ["CS", f"{report.summary['CS_mean']:.6f}", "up"],
        ["GS_gen", f"{report.summary['GS_gen_mean']:.6f}", "closer_to_GT"],
        ["GS_ref", f"{report.summary['GS_ref_mean']:.6f}", "reference"],
        ["GS_alignment", f"{report.summary['GS_alignment_mean']:.6f}", "up"],
    ]
    save_markdown_table(out_dir / f"{table_name}.md", headers=headers, rows=rows)


def create_d3pia_subjective_package(
    generated_midi_dir: Path,
    out_dir: Path,
    seed: int,
    num_excerpts: int,
    reference_midi_dir: Path | None = None,
    render_audio: bool = False,
    soundfont_path: str | None = None,
) -> Dict[str, object]:
    set_global_seed(seed)

    generated_files = scan_midi_files(generated_midi_dir)
    if len(generated_files) == 0:
        raise ValueError(f"No generated MIDI files in {generated_midi_dir}")

    n = min(num_excerpts, len(generated_files))
    selected_generated = random.sample(generated_files, n)
    selected_reference = []
    if reference_midi_dir is not None and reference_midi_dir.exists():
        refs = scan_midi_files(reference_midi_dir)
        if len(refs) > 0:
            selected_reference = random.sample(refs, min(n, len(refs)))

    pkg_dir = out_dir / "subjective_package"
    midi_dir = pkg_dir / "midi"
    audio_dir = pkg_dir / "audio"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    midi_dir.mkdir(parents=True, exist_ok=True)
    if render_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    hidden_key_rows = []

    item_counter = 0
    for midi_path in selected_generated:
        item_id = f"item_{item_counter:03d}"
        target_path = midi_dir / f"{item_id}.mid"
        target_path.write_bytes(midi_path.read_bytes())

        if render_audio:
            maybe_render_midi_to_wav(
                midi_path=target_path,
                wav_path=audio_dir / f"{item_id}.wav",
                soundfont_path=soundfont_path,
            )

        manifest_rows.append({"item_id": item_id, "path": str(target_path)})
        hidden_key_rows.append(
            {"item_id": item_id, "source": str(midi_path), "label": "generated"}
        )
        item_counter += 1

    for midi_path in selected_reference:
        item_id = f"item_{item_counter:03d}"
        target_path = midi_dir / f"{item_id}.mid"
        target_path.write_bytes(midi_path.read_bytes())

        if render_audio:
            maybe_render_midi_to_wav(
                midi_path=target_path,
                wav_path=audio_dir / f"{item_id}.wav",
                soundfont_path=soundfont_path,
            )

        manifest_rows.append({"item_id": item_id, "path": str(target_path)})
        hidden_key_rows.append(
            {"item_id": item_id, "source": str(midi_path), "label": "reference"}
        )
        item_counter += 1

    random.shuffle(manifest_rows)

    questionnaire_rows = [
        {
            "participant_id": "",
            "item_id": row["item_id"],
            "coherence": "",
            "harmony": "",
            "consistency": "",
            "correctness": "",
            "overall": "",
            "comment": "",
        }
        for row in manifest_rows
    ]

    save_csv(pkg_dir / "manifest.csv", manifest_rows)
    save_csv(pkg_dir / "hidden_key.csv", hidden_key_rows)
    save_csv(pkg_dir / "questionnaire_template.csv", questionnaire_rows)

    html = """<!doctype html>
<html>
<head><meta charset=\"utf-8\"><title>D3PIA Subjective Survey</title></head>
<body>
<h1>D3PIA Subjective Survey Template</h1>
<p>Rate each item with 1-5 Likert scale for:</p>
<ul>
<li>Coherence</li><li>Harmony</li><li>Consistency</li><li>Correctness</li><li>Overall</li>
</ul>
<p>Use questionnaire_template.csv to collect responses.</p>
</body>
</html>
"""
    (pkg_dir / "survey_template.html").write_text(html, encoding="utf-8")

    return {
        "package_dir": str(pkg_dir),
        "num_items": len(manifest_rows),
    }


def analyze_d3pia_subjective_results(
    survey_csv_path: Path,
    out_dir: Path,
) -> Dict[str, Dict[str, float]]:
    rows = read_csv_rows(survey_csv_path)
    criteria = ["coherence", "harmony", "consistency", "correctness", "overall"]
    summary = {}

    for criterion in criteria:
        values = []
        for row in rows:
            if criterion in row and row[criterion] != "":
                values.append(float(row[criterion]))
        summary[criterion] = mean_and_ci(values)

    save_json(out_dir / "subjective_summary.json", summary)

    flat_rows = []
    for criterion, stats in summary.items():
        flat_rows.append(
            {
                "criterion": criterion,
                "mean": stats["mean"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "std": stats["std"],
                "n": stats["n"],
            }
        )

    save_csv(out_dir / "subjective_summary.csv", flat_rows)

    headers = ["Criterion", "Mean", "95% CI"]
    md_rows = []
    for criterion, stats in summary.items():
        md_rows.append(
            [
                criterion,
                f"{stats['mean']:.3f}",
                f"[{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]",
            ]
        )
    save_markdown_table(out_dir / "subjective_summary.md", headers, md_rows)

    return summary
