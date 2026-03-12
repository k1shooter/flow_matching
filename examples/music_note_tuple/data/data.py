# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
import zipfile

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import urlretrieve

import torch

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - optional for lightweight unit tests
    OmegaConf = Any  # type: ignore
from torch.utils.data import DataLoader, Dataset as TorchDataset

from .note_tuple import (
    ATTR_NAMES,
    get_note_tuple_vocab_sizes,
    midi_file_to_segment_tuples,
    NoteTupleMetadata,
)
from .utils import cycle_loader, StatefulDistributedSampler


SUPPORTED_MIDI_SUFFIXES = (".mid", ".midi")
PREPROCESS_CACHE_FORMAT_VERSION = 1
DEFAULT_DOWNLOAD_CACHE_ROOT = "~/.cache/flow_matching_music_note_tuple/datasets"
DEFAULT_PREPROCESS_CACHE_DIR = "~/.cache/flow_matching_music_note_tuple/preprocessed"


def get_note_tuple_metadata(config: OmegaConf) -> NoteTupleMetadata:
    return NoteTupleMetadata(
        beats_per_bar=config.data.beats_per_bar,
        steps_per_beat=config.data.steps_per_beat,
        segment_bars=config.data.segment_bars,
        dur_max_steps=config.data.dur_max_steps,
        vel_bins=config.data.vel_bins,
        num_tracks=config.data.num_tracks,
        max_notes_per_segment=config.data.max_notes_per_segment,
        default_tempo=float(config.data.default_tempo),
    )


def get_vocab_sizes(config: OmegaConf) -> Dict[str, int]:
    return get_note_tuple_vocab_sizes(get_note_tuple_metadata(config))


def _optional_path(path_value: object) -> Optional[Path]:
    if path_value is None:
        return None

    path_str = str(path_value).strip()
    if path_str == "" or path_str.lower() in {"none", "null"}:
        return None

    return Path(path_str).expanduser()


def _optional_string(value: object) -> Optional[str]:
    if value is None:
        return None

    value_str = str(value).strip()
    if value_str == "" or value_str.lower() in {"none", "null"}:
        return None

    return value_str


def _scan_midi_files(root: Path, recursive: bool = True) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"MIDI directory does not exist: {root}")

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

    files = sorted(files)
    if not files:
        raise ValueError(f"No MIDI files found in {root}.")

    return files


def _sha256sum(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_file(url: str, out_path: Path, force_download: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force_download:
        return

    print(f"[data] Downloading dataset archive: {url}")
    urlretrieve(url, out_path)
    print(f"[data] Saved archive to: {out_path}")


def _extract_archive(archive_path: Path, out_dir: Path, force_extract: bool) -> None:
    if out_dir.exists() and not force_extract and any(out_dir.iterdir()):
        return

    if out_dir.exists() and force_extract:
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] Extracting archive to: {out_dir}")

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zipf:
            zipf.extractall(out_dir)
        return

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tarf:
            tarf.extractall(out_dir)
        return

    raise ValueError(
        f"Unsupported dataset archive format for {archive_path}. "
        "Use .zip or .tar(.gz/.bz2/.xz)."
    )


def _resolve_auto_download_midi_dir(config: OmegaConf) -> Optional[Path]:
    auto_cfg = getattr(config.data, "auto_download", None)
    if auto_cfg is None or not bool(getattr(auto_cfg, "enabled", False)):
        return None

    dataset_url = str(getattr(auto_cfg, "url", "")).strip()
    if dataset_url == "":
        raise ValueError(
            "data.auto_download.enabled=true but data.auto_download.url is empty."
        )

    dataset_name = (
        str(getattr(auto_cfg, "dataset_name", "dataset")).strip() or "dataset"
    )
    archive_name = (
        str(getattr(auto_cfg, "archive_name", "")).strip() or f"{dataset_name}.zip"
    )
    force_download = bool(getattr(auto_cfg, "force_download", False))
    force_extract = bool(getattr(auto_cfg, "force_extract", False))
    expected_sha256 = _optional_string(getattr(auto_cfg, "sha256", None))
    lock_timeout_seconds = float(getattr(auto_cfg, "lock_timeout_seconds", 3600.0))

    cache_root = _optional_path(
        getattr(auto_cfg, "cache_root", DEFAULT_DOWNLOAD_CACHE_ROOT)
    )
    if cache_root is None:
        raise ValueError("data.auto_download.cache_root must not be empty.")

    archive_path = cache_root / "archives" / archive_name
    extract_root = cache_root / "extracted" / dataset_name
    lock_path = cache_root / f"{dataset_name}.download.lock"

    with _file_lock(lock_path=lock_path, timeout_seconds=lock_timeout_seconds):
        _download_file(
            url=dataset_url, out_path=archive_path, force_download=force_download
        )

        if expected_sha256 is not None:
            checksum = _sha256sum(archive_path)
            if checksum.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"Dataset archive checksum mismatch. expected={expected_sha256}, got={checksum}"
                )

        _extract_archive(
            archive_path=archive_path,
            out_dir=extract_root,
            force_extract=force_extract,
        )

    midi_subdir = _optional_path(getattr(auto_cfg, "midi_subdir", None))
    midi_root = extract_root / midi_subdir if midi_subdir is not None else extract_root
    _scan_midi_files(midi_root, recursive=True)

    metadata_path = cache_root / f"{dataset_name}_download.json"
    metadata = {
        "dataset_name": dataset_name,
        "url": dataset_url,
        "archive_path": str(archive_path.resolve()),
        "extract_root": str(extract_root.resolve()),
        "midi_root": str(midi_root.resolve()),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[data] Auto dataset ready at: {midi_root}")
    return midi_root


def _resolve_train_valid_file_lists(config: OmegaConf) -> Tuple[List[Path], List[Path]]:
    recursive = bool(config.data.recursive)
    used_auto_download = False

    train_root = _optional_path(config.data.train_midi_dir)
    if train_root is None:
        base_root = _optional_path(config.data.midi_dir)
        if base_root is not None and base_root.exists():
            train_root = base_root
        else:
            train_root = _resolve_auto_download_midi_dir(config)
            used_auto_download = train_root is not None
            if train_root is None:
                if base_root is None:
                    raise ValueError(
                        "No MIDI directory provided. Set data.midi_dir (or data.train_midi_dir), "
                        "or enable data.auto_download."
                    )
                raise FileNotFoundError(
                    f"MIDI directory does not exist: {base_root}. "
                    "Provide a valid path or enable data.auto_download."
                )

    train_files = _scan_midi_files(
        train_root,
        recursive=True if used_auto_download else recursive,
    )

    valid_root = _optional_path(config.data.valid_midi_dir)
    if valid_root is not None:
        valid_files = _scan_midi_files(valid_root, recursive=recursive)
        return train_files, valid_files

    if len(train_files) == 1:
        return train_files, train_files

    split_mode = str(getattr(config.data, "pop909_split_mode", "random"))
    if split_mode == "deterministic_8_1_1":
        rng = Random(
            int(getattr(config.data, "pop909_split_seed", config.training.seed))
        )
        shuffled = list(train_files)
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = max(1, int(round(0.8 * n_total)))
        n_valid = max(1, int(round(0.1 * n_total)))
        n_valid = min(n_valid, n_total - n_train)
        if n_valid <= 0:
            n_valid = 1
            n_train = max(1, n_total - 1)

        train_split = sorted(shuffled[:n_train])
        valid_split = sorted(shuffled[n_train : n_train + n_valid])
        if len(valid_split) == 0:
            valid_split = sorted(shuffled[-1:])
        return train_split, valid_split

    rng = Random(int(config.training.seed))
    shuffled = list(train_files)
    rng.shuffle(shuffled)

    n_valid = max(1, int(round(len(shuffled) * float(config.data.valid_fraction))))
    n_valid = min(n_valid, len(shuffled) - 1)

    valid_files = sorted(shuffled[:n_valid])
    train_files = sorted(shuffled[n_valid:])

    return train_files, valid_files


def _build_cache_signature(
    midi_files: Sequence[Path],
    metadata: NoteTupleMetadata,
    segment_stride_bars: int,
    config: OmegaConf,
) -> Tuple[str, Dict[str, object]]:
    file_entries = []
    for midi_path in sorted(midi_files):
        stat = midi_path.stat()
        file_entries.append(
            {
                "path": str(midi_path.resolve()),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )

    payload = {
        "format_version": PREPROCESS_CACHE_FORMAT_VERSION,
        "user_cache_version": int(getattr(config.data, "preprocess_cache_version", 1)),
        "metadata": asdict(metadata),
        "segment_stride_bars": int(segment_stride_bars),
        "files": file_entries,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:20]
    return digest, payload


def _load_cache_file(cache_path: Path) -> List[Dict[str, torch.Tensor]]:
    try:
        loaded = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(cache_path, map_location="cpu")
    if isinstance(loaded, dict) and "samples" in loaded:
        return loaded["samples"]
    if isinstance(loaded, list):
        return loaded
    raise ValueError(f"Unsupported cache payload format in {cache_path}.")


@contextmanager
def _file_lock(lock_path: Path, timeout_seconds: float = 3600.0) -> Iterable[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    fd: Optional[int] = None

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            fd = None
            break
        except FileExistsError:
            if time.time() - start > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for dataset lock: {lock_path}")
            time.sleep(0.2)

    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _load_or_build_preprocessed_samples(
    midi_files: Sequence[Path],
    metadata: NoteTupleMetadata,
    segment_stride_bars: int,
    split_name: str,
    config: OmegaConf,
) -> List[Dict[str, torch.Tensor]]:
    cache_dir = _optional_path(
        getattr(config.data, "preprocess_cache_dir", DEFAULT_PREPROCESS_CACHE_DIR)
    )
    if cache_dir is None:
        raise ValueError("data.preprocess_cache_dir must not be empty.")
    cache_dir.mkdir(parents=True, exist_ok=True)

    rebuild_cache = bool(getattr(config.data, "rebuild_preprocess_cache", False))
    lock_timeout_seconds = float(
        getattr(config.data, "preprocess_cache_lock_timeout_seconds", 3600.0)
    )

    signature, signature_payload = _build_cache_signature(
        midi_files=midi_files,
        metadata=metadata,
        segment_stride_bars=segment_stride_bars,
        config=config,
    )
    cache_path = cache_dir / f"{split_name}_{signature}.pt"
    lock_path = cache_dir / f"{split_name}_{signature}.lock"
    manifest_path = cache_dir / f"{split_name}_{signature}.json"

    if cache_path.exists() and not rebuild_cache:
        print(f"[data] Loading preprocessed {split_name} cache: {cache_path}")
        return _load_cache_file(cache_path)

    with _file_lock(lock_path=lock_path, timeout_seconds=lock_timeout_seconds):
        if cache_path.exists() and not rebuild_cache:
            print(f"[data] Loading preprocessed {split_name} cache: {cache_path}")
            return _load_cache_file(cache_path)

        print(
            f"[data] Building preprocessed {split_name} cache from {len(midi_files)} MIDI files."
        )
        samples: List[Dict[str, torch.Tensor]] = []
        for midi_file in midi_files:
            samples.extend(
                midi_file_to_segment_tuples(
                    midi_path=midi_file,
                    metadata=metadata,
                    segment_stride_bars=segment_stride_bars,
                )
            )

        if len(samples) == 0:
            raise ValueError(
                f"Dataset contains zero segments after preprocessing for split='{split_name}'."
            )

        tmp_cache_path = cache_path.with_suffix(".pt.tmp")
        torch.save({"samples": samples}, tmp_cache_path)
        os.replace(tmp_cache_path, cache_path)

        manifest = {
            "split": split_name,
            "num_files": len(midi_files),
            "num_segments": len(samples),
            "cache_path": str(cache_path.resolve()),
            "signature": signature,
            "signature_payload": signature_payload,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"[data] Saved preprocessed {split_name} cache: {cache_path} "
            f"(segments={len(samples)})"
        )
        return samples


class MidiNoteTupleDataset(TorchDataset):
    def __init__(
        self,
        samples: Sequence[Dict[str, torch.Tensor]],
        metadata: NoteTupleMetadata,
        apply_pitch_transpose: bool = False,
        pitch_transpose_min: int = -5,
        pitch_transpose_max: int = 6,
    ) -> None:
        self.metadata = metadata
        self.apply_pitch_transpose = apply_pitch_transpose
        self.pitch_transpose_min = int(pitch_transpose_min)
        self.pitch_transpose_max = int(pitch_transpose_max)
        self.samples = list(samples)

        if len(self.samples) == 0:
            raise ValueError("Dataset contains zero segments.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = {key: value.clone() for key, value in self.samples[index].items()}

        if self.apply_pitch_transpose:
            shift = int(
                torch.randint(
                    low=self.pitch_transpose_min,
                    high=self.pitch_transpose_max + 1,
                    size=(1,),
                ).item()
            )
            pitch = sample["pitch"]
            non_pad = pitch > 0
            shifted = torch.clamp((pitch[non_pad] - 1) + shift, min=0, max=127) + 1
            pitch[non_pad] = shifted
            sample["pitch"] = pitch
            sample["attention_mask"] = (sample["pitch"] != 0).long()

        return sample


@dataclass
class Dataset:
    dataset: TorchDataset = field(metadata={"help": "MIDI note tuple dataset"})
    sampler: StatefulDistributedSampler = field(
        metadata={"help": "Stateful sampler for `dataset`"}
    )


@dataclass
class DataState:
    train: Dataset = field(metadata={"help": "Train dataset"})
    test: Dataset = field(metadata={"help": "Validation dataset"})


def _build_dataset(
    midi_files: Sequence[Path],
    metadata: NoteTupleMetadata,
    segment_stride_bars: int,
    split_name: str,
    config: OmegaConf,
    batch_size: int,
    ngpus: int,
    apply_pitch_transpose: bool = False,
    pitch_transpose_min: int = -5,
    pitch_transpose_max: int = 6,
) -> Dataset:
    assert batch_size % ngpus == 0, "Batch size must be divisible by number of gpus."

    samples = _load_or_build_preprocessed_samples(
        midi_files=midi_files,
        metadata=metadata,
        segment_stride_bars=segment_stride_bars,
        split_name=split_name,
        config=config,
    )

    dataset = MidiNoteTupleDataset(
        samples=samples,
        metadata=metadata,
        apply_pitch_transpose=apply_pitch_transpose,
        pitch_transpose_min=pitch_transpose_min,
        pitch_transpose_max=pitch_transpose_max,
    )

    sampler = StatefulDistributedSampler(dataset=dataset)
    return Dataset(dataset=dataset, sampler=sampler)


def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key in [*ATTR_NAMES, "attention_mask"]:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def get_data_state(config: OmegaConf) -> DataState:
    metadata = get_note_tuple_metadata(config)
    train_files, valid_files = _resolve_train_valid_file_lists(config)

    train = _build_dataset(
        midi_files=train_files,
        metadata=metadata,
        segment_stride_bars=config.data.segment_stride_bars,
        split_name="train",
        config=config,
        batch_size=config.training.batch_size,
        ngpus=config.compute.ngpus,
        apply_pitch_transpose=bool(
            getattr(config.data, "pop909_pitch_transpose_train", False)
        ),
        pitch_transpose_min=int(getattr(config.data, "pop909_pitch_transpose_min", -5)),
        pitch_transpose_max=int(getattr(config.data, "pop909_pitch_transpose_max", 6)),
    )

    test = _build_dataset(
        midi_files=valid_files,
        metadata=metadata,
        segment_stride_bars=config.data.segment_stride_bars,
        split_name="valid",
        config=config,
        batch_size=config.eval.batch_size,
        ngpus=config.compute.ngpus,
    )

    return DataState(train=train, test=test)


def get_data_loaders(
    config: OmegaConf,
    data_state: DataState,
) -> Tuple[Iterable, Iterable]:
    train_loader = cycle_loader(
        DataLoader(
            data_state.train.dataset,
            batch_size=config.training.batch_size // config.compute.ngpus,
            sampler=data_state.train.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            persistent_workers=config.data.num_workers > 0,
            collate_fn=collate_fn,
        )
    )

    valid_loader = cycle_loader(
        DataLoader(
            data_state.test.dataset,
            batch_size=config.eval.batch_size // config.compute.ngpus,
            sampler=data_state.test.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            persistent_workers=config.data.num_workers > 0,
            collate_fn=collate_fn,
        )
    )

    return iter(train_loader), iter(valid_loader)
