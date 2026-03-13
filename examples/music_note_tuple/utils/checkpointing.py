# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from data.data import get_vocab_sizes
from model import Transformer
from omegaconf import OmegaConf, open_dict
from torch import nn


@dataclass
class WorkDirectory:
    root: Path = field(metadata={"help": "Root work directory"})
    checkpoint: Path = field(metadata={"help": "Checkpoint directory"})
    samples: Path = field(metadata={"help": "Samples directory"})


def _resolve_root_and_checkpoint_path(work_dir: str) -> tuple[Path, Path]:
    path = Path(work_dir)

    if path.is_file():
        return path.parents[1], path

    root = path
    checkpoint = root / "checkpoints" / "checkpoint.pth"
    return root, checkpoint


def load_cfg_from_path(work_dir: str) -> OmegaConf:
    root_dir, _ = _resolve_root_and_checkpoint_path(work_dir)
    cfg_path = root_dir / ".hydra" / "config.yaml"
    return OmegaConf.load(cfg_path)


def load_model_from_path(work_dir: str, device: torch.device) -> nn.Module:
    root_dir, ckpt_path = _resolve_root_and_checkpoint_path(work_dir)
    cfg = load_cfg_from_path(str(root_dir))
    with open_dict(cfg):
        cfg.model.enable_edit_flow = bool(getattr(cfg.flow, "use_edit_flow", False))
        cfg.model.enable_velocity = (
            str(getattr(cfg.flow, "parameterization", "posterior")) == "velocity"
        )

    vocab_sizes = get_vocab_sizes(cfg)
    model = Transformer(config=cfg.model, vocab_sizes=vocab_sizes).to(device)

    loaded_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(loaded_state["model"])

    return model


def get_work_dirs(work_dir: str, rank: int) -> WorkDirectory:
    root = Path(work_dir)
    sample_dir = root / "samples"
    checkpoint_dir = root / "checkpoints" / "checkpoint.pth"

    if rank == 0:
        sample_dir.mkdir(exist_ok=True)
        checkpoint_dir.parents[0].mkdir(exist_ok=True)

    return WorkDirectory(root=root, checkpoint=checkpoint_dir, samples=sample_dir)
