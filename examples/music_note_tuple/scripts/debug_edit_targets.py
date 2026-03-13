# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import data
from data.note_tuple import ATTR_NAMES
from logic import flow
from logic.edit_flow import derive_edit_targets
from utils import checkpointing


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_single_process_dist() -> bool:
    if not dist.is_available() or dist.is_initialized():
        return False
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug EditFlow target ratios (insert/delete/substitute)."
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Optional run directory to load config from .hydra/config.yaml",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    initialized_dist = _ensure_single_process_dist()

    try:
        if args.work_dir is not None:
            cfg = checkpointing.load_cfg_from_path(args.work_dir)
        else:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")

        cfg.training.batch_size = int(args.batch_size)
        cfg.data.num_workers = 0
        cfg.compute.ngpus = 1

        vocab_sizes = data.get_vocab_sizes(config=cfg)
        source_distribution = flow.get_source_distribution(
            source_distribution=cfg.flow.source_distribution,
            vocab_sizes=vocab_sizes,
            include_pad=cfg.data.source_include_pad,
            pitch_pad_prob=getattr(cfg.flow, "source_pitch_pad_prob", None),
        )
        path = flow.get_path(flow_cfg=cfg.flow)

        data_state = data.get_data_state(config=cfg)
        train_iter, _ = data.get_data_loaders(config=cfg, data_state=data_state)
        batch = next(train_iter)

        x_1 = {
            attr: batch[attr].to(device=device, dtype=torch.long) for attr in ATTR_NAMES
        }
        x_0 = source_distribution.sample_like(x_1)
        t = torch.rand(x_1["pitch"].shape[0], device=device) * (
            1.0 - float(getattr(cfg.flow, "time_epsilon", 1e-3))
        )
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
        inactive = path_sample.x_t["pitch"] == 0
        for attr in ATTR_NAMES:
            if attr == "pitch":
                continue
            path_sample.x_t[attr][inactive] = 0

        targets = derive_edit_targets(x_t=path_sample.x_t, x_1=x_1)

        print("==== Edit Target Debug ====")
        print(f"source_distribution: {cfg.flow.source_distribution}")
        print(
            f"source_pitch_pad_prob: {getattr(cfg.flow, 'source_pitch_pad_prob', None)}"
        )
        print(f"batch_size: {x_1['pitch'].shape[0]}, seq_len: {x_1['pitch'].shape[1]}")
        print(
            f"active_t_mean: {(path_sample.x_t['pitch'] != 0).float().mean().item():.6f}"
        )
        print(f"active_1_mean: {(x_1['pitch'] != 0).float().mean().item():.6f}")
        print(f"insert_mean: {targets.insert.mean().item():.6f}")
        print(f"delete_mean: {targets.delete.mean().item():.6f}")
        print(f"substitute_mean: {targets.substitute.mean().item():.6f}")
    finally:
        if initialized_dist and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
