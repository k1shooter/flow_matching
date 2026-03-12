# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

import torch.multiprocessing as mp

try:
    import hydra
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency: hydra-core. Install with "
        "`pip install hydra-core omegaconf hydra-submitit-launcher`."
    ) from exc

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig

from train import run_mp_training


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    work_dir = (
        hydra_cfg.run.dir
        if hydra_cfg.mode == RunMode.RUN
        else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    )
    os.makedirs(work_dir, exist_ok=True)

    with open_dict(cfg):
        cfg.work_dir = work_dir

    port = int(cfg.compute.port)

    if cfg.compute.ngpus == 1:
        run_mp_training(rank=0, world_size=1, cfg=cfg, port=port)
    else:
        mp.set_start_method("forkserver")
        mp.spawn(
            run_mp_training,
            args=(cfg.compute.ngpus, cfg, port),
            nprocs=cfg.compute.ngpus,
            join=True,
        )


if __name__ == "__main__":
    main()
