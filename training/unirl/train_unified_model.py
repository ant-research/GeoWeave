#!/usr/bin/env python
"""UniRL v2 HunyuanImage3 training entry point (Hydra-native).

Thin wrapper around :class:`unirl.trainer.unified_model.UnifiedModelTrainer`. The trainer
owns the placement scope, sibling Remote wiring, and the ``train_step → train``
loop; this module just maps the loaded Hydra config blocks to constructor
kwargs.

Pairs with ``examples/unified_model/hi3_vllmomni.yaml``::

    python -m unirl.train_unified_model --config-name unified_model/hi3_vllmomni
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from unirl.trainer.unified_model import UnifiedModelTrainer


def _optimizer_steps(
    num_rollouts: int,
    gradient_accumulation_steps: int = 1,
    num_updates_per_batch: int = 1,
) -> int:
    """Resolve scheduler length in optimizer-step units from rollout settings."""
    rollouts = int(num_rollouts)
    accumulation = int(gradient_accumulation_steps)
    updates = int(num_updates_per_batch)
    if rollouts < 1 or accumulation < 1 or updates < 1:
        raise ValueError(
            "optimizer_steps requires positive num_rollouts, "
            "gradient_accumulation_steps, and num_updates_per_batch"
        )
    if accumulation > 1 and updates > 1:
        raise ValueError(
            "gradient_accumulation_steps>1 requires num_updates_per_batch=1"
        )
    if accumulation > 1:
        return (rollouts + accumulation - 1) // accumulation
    return rollouts * updates


OmegaConf.register_new_resolver(
    "optimizer_steps",
    _optimizer_steps,
    replace=True,
)


@hydra.main(version_base=None, config_path="../examples", config_name="unified_model/hi3_vllmomni")
def main(cfg: DictConfig) -> None:
    trainer = UnifiedModelTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        ar_rollout_cfg=cfg.get("ar_rollout"),
        dit_rollout_cfg=cfg.get("dit_rollout"),
        rollout_cfg=cfg.get("rollout"),
        reward_cfg=cfg.reward,
        ar_algorithm_cfg=cfg.algorithm.ar,
        image_algorithm_cfg=cfg.algorithm.image,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        sync_cfg=cfg.get("sync"),
        dump_dir=cfg.get("dump_dir"),
        dump_async=cfg.get("dump_async", True),
        dump_image_workers=cfg.get("dump_image_workers", 8),
        dump_jpeg_quality=cfg.get("dump_jpeg_quality", 90),
        dump_max_pending=cfg.get("dump_max_pending", 1),
        logging_cfg=cfg.get("logging"),
        enable_fsdp_offload=cfg.get("enable_fsdp_offload", True),
        stage_config=cfg.get("stage_config"),
    )
    trainer.train(
        num_rollouts=cfg.get("num_rollouts", 100),
        weight_sync_interval=cfg.get("weight_sync_interval", 1),
        save_interval=cfg.get("save_interval", 0),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=cfg.get("save_mode", "auto"),
    )


if __name__ == "__main__":
    main()
