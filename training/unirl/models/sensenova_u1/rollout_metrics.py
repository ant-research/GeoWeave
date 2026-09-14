"""Lightweight rollout metrics for GeoWeave.

Metrics are emitted without distributed collectives: every rank writes one
atomic JSON file per local rollout step, and whichever rank observes all rank
files writes the aggregate summary. Set ``SENSENOVA_ROLLOUT_METRICS_DIR`` to an
empty string to disable file output.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_COUNTER_LOCK = threading.Lock()
_ROLLOUT_COUNTER = 0


def next_rollout_step() -> int:
    global _ROLLOUT_COUNTER
    with _COUNTER_LOCK:
        step = _ROLLOUT_COUNTER
        _ROLLOUT_COUNTER += 1
    return step


def distributed_rank_world() -> Tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def gpu_utilization(device: torch.device) -> Optional[float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        return float(torch.cuda.utilization(device))
    except Exception:
        return None


@dataclass
class SampleRolloutMetrics:
    rollout_id: str
    prompt_id: str
    group_id: str
    rewrite_id: int
    rank: int
    local_sample_index: int
    prompt_tokens: int = 0
    generated_text_tokens: int = 0
    generated_images: int = 0
    image_context_tokens: int = 0
    image_sizes: List[Tuple[int, int]] = field(default_factory=list)
    prefix_time: float = 0.0
    text_decode_time: float = 0.0
    diffusion_time: float = 0.0
    image_reencode_time: float = 0.0
    total_time: float = 0.0
    diffusion_forwards: int = 0
    text_forward_calls_share: float = 0.0
    text_forward_rows: int = 0
    text_cfg_forward_calls_share: float = 0.0
    text_cfg_forward_rows: int = 0
    diffusion_forward_calls_share: float = 0.0
    diffusion_forward_rows: int = 0
    reencode_forward_calls_share: float = 0.0
    reencode_forward_rows: int = 0
    reencode_cfg_forward_calls_share: float = 0.0
    reencode_cfg_forward_rows: int = 0
    continuous_refill_count: float = 0.0
    continuous_coalesce_calls: float = 0.0
    continuous_ready_groups: float = 0.0
    continuous_compatible_groups: float = 0.0
    continuous_underfilled_slots: float = 0.0
    continuous_refilled_slots: float = 0.0
    continuous_incompatible_groups: float = 0.0
    stop_reason: str = "max_new_tokens"


@dataclass
class RankRolloutMetrics:
    run_id: str
    rollout_step: int
    rank: int
    world_size: int
    rank_total_time: float
    rank_total_tokens: int
    rank_total_images: int
    rank_total_image_pixels: int
    rank_total_diffusion_forwards: int
    rank_max_sample_time: float
    tokens_per_second: float
    images_per_second: float
    diffusion_forwards_per_second: float
    text_decode_time: float
    diffusion_time: float
    image_reencode_time: float
    tokens_per_text_decode_second: float
    images_per_diffusion_second: float
    images_per_reencode_second: float
    gpu_utilization: Optional[float]
    samples: List[Dict[str, Any]]


def build_rank_metrics(
    *, rollout_step: int, elapsed: float, samples: List[SampleRolloutMetrics],
    gpu_util: Optional[float],
) -> RankRolloutMetrics:
    rank, world_size = distributed_rank_world()
    tokens = sum(s.generated_text_tokens for s in samples)
    images = sum(s.generated_images for s in samples)
    pixels = sum(h * w for s in samples for h, w in s.image_sizes)
    forwards = sum(s.diffusion_forwards for s in samples)
    text_time = sum(s.text_decode_time for s in samples)
    diffusion_time = sum(s.diffusion_time for s in samples)
    reencode_time = sum(s.image_reencode_time for s in samples)
    denom = max(elapsed, 1e-12)
    run_id = os.environ.get(
        "SENSENOVA_ROLLOUT_RUN_ID",
        os.environ.get("TORCHELASTIC_RUN_ID", f"pid-{os.getpid()}"),
    )
    return RankRolloutMetrics(
        run_id=run_id, rollout_step=rollout_step, rank=rank, world_size=world_size,
        rank_total_time=elapsed, rank_total_tokens=tokens,
        rank_total_images=images, rank_total_image_pixels=pixels,
        rank_total_diffusion_forwards=forwards,
        rank_max_sample_time=max((s.total_time for s in samples), default=0.0),
        tokens_per_second=tokens / denom, images_per_second=images / denom,
        diffusion_forwards_per_second=forwards / denom,
        text_decode_time=text_time,
        diffusion_time=diffusion_time,
        image_reencode_time=reencode_time,
        tokens_per_text_decode_second=tokens / max(text_time, 1e-12),
        images_per_diffusion_second=images / max(diffusion_time, 1e-12),
        images_per_reencode_second=images / max(reencode_time, 1e-12),
        gpu_utilization=gpu_util, samples=[asdict(s) for s in samples],
    )


def aggregate_rank_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rank_times = [float(r["rank_total_time"]) for r in records]
    max_time = max(rank_times, default=0.0)
    mean_time = sum(rank_times) / len(rank_times) if rank_times else 0.0
    total_time_capacity = len(rank_times) * max_time
    return {
        "rollout_step": records[0]["rollout_step"] if records else -1,
        "world_size": len(records),
        "rank_times": rank_times,
        "imbalance_ratio": max_time / mean_time if mean_time > 0 else 0.0,
        "straggler_waste": (
            1.0 - sum(rank_times) / total_time_capacity
            if total_time_capacity > 0 else 0.0
        ),
        "total_tokens": sum(int(r["rank_total_tokens"]) for r in records),
        "total_images": sum(int(r["rank_total_images"]) for r in records),
        "total_image_pixels": sum(int(r["rank_total_image_pixels"]) for r in records),
        "total_diffusion_forwards": sum(
            int(r["rank_total_diffusion_forwards"]) for r in records
        ),
    }


def emit_rank_metrics(metrics: RankRolloutMetrics) -> None:
    payload = asdict(metrics)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[DEBUG][ROLLOUT][RANK] %s",
            json.dumps(payload, sort_keys=True),
        )
    metrics_dir = os.environ.get(
        "SENSENOVA_ROLLOUT_METRICS_DIR", "/tmp/geoweave_rollout_metrics"
    )
    if not metrics_dir:
        return
    safe_run_id = metrics.run_id.replace("/", "_")
    root = Path(metrics_dir) / safe_run_id
    root.mkdir(parents=True, exist_ok=True)
    stem = f"step_{metrics.rollout_step:06d}"
    rank_path = root / f"{stem}_rank_{metrics.rank:04d}.json"
    tmp_path = rank_path.with_suffix(f".tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(tmp_path, rank_path)

    rank_paths = [root / f"{stem}_rank_{rank:04d}.json" for rank in range(metrics.world_size)]
    if not all(path.exists() for path in rank_paths):
        return
    try:
        records = [json.loads(path.read_text()) for path in rank_paths]
        aggregate = aggregate_rank_metrics(records)
        summary_path = root / f"{stem}_summary.json"
        summary_tmp = summary_path.with_suffix(f".tmp.{os.getpid()}")
        summary_tmp.write_text(json.dumps(aggregate, sort_keys=True) + "\n")
        os.replace(summary_tmp, summary_path)
        rank_times = aggregate["rank_times"]
        logger.info(
            "[RUN][ROLLOUT] step=%d ranks=%d tokens=%d images=%d "
            "elapsed_s=%.1f imbalance=%.2f idle=%.1f%%",
            aggregate["rollout_step"],
            aggregate["world_size"],
            aggregate["total_tokens"],
            aggregate["total_images"],
            max(rank_times, default=0.0),
            aggregate["imbalance_ratio"],
            100.0 * aggregate["straggler_waste"],
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[DEBUG][ROLLOUT][AGGREGATE] %s",
                json.dumps(aggregate, sort_keys=True),
            )
    except (OSError, ValueError, json.JSONDecodeError):
        logger.exception("Failed to aggregate GeoWeave rollout metrics")


__all__ = [
    "SampleRolloutMetrics", "RankRolloutMetrics", "aggregate_rank_metrics",
    "build_rank_metrics", "distributed_rank_world", "emit_rank_metrics",
    "gpu_utilization", "next_rollout_step",
]
