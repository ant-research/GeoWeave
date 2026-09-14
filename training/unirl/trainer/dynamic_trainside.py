"""Driver-side dynamic scheduling for trainside single-trajectory rollouts.

The scheduler keeps rollout on-policy while supporting two kinds of refill:

* a truncated candidate slot is retried under the same prompt; when its retry
  budget is exhausted the first truncated trajectory is kept with forced
  reward zero (and is not sent to the judge);
* a completed all-zero/all-one prompt group is replaced from a reserve prompt
  pool until the target number of informative groups or a refill budget is hit.

The final train batch remains prompt-major with a fixed ``P * N`` shape.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import math
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import ray
import torch

from unirl.distributed.tensor import WorkerLocalTransport
from unirl.distributed.tensor.ref import hydrate
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack, _track_with_field
from unirl.types.segments import TextSegment
from unirl.utils.repetition import detect_tandem_token_repetition

logger = logging.getLogger(__name__)

_TRUNCATED_REASONS = frozenset({"max_new_tokens", "max_images"})


def _make_session_queue():
    from ray.util.queue import Queue

    return Queue()


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile without adding a dependency."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True)
class TrajectoryJob:
    job_index: int
    prompt_index: int
    rewrite_indices: Tuple[int, ...]
    attempt_indices: Tuple[int, ...]
    prompt_id: str
    candidate_ids: Tuple[str, ...]
    req: RolloutReq

    @property
    def trajectory_count(self) -> int:
        return len(self.rewrite_indices)


@dataclass
class InFlightJob:
    job: TrajectoryJob
    dp_rank: int
    worker_indices: List[int]
    refs: List[Any]
    started_at: float
    jobs: Tuple[TrajectoryJob, ...] = ()

    @property
    def trajectory_count(self) -> int:
        jobs = self.jobs or (self.job,)
        return sum(job.trajectory_count for job in jobs)


@dataclass
class PersistentWorkerSession:
    dp_rank: int
    worker_indices: List[int]
    input_queues: List[Any]
    output_queues: List[Any]
    refs: List[Any]


@dataclass
class CandidateSlotState:
    rewrite_index: int
    attempt_count: int = 0
    fallback_track: Optional[RolloutTrack] = None
    fallback_owner_rank: Optional[int] = None
    fallback_reason: str = ""
    selected_track: Optional[RolloutTrack] = None
    selected_owner_rank: Optional[int] = None
    selected_reason: str = ""
    forced_zero: bool = False


@dataclass
class PromptGroupState:
    prompt_index: int
    prompt_id: str
    slots: List[CandidateSlotState]
    truncated_attempts: int = 0
    track: Optional[RolloutTrack] = None
    owner_ranks: List[int] = field(default_factory=list)
    status: str = "rolling_out"

    @property
    def complete(self) -> bool:
        return all(slot.selected_track is not None for slot in self.slots)


@dataclass(frozen=True)
class DynamicRolloutResult:
    resp: RolloutResp
    owner_ranks: List[int]
    selected_prompt_indices: List[int]
    consumed_prompt_indices: List[int]
    metrics: Dict[str, float]


def _stable_trajectory_seed(
    *, base_seed: int, rollout_id: int, candidate_id: str
) -> int:
    payload = f"{int(base_seed)}:{int(rollout_id)}:{candidate_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def _dp_worker_groups(handle: Any) -> List[Tuple[int, List[int]]]:
    groups: Dict[int, List[int]] = {}
    for worker_index, rank_info in enumerate(handle.rank_infos):
        groups.setdefault(int(rank_info.dp_rank), []).append(worker_index)
    return [(dp_rank, groups[dp_rank]) for dp_rank in sorted(groups)]


def _trajectory_req(
    req: RolloutReq,
    *,
    prompt_index: int,
    rewrite_indices: Sequence[int],
    attempt_indices: Sequence[int],
    rollout_id: int,
) -> Tuple[str, Tuple[str, ...], RolloutReq]:
    ar_params = req.sampling_params.get("ar")
    if ar_params is None:
        raise TypeError("Dynamic trainside rollout requires AR sampling params.")
    if not rewrite_indices or len(rewrite_indices) != len(attempt_indices):
        raise ValueError(
            "Dynamic trajectory job requires aligned non-empty rewrite/attempt indices"
        )
    prompt_req = req.slice(prompt_index, prompt_index + 1)
    prompt_id = str(req.sample_ids[prompt_index])
    candidate_ids = []
    for rewrite_index, attempt_index in zip(rewrite_indices, attempt_indices):
        candidate_id = f"{prompt_id}/a{int(rewrite_index)}"
        if attempt_index > 0:
            candidate_id = f"{candidate_id}/attempt-{int(attempt_index)}"
        candidate_ids.append(candidate_id)

    job_ar = dataclasses.replace(ar_params, samples_per_prompt=len(candidate_ids))
    diff_params = req.sampling_params.get("diffusion")
    base_seed = int(getattr(diff_params, "seed", 0) or 0)
    init_same_noise = bool(getattr(diff_params, "init_same_noise", False))
    stage_config = dict(prompt_req.stage_config)
    trajectory_seeds = [
        _stable_trajectory_seed(
            base_seed=base_seed,
            rollout_id=rollout_id,
            candidate_id=candidate_id,
        )
        for candidate_id in candidate_ids
    ]
    stage_config.update(
        {
            "disable_rollout_metrics": True,
            "trajectory_candidate_ids": list(candidate_ids),
            "trajectory_rewrite_indices": [int(index) for index in rewrite_indices],
        }
    )
    continuous_batching = bool(stage_config.get("continuous_batching", False))
    requested_text_batch_size = max(
        1, int(stage_config.get("continuous_text_batch_size", len(candidate_ids)))
    )
    active_text_batch_size = (
        min(len(candidate_ids), requested_text_batch_size)
        if continuous_batching
        else len(candidate_ids)
    )
    if len(candidate_ids) == 1:
        stage_config.update(
            {
                "trajectory_seed": trajectory_seeds[0],
                "rollout_text_batch_size": 1,
            }
        )
        stage_config.pop("trajectory_seeds", None)
    else:
        stage_config.update(
            {
                "trajectory_seeds": trajectory_seeds,
                "rollout_text_batch_size": active_text_batch_size,
            }
        )
        stage_config.pop("trajectory_seed", None)
    noise_ids = [
        prompt_id if init_same_noise else candidate_id for candidate_id in candidate_ids
    ]
    noise_group_ids = (
        [f"r{int(rollout_id)}:{noise_id}" for noise_id in noise_ids]
        if diff_params is not None and getattr(diff_params, "seed", None) is not None
        else []
    )
    job_req = RolloutReq(
        sample_ids=list(prompt_req.sample_ids),
        group_ids=list(prompt_req.group_ids),
        primitives=dict(prompt_req.primitives),
        request_conditions=dict(prompt_req.request_conditions),
        sampling_params={**prompt_req.sampling_params, "ar": job_ar},
        stage_config=stage_config,
        sigmas=prompt_req.sigmas,
        metadata=list(prompt_req.metadata) if prompt_req.metadata else [],
        init_noise_group_ids=noise_group_ids,
        init_noise_latent_shape=prompt_req.init_noise_latent_shape,
    )
    return prompt_id, tuple(candidate_ids), job_req


def _merge_session_jobs(
    jobs: Sequence[TrajectoryJob], *, text_batch_size: int
) -> RolloutReq:
    """Merge single-trajectory jobs into one model-side admission window."""
    if not jobs:
        raise ValueError("cannot merge an empty persistent-session job window")
    if any(job.trajectory_count != 1 for job in jobs):
        raise ValueError("request admission windows require single-trajectory jobs")

    merged = RolloutReq.concat([job.req for job in jobs])
    first_ar = merged.sampling_params.get("ar")
    if first_ar is None:
        raise TypeError("request admission window requires AR sampling params")
    merged.sampling_params = {
        **merged.sampling_params,
        "ar": dataclasses.replace(first_ar, samples_per_prompt=1),
    }

    stage_config = dict(jobs[0].req.stage_config)
    candidate_ids: List[str] = []
    rewrite_indices: List[int] = []
    trajectory_seeds: List[int] = []
    for job in jobs:
        candidate_ids.extend(job.candidate_ids)
        rewrite_indices.extend(int(index) for index in job.rewrite_indices)
        job_cfg = job.req.stage_config
        if "trajectory_seed" in job_cfg:
            trajectory_seeds.append(int(job_cfg["trajectory_seed"]))
        else:
            seeds = list(job_cfg.get("trajectory_seeds", []))
            if len(seeds) != 1:
                raise ValueError(
                    "single-trajectory admission job is missing trajectory_seed"
                )
            trajectory_seeds.append(int(seeds[0]))
    stage_config.update(
        {
            "trajectory_candidate_ids": candidate_ids,
            "trajectory_rewrite_indices": rewrite_indices,
            "trajectory_seeds": trajectory_seeds,
            "rollout_text_batch_size": max(1, int(text_batch_size)),
            "continuous_request_admission": True,
        }
    )
    stage_config.pop("trajectory_seed", None)
    merged.stage_config = stage_config
    return merged


def _single_trajectory_req(
    req: RolloutReq,
    *,
    prompt_index: int,
    rewrite_index: int,
    attempt_index: int,
    rollout_id: int,
) -> Tuple[str, str, RolloutReq]:
    prompt_id, candidate_ids, job_req = _trajectory_req(
        req,
        prompt_index=prompt_index,
        rewrite_indices=[rewrite_index],
        attempt_indices=[attempt_index],
        rollout_id=rollout_id,
    )
    return prompt_id, candidate_ids[0], job_req


def _build_jobs(req: RolloutReq, *, rollout_id: int) -> Tuple[List[TrajectoryJob], int]:
    """Build the initial, attempt-0 jobs (kept as a test/debug helper)."""
    texts = req.primitives.get("text")
    if not isinstance(texts, Texts):
        raise TypeError(
            "Dynamic trainside rollout requires req.primitives['text'] to be Texts."
        )
    ar_params = req.sampling_params.get("ar")
    if ar_params is None:
        raise TypeError("Dynamic trainside rollout requires AR sampling params.")
    candidates_per_prompt = int(ar_params.samples_per_prompt)
    if candidates_per_prompt < 1:
        raise ValueError("AR samples_per_prompt must be >= 1")
    jobs: List[TrajectoryJob] = []
    for prompt_index in range(len(req.sample_ids)):
        for rewrite_index in range(candidates_per_prompt):
            prompt_id, candidate_id, job_req = _single_trajectory_req(
                req,
                prompt_index=prompt_index,
                rewrite_index=rewrite_index,
                attempt_index=0,
                rollout_id=rollout_id,
            )
            jobs.append(
                TrajectoryJob(
                    job_index=len(jobs),
                    prompt_index=prompt_index,
                    rewrite_indices=(rewrite_index,),
                    attempt_indices=(0,),
                    prompt_id=prompt_id,
                    candidate_ids=(candidate_id,),
                    req=job_req,
                )
            )
    return jobs, candidates_per_prompt




def _pop_live_microbundle(
    queue: Deque[TrajectoryJob], *, max_size: int
) -> List[TrajectoryJob]:
    """Pop one atomic same-prompt bundle from the live admission queue."""
    if not queue:
        return []
    jobs = [queue.popleft()]
    while (
        queue
        and len(jobs) < max(1, int(max_size))
        and queue[0].prompt_id == jobs[0].prompt_id
    ):
        jobs.append(queue.popleft())
    return jobs


def _collect_one(handle: Any, rec: InFlightJob) -> RolloutResp:
    raw_results = ray.get(rec.refs)
    worker_local = issubclass(handle.pool.transport_cls, WorkerLocalTransport)
    rebound = [
        handle._rebind_tree(
            result, handle.workers[worker_index], worker_local=worker_local
        )
        for result, worker_index in zip(raw_results, rec.worker_indices)
    ]
    heads = []
    for result, worker_index in zip(rebound, rec.worker_indices):
        rank_info = handle.rank_infos[worker_index]
        if (
            rank_info.tp_rank == 0
            and rank_info.is_pipeline_last_stage
            and rank_info.sp_rank == 0
        ):
            heads.append(result)
    if len(heads) != 1:
        raise RuntimeError(
            f"Dynamic trainside DP group {rec.dp_rank} produced {len(heads)} head results; expected 1."
        )
    return heads[0]


def _collect_session_one(handle: Any, rec: InFlightJob) -> RolloutResp:
    raw_results = []
    for future in rec.refs:
        status, request_id, payload = future.result()
        if request_id != rec.job.job_index:
            raise RuntimeError(
                f"Persistent rollout session returned request {request_id}, "
                f"expected {rec.job.job_index}."
            )
        if status != "ok":
            raise RuntimeError(
                f"Persistent rollout session request {request_id} failed: {payload}"
            )
        raw_results.append(payload)
    worker_local = issubclass(handle.pool.transport_cls, WorkerLocalTransport)
    rebound = [
        handle._rebind_tree(
            result, handle.workers[worker_index], worker_local=worker_local
        )
        for result, worker_index in zip(raw_results, rec.worker_indices)
    ]
    heads = []
    for result, worker_index in zip(rebound, rec.worker_indices):
        rank_info = handle.rank_infos[worker_index]
        if (
            rank_info.tp_rank == 0
            and rank_info.is_pipeline_last_stage
            and rank_info.sp_rank == 0
        ):
            heads.append(result)
    if len(heads) != 1:
        raise RuntimeError(
            f"Persistent trainside DP group {rec.dp_rank} produced "
            f"{len(heads)} head results; expected 1."
        )
    return heads[0]


def _rewrite_job_track_identity(resp: RolloutResp, job: TrajectoryJob) -> RolloutTrack:
    if set(resp.tracks) != {"ar"}:
        raise RuntimeError(
            "Dynamic trainside scheduling currently requires a single 'ar' track; "
            f"got {sorted(resp.tracks)}"
        )
    track = resp.tracks["ar"]
    if track.batch_size != job.trajectory_count:
        raise RuntimeError(
            f"Dynamic trajectory RPC returned track batch_size={track.batch_size}, "
            f"expected {job.trajectory_count}"
        )
    track.sample_ids = list(job.candidate_ids)
    track.parent_ids = [job.prompt_id] * job.trajectory_count
    return track


def _stop_reason(track: RolloutTrack) -> str:
    for condition in dict(track.conditions or {}).values():
        reasons = getattr(condition, "stop_reasons", None)
        if isinstance(reasons, list) and reasons:
            return str(reasons[0])
    return "unknown"


def _set_stop_reason(track: RolloutTrack, reason: str) -> RolloutTrack:
    """Copy condition containers and replace their single-row stop reason."""
    updated = copy.copy(track)
    conditions: Dict[str, Any] = {}
    for name, condition in dict(track.conditions or {}).items():
        copied = copy.copy(condition)
        reasons = list(getattr(condition, "stop_reasons", []) or [])
        if reasons:
            reasons[0] = str(reason)
            copied.stop_reasons = reasons
        conditions[name] = copied
    updated.conditions = conditions
    return updated


def _step_aligned_prefix_token_count(
    *,
    text: str,
    decoded_starts: torch.Tensor,
    decoded_ends: torch.Tensor,
    detected_cutoff_token: int,
    require_complete_step: bool,
) -> int:
    """Move a repetition cutoff back to the last complete SCA reasoning unit."""
    cutoff = max(0, min(int(detected_cutoff_token), int(decoded_ends.numel())))
    if cutoff <= 0 or not require_complete_step:
        return cutoff

    boundary_char = (
        int(decoded_starts[cutoff].item())
        if cutoff < int(decoded_starts.numel())
        else len(text)
    )
    from unirl.reward.local.sca_genprm import split_sca_steps

    complete_ends = [
        int(step["char_end"])
        for step in split_sca_steps(text)
        if int(step.get("char_end", 0)) <= boundary_char
    ]
    if not complete_ends:
        return 0
    complete_char_end = max(complete_ends)
    # Exclude a token crossing the selected character boundary. Empty-span
    # special tokens are excluded later by SCA's visibility mask.
    eligible = (decoded_ends <= complete_char_end) & (decoded_ends > decoded_starts)
    indices = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    return int(indices[-1].item()) + 1 if indices.numel() else 0


def _recover_repetition_truncation(
    track: RolloutTrack,
    *,
    enabled: bool,
    min_block_tokens: int,
    max_block_tokens: int,
    min_repeats: int,
    min_prefix_tokens: int,
    tail_tolerance_tokens: int,
    require_complete_step: bool,
) -> RolloutTrack:
    """Expose and train only a safe prefix of a repeated max-length response."""
    if track.batch_size != 1:
        raise ValueError("repetition recovery expects a single-sample track")
    segment = track.segment
    token_count = 0
    if isinstance(segment, TextSegment) and segment.tokens is not None:
        token_count = int(hydrate(segment.tokens).reshape(-1).numel())
        if segment.loss_mask is None:
            updated_segment = copy.copy(segment)
            updated_segment.loss_mask = torch.ones(token_count, dtype=torch.float32)
            track = _track_with_field(track, "segment", updated_segment)
            segment = updated_segment
    original_truncated = float(_stop_reason(track) == "max_new_tokens")

    def attach_metadata(
        current: RolloutTrack, *, recovered: bool, effective: int, removed: int, period: int
    ) -> RolloutTrack:
        current = _track_with_field(
            current, "repetition_recovered", torch.tensor([float(recovered)])
        )
        current = _track_with_field(
            current, "original_text_truncated", torch.tensor([original_truncated])
        )
        current = _track_with_field(
            current, "effective_text_token_count", torch.tensor([float(effective)])
        )
        current = _track_with_field(
            current, "removed_repetition_token_count", torch.tensor([float(removed)])
        )
        return _track_with_field(
            current, "repetition_period_token_count", torch.tensor([float(period)])
        )

    if not enabled or not original_truncated or not isinstance(segment, TextSegment):
        return attach_metadata(
            track, recovered=False, effective=token_count, removed=0, period=0
        )
    if (
        segment.tokens is None
        or segment.decoded_char_starts is None
        or segment.decoded_char_ends is None
        or not isinstance(track.decoded, Texts)
        or len(track.decoded.texts) != 1
    ):
        logger.warning(
            "Repetition recovery skipped sample=%s: missing tokens/text/character spans",
            track.sample_ids[0] if track.sample_ids else "unknown",
        )
        return attach_metadata(
            track, recovered=False, effective=token_count, removed=0, period=0
        )

    tokens = hydrate(segment.tokens).reshape(-1).to(torch.long).cpu()
    starts = hydrate(segment.decoded_char_starts).reshape(-1).to(torch.long).cpu()
    ends = hydrate(segment.decoded_char_ends).reshape(-1).to(torch.long).cpu()
    if starts.numel() != tokens.numel() or ends.numel() != tokens.numel():
        logger.warning(
            "Repetition recovery skipped sample=%s: token/span length mismatch",
            track.sample_ids[0] if track.sample_ids else "unknown",
        )
        return attach_metadata(
            track, recovered=False, effective=token_count, removed=0, period=0
        )

    loop = detect_tandem_token_repetition(
        tokens.tolist(),
        min_block_tokens=min_block_tokens,
        max_block_tokens=max_block_tokens,
        min_repeats=min_repeats,
        min_prefix_tokens=min_prefix_tokens,
        tail_tolerance_tokens=tail_tolerance_tokens,
    )
    if loop is None:
        return attach_metadata(
            track, recovered=False, effective=token_count, removed=0, period=0
        )

    text = str(track.decoded.texts[0])
    effective_tokens = _step_aligned_prefix_token_count(
        text=text,
        decoded_starts=starts,
        decoded_ends=ends,
        detected_cutoff_token=loop.cutoff_token,
        require_complete_step=require_complete_step,
    )
    if effective_tokens < int(min_prefix_tokens) or effective_tokens >= token_count:
        return attach_metadata(
            track, recovered=False, effective=token_count, removed=0, period=0
        )

    visible_indices = torch.nonzero(
        (ends[:effective_tokens] > starts[:effective_tokens]), as_tuple=False
    ).reshape(-1)
    if visible_indices.numel() == 0:
        return attach_metadata(
            track, recovered=False, effective=token_count, removed=0, period=0
        )
    effective_char_end = int(ends[int(visible_indices[-1].item())].item())
    if effective_char_end <= 0:
        return attach_metadata(
            track, recovered=False, effective=token_count, removed=0, period=0
        )

    updated_segment = copy.copy(segment)
    prefix_mask = torch.zeros(token_count, dtype=torch.float32)
    prefix_mask[:effective_tokens] = 1.0
    if segment.loss_mask is not None:
        existing_mask = hydrate(segment.loss_mask).to(torch.float32).cpu().reshape(-1)
        if existing_mask.numel() != token_count:
            raise ValueError("repetition recovery loss_mask must align with tokens")
        prefix_mask *= (existing_mask > 0.5).to(torch.float32)
    updated_segment.loss_mask = prefix_mask
    updated = _track_with_field(track, "segment", updated_segment)
    updated = _track_with_field(updated, "decoded", Texts(texts=[text[:effective_char_end].rstrip()]))
    updated = _set_stop_reason(updated, "repetition")
    updated = attach_metadata(
        updated,
        recovered=True,
        effective=effective_tokens,
        removed=token_count - effective_tokens,
        period=loop.period_tokens,
    )
    logger.info(
        "Recovered repetition-truncated sample=%s tokens=%d->%d removed=%d period=%d",
        updated.sample_ids[0] if updated.sample_ids else "unknown",
        token_count,
        effective_tokens,
        token_count - effective_tokens,
        loop.period_tokens,
    )
    return updated


def _rollout_phase_profile(track: RolloutTrack) -> Dict[str, float]:
    """Return the lightweight profile attached by the GeoWeave AR stage."""
    for condition in dict(track.conditions or {}).values():
        profiles = getattr(condition, "rollout_phase_metrics", None)
        if isinstance(profiles, list) and profiles and isinstance(profiles[0], dict):
            return {
                str(key): float(value)
                for key, value in profiles[0].items()
                if isinstance(value, (int, float))
            }
    return {}


def _make_padding_group(
    source: PromptGroupState,
    *,
    rollout_id: int,
    padding_index: int,
) -> PromptGroupState:
    """Clone a fixed-shape padding group with independent GRPO lineage.

    Reusing ``source`` directly would duplicate its parent id, causing
    ``compute_advantages`` to merge two logical prompt slots into one larger
    group.  Padding is masked out of training, but it must still preserve the
    fixed number and width of groups expected by GRPO.
    """
    if source.track is None:
        raise RuntimeError("cannot pad from a prompt group without a track")
    suffix = f"__padding_r{int(rollout_id)}_{int(padding_index)}"
    padding_prompt_id = f"{source.prompt_id}/{suffix}"
    padding_track = copy.copy(source.track)
    padding_track.parent_ids = [padding_prompt_id] * source.track.batch_size
    padding_track.sample_ids = [
        f"{sample_id}/{suffix}" for sample_id in source.track.sample_ids
    ]

    padding_state = copy.copy(source)
    padding_state.prompt_id = padding_prompt_id
    padding_state.track = padding_track
    padding_state.owner_ranks = list(source.owner_ranks)
    padding_state.status = "padding"
    return padding_state


def _reward_only_track(track: RolloutTrack) -> RolloutTrack:
    """Drop heavy TensorRefs while preserving judge-visible token lengths."""
    light = copy.copy(track)
    light.conditions = {}
    light.media_preview = None
    if track.segment is not None and track.segment.cu_seqlens is not None:
        cu = hydrate(track.segment.cu_seqlens).to(torch.long).cpu()
        lengths = cu[1:] - cu[:-1]
        effective = track.effective_text_token_count
        recovered = track.repetition_recovered
        if effective is not None and recovered is not None:
            effective_local = hydrate(effective).to(torch.long).cpu().reshape(-1)
            recovered_local = hydrate(recovered).to(torch.float32).cpu().reshape(-1) > 0.5
            if effective_local.numel() == lengths.numel() == recovered_local.numel():
                lengths = torch.where(recovered_local, effective_local, lengths)
        segment = TextSegment()
        effective_cu = torch.cat(
            [torch.zeros(1, dtype=torch.long), lengths.cumsum(dim=0)]
        )
        object.__setattr__(segment, "_packed_cu_seqlens", effective_cu)
        light.segment = segment
    else:
        light.segment = None
    return light


def _score_group(
    *,
    driver_reward: Any,
    req: RolloutReq,
    prompt_index: int,
    track: RolloutTrack,
    forced_zero: Sequence[bool],
    stop_reasons: Optional[Sequence[str]] = None,
) -> Tuple[
    int,
    torch.Tensor,
    Dict[str, torch.Tensor],
    List[Optional[Dict[str, Any]]],
    float,
    bool,
]:
    """Judge non-forced rows and attach truncation/correctness components."""
    started = time.perf_counter()
    batch_size = track.batch_size
    if len(forced_zero) != batch_size:
        raise ValueError("forced_zero mask must align with the prompt group")
    judge_indices = [i for i, forced in enumerate(forced_zero) if not forced]
    rewards = torch.zeros(batch_size, dtype=torch.float32)
    merged_components: Dict[str, torch.Tensor] = {}
    process_annotations: List[Optional[Dict[str, Any]]] = [None] * batch_size

    if judge_indices:
        judge_track = track.select(torch.tensor(judge_indices, dtype=torch.long))
        prompt_req = req.slice(prompt_index, prompt_index + 1)
        ar_params = prompt_req.sampling_params.get("ar")
        sampling_params = dict(prompt_req.sampling_params)
        if ar_params is not None:
            sampling_params["ar"] = dataclasses.replace(
                ar_params, samples_per_prompt=len(judge_indices)
            )
        judge_req = RolloutReq(
            sample_ids=list(prompt_req.sample_ids),
            group_ids=list(prompt_req.group_ids),
            primitives=dict(prompt_req.primitives),
            request_conditions=dict(prompt_req.request_conditions),
            sampling_params=sampling_params,
            stage_config=dict(prompt_req.stage_config),
            sigmas=prompt_req.sigmas,
            metadata=list(prompt_req.metadata) if prompt_req.metadata else [],
            init_noise_group_ids=list(prompt_req.init_noise_group_ids),
            init_noise_latent_shape=prompt_req.init_noise_latent_shape,
        )
        scored = driver_reward.score_and_attach(
            req=judge_req, track=_reward_only_track(judge_track)
        )
        if scored.rewards is None:
            raise RuntimeError("Driver reward returned rewards=None")
        rewards[judge_indices] = scored.rewards.to(dtype=torch.float32, device="cpu")
        for name, values in dict(scored.component_rewards or {}).items():
            full = torch.zeros(batch_size, dtype=torch.float32)
            full[judge_indices] = values.to(dtype=torch.float32, device="cpu")
            merged_components[str(name)] = full
        judged_annotations = list(getattr(scored, "process_annotations", None) or [])
        if judged_annotations:
            if len(judged_annotations) != len(judge_indices):
                raise RuntimeError(
                    "Driver reward returned process_annotations with length "
                    f"{len(judged_annotations)} for {len(judge_indices)} judged rows"
                )
            for sample_index, annotation in zip(judge_indices, judged_annotations):
                process_annotations[sample_index] = annotation

    forced = torch.tensor(list(forced_zero), dtype=torch.float32)
    reasons = list(stop_reasons or [""] * batch_size)
    if len(reasons) != batch_size:
        raise ValueError("stop_reasons must align with the prompt group")
    merged_components.setdefault("answer_correctness", rewards.clone())
    text_truncated = torch.tensor(
        [float(reason == "max_new_tokens") for reason in reasons],
        dtype=torch.float32,
    )
    image_truncated = torch.tensor(
        [float(reason == "max_images") for reason in reasons],
        dtype=torch.float32,
    )
    merged_components["truncated"] = torch.maximum(text_truncated, image_truncated)
    merged_components["text_truncated"] = text_truncated
    merged_components["image_truncated"] = image_truncated

    def metadata_component(name: str) -> torch.Tensor:
        value = getattr(track, name, None)
        if value is None:
            return torch.zeros(batch_size, dtype=torch.float32)
        local = hydrate(value).to(torch.float32).cpu().reshape(-1)
        if local.numel() != batch_size:
            raise ValueError(f"{name} must align with the prompt group")
        return local

    merged_components["repetition_recovered"] = metadata_component(
        "repetition_recovered"
    )
    merged_components["original_text_truncated"] = metadata_component(
        "original_text_truncated"
    )
    merged_components["effective_text_token_count"] = metadata_component(
        "effective_text_token_count"
    )
    merged_components["removed_repetition_token_count"] = metadata_component(
        "removed_repetition_token_count"
    )
    merged_components["repetition_period_token_count"] = metadata_component(
        "repetition_period_token_count"
    )
    merged_components["forced_truncation_zero"] = forced.clone()
    merged_components["judge_evaluated"] = 1.0 - forced
    judge_failed = merged_components.get(
        "judge_failed",
        merged_components.get("sca_judge_failed", torch.zeros(batch_size)),
    ).to(torch.float32)
    has_judge_failure = bool((judge_failed > 0.5).any().item())
    return (
        prompt_index,
        rewards,
        merged_components,
        process_annotations,
        time.perf_counter() - started,
        has_judge_failure,
    )


def _is_informative(rewards: torch.Tensor, eps: float = 1e-8) -> bool:
    flat = rewards.to(torch.float32).reshape(-1)
    return bool(flat.numel() > 1 and float((flat.max() - flat.min()).item()) > eps)


def _sca_annotations_informative(
    annotations: Sequence[Optional[Dict[str, Any]]],
) -> bool:
    """SCA can carry useful within-sequence credit despite uniform outcomes."""
    valid = [
        annotation
        for annotation in annotations
        if isinstance(annotation, dict) and bool(annotation.get("valid", False))
    ]
    if not valid:
        return False
    outcomes = {bool(annotation.get("answer_correct", False)) for annotation in valid}
    if len(outcomes) > 1:
        return True
    return any(
        any(float(weight) > 0.0 for weight in annotation.get("step_error_weights", []))
        for annotation in valid
    )


def post_advantage_owner_affine_train_permutation(
    owner_ranks: Sequence[int], dp_size: int
) -> List[int]:
    """Build equal train shards that maximize source-worker locality."""
    total = len(owner_ranks)
    if dp_size < 1 or total % dp_size != 0:
        raise ValueError(
            f"owner-affine train ordering requires total={total} divisible by dp_size={dp_size}"
        )
    shard_size = total // dp_size
    by_owner: Dict[int, Deque[int]] = {rank: deque() for rank in range(dp_size)}
    for sample_index, owner_rank in enumerate(owner_ranks):
        if owner_rank not in by_owner:
            raise ValueError(f"owner rank {owner_rank} outside [0, {dp_size})")
        by_owner[owner_rank].append(sample_index)

    shards: List[List[int]] = [[] for _ in range(dp_size)]
    for rank in range(dp_size):
        while by_owner[rank] and len(shards[rank]) < shard_size:
            shards[rank].append(by_owner[rank].popleft())
    leftovers = deque(
        sample_index for rank in range(dp_size) for sample_index in by_owner[rank]
    )
    for rank in range(dp_size):
        while len(shards[rank]) < shard_size:
            if not leftovers:
                raise RuntimeError("owner-affine permutation ran out of samples")
            shards[rank].append(leftovers.popleft())
    if leftovers:
        raise RuntimeError("owner-affine permutation left unassigned samples")
    return [sample_index for shard in shards for sample_index in shard]


def run_dynamic_trainside_rollout(
    *,
    rollout_handle: Any,
    driver_reward: Any,
    req: RolloutReq,
    rollout_id: int,
    reward_workers: int = 4,
    target_prompt_count: Optional[int] = None,
) -> DynamicRolloutResult:
    """Run a global FIFO with candidate retry and reward-driven prompt refill."""
    texts = req.primitives.get("text")
    if not isinstance(texts, Texts):
        raise TypeError(
            "Dynamic trainside rollout requires req.primitives['text'] to be Texts."
        )
    ar_params = req.sampling_params.get("ar")
    if ar_params is None:
        raise TypeError("Dynamic trainside rollout requires AR sampling params.")
    candidates_per_prompt = int(ar_params.samples_per_prompt)
    target_prompts = int(target_prompt_count or len(req.sample_ids))
    if target_prompts < 1 or target_prompts > len(req.sample_ids):
        raise ValueError(
            f"target_prompt_count must be in [1, {len(req.sample_ids)}], got {target_prompts}"
        )

    cfg = dict(req.stage_config or {})
    retry_truncated = bool(cfg.get("retry_truncated_trajectories", True))
    ignore_truncated_samples = bool(cfg.get("ignore_truncated_samples", False))
    exclude_truncated_from_advantage_stats = bool(
        cfg.get("exclude_truncated_from_advantage_stats", False)
    )
    recover_repetition_truncations = bool(
        cfg.get("recover_repetition_truncations", True)
    )
    repetition_min_block_tokens = int(cfg.get("repetition_min_block_tokens", 16))
    repetition_max_block_tokens = int(cfg.get("repetition_max_block_tokens", 256))
    repetition_min_repeats = int(cfg.get("repetition_min_repeats", 3))
    repetition_min_prefix_tokens = int(cfg.get("repetition_min_prefix_tokens", 64))
    repetition_tail_tolerance_tokens = int(
        cfg.get("repetition_tail_tolerance_tokens", 32)
    )
    repetition_require_complete_step = bool(
        cfg.get("repetition_require_complete_step", True)
    )
    if repetition_min_block_tokens <= 0 or repetition_max_block_tokens < repetition_min_block_tokens:
        raise ValueError("invalid repetition block token thresholds")
    if repetition_min_repeats < 2 or repetition_min_prefix_tokens < 0 or repetition_tail_tolerance_tokens < 0:
        raise ValueError("invalid repetition recovery thresholds")
    if exclude_truncated_from_advantage_stats and not ignore_truncated_samples:
        raise ValueError(
            "stage_config.exclude_truncated_from_advantage_stats=true requires "
            "stage_config.ignore_truncated_samples=true"
        )
    max_attempts = max(1, int(cfg.get("max_attempts_per_candidate", 3)))
    max_truncated_per_prompt = max(
        0, int(cfg.get("max_truncated_refills_per_prompt", candidates_per_prompt))
    )
    prompt_refill = bool(cfg.get("dynamic_prompt_group_refill", True))
    max_replacements = max(0, int(cfg.get("max_replacement_prompts", target_prompts)))
    trajectory_multiplier = max(1.0, float(cfg.get("max_trajectory_multiplier", 2.5)))
    max_trajectory_jobs = max(
        target_prompts * candidates_per_prompt,
        int(target_prompts * candidates_per_prompt * trajectory_multiplier),
    )
    rollout_chunk_size = max(
        1, min(candidates_per_prompt, int(cfg.get("dynamic_rollout_chunk_size", 1)))
    )
    continuous_batching = bool(cfg.get("continuous_batching", False))
    persistent_worker_session = bool(
        cfg.get("persistent_worker_session", False)
    )
    continuous_request_admission = bool(
        cfg.get("continuous_request_admission", False)
    )
    continuous_live_admission = bool(
        cfg.get("continuous_live_admission", False)
    )
    live_prompt_microbundle_size = max(
        1, int(cfg.get("continuous_prompt_microbundle_size", 1))
    )
    if persistent_worker_session and not continuous_batching:
        raise ValueError(
            "persistent_worker_session requires continuous_batching=true"
        )
    if continuous_request_admission and not persistent_worker_session:
        raise ValueError(
            "continuous_request_admission requires persistent_worker_session=true"
        )
    if continuous_live_admission and not continuous_request_admission:
        raise ValueError(
            "continuous_live_admission requires continuous_request_admission=true"
        )
    if live_prompt_microbundle_size > 1 and not continuous_live_admission:
        raise ValueError(
            "continuous_prompt_microbundle_size>1 requires continuous_live_admission=true"
        )
    continuous_pool_size = max(
        1,
        min(
            candidates_per_prompt,
            int(cfg.get("continuous_rollout_pool_size", rollout_chunk_size)),
        ),
    )
    rpc_pool_size = (
        1
        if continuous_request_admission
        else (continuous_pool_size if continuous_batching else rollout_chunk_size)
    )
    text_pool_size = (
        continuous_pool_size if continuous_request_admission else rpc_pool_size
    )
    continuous_text_batch_size = max(
        1,
        min(
            text_pool_size,
            int(cfg.get("continuous_text_batch_size", rollout_chunk_size)),
        ),
    )
    session_window_size = max(
        continuous_pool_size,
        int(cfg.get("continuous_session_window_size", continuous_pool_size * 2)),
    )
    if live_prompt_microbundle_size > continuous_pool_size:
        raise ValueError(
            "continuous_prompt_microbundle_size cannot exceed continuous_rollout_pool_size"
        )

    worker_groups = _dp_worker_groups(rollout_handle)
    if not worker_groups:
        raise RuntimeError("Dynamic trainside rollout has no DP worker groups")
    if target_prompts * candidates_per_prompt % len(worker_groups) != 0:
        raise ValueError(
            f"Final dynamic trajectory count {target_prompts * candidates_per_prompt} must be divisible by "
            f"train/rollout dp_size {len(worker_groups)}."
        )

    queue: Deque[TrajectoryJob] = deque()
    inflight: Dict[int, InFlightJob] = {}
    ref_to_dp_rank: Dict[Any, int] = {}
    idle_workers: Dict[int, List[int]] = {
        rank: indices for rank, indices in worker_groups
    }
    persistent_sessions: Dict[int, PersistentWorkerSession] = {}
    session_executor: Optional[ThreadPoolExecutor] = None
    live_input_queue: Optional[Any] = None
    live_jobs: Dict[int, TrajectoryJob] = {}
    live_output_futures: Dict[Future, int] = {}
    live_session_started: Dict[int, float] = {}
    live_close_sent = False
    streamed_trajectory_results = 0
    live_microbundles_published = 0
    live_microbundle_rows = 0
    prompt_states: Dict[int, PromptGroupState] = {}
    reserve_prompts: Deque[int] = deque(range(target_prompts, len(req.sample_ids)))
    consumed_prompt_indices: List[int] = []
    accepted: List[PromptGroupState] = []
    uniform_groups: List[PromptGroupState] = []
    scored_groups: List[PromptGroupState] = []
    reward_futures: Dict[Future, int] = {}
    reward_times: Dict[int, float] = {}
    worker_busy = {dp_rank: 0.0 for dp_rank, _ in worker_groups}
    worker_jobs = {dp_rank: 0 for dp_rank, _ in worker_groups}
    rpc_durations: List[float] = []
    rpc_trajectory_counts: List[int] = []
    replacement_count = 0
    next_job_index = 0
    total_jobs_created = 0
    total_rpc_jobs_created = 0
    truncated_attempts = 0
    soft_text_truncations = 0
    repetition_recovered_samples = 0
    repetition_removed_tokens = 0
    repetition_recovery_failures = 0
    forced_zero_count = 0
    profiled_trajectories = 0
    profile_totals = {
        "prefix_s": 0.0,
        "text_decode_s": 0.0,
        "diffusion_s": 0.0,
        "image_reencode_s": 0.0,
        "total_s": 0.0,
        "generated_text_tokens": 0.0,
        "generated_images": 0.0,
        "diffusion_forwards": 0.0,
        "text_forward_calls_share": 0.0,
        "text_forward_rows": 0.0,
        "continuous_refill_count": 0.0,
        "continuous_coalesce_calls": 0.0,
        "continuous_ready_groups": 0.0,
        "continuous_compatible_groups": 0.0,
        "continuous_underfilled_slots": 0.0,
        "continuous_refilled_slots": 0.0,
        "continuous_incompatible_groups": 0.0,
    }
    truncated_reward_policy = str(getattr(driver_reward, "truncated_reward", "zero"))
    if truncated_reward_policy not in {"zero", "keep", "soft"}:
        raise ValueError(
            "driver reward truncated_reward must be zero|keep|soft, "
            f"got {truncated_reward_policy!r}"
        )
    if ignore_truncated_samples and truncated_reward_policy != "zero":
        raise ValueError(
            "stage_config.ignore_truncated_samples=true requires "
            "reward.truncated_reward=zero so truncated rows keep the intended "
            "zero-reward advantage semantics before their own gradients are masked."
        )
    reward_backend = getattr(driver_reward, "backend", None)
    reward_thread_safe = bool(getattr(reward_backend, "thread_safe", False))
    scheduler_started = time.perf_counter()

    def enqueue_slots(
        state: PromptGroupState, slots: Sequence[CandidateSlotState]
    ) -> bool:
        nonlocal next_job_index, total_jobs_created, total_rpc_jobs_created
        if not slots or total_jobs_created + len(slots) > max_trajectory_jobs:
            return False
        attempts = [slot.attempt_count for slot in slots]
        rewrite_indices = [slot.rewrite_index for slot in slots]
        prompt_id, candidate_ids, job_req = _trajectory_req(
            req,
            prompt_index=state.prompt_index,
            rewrite_indices=rewrite_indices,
            attempt_indices=attempts,
            rollout_id=rollout_id,
        )
        queue.append(
            TrajectoryJob(
                job_index=next_job_index,
                prompt_index=state.prompt_index,
                rewrite_indices=tuple(rewrite_indices),
                attempt_indices=tuple(attempts),
                prompt_id=prompt_id,
                candidate_ids=candidate_ids,
                req=job_req,
            )
        )
        next_job_index += 1
        total_rpc_jobs_created += 1
        total_jobs_created += len(slots)
        for slot in slots:
            slot.attempt_count += 1
        return True

    def enqueue_job(state: PromptGroupState, slot: CandidateSlotState) -> bool:
        return enqueue_slots(state, [slot])

    def activate_prompt(prompt_index: int) -> bool:
        if prompt_index in prompt_states:
            return False
        if total_jobs_created + candidates_per_prompt > max_trajectory_jobs:
            return False
        state = PromptGroupState(
            prompt_index=prompt_index,
            prompt_id=str(req.sample_ids[prompt_index]),
            slots=[CandidateSlotState(i) for i in range(candidates_per_prompt)],
        )
        prompt_states[prompt_index] = state
        consumed_prompt_indices.append(prompt_index)
        for start in range(0, candidates_per_prompt, rpc_pool_size):
            slots = state.slots[start : start + rpc_pool_size]
            if not enqueue_slots(state, slots):
                raise RuntimeError(
                    "trajectory budget changed while activating a prompt"
                )
        return True

    for prompt_index in range(target_prompts):
        activate_prompt(prompt_index)

    executor = ThreadPoolExecutor(
        max_workers=max(1, min(int(reward_workers), max(1, len(req.sample_ids)))),
        thread_name_prefix="unirl-reward",
    )
    if persistent_worker_session:
        session_executor = ThreadPoolExecutor(
            max_workers=sum(len(indices) for _, indices in worker_groups),
            thread_name_prefix="unirl-rollout-session",
        )
        if continuous_live_admission:
            live_input_queue = _make_session_queue()
            for dp_rank, worker_indices in worker_groups:
                if len(worker_indices) != 1:
                    raise ValueError(
                        "continuous_live_admission currently requires exactly one "
                        "physical worker per DP group (tp_size=sp_size=1)"
                    )
                output_queue = _make_session_queue()
                worker_index = worker_indices[0]
                ref = rollout_handle.workers[worker_index].live_session_call.remote(
                    rollout_handle.role_name,
                    "generate_live",
                    live_input_queue,
                    output_queue,
                )
                persistent_sessions[dp_rank] = PersistentWorkerSession(
                    dp_rank=dp_rank,
                    worker_indices=list(worker_indices),
                    input_queues=[live_input_queue],
                    output_queues=[output_queue],
                    refs=[ref],
                )
                live_session_started[dp_rank] = time.perf_counter()
                live_output_futures[
                    session_executor.submit(output_queue.get)
                ] = dp_rank
        else:
            for dp_rank, worker_indices in worker_groups:
                input_queues = [_make_session_queue() for _ in worker_indices]
                output_queues = [_make_session_queue() for _ in worker_indices]
                refs = [
                    rollout_handle.workers[worker_index].session_call.remote(
                        rollout_handle.role_name,
                        "generate",
                        input_queue,
                        output_queue,
                    )
                    for worker_index, input_queue, output_queue in zip(
                        worker_indices, input_queues, output_queues
                    )
                ]
                persistent_sessions[dp_rank] = PersistentWorkerSession(
                    dp_rank=dp_rank,
                    worker_indices=list(worker_indices),
                    input_queues=input_queues,
                    output_queues=output_queues,
                    refs=refs,
                )

    def maybe_activate_replacement() -> None:
        nonlocal replacement_count
        if (
            not prompt_refill
            or len(accepted) >= target_prompts
            or replacement_count >= max_replacements
        ):
            return
        while reserve_prompts:
            prompt_index = reserve_prompts.popleft()
            if activate_prompt(prompt_index):
                replacement_count += 1
                return

    def handle_score_result(
        result: Tuple[
            int,
            torch.Tensor,
            Dict[str, torch.Tensor],
            List[Optional[Dict[str, Any]]],
            float,
            bool,
        ],
    ) -> None:
        prompt_index, rewards, components, annotations, reward_time, judge_failed = result
        reward_times[prompt_index] = reward_time
        state = prompt_states[prompt_index]
        if state.track is None:
            raise RuntimeError("reward completed before group track assembly")
        state.track = _track_with_field(state.track, "rewards", rewards)
        state.track = _track_with_field(state.track, "component_rewards", components)
        if any(annotation is not None for annotation in annotations):
            state.track = _track_with_field(
                state.track, "process_annotations", annotations
            )
        scored_groups.append(state)
        if judge_failed:
            state.status = "reward_invalid"
            maybe_activate_replacement()
        else:
            correctness = components.get("answer_correctness")
            filter_values = correctness if correctness is not None else rewards
            if _is_informative(filter_values) or _sca_annotations_informative(annotations):
                state.status = "accepted"
                accepted.append(state)
            else:
                state.status = (
                    "uniform_one"
                    if bool((filter_values > 0.5).all().item())
                    else "uniform_zero"
                )
                uniform_groups.append(state)
                maybe_activate_replacement()

    def submit_group(state: PromptGroupState) -> None:
        tracks = [slot.selected_track for slot in state.slots]
        if any(track is None for track in tracks):
            raise RuntimeError("cannot score an incomplete prompt group")
        state.track = RolloutTrack.concat(
            [track for track in tracks if track is not None]
        )
        state.owner_ranks = [
            int(slot.selected_owner_rank)
            for slot in state.slots
            if slot.selected_owner_rank is not None
        ]
        if len(state.owner_ranks) != candidates_per_prompt:
            raise RuntimeError("prompt group is missing trajectory owner ranks")
        forced = [slot.forced_zero for slot in state.slots]
        stop_reasons = [slot.selected_reason for slot in state.slots]
        state.status = "scoring"
        kwargs = dict(
            driver_reward=driver_reward,
            req=req,
            prompt_index=state.prompt_index,
            track=state.track,
            forced_zero=forced,
            stop_reasons=stop_reasons,
        )
        if reward_thread_safe:
            future = executor.submit(_score_group, **kwargs)
            reward_futures[future] = state.prompt_index
        else:
            handle_score_result(_score_group(**kwargs))

    def process_trajectory(
        *,
        prompt_index: int,
        rewrite_index: int,
        track: RolloutTrack,
        owner_rank: int,
    ) -> None:
        nonlocal truncated_attempts, soft_text_truncations, forced_zero_count, profiled_trajectories
        nonlocal repetition_recovered_samples, repetition_removed_tokens, repetition_recovery_failures
        profile = _rollout_phase_profile(track)
        if profile:
            profiled_trajectories += 1
            for key in profile_totals:
                profile_totals[key] += float(profile.get(key, 0.0))
        state = prompt_states[prompt_index]
        slot = state.slots[rewrite_index]
        # Attempts carry a unique RPC identity, but the selected training row
        # must retain the stable candidate identity for this prompt slot.
        track.sample_ids = [f"{state.prompt_id}/a{int(rewrite_index)}"]
        track.parent_ids = [state.prompt_id]
        try:
            track = _recover_repetition_truncation(
                track,
                enabled=recover_repetition_truncations,
                min_block_tokens=repetition_min_block_tokens,
                max_block_tokens=repetition_max_block_tokens,
                min_repeats=repetition_min_repeats,
                min_prefix_tokens=repetition_min_prefix_tokens,
                tail_tolerance_tokens=repetition_tail_tolerance_tokens,
                require_complete_step=repetition_require_complete_step,
            )
        except Exception:
            # Recovery is optional safety logic: malformed metadata must fall
            # back to the existing truncation policy instead of killing training.
            repetition_recovery_failures += 1
            logger.exception(
                "Repetition recovery failed for sample=%s; using original truncation policy",
                track.sample_ids[0],
            )
            track = _recover_repetition_truncation(
                track,
                enabled=False,
                min_block_tokens=repetition_min_block_tokens,
                max_block_tokens=repetition_max_block_tokens,
                min_repeats=repetition_min_repeats,
                min_prefix_tokens=repetition_min_prefix_tokens,
                tail_tolerance_tokens=repetition_tail_tolerance_tokens,
                require_complete_step=repetition_require_complete_step,
            )
        if track.repetition_recovered is not None and float(hydrate(track.repetition_recovered)[0]) > 0.5:
            repetition_recovered_samples += 1
            repetition_removed_tokens += int(hydrate(track.removed_repetition_token_count)[0])
        reason = _stop_reason(track)
        if reason not in _TRUNCATED_REASONS:
            slot.selected_track = track
            slot.selected_owner_rank = owner_rank
            slot.selected_reason = reason
        elif reason == "max_new_tokens" and truncated_reward_policy in {"soft", "keep"}:
            # Soft/keep must score the sampled long response as-is. Retrying it
            # would rejection-sample long responses before reward shaping.
            truncated_attempts += 1
            state.truncated_attempts += 1
            soft_text_truncations += 1
            slot.selected_track = track
            slot.selected_owner_rank = owner_rank
            slot.selected_reason = reason
        else:
            # Preserve retry/forced-zero for max_images and for hard-zero text truncation.
            truncated_attempts += 1
            state.truncated_attempts += 1
            if slot.fallback_track is None:
                slot.fallback_track = track
                slot.fallback_owner_rank = owner_rank
                slot.fallback_reason = reason
            can_retry = (
                retry_truncated
                and slot.attempt_count < max_attempts
                and state.truncated_attempts <= max_truncated_per_prompt
            )
            if can_retry and enqueue_job(state, slot):
                return
            slot.selected_track = slot.fallback_track
            slot.selected_owner_rank = slot.fallback_owner_rank
            slot.selected_reason = slot.fallback_reason
            slot.forced_zero = True
            forced_zero_count += 1
        if state.complete and state.status == "rolling_out":
            submit_group(state)

    def process_track(job: TrajectoryJob, track: RolloutTrack, owner_rank: int) -> None:
        if track.batch_size != job.trajectory_count:
            raise RuntimeError(
                "dynamic job track does not align with trajectory identities"
            )
        for row, rewrite_index in enumerate(job.rewrite_indices):
            row_track = track.select(torch.tensor([row], dtype=torch.long))
            process_trajectory(
                prompt_index=job.prompt_index,
                rewrite_index=rewrite_index,
                track=row_track,
                owner_rank=owner_rank,
            )

    def publish_live_jobs() -> None:
        nonlocal live_microbundles_published, live_microbundle_rows
        if live_input_queue is None:
            raise RuntimeError("live rollout input queue was not initialized")
        while queue:
            jobs = _pop_live_microbundle(
                queue, max_size=live_prompt_microbundle_size
            )
            items = []
            for job in jobs:
                if job.job_index in live_jobs:
                    raise RuntimeError(f"duplicate live trajectory job {job.job_index}")
                live_jobs[job.job_index] = job
                items.append((job.job_index, (job.req,), {}))
            live_microbundles_published += 1
            live_microbundle_rows += len(items)
            live_input_queue.put(items if len(items) > 1 else items[0])

    def run_live_sessions() -> None:
        nonlocal live_close_sent, streamed_trajectory_results
        if session_executor is None or live_input_queue is None:
            raise RuntimeError("live rollout sessions were not initialized")
        closed_sessions = set()
        session_count = len(persistent_sessions)
        worker_local = issubclass(
            rollout_handle.pool.transport_cls, WorkerLocalTransport
        )
        while (
            queue
            or live_jobs
            or reward_futures
            or len(closed_sessions) < session_count
        ):
            done_rewards = [future for future in reward_futures if future.done()]
            for future in done_rewards:
                reward_futures.pop(future, None)
                handle_score_result(future.result())

            publish_live_jobs()
            if (
                not live_close_sent
                and not queue
                and not live_jobs
                and not reward_futures
            ):
                for _ in range(session_count):
                    live_input_queue.put(None)
                live_close_sent = True

            completed_outputs = [
                future for future in live_output_futures if future.done()
            ]
            if not completed_outputs:
                time.sleep(0.01)
                continue
            for future in completed_outputs:
                dp_rank = live_output_futures.pop(future)
                status, request_id, payload = future.result()
                session = persistent_sessions[dp_rank]
                if status == "closed":
                    closed_sessions.add(dp_rank)
                    duration = time.perf_counter() - live_session_started[dp_rank]
                    worker_busy[dp_rank] += duration
                    rpc_durations.append(duration)
                    rpc_trajectory_counts.append(worker_jobs[dp_rank])
                    continue
                if status == "error":
                    raise RuntimeError(
                        f"Live rollout session {dp_rank} failed for request "
                        f"{request_id}: {payload}"
                    )
                if status != "ok":
                    raise RuntimeError(
                        f"Live rollout session {dp_rank} returned unknown status "
                        f"{status!r}"
                    )
                job = live_jobs.pop(int(request_id), None)
                if job is None:
                    raise RuntimeError(
                        f"Live rollout returned unknown request {request_id}"
                    )
                worker_index = session.worker_indices[0]
                response = rollout_handle._rebind_tree(
                    payload,
                    rollout_handle.workers[worker_index],
                    worker_local=worker_local,
                )
                process_track(
                    job,
                    _rewrite_job_track_identity(response, job),
                    dp_rank,
                )
                streamed_trajectory_results += job.trajectory_count
                worker_jobs[dp_rank] += job.trajectory_count
                live_output_futures[
                    session_executor.submit(session.output_queues[0].get)
                ] = dp_rank

        ray.get([ref for session in persistent_sessions.values() for ref in session.refs])

    def launch_available() -> None:
        for dp_rank in list(idle_workers):
            if not queue:
                break
            worker_indices = idle_workers.pop(dp_rank)
            jobs = [queue.popleft()]
            if continuous_request_admission:
                while queue and len(jobs) < session_window_size:
                    jobs.append(queue.popleft())
            job = jobs[0]
            request = (
                _merge_session_jobs(
                    jobs, text_batch_size=continuous_text_batch_size
                )
                if continuous_request_admission
                else job.req
            )
            if persistent_worker_session:
                if session_executor is None:
                    raise RuntimeError("persistent session executor was not initialized")
                session = persistent_sessions[dp_rank]
                for input_queue in session.input_queues:
                    input_queue.put((job.job_index, (request,), {}))
                refs = [
                    session_executor.submit(output_queue.get)
                    for output_queue in session.output_queues
                ]
            else:
                refs = [
                    rollout_handle.workers[worker_index].call.remote(
                        rollout_handle.role_name,
                        "generate",
                        (request,),
                        {},
                        False,
                        None,
                    )
                    for worker_index in worker_indices
                ]
            rec = InFlightJob(
                job,
                dp_rank,
                list(worker_indices),
                refs,
                time.perf_counter(),
                jobs=tuple(jobs),
            )
            inflight[dp_rank] = rec
            if not persistent_worker_session:
                for ref in refs:
                    ref_to_dp_rank[ref] = dp_rank

    try:
        if continuous_live_admission:
            run_live_sessions()
        while not continuous_live_admission and (queue or inflight or reward_futures):
            done_rewards = [future for future in reward_futures if future.done()]
            for future in done_rewards:
                reward_futures.pop(future, None)
                handle_score_result(future.result())

            launch_available()
            if inflight:
                completed_dp_rank = None
                if persistent_worker_session:
                    completed_dp_rank = next(
                        (
                            dp_rank
                            for dp_rank, rec in inflight.items()
                            if all(future.done() for future in rec.refs)
                        ),
                        None,
                    )
                    if completed_dp_rank is None:
                        time.sleep(0.05)
                else:
                    all_refs = [ref for rec in inflight.values() for ref in rec.refs]
                    ready, _ = ray.wait(all_refs, num_returns=1, timeout=0.1)
                    if ready:
                        completed_dp_rank = ref_to_dp_rank[ready[0]]
                if completed_dp_rank is not None:
                    rec = inflight.pop(completed_dp_rank)
                    if persistent_worker_session:
                        resp = _collect_session_one(rollout_handle, rec)
                    else:
                        for ref in rec.refs:
                            ref_to_dp_rank.pop(ref, None)
                        resp = _collect_one(rollout_handle, rec)
                    duration = time.perf_counter() - rec.started_at
                    worker_busy[completed_dp_rank] += duration
                    worker_jobs[completed_dp_rank] += 1
                    rpc_durations.append(duration)
                    rpc_trajectory_counts.append(rec.trajectory_count)
                    idle_workers[completed_dp_rank] = rec.worker_indices
                    jobs = rec.jobs or (rec.job,)
                    if len(jobs) == 1:
                        process_track(
                            rec.job,
                            _rewrite_job_track_identity(resp, rec.job),
                            rec.dp_rank,
                        )
                    else:
                        track = resp.tracks.get("ar")
                        if track is None or track.batch_size != rec.trajectory_count:
                            raise RuntimeError(
                                "admission window response does not align with jobs"
                            )
                        offset = 0
                        for child_job in jobs:
                            child_track = track.select(
                                torch.arange(
                                    offset,
                                    offset + child_job.trajectory_count,
                                    dtype=torch.long,
                                )
                            )
                            child_resp = RolloutResp(tracks={"ar": child_track})
                            process_track(
                                child_job,
                                _rewrite_job_track_identity(child_resp, child_job),
                                rec.dp_rank,
                            )
                            offset += child_job.trajectory_count
                continue
            if reward_futures:
                done, _ = wait(list(reward_futures), return_when=FIRST_COMPLETED)
                for future in done:
                    reward_futures.pop(future, None)
                    handle_score_result(future.result())
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        if persistent_sessions:
            if continuous_live_admission:
                if not live_close_sent and live_input_queue is not None:
                    for _ in persistent_sessions:
                        live_input_queue.put(None)
                    live_close_sent = True
            else:
                for session in persistent_sessions.values():
                    for input_queue in session.input_queues:
                        input_queue.put(None)
            session_errors = []
            for session in persistent_sessions.values():
                try:
                    ray.get(session.refs)
                except BaseException as exc:
                    session_errors.append(exc)
            queue_objects = {
                id(queue_obj): queue_obj
                for session in persistent_sessions.values()
                for queue_obj in [*session.input_queues, *session.output_queues]
            }
            for queue_obj in queue_objects.values():
                queue_obj.shutdown(force=True)
            if session_executor is not None:
                session_executor.shutdown(wait=True, cancel_futures=True)
            if session_errors:
                raise session_errors[0]

    accepted_selected = sorted(accepted, key=lambda state: state.prompt_index)[
        :target_prompts
    ]
    selected: List[PromptGroupState] = list(accepted_selected)
    padding_needed = target_prompts - len(selected)
    if padding_needed > 0:
        selected.extend(
            sorted(uniform_groups, key=lambda state: state.prompt_index)[
                :padding_needed
            ]
        )
        padding_needed = target_prompts - len(selected)
    if padding_needed > 0:
        padding_sources = sorted(
            (state for state in scored_groups if state.track is not None),
            key=lambda state: state.prompt_index,
        )
        if not padding_sources:
            raise RuntimeError(
                "Dynamic refill produced no scored group for fixed-shape padding"
            )
        for i in range(padding_needed):
            selected.append(
                _make_padding_group(
                    padding_sources[i % len(padding_sources)],
                    rollout_id=rollout_id,
                    padding_index=i,
                )
            )

    ordered_tracks: List[RolloutTrack] = []
    stable_owner_ranks: List[int] = []
    selected_prompt_indices: List[int] = []
    accepted_ids = {id(state) for state in accepted_selected}
    masked_truncated_samples = 0
    effective_train_samples = 0
    effective_train_tokens = 0
    groups_with_zero_trainable_samples = 0
    groups_with_one_trainable_sample = 0
    component_names = {
        str(name)
        for state in selected
        if state.track is not None
        for name in dict(state.track.component_rewards or {})
    }
    has_process_annotations = any(
        state.track is not None and state.track.process_annotations is not None
        for state in selected
    )
    for state in selected:
        if state.track is None or state.track.rewards is None:
            raise RuntimeError("selected prompt group is missing scored track data")
        # Batch.concat cannot infer sample-aligned zeroes for a missing dict key.
        # Normalize every selected group to the same component schema first.
        components = dict(state.track.component_rewards or {})
        for name in component_names:
            components.setdefault(
                name, torch.zeros(candidates_per_prompt, dtype=torch.float32)
            )
        state_track = _track_with_field(state.track, "component_rewards", components)
        if has_process_annotations:
            annotations = state_track.process_annotations
            if annotations is None:
                annotations = [None] * candidates_per_prompt
            elif len(annotations) != candidates_per_prompt:
                raise RuntimeError(
                    "selected prompt group has misaligned process_annotations: "
                    f"annotations={len(annotations)}, samples={candidates_per_prompt}"
                )
            state_track = _track_with_field(
                state_track, "process_annotations", list(annotations)
            )
        group_train_mask = 1.0 if id(state) in accepted_ids else 0.0
        sample_train_mask = torch.full(
            (candidates_per_prompt,), group_train_mask, dtype=torch.float32
        )
        if ignore_truncated_samples and group_train_mask > 0.0:
            text_truncated = components.get(
                "text_truncated",
                torch.zeros(candidates_per_prompt, dtype=torch.float32),
            ).to(torch.float32)
            truncated_mask = text_truncated > 0.5
            masked_truncated_samples += int(truncated_mask.sum().item())
            sample_train_mask = sample_train_mask * (~truncated_mask).to(torch.float32)
        trainable_count = int(sample_train_mask.sum().item())
        effective_train_samples += trainable_count
        groups_with_zero_trainable_samples += int(trainable_count == 0)
        groups_with_one_trainable_sample += int(trainable_count == 1)
        lengths = getattr(state_track.segment, "lengths", None) if state_track.segment is not None else None
        if lengths is not None and int(lengths.numel()) == candidates_per_prompt:
            effective_train_tokens += int(
                (lengths.to(torch.float32).cpu() * sample_train_mask).sum().item()
            )
        state_track = _track_with_field(state_track, "status", sample_train_mask)
        ordered_tracks.append(state_track)
        stable_owner_ranks.extend(state.owner_ranks)
        selected_prompt_indices.append(state.prompt_index)

    ordered_track = RolloutTrack.concat(ordered_tracks)
    expected_trajectories = target_prompts * candidates_per_prompt
    if ordered_track.batch_size != expected_trajectories:
        raise RuntimeError(
            "Dynamic scheduler produced an invalid fixed-shape batch: "
            f"samples={ordered_track.batch_size}, expected={expected_trajectories}"
        )
    parent_ids = list(ordered_track.parent_ids or [])
    unique_parent_ids = list(dict.fromkeys(parent_ids))
    if (
        len(parent_ids) != expected_trajectories
        or len(unique_parent_ids) != target_prompts
        or any(
            parent_ids.count(parent_id) != candidates_per_prompt
            for parent_id in unique_parent_ids
        )
    ):
        raise RuntimeError(
            "Dynamic scheduler produced invalid GRPO group lineage: "
            f"samples={len(parent_ids)}, groups={len(unique_parent_ids)}, "
            f"expected_groups={target_prompts}, group_size={candidates_per_prompt}"
        )
    if len(set(ordered_track.sample_ids)) != expected_trajectories:
        raise RuntimeError("Dynamic scheduler produced duplicate trajectory sample ids")

    elapsed = time.perf_counter() - scheduler_started
    total_capacity = len(worker_groups) * elapsed
    total_worker_busy = sum(worker_busy.values())
    profiled_compute = profile_totals["total_s"]
    max_busy = max(worker_busy.values(), default=0.0)
    rpc_duration_mean = (
        sum(rpc_durations) / len(rpc_durations) if rpc_durations else 0.0
    )
    rpc_duration_variance = (
        sum((duration - rpc_duration_mean) ** 2 for duration in rpc_durations)
        / len(rpc_durations)
        if rpc_durations
        else 0.0
    )
    total_rpc_trajectories = sum(rpc_trajectory_counts)
    scheduler_metrics = {
        "rollout_id": int(rollout_id),
        "target_prompts": target_prompts,
        "prompt_pool": len(req.sample_ids),
        "consumed_prompts": len(consumed_prompt_indices),
        "accepted_groups": len(accepted),
        "uniform_groups": len(uniform_groups),
        "padding_groups": sum(
            1
            for track in ordered_tracks
            if track.status is not None and float(track.status.sum()) == 0.0
        ),
        "replacement_prompts": replacement_count,
        "candidates_per_prompt": candidates_per_prompt,
        "trajectory_jobs": total_jobs_created,
        "rpc_jobs": len(rpc_durations),
        "logical_rpc_jobs": total_rpc_jobs_created,
        "rollout_chunk_size": rollout_chunk_size,
        "continuous_batching": continuous_batching,
        "persistent_worker_session": persistent_worker_session,
        "persistent_session_count": len(persistent_sessions),
        "continuous_request_admission": continuous_request_admission,
        "continuous_live_admission": continuous_live_admission,
        "live_session_count": (
            len(persistent_sessions) if continuous_live_admission else 0
        ),
        "streamed_trajectory_results": streamed_trajectory_results,
        "live_queue_admissions": (
            total_rpc_jobs_created if continuous_live_admission else 0
        ),
        "continuous_prompt_microbundle_size": live_prompt_microbundle_size,
        "live_microbundles_published": live_microbundles_published,
        "live_microbundle_average_rows": (
            live_microbundle_rows / live_microbundles_published
            if live_microbundles_published > 0
            else 0.0
        ),
        "live_session_trajectories_per_session": (
            streamed_trajectory_results / len(persistent_sessions)
            if continuous_live_admission and persistent_sessions
            else 0.0
        ),
        "continuous_session_window_size": session_window_size,
        "continuous_rollout_pool_size": continuous_pool_size,
        "continuous_text_batch_size": continuous_text_batch_size,
        "average_rpc_batch_size": (
            total_jobs_created / len(rpc_durations)
            if rpc_durations
            else 0.0
        ),
        "rpc_duration_mean_s": rpc_duration_mean,
        "rpc_duration_p50_s": _percentile(rpc_durations, 0.50),
        "rpc_duration_p90_s": _percentile(rpc_durations, 0.90),
        "rpc_duration_max_s": max(rpc_durations, default=0.0),
        "rpc_duration_cv": (
            math.sqrt(rpc_duration_variance) / rpc_duration_mean
            if rpc_duration_mean > 0.0
            else 0.0
        ),
        "rpc_seconds_per_trajectory": (
            sum(rpc_durations) / total_rpc_trajectories
            if total_rpc_trajectories > 0
            else 0.0
        ),
        "truncated_attempts": truncated_attempts,
        "soft_text_truncations": soft_text_truncations,
        "repetition_recovery_enabled": recover_repetition_truncations,
        "repetition_recovered_samples": repetition_recovered_samples,
        "repetition_removed_tokens": repetition_removed_tokens,
        "repetition_recovery_failures": repetition_recovery_failures,
        "forced_truncation_zero": forced_zero_count,
        "ignore_truncated_samples": ignore_truncated_samples,
        "exclude_truncated_from_advantage_stats": exclude_truncated_from_advantage_stats,
        "masked_truncated_samples": masked_truncated_samples,
        "effective_train_samples": effective_train_samples,
        "effective_train_tokens": effective_train_tokens,
        "groups_with_zero_trainable_samples": groups_with_zero_trainable_samples,
        "groups_with_one_trainable_sample": groups_with_one_trainable_sample,
        "trajectory_multiplier": total_jobs_created
        / float(target_prompts * candidates_per_prompt),
        "dp_workers": len(worker_groups),
        "elapsed_s": elapsed,
        "worker_busy_s": worker_busy,
        "worker_jobs": worker_jobs,
        "busy_imbalance_ratio": (
            max_busy / (sum(worker_busy.values()) / len(worker_busy))
            if worker_busy and sum(worker_busy.values()) > 0
            else 0.0
        ),
        "scheduler_idle_fraction": (
            1.0 - sum(worker_busy.values()) / total_capacity
            if total_capacity > 0
            else 0.0
        ),
        "reward_group_time_s": reward_times,
        "reward_group_time_sum_s": sum(reward_times.values()),
        "reward_group_time_mean_s": (
            sum(reward_times.values()) / len(reward_times) if reward_times else 0.0
        ),
        "reward_group_time_p50_s": _percentile(list(reward_times.values()), 0.50),
        "reward_group_time_p90_s": _percentile(list(reward_times.values()), 0.90),
        "reward_group_time_max_s": max(reward_times.values(), default=0.0),
        "reward_completed_groups": len(reward_times),
        "reward_thread_safe": reward_thread_safe,
        "profiled_trajectories": profiled_trajectories,
        "profiled_prefix_worker_s": profile_totals["prefix_s"],
        "profiled_text_decode_worker_s": profile_totals["text_decode_s"],
        "profiled_diffusion_worker_s": profile_totals["diffusion_s"],
        "profiled_image_reencode_worker_s": profile_totals["image_reencode_s"],
        "profiled_compute_worker_s": profiled_compute,
        "profiled_other_busy_worker_s": max(0.0, total_worker_busy - profiled_compute),
        "profiled_generated_text_tokens": profile_totals["generated_text_tokens"],
        "profiled_generated_images": profile_totals["generated_images"],
        "profiled_diffusion_forwards": profile_totals["diffusion_forwards"],
        "profiled_text_forward_calls": profile_totals[
            "text_forward_calls_share"
        ],
        "profiled_text_forward_rows": profile_totals["text_forward_rows"],
        "profiled_average_text_batch_size": (
            profile_totals["text_forward_rows"]
            / profile_totals["text_forward_calls_share"]
            if profile_totals["text_forward_calls_share"] > 0
            else 0.0
        ),
        "profiled_continuous_refills": profile_totals[
            "continuous_refill_count"
        ],
        "profiled_continuous_coalesce_calls": profile_totals[
            "continuous_coalesce_calls"
        ],
        "profiled_continuous_ready_groups": profile_totals[
            "continuous_ready_groups"
        ],
        "profiled_continuous_compatible_groups": profile_totals[
            "continuous_compatible_groups"
        ],
        "profiled_continuous_underfilled_slots": profile_totals[
            "continuous_underfilled_slots"
        ],
        "profiled_continuous_refilled_slots": profile_totals[
            "continuous_refilled_slots"
        ],
        "profiled_continuous_incompatible_groups": profile_totals[
            "continuous_incompatible_groups"
        ],
        "profiled_strict_compatibility_group_rate": (
            profile_totals["continuous_compatible_groups"]
            / profile_totals["continuous_ready_groups"]
            if profile_totals["continuous_ready_groups"] > 0
            else 0.0
        ),
        "profiled_slot_refill_efficiency": (
            profile_totals["continuous_refilled_slots"]
            / profile_totals["continuous_underfilled_slots"]
            if profile_totals["continuous_underfilled_slots"] > 0
            else 0.0
        ),
        "profiled_compute_busy_fraction": (
            profiled_compute / total_worker_busy if total_worker_busy > 0 else 0.0
        ),
        "profiled_diffusion_fraction": (
            profile_totals["diffusion_s"] / profiled_compute
            if profiled_compute > 0
            else 0.0
        ),
    }
    logger.info(
        "[RUN][SCHEDULER] rollout=%d trajectories=%d rpc_jobs=%d "
        "logical_rpc_jobs=%d chunk=%d accepted=%d/%d replacements=%d "
        "truncated=%d soft_text=%d forced_zero=%d padding=%d elapsed_s=%.1f",
        rollout_id,
        total_jobs_created,
        len(rpc_durations),
        total_rpc_jobs_created,
        rollout_chunk_size,
        min(len(accepted), target_prompts),
        target_prompts,
        replacement_count,
        truncated_attempts,
        soft_text_truncations,
        forced_zero_count,
        scheduler_metrics["padding_groups"],
        elapsed,
    )
    if profiled_trajectories:
        logger.info(
            "[RUN][ROLLOUT_PHASES] rollout=%d trajectories=%d "
            "prefix_worker_s=%.1f text_worker_s=%.1f diffusion_worker_s=%.1f "
            "reencode_worker_s=%.1f "
            "other_busy_worker_s=%.1f diffusion_share=%.1f%%",
            rollout_id,
            profiled_trajectories,
            profile_totals["prefix_s"],
            profile_totals["text_decode_s"],
            profile_totals["diffusion_s"],
            profile_totals["image_reencode_s"],
            scheduler_metrics["profiled_other_busy_worker_s"],
            100.0 * scheduler_metrics["profiled_diffusion_fraction"],
        )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[DEBUG][SCHEDULER] %s", json.dumps(scheduler_metrics, sort_keys=True)
        )

    return DynamicRolloutResult(
        resp=RolloutResp(tracks={"ar": ordered_track}),
        owner_ranks=stable_owner_ranks,
        selected_prompt_indices=selected_prompt_indices,
        consumed_prompt_indices=consumed_prompt_indices,
        metrics={
            key: float(value)
            for key, value in scheduler_metrics.items()
            if isinstance(value, (int, float, bool))
        },
    )


__all__ = [
    "CandidateSlotState",
    "DynamicRolloutResult",
    "PromptGroupState",
    "TrajectoryJob",
    "_build_jobs",
    "_pop_live_microbundle",
    "_dp_worker_groups",
    "_is_informative",
    "_percentile",
    "_score_group",
    "_stable_trajectory_seed",
    "_stop_reason",
    "post_advantage_owner_affine_train_permutation",
    "run_dynamic_trainside_rollout",
]
