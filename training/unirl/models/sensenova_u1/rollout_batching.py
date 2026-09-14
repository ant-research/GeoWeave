"""Single-GPU batching scheduler for GeoWeave interleaved rollout."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
from torch import Tensor

from . import rl_ops
from .rollout_metrics import SampleRolloutMetrics


class RolloutPhase(str, Enum):
    """Resumable phase of one interleaved rollout trajectory."""

    READY_TEXT = "ready_text"
    READY_IMAGE_START = "ready_image_start"
    READY_DIFFUSION = "ready_diffusion"
    READY_REENCODE = "ready_reencode"
    FINISHED = "finished"


@dataclass
class _State:
    sample_index: int
    metrics: SampleRolloutMetrics
    image_size: Tuple[int, int]
    tokens: List[int] = field(default_factory=list)
    logps: List[float] = field(default_factory=list)
    images: List[Tensor] = field(default_factory=list)
    boundaries: List[Tuple[int, int]] = field(default_factory=list)
    latent_segments: List[Any] = field(default_factory=list)
    segment_start: int = 0
    num_images: int = 0
    phase: RolloutPhase = RolloutPhase.READY_TEXT

    @property
    def finished(self) -> bool:
        return self.phase is RolloutPhase.FINISHED

    @finished.setter
    def finished(self, value: bool) -> None:
        if value:
            self.phase = RolloutPhase.FINISHED
        elif self.phase is RolloutPhase.FINISHED:
            self.phase = RolloutPhase.READY_TEXT


@dataclass
class _TextGroup:
    states: List[_State]
    cache: Any
    t_idx: int
    uncond_cache: Any | None
    uncond_t_idx: int | None
    next_tokens: Tensor
    next_logps: Tensor


@dataclass
class _CoalesceStats:
    calls: int = 0
    ready_groups: int = 0
    compatible_groups: int = 0
    underfilled_slots: int = 0
    refilled_slots: int = 0
    incompatible_groups: int = 0
    compactions: int = 0

    def add_(self, other: "_CoalesceStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class _ImageGroup:
    states: List[_State]
    cache: Any
    t_idx: int
    uncond_cache: Any | None
    uncond_t_idx: int | None


@dataclass
class _ReencodeGroup:
    states: List[_State]
    cache: Any
    t_idx: int
    uncond_cache: Any | None
    uncond_t_idx: int | None
    images: Tensor


class _PhaseTimer:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.pending = []

    def start(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(self, started, states: List[_State], attr: str) -> None:
        if isinstance(started, torch.cuda.Event):
            ended = torch.cuda.Event(enable_timing=True)
            ended.record()
            self.pending.append((started, ended, list(states), attr))
            return
        elapsed = time.perf_counter() - started
        share = elapsed / max(len(states), 1)
        for state in states:
            setattr(state.metrics, attr, getattr(state.metrics, attr) + share)

    def finalize(self) -> None:
        if not self.pending:
            return
        torch.cuda.synchronize(self.device)
        pending, self.pending = self.pending, []
        for started, ended, states, attr in pending:
            share = started.elapsed_time(ended) / 1000.0 / max(len(states), 1)
            for state in states:
                setattr(state.metrics, attr, getattr(state.metrics, attr) + share)


def _chunks(size: int, chunk_size: int) -> Sequence[Tuple[int, int]]:
    return [
        (start, min(start + chunk_size, size)) for start in range(0, size, chunk_size)
    ]


def _subcache(cache: Any, start: int, end: int, total: int) -> Any:
    if start == 0 and end == total:
        return cache
    device = cache.layers[0].keys.device
    indices = torch.arange(start, end, dtype=torch.long, device=device)
    return rl_ops.clone_select_kv_cache(cache, indices)


def _partition_cache(cache: Any, rows: List[int], total: int) -> Any:
    if len(rows) == total and rows == list(range(total)):
        return cache
    device = cache.layers[0].keys.device
    return rl_ops.clone_select_kv_cache(
        cache, torch.tensor(rows, dtype=torch.long, device=device)
    )


def _partition_optional_cache(
    cache: Any | None, rows: List[int], total: int
) -> Any | None:
    if cache is None:
        return None
    return _partition_cache(cache, rows, total)


def _subcache_optional(
    cache: Any | None, start: int, end: int, total: int
) -> Any | None:
    if cache is None:
        return None
    return _subcache(cache, start, end, total)


def _cache_seq_len(cache: Any | None) -> int:
    return -1 if cache is None else int(cache.get_seq_length())


def _concat_caches(caches: Sequence[Any]) -> Any:
    """Concatenate equal-length DynamicCache objects along their batch axis."""
    if not caches:
        raise ValueError("cannot concatenate an empty cache list")
    if len(caches) == 1:
        return caches[0]
    from transformers.cache_utils import DynamicCache

    layer_count = len(caches[0].layers)
    if any(len(cache.layers) != layer_count for cache in caches):
        raise ValueError("continuous batching received caches with different layers")
    merged = []
    for layer_index in range(layer_count):
        keys = [cache.layers[layer_index].keys for cache in caches]
        values = [cache.layers[layer_index].values for cache in caches]
        if len({tuple(tensor.shape[1:]) for tensor in keys}) != 1:
            raise ValueError("continuous batching can only merge equal-shaped KV caches")
        merged.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
    return DynamicCache(ddp_cache_data=merged)


def _concat_optional_caches(caches: Sequence[Any | None]) -> Any | None:
    if all(cache is None for cache in caches):
        return None
    if any(cache is None for cache in caches):
        raise ValueError("continuous batching mixed conditional and non-CFG caches")
    return _concat_caches([cache for cache in caches if cache is not None])


def _text_group_key(group: _TextGroup) -> Tuple[int, int, int, int]:
    return (
        int(group.t_idx),
        _cache_seq_len(group.cache),
        -1 if group.uncond_t_idx is None else int(group.uncond_t_idx),
        _cache_seq_len(group.uncond_cache),
    )


def _coalesce_text_groups(
    groups: List[_TextGroup], text_batch_size: int
) -> Tuple[List[_TextGroup], _CoalesceStats]:
    """Regroup compatible rows so completed/diverged rows can free text slots.

    This is the low-risk continuous-batching path: it never pads KV. Rows are
    merged only when both conditional and unconditional caches have identical
    sequence lengths and temporal indices.
    """
    stats = _CoalesceStats(calls=1, ready_groups=len(groups))
    if len(groups) < 2:
        stats.incompatible_groups = len(groups)
        return groups, stats

    buckets: Dict[Tuple[int, int, int, int], List[_TextGroup]] = {}
    order: List[Tuple[int, int, int, int]] = []
    for group in groups:
        key = _text_group_key(group)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(group)

    regrouped: List[_TextGroup] = []
    for key in order:
        bucket = buckets[key]
        if len(bucket) > 1:
            stats.compatible_groups += len(bucket)
        else:
            stats.incompatible_groups += 1
        states = [state for group in bucket for state in group.states]
        empty_before = sum(
            max(text_batch_size - len(group.states), 0) for group in bucket
        )
        stats.underfilled_slots += empty_before
        expected_sizes = [
            end - start for start, end in _chunks(len(states), text_batch_size)
        ]
        if [len(group.states) for group in bucket] == expected_sizes:
            regrouped.extend(bucket)
            continue

        cache = _concat_caches([group.cache for group in bucket])
        uncond_cache = _concat_optional_caches(
            [group.uncond_cache for group in bucket]
        )
        next_tokens = torch.cat([group.next_tokens for group in bucket], dim=0)
        next_logps = torch.cat([group.next_logps for group in bucket], dim=0)
        if len(bucket) > 1:
            stats.compactions += len(bucket) - 1
        total = len(states)
        empty_after = sum(
            text_batch_size - (end - start)
            for start, end in _chunks(total, text_batch_size)
        )
        stats.refilled_slots += max(empty_before - empty_after, 0)
        for start, end in _chunks(total, text_batch_size):
            regrouped.append(
                _TextGroup(
                    states=states[start:end],
                    cache=_subcache(cache, start, end, total),
                    t_idx=key[0],
                    uncond_cache=_subcache_optional(
                        uncond_cache, start, end, total
                    ),
                    uncond_t_idx=None if key[2] < 0 else key[2],
                    next_tokens=next_tokens[start:end],
                    next_logps=next_logps[start:end],
                )
            )
    return regrouped, stats


def _record_forward(states: List[_State], phase: str) -> None:
    batch_size = max(len(states), 1)
    calls_share_attr = f"{phase}_forward_calls_share"
    rows_attr = f"{phase}_forward_rows"
    for state in states:
        setattr(
            state.metrics,
            calls_share_attr,
            getattr(state.metrics, calls_share_attr) + 1.0 / batch_size,
        )
        setattr(state.metrics, rows_attr, getattr(state.metrics, rows_attr) + 1)


def batched_interleave_decode(
    model: Any,
    tokenizer: Any,
    base_cache: Any,
    t_idx: int,
    *,
    base_uncond_cache: Any | None,
    uncond_t_idx: int | None,
    start_logits: Tensor,
    states: List[_State],
    sample_fn: Callable[..., Tuple[Tensor, Tensor]],
    max_new_tokens: int,
    max_images: int,
    stop_ids: List[int],
    img_start_token_id: int,
    diffuse_batch_fn: Callable[..., Tuple[Tensor, Any, int]],
    text_batch_size: int,
    diffusion_batch_size: int,
    reencode_batch_size: int,
    num_diffusion_steps: int,
    device: torch.device,
    continuous_batching: bool = False,
    initial_text_groups: List[_TextGroup] | None = None,
    admit_fn: Callable[[int, bool], Tuple[List[_TextGroup], bool]] | None = None,
    max_active_states: int | None = None,
    finish_fn: Callable[[List[_State]], None] | None = None,
) -> None:
    """Decode one request pool with resumable phase queues.

    The compatibility path preserves the original static text groups. With
    ``continuous_batching=True``, compatible text groups are coalesced before
    each token step, allowing rows freed by EOS/image divergence to be filled
    from the rest of the local request pool without padded/ragged KV support.
    """
    stop_set = set(stop_ids)
    timer = _PhaseTimer(device)
    text_groups: List[_TextGroup] = list(initial_text_groups or [])
    if initial_text_groups is None:
        logits_2d = (
            start_logits[:, -1, :] if start_logits.dim() == 3 else start_logits
        )
        for start, end in _chunks(len(states), text_batch_size):
            chunk_states = states[start:end]
            cache = rl_ops.repeat_kv_cache(base_cache, len(chunk_states))
            uncond_cache = (
                rl_ops.repeat_kv_cache(base_uncond_cache, len(chunk_states))
                if base_uncond_cache is not None
                else None
            )
            logits = logits_2d.expand(len(chunk_states), -1)
            next_tokens, next_logps = sample_fn(
                logits, [state.sample_index for state in chunk_states]
            )
            text_groups.append(
                _TextGroup(
                    chunk_states,
                    cache,
                    t_idx,
                    uncond_cache,
                    uncond_t_idx,
                    next_tokens,
                    next_logps,
                )
            )

    active_limit = max_active_states
    if active_limit is not None and active_limit < 1:
        raise ValueError("max_active_states must be >= 1 when set")
    if (
        active_limit is not None
        and sum(len(group.states) for group in text_groups) > active_limit
    ):
        raise ValueError("initial text groups exceed max_active_states")
    admission_closed = admit_fn is None

    def admit(*, block_when_idle: bool) -> None:
        nonlocal admission_closed
        if admit_fn is None or admission_closed:
            return
        active = sum(len(group.states) for group in text_groups)
        capacity = (active_limit - active) if active_limit is not None else 0
        if capacity <= 0:
            return
        admitted, admission_closed = admit_fn(capacity, block_when_idle)
        admitted_count = sum(len(group.states) for group in admitted)
        if admitted_count > capacity:
            raise ValueError(
                f"admit_fn returned {admitted_count} states for capacity {capacity}"
            )
        text_groups.extend(admitted)

    coalesce_stats = _CoalesceStats()
    emitted_finished: set[int] = set()

    def emit_finished() -> None:
        if finish_fn is None:
            return
        finished = [
            state
            for state in states
            if state.finished and id(state) not in emitted_finished
        ]
        if not finished:
            return
        timer.finalize()
        for state in finished:
            state.metrics.generated_text_tokens = len(state.tokens)
            state.metrics.generated_images = len(state.images)
            state.metrics.total_time = (
                state.metrics.prefix_time
                + state.metrics.text_decode_time
                + state.metrics.diffusion_time
                + state.metrics.image_reencode_time
            )
            emitted_finished.add(id(state))
        finish_fn(finished)

    while text_groups or not admission_closed:
        admit(block_when_idle=not text_groups)
        if not text_groups:
            if admission_closed:
                break
            continue
        if continuous_batching:
            text_groups, step_stats = _coalesce_text_groups(
                text_groups, text_batch_size
            )
            coalesce_stats.add_(step_stats)

        image_groups: List[_ImageGroup] = []
        next_text_groups: List[_TextGroup] = []

        for group in text_groups:
            token_values = group.next_tokens.detach().tolist()
            logp_values = group.next_logps.detach().float().tolist()
            regular_rows: List[int] = []
            image_rows: List[int] = []

            for row, (state, token_id, logp) in enumerate(
                zip(group.states, token_values, logp_values)
            ):
                if state.phase is not RolloutPhase.READY_TEXT:
                    raise RuntimeError(
                        f"text queue contains trajectory in phase {state.phase}"
                    )
                if token_id in stop_set:
                    state.tokens.append(token_id)
                    state.logps.append(logp)
                    state.boundaries.append((state.segment_start, len(state.tokens)))
                    state.metrics.stop_reason = "eos"
                    state.phase = RolloutPhase.FINISHED
                    continue

                if token_id == img_start_token_id:
                    if state.num_images >= max_images:
                        state.boundaries.append(
                            (state.segment_start, len(state.tokens))
                        )
                        state.metrics.stop_reason = "max_images"
                        state.phase = RolloutPhase.FINISHED
                        continue
                    state.tokens.append(token_id)
                    state.logps.append(logp)
                    state.boundaries.append((state.segment_start, len(state.tokens)))
                    state.segment_start = len(state.tokens)
                    state.phase = RolloutPhase.READY_IMAGE_START
                    image_rows.append(row)
                    continue

                state.tokens.append(token_id)
                state.logps.append(logp)
                if len(state.tokens) >= max_new_tokens:
                    state.boundaries.append((state.segment_start, len(state.tokens)))
                    state.metrics.stop_reason = "max_new_tokens"
                    state.phase = RolloutPhase.FINISHED
                else:
                    regular_rows.append(row)

            if regular_rows:
                regular_cache = _partition_cache(
                    group.cache, regular_rows, len(group.states)
                )
                regular_uncond_cache = _partition_optional_cache(
                    group.uncond_cache, regular_rows, len(group.states)
                )
                regular_states = [group.states[row] for row in regular_rows]
                input_ids = group.next_tokens.index_select(
                    0, torch.tensor(regular_rows, device=device)
                ).unsqueeze(1)
                started = timer.start()
                model.language_model.model.current_index = group.t_idx
                outputs = model.language_model(
                    input_ids=input_ids,
                    past_key_values=regular_cache,
                    use_cache=True,
                )
                _record_forward(regular_states, "text")
                next_tokens, next_logps = sample_fn(
                    outputs.logits[:, -1, :],
                    [state.sample_index for state in regular_states],
                )
                timer.stop(started, regular_states, "text_decode_time")
                next_text_groups.append(
                    _TextGroup(
                        regular_states,
                        outputs.past_key_values,
                        group.t_idx + 1,
                        regular_uncond_cache,
                        group.uncond_t_idx,
                        next_tokens,
                        next_logps,
                    )
                )

            if image_rows:
                image_cache = _partition_cache(
                    group.cache, image_rows, len(group.states)
                )
                image_uncond_cache = _partition_optional_cache(
                    group.uncond_cache, image_rows, len(group.states)
                )
                image_states = [group.states[row] for row in image_rows]
                img_ids = torch.full(
                    (len(image_states), 1),
                    img_start_token_id,
                    dtype=torch.long,
                    device=device,
                )
                started = timer.start()
                model.language_model.model.current_index = group.t_idx
                outputs = model.language_model(
                    input_ids=img_ids,
                    past_key_values=image_cache,
                    use_cache=True,
                )
                _record_forward(image_states, "text")
                image_uncond_t_idx = group.uncond_t_idx
                if image_uncond_cache is not None:
                    if image_uncond_t_idx is None:
                        raise RuntimeError(
                            "text CFG cache is missing its temporal index"
                        )
                    model.language_model.model.current_index = image_uncond_t_idx
                    uncond_outputs = model.language_model(
                        input_ids=img_ids,
                        past_key_values=image_uncond_cache,
                        use_cache=True,
                    )
                    _record_forward(image_states, "text_cfg")
                    image_uncond_cache = uncond_outputs.past_key_values
                    image_uncond_t_idx += 1
                timer.stop(started, image_states, "text_decode_time")
                for state in image_states:
                    state.phase = RolloutPhase.READY_DIFFUSION
                image_groups.append(
                    _ImageGroup(
                        image_states,
                        outputs.past_key_values,
                        group.t_idx + 1,
                        image_uncond_cache,
                        image_uncond_t_idx,
                    )
                )

        reencode_groups: List[_ReencodeGroup] = []
        for group in image_groups:
            if any(
                state.phase is not RolloutPhase.READY_DIFFUSION
                for state in group.states
            ):
                raise RuntimeError("diffusion queue contains an invalid trajectory phase")
            image_size = group.states[0].image_size
            if any(state.image_size != image_size for state in group.states):
                raise ValueError("diffusion batch contains mixed image sizes")
            total = len(group.states)
            for start, end in _chunks(total, diffusion_batch_size):
                chunk_states = group.states[start:end]
                chunk_cache = _subcache(group.cache, start, end, total)
                chunk_uncond_cache = _subcache_optional(
                    group.uncond_cache, start, end, total
                )
                started = timer.start()
                images, segment, cfg_steps = diffuse_batch_fn(
                    chunk_cache,
                    group.t_idx,
                    chunk_uncond_cache,
                    group.uncond_t_idx,
                    image_size,
                    len(chunk_states),
                    [state.sample_index for state in chunk_states],
                    [state.num_images for state in chunk_states],
                )
                _record_forward(chunk_states, "diffusion")
                timer.stop(started, chunk_states, "diffusion_time")
                for row, state in enumerate(chunk_states):
                    state.metrics.diffusion_forwards += num_diffusion_steps + cfg_steps
                    state.images.append(images[row].detach().cpu())
                    state.latent_segments.append(segment.slice(row, row + 1))
                    state.num_images += 1
                    state.metrics.image_sizes.append(image_size)
                    state.phase = RolloutPhase.READY_REENCODE
                reencode_groups.append(
                    _ReencodeGroup(
                        chunk_states,
                        chunk_cache,
                        group.t_idx,
                        chunk_uncond_cache,
                        group.uncond_t_idx,
                        images,
                    )
                )

        for group in reencode_groups:
            if any(
                state.phase is not RolloutPhase.READY_REENCODE
                for state in group.states
            ):
                raise RuntimeError("reencode queue contains an invalid trajectory phase")
            total = len(group.states)
            for start, end in _chunks(total, reencode_batch_size):
                sub_states = group.states[start:end]
                sub_cache = _subcache(group.cache, start, end, total)
                sub_uncond_cache = _subcache_optional(
                    group.uncond_cache, start, end, total
                )
                sub_images = group.images[start:end]
                started = timer.start()
                cache_len_before = int(sub_cache.get_seq_length())
                cache_out, new_t_idx, logits = rl_ops.append_images_to_cache(
                    model,
                    tokenizer,
                    sub_cache,
                    group.t_idx,
                    sub_images,
                    device=device,
                )
                _record_forward(sub_states, "reencode")
                image_context_tokens = (
                    int(cache_out.get_seq_length()) - cache_len_before
                )
                for state in sub_states:
                    state.metrics.image_context_tokens += image_context_tokens
                new_uncond_t_idx = group.uncond_t_idx
                if sub_uncond_cache is not None:
                    if new_uncond_t_idx is None:
                        raise RuntimeError(
                            "text CFG cache is missing its temporal index"
                        )
                    sub_uncond_cache, new_uncond_t_idx, _ = (
                        rl_ops.append_images_to_cache(
                            model,
                            tokenizer,
                            sub_uncond_cache,
                            new_uncond_t_idx,
                            sub_images,
                            device=device,
                        )
                    )
                    _record_forward(sub_states, "reencode_cfg")
                timer.stop(started, sub_states, "image_reencode_time")
                if all(len(state.tokens) >= max_new_tokens for state in sub_states):
                    for state in sub_states:
                        state.metrics.stop_reason = "max_new_tokens"
                        state.phase = RolloutPhase.FINISHED
                    continue
                next_tokens, next_logps = sample_fn(
                    logits[:, -1, :],
                    [state.sample_index for state in sub_states],
                )
                for state in sub_states:
                    state.phase = RolloutPhase.READY_TEXT
                next_text_groups.append(
                    _TextGroup(
                        sub_states,
                        cache_out,
                        new_t_idx,
                        sub_uncond_cache,
                        new_uncond_t_idx,
                        next_tokens,
                        next_logps,
                    )
                )

        text_groups = next_text_groups
        emit_finished()

    timer.finalize()
    emit_finished()
    for state in states:
        state.metrics.generated_text_tokens = len(state.tokens)
        state.metrics.generated_images = len(state.images)
        share = 1.0 / max(len(states), 1)
        state.metrics.continuous_refill_count = coalesce_stats.compactions * share
        state.metrics.continuous_coalesce_calls = coalesce_stats.calls * share
        state.metrics.continuous_ready_groups = coalesce_stats.ready_groups * share
        state.metrics.continuous_compatible_groups = (
            coalesce_stats.compatible_groups * share
        )
        state.metrics.continuous_underfilled_slots = (
            coalesce_stats.underfilled_slots * share
        )
        state.metrics.continuous_refilled_slots = coalesce_stats.refilled_slots * share
        state.metrics.continuous_incompatible_groups = (
            coalesce_stats.incompatible_groups * share
        )
        state.metrics.total_time = (
            state.metrics.prefix_time
            + state.metrics.text_decode_time
            + state.metrics.diffusion_time
            + state.metrics.image_reencode_time
        )


__all__ = [
    "RolloutPhase",
    "_CoalesceStats",
    "_State",
    "_TextGroup",
    "_coalesce_text_groups",
    "batched_interleave_decode",
]
