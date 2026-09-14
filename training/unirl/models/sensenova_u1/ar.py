"""GeoWeave AR stage — text thinking/reasoning generation and replay.

Mirrors :class:`unirl.models.bagel.ar.BagelARStage`.

Supports two modes:
- **Standard**: generate text until EOS (for t2t/VQA tasks)
- **Interleave**: alternate text generation and image generation, with
  images re-encoded through the understanding ViT between text segments
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype
from unirl.utils.memory_utils import log_memory_usage

from . import rl_ops
from .conditions import SensenovaU1ARConditions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass
class SensenovaU1LiveSample:
    query: str
    rollout_id: str
    prompt_id: str
    rewrite_id: int
    image_size: Tuple[int, int]
    trajectory_seed: int
    noise_group_id: str
    pixel_values: Optional[Tensor] = None
    grid_hw: Optional[Tensor] = None
    payload: Any = None
    sample_index: int = -1


@dataclass
class SensenovaU1ARParams:
    """Per-request AR knobs."""

    stop_token_ids: List[int] = field(default_factory=list)
    system_message: str = ""
    interleave: bool = True
    max_images: int = 10
    image_size: Tuple[int, int] = (256, 256)
    rollout_text_batch_size: int = 8
    continuous_batching: bool = False
    continuous_request_admission: bool = False
    continuous_rollout_pool_size: int = 4


# ---------------------------------------------------------------------------
# Step kernel
# ---------------------------------------------------------------------------


class SensenovaU1ARStep:
    """Per-token sampling kernel.

    Computes full-softmax log-prob BEFORE top-k/top-p truncation so rollout
    ``old_logp`` matches replay ``new_logp`` at the same weights.
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        generators: Optional[Dict[int, torch.Generator]] = None,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.generators = generators

    def step(
        self, logits: Tensor, sample_indices: Optional[Sequence[int]] = None
    ) -> Tuple[Tensor, Tensor]:
        """``(logits [B, V]) -> (token_id [B], log_prob [B])``."""
        if logits.dim() == 3:
            logits = logits[:, -1, :]  # [B, V]

        if self.temperature <= 0:
            log_probs_full = torch.log_softmax(logits.float(), dim=-1)
            token_ids = log_probs_full.argmax(dim=-1)
            log_prob = log_probs_full.gather(1, token_ids.unsqueeze(1)).squeeze(1)
            return token_ids, log_prob

        scaled = logits.float() / self.temperature
        log_probs_full = torch.log_softmax(scaled, dim=-1)

        # Apply top-k
        filtered = scaled.clone()
        if self.top_k > 0:
            topk_vals, _ = torch.topk(filtered, min(self.top_k, filtered.shape[-1]))
            threshold = topk_vals[:, -1].unsqueeze(-1)
            filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

        # Apply top-p (nucleus)
        if self.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(filtered, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumprobs - torch.softmax(sorted_logits, dim=-1) >= self.top_p
            sorted_logits[remove_mask] = float("-inf")
            filtered = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        # Sample from truncated distribution. Dynamic chunk jobs optionally
        # carry one generator per candidate so scheduling/chunk composition does
        # not couple their random streams. The fixed path stays vectorized.
        probs = torch.softmax(filtered, dim=-1)
        if self.generators is None:
            token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            if sample_indices is None or len(sample_indices) != probs.shape[0]:
                raise ValueError(
                    "per-candidate sampling requires one sample index per logits row"
                )
            token_ids = torch.stack(
                [
                    torch.multinomial(
                        probs[row],
                        num_samples=1,
                        generator=self.generators[int(sample_index)],
                    ).squeeze(0)
                    for row, sample_index in enumerate(sample_indices)
                ]
            )

        # Return log-prob from the FULL (pre-truncation) distribution
        log_prob = log_probs_full.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        return token_ids, log_prob


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class SensenovaU1ARStage:
    """AR stage for GeoWeave text generation and replay.

    Implements the ``ARStage[SensenovaU1ARConditions]`` protocol.
    """

    def __init__(
        self,
        *,
        model: Any,  # GeoWeave model bundle
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model  # GeoWeave model bundle
        self._autocast_dtype = parse_torch_dtype(autocast_precision)
        self._logprob_dtype = parse_torch_dtype(logprob_precision)

    def _autocast_ctx(self, device: Any = "cuda"):
        if self._autocast_dtype in (torch.bfloat16, torch.float16):
            return torch.autocast("cuda", dtype=self._autocast_dtype)
        from contextlib import nullcontext

        return nullcontext()

    def _resolve_stop_ids(
        self, params: SensenovaU1ARParams, sampling_params: ARSamplingParams
    ) -> List[int]:
        """Union of configured stop IDs + EOS."""
        stop = set(params.stop_token_ids)
        tokenizer = self.model.tokenizer
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            stop.add(eos_id)
        for tok_str in ["<|im_end|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
            if tid is not None and tid != tokenizer.unk_token_id:
                stop.add(tid)
        if hasattr(sampling_params, "stop_token_id") and sampling_params.stop_token_id:
            stop.add(sampling_params.stop_token_id)
        return list(stop)

    # ------------------------------------------------------------------
    # Rollout: autoregress
    # ------------------------------------------------------------------

    def autoregress(
        self,
        conditions: SensenovaU1ARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[SensenovaU1ARParams] = None,
        diffuse_fn: Optional[Any] = None,
        diffuse_batch_fn: Optional[Any] = None,
        diffusion_batch_size: int = 2,
        reencode_batch_size: int = 2,
        num_diffusion_steps: int = 30,
        guidance_scale: float = 1.0,
        rollout_step: int = 0,
        trajectory_seeds: Optional[Sequence[int]] = None,
        live_admit_fn: Optional[
            Callable[[int, bool], Tuple[List[SensenovaU1LiveSample], bool]]
        ] = None,
        live_finish_fn: Optional[
            Callable[[SensenovaU1LiveSample, Any], None]
        ] = None,
        **_kwargs: Any,
    ) -> TextSegment:
        """Generate text (with optional interleaved images) under ``torch.no_grad``.

        Args:
            conditions: per-sample prompt queries.
            sampling_params: temperature, top_k, top_p, max_new_tokens, etc.
            params: AR-specific knobs (interleave, max_images, etc.).
            diffuse_fn: for interleave mode, callable to generate images.
                Signature: ``(past_kv, text_len, image_size) -> (image [3,H,W], info)``

        Returns:
            TextSegment with packed tokens and log-probs.
            For interleave mode, also populates conditions.generated_images
            and conditions.text_segment_boundaries.
        """
        if params is None:
            params = SensenovaU1ARParams()

        bundle = self.model
        model = bundle.model
        tokenizer = bundle.tokenizer
        device = next(model.parameters()).device

        sample_generators = {} if live_admit_fn is not None else None
        if trajectory_seeds is not None:
            if len(trajectory_seeds) != conditions.batch_size:
                raise ValueError(
                    "trajectory_seeds must align with the expanded AR batch"
                )
            sample_generators = {}
            for sample_index, seed in enumerate(trajectory_seeds):
                generator = torch.Generator(device=device)
                generator.manual_seed(int(seed))
                sample_generators[sample_index] = generator

        step_kernel = SensenovaU1ARStep(
            temperature=sampling_params.temperature,
            top_p=getattr(sampling_params, "top_p", 1.0),
            top_k=getattr(sampling_params, "top_k", 0),
            generators=sample_generators,
        )

        stop_ids = self._resolve_stop_ids(params, sampling_params)
        max_new = getattr(sampling_params, "max_new_tokens", 1024)

        all_sample_tokens: List[Tensor] = []
        all_sample_logps: List[Tensor] = []
        sample_metrics = []

        if params.rollout_text_batch_size < 1:
            raise ValueError("rollout_text_batch_size must be >= 1")

        from .rollout_batching import _State, batched_interleave_decode
        from .rollout_metrics import SampleRolloutMetrics, distributed_rank_world

        rank, _ = distributed_rank_world()
        # Consecutive samples with the same prompt_id are rewrites of one prompt.
        prompt_ids = (
            conditions.prompt_ids
            if len(conditions.prompt_ids) == conditions.batch_size
            else [f"local-prompt-{i}" for i in range(conditions.batch_size)]
        )
        rewrite_ids = (
            conditions.rewrite_ids
            if len(conditions.rewrite_ids) == conditions.batch_size
            else [0] * conditions.batch_size
        )
        groups: List[List[int]] = []
        for sample_index, prompt_id in enumerate(prompt_ids):
            if not groups or prompt_ids[groups[-1][0]] != prompt_id:
                groups.append([])
            groups[-1].append(sample_index)

        if params.continuous_request_admission:
            if not params.interleave or diffuse_batch_fn is None:
                raise ValueError(
                    "continuous_request_admission requires batched interleave decode"
                )
            active_pool_size = max(1, int(params.continuous_rollout_pool_size))
            from collections import deque

            from .rollout_batching import _TextGroup

            fixed_batch_size = conditions.batch_size
            pending_groups = deque(groups)
            prepared_pending: Deque[_TextGroup] = deque()
            window_states: List[_State] = []
            live_samples: List[SensenovaU1LiveSample] = []
            live_closed = live_admit_fn is None

            def sample_fields(sample_index: int):
                if sample_index < fixed_batch_size:
                    return (
                        conditions.prompt_queries[sample_index],
                        conditions.pixel_values[sample_index]
                        if conditions.pixel_values
                        else None,
                        conditions.grid_hws[sample_index]
                        if conditions.grid_hws
                        else None,
                        conditions.image_sizes[sample_index]
                        if conditions.image_sizes
                        else params.image_size,
                        str(prompt_ids[sample_index]),
                        int(rewrite_ids[sample_index]),
                        conditions.rollout_ids[sample_index]
                        if len(conditions.rollout_ids) == fixed_batch_size
                        else f"{rollout_step}:{rank}:{sample_index}",
                    )
                spec = live_samples[sample_index - fixed_batch_size]
                return (
                    spec.query,
                    spec.pixel_values,
                    spec.grid_hw,
                    spec.image_size,
                    spec.prompt_id,
                    spec.rewrite_id,
                    spec.rollout_id,
                )

            def prepare_group(sample_indices: List[int]) -> List[_TextGroup]:
                first = sample_indices[0]
                query, pv, ghw, _, _, _, _ = sample_fields(first)

                prefix_started = time.perf_counter()
                if pv is not None:
                    input_embeds, indexes, attn_mask = rl_ops.build_it2i_inputs(
                        model, tokenizer, query, pv, ghw
                    )
                    base_cache, hidden = rl_ops.prefix_forward_embeds(
                        model, input_embeds, indexes, attn_mask
                    )
                    start_logits = model.language_model.lm_head(hidden)
                    prompt_tokens = int(input_embeds.shape[1])
                else:
                    input_ids, indexes, attn_mask = rl_ops.build_text_inputs(
                        model, tokenizer, query
                    )
                    base_cache, hidden = rl_ops.prefix_forward(
                        model, input_ids, indexes, attn_mask
                    )
                    start_logits = model.language_model.lm_head(hidden)
                    prompt_tokens = int(input_ids.shape[1])
                start_t_idx = int(indexes[0].max().item())

                use_text_cfg = (
                    guidance_scale > 1.0
                    and params.interleave
                    and params.max_images > 0
                )
                base_uncond_cache = None
                start_uncond_t_idx = None
                if use_text_cfg:
                    uncond_prompt = "<image>" if pv is not None else ""
                    uncond_query = rl_ops.build_query(
                        model,
                        uncond_prompt,
                        system_message=params.system_message,
                    )
                    if pv is not None:
                        if ghw is None:
                            raise RuntimeError("input image is missing grid_hw")
                        uncond_query = rl_ops.insert_image_tokens(
                            uncond_query, [ghw], bundle.downsample_ratio
                        )
                        uncond_embeds, uncond_indexes, uncond_mask = (
                            rl_ops.build_it2i_inputs(
                                model, tokenizer, uncond_query, pv, ghw
                            )
                        )
                        base_uncond_cache, _ = rl_ops.prefix_forward_embeds(
                            model, uncond_embeds, uncond_indexes, uncond_mask
                        )
                    else:
                        uncond_ids, uncond_indexes, uncond_mask = (
                            rl_ops.build_text_inputs(model, tokenizer, uncond_query)
                        )
                        base_uncond_cache, _ = rl_ops.prefix_forward(
                            model, uncond_ids, uncond_indexes, uncond_mask
                        )
                    start_uncond_t_idx = int(uncond_indexes[0].max().item())

                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                prefix_elapsed = time.perf_counter() - prefix_started

                prepared_states: List[_State] = []
                for sample_index in sample_indices:
                    _, _, _, image_size, prompt_id, rewrite_id, rollout_id = (
                        sample_fields(sample_index)
                    )
                    metrics = SampleRolloutMetrics(
                        rollout_id=str(rollout_id),
                        prompt_id=str(prompt_id),
                        group_id=str(prompt_id),
                        rewrite_id=int(rewrite_id),
                        rank=rank,
                        local_sample_index=sample_index,
                        prompt_tokens=prompt_tokens,
                        prefix_time=prefix_elapsed / len(sample_indices),
                    )
                    prepared_states.append(
                        _State(
                            sample_index=sample_index,
                            metrics=metrics,
                            image_size=image_size,
                        )
                    )
                window_states.extend(prepared_states)

                logits_2d = (
                    start_logits[:, -1, :]
                    if start_logits.dim() == 3
                    else start_logits
                )
                prepared_groups: List[_TextGroup] = []
                for state in prepared_states:
                    cache = rl_ops.repeat_kv_cache(base_cache, 1)
                    uncond_cache = (
                        rl_ops.repeat_kv_cache(base_uncond_cache, 1)
                        if base_uncond_cache is not None
                        else None
                    )
                    logits = logits_2d.expand(1, -1)
                    next_tokens, next_logps = step_kernel.step(
                        logits, [state.sample_index]
                    )
                    prepared_groups.append(
                        _TextGroup(
                            states=[state],
                            cache=cache,
                            t_idx=start_t_idx,
                            uncond_cache=uncond_cache,
                            uncond_t_idx=start_uncond_t_idx,
                            next_tokens=next_tokens,
                            next_logps=next_logps,
                        )
                    )
                return prepared_groups

            def admit_groups(
                capacity: int, block_when_idle: bool
            ) -> Tuple[List[_TextGroup], bool]:
                nonlocal live_closed
                admitted: List[_TextGroup] = []
                while len(admitted) < capacity:
                    if prepared_pending:
                        admitted.append(prepared_pending.popleft())
                        continue
                    if pending_groups:
                        prepared_pending.extend(prepare_group(pending_groups.popleft()))
                        continue
                    if live_admit_fn is None or live_closed:
                        break
                    specs, live_closed = live_admit_fn(
                        capacity - len(admitted), block_when_idle and not admitted
                    )
                    if not specs:
                        break
                    grouped_indices: List[List[int]] = []
                    for spec in specs:
                        sample_index = fixed_batch_size + len(live_samples)
                        spec.sample_index = sample_index
                        live_samples.append(spec)
                        if sample_generators is None:
                            raise RuntimeError("live admission requires sample generators")
                        generator = torch.Generator(device=device)
                        generator.manual_seed(int(spec.trajectory_seed))
                        sample_generators[sample_index] = generator
                        if (
                            not grouped_indices
                            or live_samples[
                                grouped_indices[-1][0] - fixed_batch_size
                            ].prompt_id
                            != spec.prompt_id
                        ):
                            grouped_indices.append([])
                        grouped_indices[-1].append(sample_index)
                    for indices in grouped_indices:
                        prepared_pending.extend(prepare_group(indices))
                return (
                    admitted,
                    not pending_groups
                    and not prepared_pending
                    and live_closed,
                )

            def finish_states(finished: List[_State]) -> None:
                if live_finish_fn is None:
                    return
                for state in finished:
                    if state.sample_index < fixed_batch_size:
                        continue
                    spec = live_samples[state.sample_index - fixed_batch_size]
                    live_finish_fn(spec, state)

            with torch.no_grad(), self._autocast_ctx(device):
                batched_interleave_decode(
                    model,
                    tokenizer,
                    None,
                    0,
                    base_uncond_cache=None,
                    uncond_t_idx=None,
                    start_logits=torch.empty(0, device=device),
                    states=window_states,
                    sample_fn=step_kernel.step,
                    max_new_tokens=max_new,
                    max_images=params.max_images,
                    stop_ids=stop_ids,
                    img_start_token_id=bundle.img_start_token_id,
                    diffuse_batch_fn=diffuse_batch_fn,
                    text_batch_size=params.rollout_text_batch_size,
                    diffusion_batch_size=diffusion_batch_size,
                    reencode_batch_size=reencode_batch_size,
                    num_diffusion_steps=num_diffusion_steps,
                    device=device,
                    continuous_batching=params.continuous_batching,
                    initial_text_groups=[],
                    admit_fn=admit_groups,
                    max_active_states=active_pool_size,
                    finish_fn=finish_states,
                )

            for state in sorted(window_states, key=lambda item: item.sample_index):
                i = state.sample_index
                if i < fixed_batch_size:
                    conditions.generated_images[i] = state.images
                    conditions.text_segment_boundaries[i] = state.boundaries
                    conditions.latent_segments[i] = state.latent_segments
                sample_metrics.append(state.metrics)
                all_sample_tokens.append(
                    torch.tensor(state.tokens, dtype=torch.long, device=device)
                )
                all_sample_logps.append(
                    torch.tensor(state.logps, dtype=torch.float32, device=device)
                )
            groups = []

        with torch.no_grad(), self._autocast_ctx(device):
            for sample_indices in groups:
                first = sample_indices[0]
                query = conditions.prompt_queries[first]
                pv = conditions.pixel_values[first] if conditions.pixel_values else None
                ghw = conditions.grid_hws[first] if conditions.grid_hws else None

                prefix_started = time.perf_counter()
                if pv is not None:
                    input_embeds, indexes, attn_mask = rl_ops.build_it2i_inputs(
                        model, tokenizer, query, pv, ghw
                    )
                    base_cache, hidden = rl_ops.prefix_forward_embeds(
                        model, input_embeds, indexes, attn_mask
                    )
                    start_logits = model.language_model.lm_head(hidden)
                    prompt_tokens = int(input_embeds.shape[1])
                else:
                    input_ids, indexes, attn_mask = rl_ops.build_text_inputs(
                        model, tokenizer, query
                    )
                    base_cache, hidden = rl_ops.prefix_forward(
                        model, input_ids, indexes, attn_mask
                    )
                    start_logits = model.language_model.lm_head(hidden)
                    prompt_tokens = int(input_ids.shape[1])
                t_idx = int(indexes[0].max().item())

                # The inference-default CFG branch removes user/generated text
                # while preserving input and previously generated images.
                use_text_cfg = (
                    guidance_scale > 1.0 and params.interleave and params.max_images > 0
                )
                base_uncond_cache = None
                uncond_t_idx = None
                if use_text_cfg:
                    uncond_prompt = "<image>" if pv is not None else ""
                    uncond_query = rl_ops.build_query(
                        model,
                        uncond_prompt,
                        system_message=params.system_message,
                    )
                    if pv is not None:
                        if ghw is None:
                            raise RuntimeError("input image is missing grid_hw")
                        uncond_query = rl_ops.insert_image_tokens(
                            uncond_query, [ghw], bundle.downsample_ratio
                        )
                        uncond_embeds, uncond_indexes, uncond_mask = (
                            rl_ops.build_it2i_inputs(
                                model, tokenizer, uncond_query, pv, ghw
                            )
                        )
                        base_uncond_cache, _ = rl_ops.prefix_forward_embeds(
                            model, uncond_embeds, uncond_indexes, uncond_mask
                        )
                    else:
                        uncond_ids, uncond_indexes, uncond_mask = (
                            rl_ops.build_text_inputs(model, tokenizer, uncond_query)
                        )
                        base_uncond_cache, _ = rl_ops.prefix_forward(
                            model, uncond_ids, uncond_indexes, uncond_mask
                        )
                    uncond_t_idx = int(uncond_indexes[0].max().item())

                if device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                prefix_elapsed = time.perf_counter() - prefix_started

                states = []
                for sample_index in sample_indices:
                    image_size = (
                        conditions.image_sizes[sample_index]
                        if conditions.image_sizes
                        and sample_index < len(conditions.image_sizes)
                        else params.image_size
                    )
                    metrics = SampleRolloutMetrics(
                        rollout_id=(
                            conditions.rollout_ids[sample_index]
                            if len(conditions.rollout_ids) == conditions.batch_size
                            else f"{rollout_step}:{rank}:{sample_index}"
                        ),
                        prompt_id=str(prompt_ids[sample_index]),
                        group_id=str(prompt_ids[sample_index]),
                        rewrite_id=int(rewrite_ids[sample_index]),
                        rank=rank,
                        local_sample_index=sample_index,
                        prompt_tokens=prompt_tokens,
                        prefix_time=prefix_elapsed / len(sample_indices),
                    )
                    states.append(
                        _State(
                            sample_index=sample_index,
                            metrics=metrics,
                            image_size=image_size,
                        )
                    )

                if params.interleave and diffuse_batch_fn is not None:
                    batched_interleave_decode(
                        model,
                        tokenizer,
                        base_cache,
                        t_idx,
                        base_uncond_cache=base_uncond_cache,
                        uncond_t_idx=uncond_t_idx,
                        start_logits=start_logits,
                        states=states,
                        sample_fn=step_kernel.step,
                        max_new_tokens=max_new,
                        max_images=params.max_images,
                        stop_ids=stop_ids,
                        img_start_token_id=bundle.img_start_token_id,
                        diffuse_batch_fn=diffuse_batch_fn,
                        text_batch_size=params.rollout_text_batch_size,
                        diffusion_batch_size=diffusion_batch_size,
                        reencode_batch_size=reencode_batch_size,
                        num_diffusion_steps=num_diffusion_steps,
                        device=device,
                        continuous_batching=params.continuous_batching,
                    )
                    for state in states:
                        i = state.sample_index
                        conditions.generated_images[i] = state.images
                        conditions.text_segment_boundaries[i] = state.boundaries
                        conditions.latent_segments[i] = state.latent_segments
                else:
                    # Compatibility fallback for direct stage users. Prefix is
                    # still reused, but each rewrite owns an independent cache.
                    for state in states:
                        cache = rl_ops.repeat_kv_cache(base_cache, 1)
                        uncond_cache = (
                            rl_ops.repeat_kv_cache(base_uncond_cache, 1)
                            if base_uncond_cache is not None
                            else None
                        )
                        if params.interleave and diffuse_fn is not None:
                            tokens, logps, gen_imgs, boundaries, _, _ = (
                                rl_ops.interleave_decode(
                                    model,
                                    tokenizer,
                                    cache,
                                    t_idx,
                                    text_uncond_cache=uncond_cache,
                                    text_uncond_t_idx=uncond_t_idx,
                                    start_logits=start_logits,
                                    sample_fn=(
                                        lambda logits, sample_index=state.sample_index: step_kernel.step(
                                            logits, [sample_index]
                                        )
                                    ),
                                    max_new_tokens=max_new,
                                    max_images=params.max_images,
                                    stop_ids=stop_ids,
                                    img_start_token_id=bundle.img_start_token_id,
                                    diffuse_fn=diffuse_fn,
                                    image_size=state.image_size,
                                    device=device,
                                    rollout_metrics=state.metrics,
                                )
                            )
                            state.images, state.boundaries = gen_imgs, boundaries
                            conditions.generated_images[state.sample_index] = gen_imgs
                            conditions.text_segment_boundaries[state.sample_index] = (
                                boundaries
                            )
                        else:
                            tokens, logps, _, _ = rl_ops.decode_text(
                                model,
                                tokenizer,
                                cache,
                                t_idx,
                                start_logits=start_logits,
                                sample_fn=(
                                    lambda logits, sample_index=state.sample_index: step_kernel.step(
                                        logits, [sample_index]
                                    )
                                ),
                                max_new_tokens=max_new,
                                stop_ids=stop_ids,
                                device=device,
                            )
                            state.metrics.stop_reason = (
                                "eos"
                                if tokens and tokens[-1] in set(stop_ids)
                                else "max_new_tokens"
                            )
                        state.tokens, state.logps = tokens, logps
                        state.metrics.generated_text_tokens = len(tokens)
                        state.metrics.generated_images = len(state.images)

                sample_metrics.extend(state.metrics for state in states)
                for state in states:
                    all_sample_tokens.append(
                        torch.tensor(state.tokens, dtype=torch.long, device=device)
                    )
                    all_sample_logps.append(
                        torch.tensor(state.logps, dtype=torch.float32, device=device)
                    )

        self.last_rollout_metrics = sample_metrics
        metrics_by_sample = {
            int(metric.local_sample_index): metric for metric in sample_metrics
        }
        conditions.stop_reasons = [
            metrics_by_sample[i].stop_reason if i in metrics_by_sample else "unknown"
            for i in range(conditions.batch_size)
        ]
        conditions.image_context_token_counts = [
            (
                int(metrics_by_sample[i].image_context_tokens)
                if i in metrics_by_sample
                else 0
            )
            for i in range(conditions.batch_size)
        ]
        conditions.rollout_phase_metrics = [
            (
                {
                    "prefix_s": float(metrics_by_sample[i].prefix_time),
                    "text_decode_s": float(metrics_by_sample[i].text_decode_time),
                    "diffusion_s": float(metrics_by_sample[i].diffusion_time),
                    "image_reencode_s": float(metrics_by_sample[i].image_reencode_time),
                    "total_s": float(metrics_by_sample[i].total_time),
                    "generated_text_tokens": int(
                        metrics_by_sample[i].generated_text_tokens
                    ),
                    "generated_images": int(metrics_by_sample[i].generated_images),
                    "diffusion_forwards": int(metrics_by_sample[i].diffusion_forwards),
                    "text_forward_calls_share": float(
                        metrics_by_sample[i].text_forward_calls_share
                    ),
                    "text_forward_rows": int(metrics_by_sample[i].text_forward_rows),
                    "continuous_refill_count": float(
                        metrics_by_sample[i].continuous_refill_count
                    ),
                    "continuous_coalesce_calls": float(
                        metrics_by_sample[i].continuous_coalesce_calls
                    ),
                    "continuous_ready_groups": float(
                        metrics_by_sample[i].continuous_ready_groups
                    ),
                    "continuous_compatible_groups": float(
                        metrics_by_sample[i].continuous_compatible_groups
                    ),
                    "continuous_underfilled_slots": float(
                        metrics_by_sample[i].continuous_underfilled_slots
                    ),
                    "continuous_refilled_slots": float(
                        metrics_by_sample[i].continuous_refilled_slots
                    ),
                    "continuous_incompatible_groups": float(
                        metrics_by_sample[i].continuous_incompatible_groups
                    ),
                }
                if i in metrics_by_sample
                else {}
            )
            for i in range(conditions.batch_size)
        ]
        if not all_sample_tokens:
            return TextSegment()
        return TextSegment.pack(tokens=all_sample_tokens, log_probs=all_sample_logps)

    # ------------------------------------------------------------------
    # Replay: teacher-forced new_logp
    # ------------------------------------------------------------------

    def replay(
        self,
        conditions: SensenovaU1ARConditions,
        *,
        segment: TextSegment,
        temperature: Optional[float] = None,
        compute_policy_entropy: bool = False,
        **_kwargs: Any,
    ) -> Any:
        """Compute new per-token log-probs for GRPO ratio.

        Returns packed ``[total_tokens]`` tensor aligned with ``segment.log_probs``.
        """
        entropy_acc = (
            rl_ops.PolicyEntropyAccumulator() if compute_policy_entropy else None
        )
        if segment.tokens is None or segment.log_probs is None:
            zero = torch.zeros(0, device="cuda", dtype=torch.float32)
            return (zero, None) if compute_policy_entropy else zero

        bundle = self.model
        model = bundle.model
        tokenizer = bundle.tokenizer
        device = next(model.parameters()).device
        temp = temperature or 1.0

        cu_seqlens = segment.cu_seqlens
        lengths = segment.lengths

        if cu_seqlens is None or lengths is None:
            zero = torch.zeros(0, device=device, dtype=torch.float32)
            return (zero, None) if compute_policy_entropy else zero

        has_interleave = bool(conditions.generated_images) and any(
            len(imgs) > 0 for imgs in conditions.generated_images
        )

        all_logps: List[Tensor] = []
        _debug_timing = logger.isEnabledFor(logging.DEBUG)
        _rank = (
            torch.distributed.get_rank()
            if _debug_timing and torch.distributed.is_initialized()
            else 0
        )
        if _debug_timing:
            logger.debug(
                "[DEBUG][TRAIN] rank=%d phase=ar_replay batch=%d interleave=%s "
                "tokens=%s boundaries=%s images=%s input_images=%s",
                _rank,
                conditions.batch_size,
                has_interleave,
                [
                    int(cu_seqlens[i + 1].item() - cu_seqlens[i].item())
                    for i in range(conditions.batch_size)
                ],
                [len(b) for b in (conditions.text_segment_boundaries or [])],
                [len(imgs) for imgs in (conditions.generated_images or [])],
                [
                    (
                        conditions.pixel_values[i] is not None
                        if conditions.pixel_values
                        else False
                    )
                    for i in range(conditions.batch_size)
                ],
            )

        with self._autocast_ctx(device):
            for i in range(conditions.batch_size):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                sample_tokens = segment.tokens[start:end]

                if len(sample_tokens) == 0:
                    continue

                _t_sample = time.perf_counter() if _debug_timing else 0.0
                query = conditions.prompt_queries[i]
                pv = conditions.pixel_values[i] if conditions.pixel_values else None
                ghw = conditions.grid_hws[i] if conditions.grid_hws else None

                if i < len(conditions.generated_images):
                    gen_imgs = conditions.generated_images[i]
                    boundaries = (
                        conditions.text_segment_boundaries[i]
                        if i < len(conditions.text_segment_boundaries)
                        else []
                    )

                    # Route to interleave scorer whenever boundaries exist
                    # (rollout produced structured segments) OR pv!=None
                    # (input image needs proper build_it2i_inputs). The
                    # "standard" fallback below only supports pv=None
                    # text-only; feeding it pv-present samples yields an
                    # empty float32 cat that breaks embed_tokens.
                    if boundaries or pv is not None:
                        logps = rl_ops.score_packed_interleaved_response(
                            model,
                            tokenizer,
                            query=query,
                            all_tokens=sample_tokens.tolist(),
                            text_boundaries=boundaries,
                            generated_images=gen_imgs,
                            temperature=temp,
                            pixel_values=pv,
                            grid_hw=ghw,
                            device=device,
                            entropy_accumulator=entropy_acc,
                        )
                        all_logps.append(logps)
                        if _debug_timing and torch.cuda.is_available():
                            torch.cuda.synchronize()
                        if _debug_timing:
                            logger.debug(
                                "[DEBUG][TRAIN] rank=%d phase=ar_replay_sample "
                                "sample=%d interleave=true images=%d tokens=%d elapsed_s=%.1f",
                                _rank,
                                i,
                                len(gen_imgs),
                                len(sample_tokens),
                                time.perf_counter() - _t_sample,
                            )
                        continue

                # Standard (non-interleave) replay
                response_ids = sample_tokens

                # Build full input: query + response[:-1]
                if pv is not None:
                    input_embeds, indexes, attn_mask = rl_ops.build_it2i_inputs(
                        model, tokenizer, query, pv, ghw
                    )
                    query_len = input_embeds.shape[1]
                else:
                    input_ids_q, indexes_q, _ = rl_ops.build_text_inputs(
                        model, tokenizer, query
                    )
                    query_len = input_ids_q.shape[1]

                # Concatenate query + response for teacher-forced scoring
                resp_input = response_ids[:-1]  # shift: input is response[:-1]
                full_ids = torch.cat(
                    [
                        (
                            input_ids_q.squeeze(0)
                            if pv is None
                            else torch.tensor([], device=device)
                        ),
                        resp_input,
                    ],
                    dim=0,
                )

                # Build 3D indexes for the full sequence
                full_len = query_len + len(resp_input)
                t_indexes = torch.arange(full_len, dtype=torch.long, device=device)
                h_indexes = torch.zeros(full_len, dtype=torch.long, device=device)
                w_indexes = torch.zeros(full_len, dtype=torch.long, device=device)
                full_indexes = torch.stack([t_indexes, h_indexes, w_indexes], dim=0)

                # Block-causal attention mask
                from .vendor.modeling_qwen3 import create_block_causal_mask

                full_attn_mask = {
                    "full_attention": create_block_causal_mask(full_indexes[0])
                }

                logps = rl_ops.score_response(
                    model,
                    input_ids=full_ids.unsqueeze(0),
                    response_ids=response_ids,
                    indexes=full_indexes,
                    attention_mask=full_attn_mask,
                    temperature=temp,
                    device=device,
                    entropy_accumulator=entropy_acc,
                )
                all_logps.append(logps)
                if _debug_timing and torch.cuda.is_available():
                    torch.cuda.synchronize()
                if _debug_timing:
                    logger.debug(
                        "[DEBUG][TRAIN] rank=%d phase=ar_replay_sample "
                        "sample=%d interleave=false tokens=%d elapsed_s=%.1f",
                        _rank,
                        i,
                        len(sample_tokens),
                        time.perf_counter() - _t_sample,
                    )

        if not all_logps:
            packed = torch.zeros(0, device=device, dtype=torch.float32)
        else:
            packed = torch.cat(all_logps, dim=0)
        if compute_policy_entropy:
            return packed, entropy_acc.mean() if entropy_acc is not None else None
        return packed

    def replay_with_reference(
        self,
        conditions: SensenovaU1ARConditions,
        *,
        segment: TextSegment,
        ref_weight_ctx: Any,
        temperature: Optional[float] = None,
        sample_mask: Optional[Tensor] = None,
        compute_policy_entropy: bool = False,
        **_kwargs: Any,
    ) -> Any:
        """Replay policy and pretrained reference on identical packed plans.

        Every local sample slot gets one reference and one policy context
        forward, including zero-length/padded slots. This keeps FSDP collective
        call order symmetric across DP ranks; ``sample_mask`` is consumed by
        the algorithm's KL reduction rather than changing forward counts here.
        """
        del sample_mask
        entropy_acc = (
            rl_ops.PolicyEntropyAccumulator() if compute_policy_entropy else None
        )
        if segment.tokens is None or segment.log_probs is None:
            zero = torch.zeros(0, device="cuda", dtype=torch.float32)
            return (zero, zero, None) if compute_policy_entropy else (zero, zero)

        bundle = self.model
        model = bundle.model
        tokenizer = bundle.tokenizer
        device = next(model.parameters()).device
        temp = temperature or 1.0
        cu_seqlens = segment.cu_seqlens
        if cu_seqlens is None:
            zero = torch.zeros(0, device=device, dtype=torch.float32)
            return (zero, zero, None) if compute_policy_entropy else (zero, zero)

        plans = []
        with self._autocast_ctx(device):
            for i in range(conditions.batch_size):
                start = int(cu_seqlens[i].item())
                end = int(cu_seqlens[i + 1].item())
                sample_tokens = segment.tokens[start:end]
                query = conditions.prompt_queries[i]
                pv = conditions.pixel_values[i] if conditions.pixel_values else None
                ghw = conditions.grid_hws[i] if conditions.grid_hws else None
                gen_imgs = (
                    conditions.generated_images[i]
                    if i < len(conditions.generated_images)
                    else []
                )
                boundaries = (
                    conditions.text_segment_boundaries[i]
                    if i < len(conditions.text_segment_boundaries)
                    else []
                )
                plans.append(
                    rl_ops.build_packed_interleave_plan(
                        model,
                        tokenizer,
                        query=query,
                        all_tokens=sample_tokens.tolist(),
                        text_boundaries=boundaries,
                        generated_images=gen_imgs,
                        pixel_values=pv,
                        grid_hw=ghw,
                        device=device,
                    )
                )

            ref_logps_per_sample = []
            with torch.no_grad():
                with ref_weight_ctx():
                    for plan in plans:
                        ref_hidden, _ = rl_ops.packed_context_forward(model, plan)
                        ref_logps_per_sample.append(
                            rl_ops.packed_logps_from_hidden(
                                model, plan, ref_hidden, temperature=temp
                            ).detach()
                        )

            policy_logps_per_sample = []
            for plan in plans:
                policy_hidden, _ = rl_ops.packed_context_forward(model, plan)
                policy_logps_per_sample.append(
                    rl_ops.packed_logps_from_hidden(
                        model, plan, policy_hidden, temperature=temp,
                        entropy_accumulator=entropy_acc,
                    )
                )

        if not policy_logps_per_sample:
            zero = torch.zeros(0, device=device, dtype=torch.float32)
            result = (zero, zero)
        else:
            result = (
                torch.cat(policy_logps_per_sample, dim=0),
                torch.cat(ref_logps_per_sample, dim=0),
            )
        if compute_policy_entropy:
            return (*result, entropy_acc.mean() if entropy_acc is not None else None)
        return result

    # ------------------------------------------------------------------
    # Replay with inline MSE: merged forward + velocity MSE at boundaries
    # ------------------------------------------------------------------

    def replay_with_mse(
        self,
        conditions: SensenovaU1ARConditions,
        *,
        segment: TextSegment,
        temperature: Optional[float] = None,
        diffusion_stage: Any,
        diffusion_params: Any,
        ref_weight_ctx: Any,
        mse_steps: int = 3,
        sample_mask: Optional[Tensor] = None,
        compute_ref_logps: bool = False,
        compute_policy_entropy: bool = False,
        **_kwargs: Any,
    ) -> Any:
        """Replay with inline velocity MSE at image boundaries.

        Returns ``(policy_logps, reference_logps_or_none, mse_loss)``. When
        ``compute_ref_logps`` is true, reference AR scoring reuses the same
        packed context forward that supplies reference boundary velocities.
        """
        entropy_acc = (
            rl_ops.PolicyEntropyAccumulator() if compute_policy_entropy else None
        )
        if segment.tokens is None or segment.log_probs is None:
            zero = torch.zeros(0, device="cuda", dtype=torch.float32)
            result = (zero, None, zero.new_zeros(()))
            return (*result, None) if compute_policy_entropy else result

        bundle = self.model
        model = bundle.model
        tokenizer = bundle.tokenizer
        device = next(model.parameters()).device
        temp = temperature or 1.0

        cu_seqlens = segment.cu_seqlens
        if cu_seqlens is None:
            zero = torch.zeros(0, device=device, dtype=torch.float32)
            result = (zero, None, zero.new_zeros(()))
            return (*result, None) if compute_policy_entropy else result

        all_logps: List[Tensor] = []
        local_mse_sum = torch.zeros((), device=device, dtype=torch.float32)
        local_real_jobs = 0
        profile_memory = os.environ.get("SENSENOVA_MEM_PROFILE", "0") == "1"

        # Per-local-sample job counts are reduced with MAX so every DP rank
        # performs the same number of generation-branch velocity forwards for
        # sample slot i. Masked padding rows advertise zero real jobs; peers may
        # still force collective-symmetric dummy jobs via the MAX reduction.
        mask_values = (
            sample_mask.detach()
            .to(dtype=torch.float32, device="cpu")
            .reshape(-1)
            .tolist()
            if sample_mask is not None
            else [1.0] * conditions.batch_size
        )
        if len(mask_values) != conditions.batch_size:
            raise ValueError(
                f"replay_with_mse sample_mask={len(mask_values)} != batch={conditions.batch_size}"
            )
        local_job_counts: List[int] = []
        for i in range(conditions.batch_size):
            lat_segs = (
                conditions.latent_segments[i]
                if mask_values[i] > 0.5 and i < len(conditions.latent_segments)
                else []
            )
            count = 0
            for lat_seg in lat_segs:
                if lat_seg is None or lat_seg.sigmas is None:
                    continue
                if lat_seg.sde_indices is not None:
                    count += min(mse_steps, len(lat_seg.sde_indices))
                else:
                    count += min(mse_steps, max(0, len(lat_seg.sigmas) - 1))
            local_job_counts.append(count)

        target_job_counts = torch.tensor(
            local_job_counts, device=device, dtype=torch.long
        )
        if torch.distributed.is_initialized() and target_job_counts.numel() > 0:
            torch.distributed.all_reduce(
                target_job_counts, op=torch.distributed.ReduceOp.MAX
            )

        packed_records = []
        with self._autocast_ctx(device):
            for i in range(conditions.batch_size):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                sample_tokens = segment.tokens[start:end]
                if len(sample_tokens) == 0:
                    continue
                query = conditions.prompt_queries[i]
                pv = conditions.pixel_values[i] if conditions.pixel_values else None
                ghw = conditions.grid_hws[i] if conditions.grid_hws else None
                gen_imgs = (
                    conditions.generated_images[i]
                    if i < len(conditions.generated_images)
                    else []
                )
                boundaries = (
                    conditions.text_segment_boundaries[i]
                    if i < len(conditions.text_segment_boundaries)
                    else []
                )
                lat_segs = (
                    conditions.latent_segments[i]
                    if mask_values[i] > 0.5 and i < len(conditions.latent_segments)
                    else []
                )
                plan = rl_ops.build_packed_interleave_plan(
                    model,
                    tokenizer,
                    query=query,
                    all_tokens=sample_tokens.tolist(),
                    text_boundaries=boundaries,
                    generated_images=gen_imgs,
                    pixel_values=pv,
                    grid_hw=ghw,
                    device=device,
                )
                jobs = rl_ops.build_packed_mse_jobs(
                    plan,
                    lat_segs,
                    mse_steps=mse_steps,
                    target_job_count=int(target_job_counts[i].item()),
                )
                packed_records.append((plan, jobs))

            # Complete every reference forward before constructing any policy
            # graph. Re-entering the weight-swap context after a policy forward
            # would bump Parameter versions and invalidate backward.
            ref_outputs_per_sample = []
            if profile_memory and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                with ref_weight_ctx():
                    for plan, jobs in packed_records:
                        ref_outputs_per_sample.append(
                            rl_ops.packed_reference_logps_and_velocities(
                                model,
                                plan,
                                jobs,
                                diffusion_stage=diffusion_stage,
                                diffusion_params=diffusion_params,
                                temperature=temp,
                                compute_logps=compute_ref_logps,
                                device=device,
                            )
                        )
            if profile_memory and torch.cuda.is_available():
                torch.cuda.synchronize()
                log_memory_usage("geoweave.reference", logger, level=logging.WARNING)
                torch.cuda.reset_peak_memory_stats()

            all_ref_logps: List[Tensor] = []
            for (plan, jobs), (ref_logps, ref_velocities) in zip(
                packed_records, ref_outputs_per_sample
            ):
                logps, mse_sum, real_jobs = rl_ops.packed_policy_logps_and_mse(
                    model,
                    plan,
                    jobs,
                    ref_velocities,
                    diffusion_stage=diffusion_stage,
                    diffusion_params=diffusion_params,
                    temperature=temp,
                    device=device,
                    entropy_accumulator=entropy_acc,
                )
                all_logps.append(logps)
                if ref_logps is not None:
                    all_ref_logps.append(ref_logps)
                local_mse_sum = local_mse_sum + mse_sum.float()
                local_real_jobs += int(real_jobs)
            if profile_memory and torch.cuda.is_available():
                torch.cuda.synchronize()
                log_memory_usage("geoweave.policy", logger, level=logging.WARNING)

        if not all_logps:
            zero = torch.zeros(0, device=device, dtype=torch.float32)
            result = (zero, None, local_mse_sum)
            return (*result, None) if compute_policy_entropy else result

        packed_logps = torch.cat(all_logps, dim=0)
        packed_ref_logps = (
            torch.cat(all_ref_logps, dim=0) if compute_ref_logps else None
        )
        global_real_jobs = torch.tensor(float(local_real_jobs), device=device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                global_real_jobs, op=torch.distributed.ReduceOp.SUM
            )
            dp_world = float(torch.distributed.get_world_size())
        else:
            dp_world = 1.0
        if float(global_real_jobs.item()) > 0:
            # FSDP averages parameter gradients across DP ranks; scale each
            # local sum so the averaged gradient equals the global job mean.
            mse_loss = local_mse_sum * (dp_world / global_real_jobs)
        else:
            mse_loss = local_mse_sum

        result = (packed_logps, packed_ref_logps, mse_loss)
        if compute_policy_entropy:
            return (*result, entropy_acc.mean() if entropy_acc is not None else None)
        return result

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def trainable_module(self) -> nn.Module:
        return self.model.transformer


__all__ = [
    "SensenovaU1LiveSample",
    "SensenovaU1ARParams",
    "SensenovaU1ARStep",
    "SensenovaU1ARStage",
]
