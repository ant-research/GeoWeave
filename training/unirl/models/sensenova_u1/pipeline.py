"""GeoWeave pipeline — interleave-capable unified reasoning+image pipeline.

Mirrors :class:`unirl.models.bagel.pipeline.BagelUniPipeline`.

The pipeline supports interleaved generation:
  text1 → <img> → [diffusion] → [ViT re-encode] → text2 → <img> → [diffusion] → ...

For RL: the AR track collects ALL text tokens + log_probs; the image track
collects ALL generated images. GRPO trains only the AR track.
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Any, Callable, Deque, List, Optional, Tuple

import torch
from torch import Tensor

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, _track_with_field
from unirl.types.segments import LatentSegment, TextSegment
from unirl.utils.memory_utils import log_memory_usage

from . import rl_ops
from .ar import SensenovaU1ARParams, SensenovaU1ARStage, SensenovaU1LiveSample
from .conditions import SensenovaU1ARConditions
from .diffusion import SensenovaU1DiffusionParams, SensenovaU1DiffusionStage
from .rollout_metrics import (
    build_rank_metrics,
    emit_rank_metrics,
    gpu_utilization,
    next_rollout_step,
)

logger = logging.getLogger(__name__)


def fit_image_size_to_pixel_budget(
    h: int,
    w: int,
    patch_size: int,
    merge_size: int,
    target_pixels: int = 1024 * 1024,
) -> Tuple[int, int]:
    """Scale ``(h, w)`` to fit within *target_pixels* while preserving aspect ratio.

    Aligns the result to ``patch_size * merge_size`` (typically 32) using the
    vendor's :func:`smart_resize`.
    """
    from .vendor.utils import smart_resize

    factor = patch_size * merge_size
    return smart_resize(
        h, w, factor=factor, min_pixels=factor * factor, max_pixels=target_pixels
    )


class SensenovaU1Pipeline(Pipeline):
    """Base pipeline for GeoWeave with task routing."""

    def __init__(
        self,
        *,
        bundle: Any,
        diffusion: Optional[SensenovaU1DiffusionStage] = None,
        strategy: Optional[FlowSDEStrategy] = None,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.bundle = bundle
        self.ar = SensenovaU1ARStage(
            model=bundle,
            autocast_precision=autocast_precision,
            logprob_precision=logprob_precision,
        )
        self.diffusion = diffusion or SensenovaU1DiffusionStage(
            model=bundle,
            strategy=strategy or FlowSDEStrategy(),
            autocast_precision=autocast_precision,
            trajectory_precision=trajectory_precision,
            logprob_precision=logprob_precision,
        )

    def build_schedule_policy(self) -> FlowMatchSchedulePolicy:
        cfg = self.bundle.config
        return FlowMatchSchedulePolicy.static_only(shift=cfg.base_shift)

    @classmethod
    def from_config(cls, config: Any) -> "SensenovaU1Pipeline":
        from .bundle import SensenovaU1Bundle

        bundle = SensenovaU1Bundle.from_config(config)
        return cls(
            bundle=bundle,
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )

    def generate(self, req: RolloutReq) -> RolloutResp:
        raise NotImplementedError(
            "SensenovaU1Pipeline.generate: use SensenovaU1UniPipeline for RL."
        )


class SensenovaU1UniPipeline(SensenovaU1Pipeline):
    """Interleave-capable unified reasoning+image pipeline for GRPO.

    Generates P*N interleaved sequences (text ↔ images) and returns a
    2-track ``{"ar", "image"}`` RolloutResp.

    The interleave loop is driven by the AR stage via ``interleave_decode``:
    - Text tokens are generated with per-token log-probs
    - When ``<img>`` is emitted, the diffusion stage generates an image
    - The image is re-encoded through the understanding ViT and appended to the KV cache
    - Text generation continues
    """

    def generate(self, req: RolloutReq) -> RolloutResp:
        trajectory_seeds = req.stage_config.get("trajectory_seeds")
        if trajectory_seeds is not None:
            return self._generate_impl(
                req, trajectory_seeds=[int(seed) for seed in trajectory_seeds]
            )
        trajectory_seed = req.stage_config.get("trajectory_seed")
        if trajectory_seed is None:
            return self._generate_impl(req)
        device = next(self.bundle.model.parameters()).device
        cuda_devices = []
        if device.type == "cuda" and torch.cuda.is_available():
            cuda_devices = [
                (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            seed = int(trajectory_seed)
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed(seed)
            return self._generate_impl(req)

    def _generate_impl(
        self, req: RolloutReq, *, trajectory_seeds: Optional[List[int]] = None
    ) -> RolloutResp:
        rollout_started = time.perf_counter()
        rollout_step = next_rollout_step()
        profile_memory = os.environ.get("SENSENOVA_MEM_PROFILE", "0") == "1"
        if profile_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        ar_params_raw = req.sampling_params.get("ar")
        diff_params = req.sampling_params.get("diffusion")
        if ar_params_raw is None:
            raise TypeError("SensenovaU1UniPipeline: missing 'ar' in sampling_params")

        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"SensenovaU1UniPipeline: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__}"
            )

        # Extract optional input images (e.g. geometry diagrams)
        images_prim = req.primitives.get("image")
        input_pil_images: Optional[List[Any]] = None
        if isinstance(images_prim, Images) and len(images_prim) > 0:
            input_pil_images = images_prim.to_pils()

        if diff_params is None:
            diff_params = SensenovaU1DiffusionParams()

        bundle = self.bundle
        model = bundle.model
        device = next(model.parameters()).device

        prompts = list(texts.texts)
        n_rewrites = int(ar_params_raw.samples_per_prompt)
        default_image_size = (int(diff_params.height), int(diff_params.width))
        target_pixels = diff_params.height * diff_params.width

        # Build AR params
        ar_stage_params = SensenovaU1ARParams(
            interleave=True,
            max_images=getattr(diff_params, "max_images", 10),
            image_size=default_image_size,
            system_message=req.stage_config.get("system_message", ""),
            rollout_text_batch_size=int(
                req.stage_config.get("rollout_text_batch_size", n_rewrites)
            ),
            continuous_batching=bool(
                req.stage_config.get("continuous_batching", False)
            ),
            continuous_request_admission=bool(
                req.stage_config.get("continuous_request_admission", False)
            ),
            continuous_rollout_pool_size=int(
                req.stage_config.get("continuous_rollout_pool_size", n_rewrites)
            ),
        )

        # ── Level 1: P → P*N interleaved sequences ──
        ar_shell = req.make_root_track(track_name="ar", branch=n_rewrites)

        # Build per-sample queries (with optional input image tokens)
        queries: List[str] = []
        pv_list: List[Optional[Tensor]] = []
        ghw_list: List[Optional[Tensor]] = []
        per_sample_image_sizes: List[Tuple[int, int]] = []
        rollout_ids: List[str] = []
        prompt_ids: List[str] = []
        rewrite_ids: List[int] = []
        requested_candidate_ids = list(
            req.stage_config.get("trajectory_candidate_ids", [])
        )
        requested_rewrite_indices = list(
            req.stage_config.get("trajectory_rewrite_indices", [])
        )
        expanded_batch_size = len(prompts) * n_rewrites
        if (
            requested_candidate_ids
            and len(requested_candidate_ids) != expanded_batch_size
        ):
            raise ValueError(
                "trajectory_candidate_ids must align with expanded AR samples"
            )
        if (
            requested_rewrite_indices
            and len(requested_rewrite_indices) != expanded_batch_size
        ):
            raise ValueError(
                "trajectory_rewrite_indices must align with expanded AR samples"
            )

        for p_idx, prompt in enumerate(prompts):
            has_image = input_pil_images is not None and p_idx < len(input_pil_images)
            if has_image:
                pil_img = input_pil_images[p_idx]
                pv, ghw = rl_ops.prepare_input_image(model, pil_img)
                img_prompt = prompt if "<image>" in prompt else "<image>\n" + prompt
                query = rl_ops.build_query(
                    model,
                    img_prompt,
                    system_message=ar_stage_params.system_message,
                )
                query = rl_ops.insert_image_tokens(
                    query,
                    [ghw],
                    bundle.downsample_ratio,
                )
                # Per-prompt image size from input resolution
                prompt_img_size = fit_image_size_to_pixel_budget(
                    pil_img.height,
                    pil_img.width,
                    bundle.patch_size,
                    bundle.merge_size,
                    target_pixels=target_pixels,
                )
            else:
                pv, ghw = None, None
                query = rl_ops.build_query(
                    model,
                    prompt,
                    system_message=ar_stage_params.system_message,
                )
                prompt_img_size = default_image_size
            for rewrite_idx in range(n_rewrites):
                sample_offset = p_idx * n_rewrites + rewrite_idx
                queries.append(query)
                rollout_ids.append(
                    str(requested_candidate_ids[sample_offset])
                    if requested_candidate_ids
                    else ar_shell.sample_ids[sample_offset]
                )
                prompt_ids.append(ar_shell.parent_ids[sample_offset])
                rewrite_ids.append(
                    int(requested_rewrite_indices[sample_offset])
                    if requested_rewrite_indices
                    else rewrite_idx
                )
                pv_list.append(pv)
                ghw_list.append(ghw)
                per_sample_image_sizes.append(prompt_img_size)

        ar_conditions = SensenovaU1ARConditions(
            prompt_queries=queries,
            rollout_ids=rollout_ids,
            prompt_ids=prompt_ids,
            rewrite_ids=rewrite_ids,
            pixel_values=pv_list,
            grid_hws=ghw_list,
            generated_images=[[] for _ in queries],
            text_segment_boundaries=[[] for _ in queries],
            latent_segments=[[] for _ in queries],
            image_sizes=per_sample_image_sizes,
            stop_reasons=["unknown" for _ in queries],
            image_context_token_counts=[0 for _ in queries],
        )

        # Build one lightweight initial-noise group ID per expanded AR sample.
        # Dynamic rollout authors these IDs on the driver. The fixed batching
        # path derives equivalent IDs here: prompt IDs are shared when
        # init_same_noise=true; rollout/sample IDs are unique otherwise.
        requested_noise_gids = list(req.init_noise_group_ids or [])
        if len(requested_noise_gids) == len(queries):
            noise_group_ids = requested_noise_gids
        elif len(requested_noise_gids) == len(prompts):
            noise_group_ids = [
                (
                    requested_noise_gids[p_idx]
                    if bool(diff_params.init_same_noise)
                    else f"{requested_noise_gids[p_idx]}/a{rewrite_idx}"
                )
                for p_idx in range(len(prompts))
                for rewrite_idx in range(n_rewrites)
            ]
        else:
            noise_scope = str(req.stage_config.get("noise_rollout_id", rollout_step))
            noise_ids = prompt_ids if bool(diff_params.init_same_noise) else rollout_ids
            noise_group_ids = [f"r{noise_scope}:{noise_id}" for noise_id in noise_ids]

        # Define diffuse_fn for interleave mode.
        #
        # Under the E1 replay design the diffusion conditions object no longer
        # carries the AR-side KV cache; ``diffuse()`` takes past_kv / text_len /
        # image_shape directly. The image ``RolloutTrack``'s conditions are
        # built AFTER ``autoregress`` completes from the AR interleave state
        # (see the block below the AR call), so the trainer can rebuild KV via
        # a no_grad AR replay at train time without shipping GiB of KV per
        # sample through the rollout→train handoff.
        schedule = req.sigmas.to(device) if req.sigmas is not None else None

        def make_diffuse_batch_fn(
            past_kv,
            text_len,
            past_kv_uncond,
            text_len_uncond,
            img_size,
            batch_size,
            sample_indices=None,
            image_indices=None,
        ):
            """Generate an equal-shaped image micro-batch for active rewrites."""
            H, W = img_size
            sample_indices = list(sample_indices or range(batch_size))
            image_indices = list(image_indices or [0] * batch_size)
            if len(sample_indices) != batch_size or len(image_indices) != batch_size:
                raise ValueError("diffusion noise identity count must match batch_size")

            initial_latents = None
            if diff_params.seed is not None:
                row_gids = [
                    f"{noise_group_ids[sample_idx]}::image-{int(image_idx)}"
                    for sample_idx, image_idx in zip(sample_indices, image_indices)
                ]
                initial_noise = NoiseRecipe(
                    noise_group_ids=row_gids,
                    base_seed=int(diff_params.seed),
                    latent_shape=(3, H, W),
                ).resolve(device=device, dtype=self.diffusion._autocast_dtype)
                if initial_noise is None:
                    raise RuntimeError("failed to resolve GeoWeave initial noise")
                pm = bundle.patch_size * bundle.merge_size
                image_seq_len = (H // pm) * (W // pm)
                noise_scale = self.diffusion._compute_noise_scale(
                    model, diff_params, image_seq_len
                )
                initial_latents = initial_noise * noise_scale

            active_schedule = schedule
            if active_schedule is None:
                active_schedule = torch.linspace(
                    1, 0, int(diff_params.num_inference_steps) + 1, device=device
                )
            seg = self.diffusion.diffuse(
                past_kv_cond=past_kv,
                text_len=text_len,
                image_shape=(H, W),
                schedule=active_schedule,
                params=diff_params,
                past_kv_uncond=past_kv_uncond,
                text_len_uncond=text_len_uncond,
                initial_latents=initial_latents,
                batch_size=batch_size,
            )
            pm = bundle.patch_size * bundle.merge_size
            final_z = seg.latents[:, -1]
            images = rl_ops.unpatchify(model, final_z, pm, H, W)
            cfg_steps = self.diffusion.cfg_step_count(
                model,
                active_schedule,
                diff_params,
                (H // pm) * (W // pm),
            )
            return images, seg, cfg_steps

        def make_diffuse_fn(
            past_kv,
            text_len,
            past_kv_uncond,
            text_len_uncond,
            img_size,
        ):
            images, seg, _ = make_diffuse_batch_fn(
                past_kv,
                text_len,
                past_kv_uncond,
                text_len_uncond,
                img_size,
                1,
                [0],
                [0],
            )
            return images[0], seg

        ar_segment = self.ar.autoregress(
            ar_conditions,
            sampling_params=ar_params_raw,
            params=ar_stage_params,
            diffuse_fn=make_diffuse_fn,
            diffuse_batch_fn=make_diffuse_batch_fn,
            diffusion_batch_size=int(diff_params.rollout_diffusion_batch_size),
            reencode_batch_size=int(diff_params.rollout_reencode_batch_size),
            num_diffusion_steps=int(diff_params.num_inference_steps),
            guidance_scale=float(diff_params.guidance_scale),
            rollout_step=rollout_step,
            trajectory_seeds=trajectory_seeds,
        )

        # Detokenize text
        decoded_texts = self._detokenize(ar_segment)

        # Build AR track
        ar_track = _track_with_field(ar_shell, "segment", ar_segment)
        ar_track = _track_with_field(ar_track, "decoded", decoded_texts)
        ar_track = _track_with_field(ar_track, "conditions", ar_conditions.to_dict())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rollout_elapsed = time.perf_counter() - rollout_started
        if not req.stage_config.get("disable_rollout_metrics", False):
            emit_rank_metrics(
                build_rank_metrics(
                    rollout_step=rollout_step,
                    elapsed=rollout_elapsed,
                    samples=getattr(self.ar, "last_rollout_metrics", []),
                    gpu_util=gpu_utilization(device),
                )
            )
        if profile_memory and torch.cuda.is_available():
            log_memory_usage("geoweave.rollout", logger, level=logging.WARNING)

        return RolloutResp(tracks={"ar": ar_track})

    def generate_live(
        self,
        *,
        admit_fn: Callable[[int, bool], Tuple[List[Tuple[Any, tuple, dict]], bool]],
        emit_fn: Callable[[Any, RolloutResp], None],
    ) -> None:
        """Stream single-trajectory requests through one persistent AR session."""
        rollout_started = time.perf_counter()
        rollout_step = next_rollout_step()
        profile_memory = os.environ.get("SENSENOVA_MEM_PROFILE", "0") == "1"
        if profile_memory and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        first_items, admission_closed = admit_fn(1, True)
        if not first_items:
            if admission_closed:
                return
            raise RuntimeError("live rollout admission returned no initial request")
        pending: Deque[Tuple[Any, tuple, dict]] = deque(first_items)
        first_req = pending[0][1][0]
        if not isinstance(first_req, RolloutReq):
            raise TypeError("live rollout queue must contain RolloutReq arguments")
        template_ar = first_req.sampling_params.get("ar")
        template_diff = first_req.sampling_params.get("diffusion")
        if template_ar is None:
            raise TypeError("live GeoWeave rollout requires AR sampling params")
        if template_diff is None:
            template_diff = SensenovaU1DiffusionParams()
        if int(template_ar.samples_per_prompt) != 1:
            raise ValueError("live rollout requests require samples_per_prompt=1")

        bundle = self.bundle
        model = bundle.model
        device = next(model.parameters()).device
        default_image_size = (int(template_diff.height), int(template_diff.width))
        target_pixels = int(template_diff.height) * int(template_diff.width)
        ar_stage_params = SensenovaU1ARParams(
            interleave=True,
            max_images=getattr(template_diff, "max_images", 10),
            image_size=default_image_size,
            system_message=first_req.stage_config.get("system_message", ""),
            rollout_text_batch_size=int(
                first_req.stage_config.get("continuous_text_batch_size", 1)
            ),
            continuous_batching=True,
            continuous_request_admission=True,
            continuous_rollout_pool_size=int(
                first_req.stage_config.get("continuous_rollout_pool_size", 1)
            ),
        )
        schedule = first_req.sigmas.to(device) if first_req.sigmas is not None else None
        live_specs: List[SensenovaU1LiveSample] = []

        def validate_request(req: RolloutReq) -> None:
            if req.batch_size != 1:
                raise ValueError("live rollout requests require batch_size=1")
            ar_params = req.sampling_params.get("ar")
            diff_params = req.sampling_params.get("diffusion")
            if ar_params != template_ar or (
                diff_params or SensenovaU1DiffusionParams()
            ) != template_diff:
                raise ValueError("live rollout requests must share sampling parameters")
            if int(ar_params.samples_per_prompt) != 1:
                raise ValueError("live rollout requests require samples_per_prompt=1")
            if (req.sigmas is None) != (schedule is None):
                raise ValueError("live rollout requests must share a sigma schedule")
            if req.sigmas is not None and not torch.equal(
                req.sigmas.to(device), schedule
            ):
                raise ValueError("live rollout requests must share a sigma schedule")

        def make_spec(request_id: Any, req: RolloutReq) -> SensenovaU1LiveSample:
            validate_request(req)
            texts = req.primitives.get("text")
            if not isinstance(texts, Texts) or len(texts.texts) != 1:
                raise TypeError("live rollout requires exactly one text primitive")
            images_prim = req.primitives.get("image")
            input_image = None
            if isinstance(images_prim, Images) and len(images_prim) > 0:
                input_image = images_prim.to_pils()[0]
            if input_image is not None:
                pixel_values, grid_hw = rl_ops.prepare_input_image(model, input_image)
                prompt = texts.texts[0]
                image_prompt = prompt if "<image>" in prompt else "<image>\n" + prompt
                query = rl_ops.build_query(
                    model, image_prompt, system_message=ar_stage_params.system_message
                )
                query = rl_ops.insert_image_tokens(
                    query, [grid_hw], bundle.downsample_ratio
                )
                image_size = fit_image_size_to_pixel_budget(
                    input_image.height,
                    input_image.width,
                    bundle.patch_size,
                    bundle.merge_size,
                    target_pixels=target_pixels,
                )
            else:
                pixel_values, grid_hw = None, None
                query = rl_ops.build_query(
                    model,
                    texts.texts[0],
                    system_message=ar_stage_params.system_message,
                )
                image_size = default_image_size

            candidate_ids = list(req.stage_config.get("trajectory_candidate_ids", []))
            rewrite_indices = list(
                req.stage_config.get("trajectory_rewrite_indices", [])
            )
            if len(candidate_ids) != 1 or len(rewrite_indices) != 1:
                raise ValueError(
                    "live rollout request requires one candidate and rewrite identity"
                )
            if "trajectory_seed" in req.stage_config:
                trajectory_seed = int(req.stage_config["trajectory_seed"])
            else:
                seeds = list(req.stage_config.get("trajectory_seeds", []))
                if len(seeds) != 1:
                    raise ValueError("live rollout request requires one trajectory seed")
                trajectory_seed = int(seeds[0])
            noise_ids = list(req.init_noise_group_ids or [])
            noise_group_id = str(noise_ids[0]) if noise_ids else ""
            prompt_id = str(req.sample_ids[0])
            shell = req.make_root_track(track_name="ar", branch=1)
            shell.sample_ids = [str(candidate_ids[0])]
            shell.parent_ids = [prompt_id]
            conditions = SensenovaU1ARConditions(
                prompt_queries=[query],
                rollout_ids=[str(candidate_ids[0])],
                prompt_ids=[prompt_id],
                rewrite_ids=[int(rewrite_indices[0])],
                pixel_values=[pixel_values],
                grid_hws=[grid_hw],
                generated_images=[[]],
                text_segment_boundaries=[[]],
                latent_segments=[[]],
                image_sizes=[image_size],
                stop_reasons=["unknown"],
                image_context_token_counts=[0],
                rollout_phase_metrics=[{}],
            )
            return SensenovaU1LiveSample(
                query=query,
                rollout_id=str(candidate_ids[0]),
                prompt_id=prompt_id,
                rewrite_id=int(rewrite_indices[0]),
                image_size=image_size,
                trajectory_seed=trajectory_seed,
                noise_group_id=noise_group_id,
                pixel_values=pixel_values,
                grid_hw=grid_hw,
                payload=(request_id, shell, conditions),
            )

        def pipeline_admit(capacity: int, block_when_idle: bool):
            nonlocal admission_closed
            items: List[Tuple[Any, tuple, dict]] = []
            while pending and len(items) < capacity:
                items.append(pending.popleft())
            if len(items) < capacity and not admission_closed:
                pulled, admission_closed = admit_fn(
                    capacity - len(items), block_when_idle and not items
                )
                items.extend(pulled)
            specs = []
            for request_id, args, kwargs in items:
                if kwargs or len(args) != 1 or not isinstance(args[0], RolloutReq):
                    raise TypeError(
                        "live rollout queue items must be positional RolloutReq calls"
                    )
                spec = make_spec(request_id, args[0])
                live_specs.append(spec)
                specs.append(spec)
            return specs, admission_closed and not pending

        def make_diffuse_batch_fn(
            past_kv,
            text_len,
            past_kv_uncond,
            text_len_uncond,
            img_size,
            batch_size,
            sample_indices=None,
            image_indices=None,
        ):
            H, W = img_size
            sample_indices = list(sample_indices or range(batch_size))
            image_indices = list(image_indices or [0] * batch_size)
            initial_latents = None
            if template_diff.seed is not None:
                row_gids = []
                for sample_index, image_index in zip(sample_indices, image_indices):
                    noise_group_id = live_specs[int(sample_index)].noise_group_id
                    if not noise_group_id:
                        raise ValueError("seeded live diffusion requires noise identity")
                    row_gids.append(
                        f"{noise_group_id}::image-{int(image_index)}"
                    )
                initial_noise = NoiseRecipe(
                    noise_group_ids=row_gids,
                    base_seed=int(template_diff.seed),
                    latent_shape=(3, H, W),
                ).resolve(device=device, dtype=self.diffusion._autocast_dtype)
                if initial_noise is None:
                    raise RuntimeError("failed to resolve GeoWeave initial noise")
                pm = bundle.patch_size * bundle.merge_size
                image_seq_len = (H // pm) * (W // pm)
                noise_scale = self.diffusion._compute_noise_scale(
                    model, template_diff, image_seq_len
                )
                initial_latents = initial_noise * noise_scale
            active_schedule = schedule
            if active_schedule is None:
                active_schedule = torch.linspace(
                    1,
                    0,
                    int(template_diff.num_inference_steps) + 1,
                    device=device,
                )
            segment = self.diffusion.diffuse(
                past_kv_cond=past_kv,
                text_len=text_len,
                image_shape=(H, W),
                schedule=active_schedule,
                params=template_diff,
                past_kv_uncond=past_kv_uncond,
                text_len_uncond=text_len_uncond,
                initial_latents=initial_latents,
                batch_size=batch_size,
            )
            pm = bundle.patch_size * bundle.merge_size
            images = rl_ops.unpatchify(
                model, segment.latents[:, -1], pm, H, W
            )
            cfg_steps = self.diffusion.cfg_step_count(
                model,
                active_schedule,
                template_diff,
                (H // pm) * (W // pm),
            )
            return images, segment, cfg_steps

        def pipeline_finish(spec: SensenovaU1LiveSample, state: Any) -> None:
            request_id, shell, conditions = spec.payload
            conditions.generated_images[0] = state.images
            conditions.text_segment_boundaries[0] = state.boundaries
            conditions.latent_segments[0] = state.latent_segments
            conditions.stop_reasons[0] = state.metrics.stop_reason
            conditions.image_context_token_counts[0] = int(
                state.metrics.image_context_tokens
            )
            conditions.rollout_phase_metrics[0] = {
                "prefix_s": float(state.metrics.prefix_time),
                "text_decode_s": float(state.metrics.text_decode_time),
                "diffusion_s": float(state.metrics.diffusion_time),
                "image_reencode_s": float(state.metrics.image_reencode_time),
                "total_s": float(state.metrics.total_time),
                "generated_text_tokens": int(state.metrics.generated_text_tokens),
                "generated_images": int(state.metrics.generated_images),
                "diffusion_forwards": int(state.metrics.diffusion_forwards),
                "text_forward_calls_share": float(
                    state.metrics.text_forward_calls_share
                ),
                "text_forward_rows": int(state.metrics.text_forward_rows),
                "continuous_refill_count": float(
                    state.metrics.continuous_refill_count
                ),
                "continuous_coalesce_calls": float(
                    state.metrics.continuous_coalesce_calls
                ),
                "continuous_ready_groups": float(
                    state.metrics.continuous_ready_groups
                ),
                "continuous_compatible_groups": float(
                    state.metrics.continuous_compatible_groups
                ),
                "continuous_underfilled_slots": float(
                    state.metrics.continuous_underfilled_slots
                ),
                "continuous_refilled_slots": float(
                    state.metrics.continuous_refilled_slots
                ),
                "continuous_incompatible_groups": float(
                    state.metrics.continuous_incompatible_groups
                ),
            }
            segment = TextSegment.pack(
                tokens=[torch.tensor(state.tokens, dtype=torch.long, device=device)],
                log_probs=[
                    torch.tensor(state.logps, dtype=torch.float32, device=device)
                ],
            )
            track = _track_with_field(shell, "segment", segment)
            track = _track_with_field(track, "decoded", self._detokenize(segment))
            track = _track_with_field(track, "conditions", conditions.to_dict())
            emit_fn(request_id, RolloutResp(tracks={"ar": track}))

        empty_conditions = SensenovaU1ARConditions()
        self.ar.autoregress(
            empty_conditions,
            sampling_params=template_ar,
            params=ar_stage_params,
            diffuse_batch_fn=make_diffuse_batch_fn,
            diffusion_batch_size=int(template_diff.rollout_diffusion_batch_size),
            reencode_batch_size=int(template_diff.rollout_reencode_batch_size),
            num_diffusion_steps=int(template_diff.num_inference_steps),
            guidance_scale=float(template_diff.guidance_scale),
            rollout_step=rollout_step,
            live_admit_fn=pipeline_admit,
            live_finish_fn=pipeline_finish,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if not first_req.stage_config.get("disable_rollout_metrics", False):
            emit_rank_metrics(
                build_rank_metrics(
                    rollout_step=rollout_step,
                    elapsed=time.perf_counter() - rollout_started,
                    samples=getattr(self.ar, "last_rollout_metrics", []),
                    gpu_util=gpu_utilization(device),
                )
            )
        if profile_memory and torch.cuda.is_available():
            log_memory_usage("geoweave.rollout", logger, level=logging.WARNING)

    def _detokenize(self, segment: TextSegment) -> Texts:
        """Decode packed text segment into strings."""
        if segment.tokens is None or segment.cu_seqlens is None:
            return Texts(texts=[])

        from unirl.utils.token_alignment import decode_with_token_char_spans

        tokenizer = self.bundle.tokenizer
        cu = segment.cu_seqlens
        results: List[str] = []
        packed_starts: List[torch.Tensor] = []
        packed_ends: List[torch.Tensor] = []
        for i in range(len(cu) - 1):
            token_ids = segment.tokens[cu[i] : cu[i + 1]].tolist()
            text, starts, ends = decode_with_token_char_spans(tokenizer, token_ids)
            results.append(text)
            packed_starts.append(starts)
            packed_ends.append(ends)
        segment.decoded_char_starts = torch.cat(packed_starts) if packed_starts else torch.empty(0, dtype=torch.long)
        segment.decoded_char_ends = torch.cat(packed_ends) if packed_ends else torch.empty(0, dtype=torch.long)
        return Texts(texts=results)

    def _simple_diffuse(
        self,
        *,
        past_kv_cond: Any,
        text_len: int,
        image_shape: Tuple[int, int],
        params: SensenovaU1DiffusionParams,
        past_kv_uncond: Optional[Any] = None,
        text_len_uncond: Optional[int] = None,
        batch_size: int = 1,
    ) -> LatentSegment:
        """Fallback Euler diffusion without a pre-built sigma schedule."""
        bundle = self.bundle
        model = bundle.model
        device = next(model.parameters()).device
        T = params.num_inference_steps

        sigmas = torch.linspace(1, 0, T + 1, device=device)
        return self.diffusion.diffuse(
            past_kv_cond=past_kv_cond,
            text_len=text_len,
            image_shape=image_shape,
            schedule=sigmas,
            params=params,
            past_kv_uncond=past_kv_uncond,
            text_len_uncond=text_len_uncond,
            batch_size=batch_size,
        )


__all__ = ["SensenovaU1Pipeline", "SensenovaU1UniPipeline"]
