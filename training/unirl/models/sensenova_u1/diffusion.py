"""GeoWeave diffusion stage — pixel-space flow-matching image generation.

Mirrors :class:`unirl.models.bagel.diffusion.BagelDiffusionStage`.

Key difference from Bagel: GeoWeave operates in pixel space (no VAE).
The ``LatentSegment`` stores patchified pixel tensors ``[B, L, patch_dim]``
where ``L = H*W / (patch_size*merge_size)^2`` and ``patch_dim = (patch_size*merge_size)^2 * 3``.
The ``FlowSDEStrategy`` operates on generic tensors, so pixel-patch tensors
work identically to latent tensors.

Sigma ↔ timestep mapping:
- UniRL uses ``sigma`` where ``sigma = 1 - t`` (sigma=1 = pure noise, sigma=0 = clean)
- GeoWeave uses ``t`` (t=0 = pure noise, t=1 = clean)
"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from unirl.sde.kernels import FlowSDEStrategy
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments import LatentSegment
from unirl.utils.dtypes import parse_torch_dtype

from . import rl_ops

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass
class SensenovaU1DiffusionParams(DiffusionSamplingParams):
    """Per-request diffusion knobs for GeoWeave."""

    num_inference_steps: int = 30
    guidance_scale: float = 1.0
    height: int = 256
    width: int = 256
    max_images: int = 10
    rollout_diffusion_batch_size: int = 2
    rollout_reencode_batch_size: int = 2
    eta: float = 1.0
    t_eps: float = 0.02
    noise_scale: float = 1.0
    noise_scale_mode: str = "resolution"
    timestep_shift: float = 1.0
    cfg_norm: str = "none"
    cfg_interval: Tuple[float, float] = (0.0, 1.0)

    def __post_init__(self):
        if self.num_inference_steps < 2:
            raise ValueError("num_inference_steps must be >= 2")
        if self.max_images < 0:
            raise ValueError("max_images must be >= 0")
        if self.rollout_diffusion_batch_size < 1:
            raise ValueError("rollout_diffusion_batch_size must be >= 1")
        if self.rollout_reencode_batch_size < 1:
            raise ValueError("rollout_reencode_batch_size must be >= 1")
        if len(self.cfg_interval) != 2:
            raise ValueError("cfg_interval must contain exactly two values")
        cfg_lo, cfg_hi = self.cfg_interval
        if not 0.0 <= cfg_lo <= cfg_hi <= 1.0:
            raise ValueError("cfg_interval must satisfy 0 <= lo <= hi <= 1")
        if self.cfg_norm not in {"none", "global", "channel"}:
            raise ValueError("cfg_norm must be one of: none, global, channel")


# ---------------------------------------------------------------------------
# Step kernel
# ---------------------------------------------------------------------------


class SensenovaU1DiffusionStep:
    """Per-step kernel: velocity prediction + SDE transition."""

    def predict_velocity(
        self,
        model: Any,
        *,
        z: Tensor,
        image_embeds: Tensor,
        indexes_image: Tensor,
        attn_mask: Any,
        past_key_values: Any,
        t: Tensor,
        image_token_num: int,
        image_size: Tuple[int, int],
    ) -> Tensor:
        """Predict velocity via the gen branch. Returns ``v_pred [B, L, patch_dim]``."""
        return rl_ops.predict_v(
            model,
            image_embeds,
            indexes_image,
            attn_mask,
            past_key_values,
            t,
            z,
            image_token_num,
            image_size=image_size,
        )

    def denoise(
        self,
        strategy: FlowSDEStrategy,
        *,
        v_t: Tensor,
        x_t: Tensor,
        sigma: Tensor,
        sigma_next: Tensor,
        sigma_max: float,
        eta: float,
        prev_sample: Optional[Tensor] = None,
        step_index: int = 0,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """SDE transition via FlowSDEStrategy."""
        return strategy.denoise(
            noise_pred=v_t,
            sample=x_t,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=eta,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            step_index=step_index,
        )


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


class SensenovaU1DiffusionStage:
    """Pixel-space flow-matching diffusion stage for GeoWeave.

    Consumed by :class:`SensenovaU1UniPipeline` at rollout and by
    :class:`InterleaveRL` at train (via :meth:`build_forward_kwargs_from_kv`,
    which takes a live grad-carrying KV cache from the AR merged forward — no
    separate condition class needed).
    """

    def __init__(
        self,
        *,
        model: Any,  # GeoWeave model bundle
        step: Optional[SensenovaU1DiffusionStep] = None,
        strategy: Optional[FlowSDEStrategy] = None,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        self.step = step or SensenovaU1DiffusionStep()
        self.strategy = strategy or FlowSDEStrategy()
        self._autocast_dtype = parse_torch_dtype(autocast_precision)
        self._traj_dtype = parse_torch_dtype(trajectory_precision)
        self._logp_dtype = parse_torch_dtype(logprob_precision)

    def _autocast_ctx(self):
        if self._autocast_dtype in (torch.bfloat16, torch.float16):
            return torch.autocast("cuda", dtype=self._autocast_dtype)
        return nullcontext()

    def _compute_noise_scale(
        self, model: Any, params: SensenovaU1DiffusionParams, image_seq_len: int
    ) -> float:
        """Compute resolution-dependent noise scale (aligned with vendor)."""
        base_ns = params.noise_scale
        mode = params.noise_scale_mode
        if mode in ("resolution", "dynamic", "dynamic_sqrt"):
            base_seq = getattr(model, "noise_scale_base_image_seq_len", 256)
            ns = math.sqrt(image_seq_len / max(base_seq, 1)) * base_ns
            if mode == "dynamic_sqrt":
                ns = math.sqrt(ns)
            max_val = getattr(model, "noise_scale_max_value", 10.0)
            return min(ns, max_val)
        return base_ns

    @staticmethod
    def _cfg_step_mask(
        timesteps: Tensor, params: SensenovaU1DiffusionParams
    ) -> List[bool]:
        if params.guidance_scale <= 1.0:
            return [False] * max(0, len(timesteps) - 1)
        cfg_lo, cfg_hi = params.cfg_interval
        mask = (timesteps[:-1] >= cfg_lo) & (timesteps[:-1] <= cfg_hi)
        return mask.detach().cpu().tolist()

    def cfg_step_count(
        self,
        model: Any,
        schedule: Tensor,
        params: SensenovaU1DiffusionParams,
        image_seq_len: int,
    ) -> int:
        """Return the number of denoising steps that execute text CFG."""
        timesteps = rl_ops.apply_time_schedule(
            model,
            1.0 - schedule.float(),
            image_seq_len,
            params.timestep_shift,
        )
        return sum(self._cfg_step_mask(timesteps, params))

    @staticmethod
    def _renorm_cfg(v_pred: Tensor, v_cond: Tensor, cfg_norm: str) -> Tensor:
        if cfg_norm == "global":
            norm_cond = torch.norm(v_cond, dim=(1, 2), keepdim=True)
            norm_cfg = torch.norm(v_pred, dim=(1, 2), keepdim=True)
        elif cfg_norm == "channel":
            norm_cond = torch.norm(v_cond, dim=-1, keepdim=True)
            norm_cfg = torch.norm(v_pred, dim=-1, keepdim=True)
        else:
            return v_pred
        scale = (norm_cond / (norm_cfg + 1e-8)).clamp(min=0, max=1.0)
        return v_pred * scale

    # ------------------------------------------------------------------
    # Diffuse (rollout)
    # ------------------------------------------------------------------

    def diffuse(
        self,
        *,
        past_kv_cond: Any,
        text_len: int,
        image_shape: Tuple[int, int],
        schedule: Tensor,
        params: SensenovaU1DiffusionParams,
        past_kv_uncond: Optional[Any] = None,
        text_len_uncond: Optional[int] = None,
        initial_latents: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> LatentSegment:
        """Full sampling loop returning a LatentSegment with SDE log-probs.

        ``past_kv_cond`` / ``past_kv_uncond`` are supplied by the rollout-side
        caller (:meth:`SensenovaU1UniPipeline.generate`'s ``make_diffuse_fn``),
        which owns the freshly-computed KV cache from the AR interleave loop.
        This entry point is rollout-only; the training-time MSE path takes a
        live grad-carrying KV cache via :meth:`build_forward_kwargs_from_kv`
        and never re-enters ``diffuse``.

        Args:
            past_kv_cond: conditioned KV cache from the AR prefix + <img> forward.
            text_len: t_idx immediately AFTER appending <img> — used for image indexes.
            image_shape: ``(H, W)`` target image size.
            schedule: sigma schedule ``[T+1]`` from 1 (noise) to 0 (clean).
            params: diffusion sampling params.
            past_kv_uncond: optional text-unconditional KV cache for CFG. It
                preserves input/generated images but excludes user/generated text.
            text_len_uncond: temporal index of ``past_kv_uncond``.
            initial_latents: optional pre-authored noise.
            batch_size: number of equal-shaped image requests in this micro-batch.
        """
        bundle = self.model
        model = bundle.model
        device = next(model.parameters()).device

        kv_cond = past_kv_cond
        kv_uncond = past_kv_uncond
        H, W = image_shape

        if params.guidance_scale > 1.0:
            if kv_uncond is None or text_len_uncond is None:
                raise RuntimeError(
                    "GeoWeave text CFG requires past_kv_uncond and text_len_uncond"
                )

        patch_size = bundle.patch_size
        merge_size = bundle.merge_size
        pm = patch_size * merge_size

        token_h = H // pm
        token_w = W // pm
        image_seq_len = token_h * token_w

        # Raw patch grid for gen ViT (asserts grid_h*grid_w == num_patches)
        patch_h = H // patch_size
        patch_w = W // patch_size

        # Compute noise scale
        noise_scale = self._compute_noise_scale(model, params, image_seq_len)

        # Convert sigma schedule to timesteps: t = 1 - sigma
        T = len(schedule) - 1
        timesteps = 1.0 - schedule.float()
        timesteps = rl_ops.apply_time_schedule(
            model, timesteps, image_seq_len, params.timestep_shift
        )
        cfg_step_mask = self._cfg_step_mask(timesteps, params)

        # Determine SDE indices from params
        sde_indices_set = set()
        if hasattr(params, "sde_indices") and params.sde_indices is not None:
            sde_indices_set = set(params.sde_indices)

        # Initialize noise in pixel space. The conditioned cache already carries
        # the batch dimension; an explicit batch_size keeps fake/test caches usable.
        if batch_size is None:
            if initial_latents is not None:
                batch_size = int(initial_latents.shape[0])
            else:
                first_layer = kv_cond.layers[0]
                batch_size = int(first_layer.keys.shape[0])
        if initial_latents is not None:
            image_pred = initial_latents.to(device=device, dtype=self._autocast_dtype)
            if int(image_pred.shape[0]) != batch_size:
                raise ValueError(
                    f"initial_latents batch={image_pred.shape[0]} != batch_size={batch_size}"
                )
        else:
            image_pred = noise_scale * torch.randn(
                batch_size, 3, H, W, device=device, dtype=self._autocast_dtype
            )

        # Build image indexes and attention mask (shared across steps)
        indexes_image = rl_ops.build_image_indexes(
            model, token_h, token_w, text_len, device
        )
        indexes_image_uncond = (
            rl_ops.build_image_indexes(
                model, token_h, token_w, int(text_len_uncond), device
            )
            if kv_uncond is not None
            else None
        )
        attn_mask = {"full_attention": None}

        # Initialize strategy
        self.strategy.init_schedule(schedule.to(device))

        # Storage
        stored_latents = []
        stored_sde_logp = []
        stored_sde_means = []
        stored_indices = []
        actual_sde_indices = []

        raw_grid_hw = torch.tensor(
            [[patch_h, patch_w]] * batch_size, device=device, dtype=torch.long
        )

        with torch.no_grad(), self._autocast_ctx():
            for step_i in range(T):
                t_cur = timesteps[step_i]
                t_next = timesteps[step_i + 1]
                sigma = schedule[step_i]
                sigma_next = schedule[step_i + 1]

                is_sde = step_i in sde_indices_set
                eta = params.eta if is_sde else 0.0

                # Patchify for SDE operations
                z = rl_ops.patchify(model, image_pred, pm)  # [1, L, patch_dim]

                # Store trajectory at SDE boundaries
                if is_sde:
                    stored_latents.append(z.to(self._traj_dtype).clone())
                    stored_indices.append(step_i)

                # Build image embeddings
                image_embeds = rl_ops.build_image_embeds(
                    model, image_pred, t_cur.reshape(1).to(device),
                    grid_hw=raw_grid_hw,
                    noise_scale_value=noise_scale if model.add_noise_scale_embedding else None,
                )

                # Predict conditional velocity.
                v_cond = self.step.predict_velocity(
                    model,
                    z=z,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image,
                    attn_mask=attn_mask,
                    past_key_values=kv_cond,
                    t=t_cur.to(device),
                    image_token_num=image_seq_len,
                    image_size=(W, H),
                )

                # Text-only CFG: the unconditional branch preserves input and
                # previously generated images, but excludes user/generated text.
                if cfg_step_mask[step_i]:
                    assert kv_uncond is not None
                    assert indexes_image_uncond is not None
                    v_uncond = self.step.predict_velocity(
                        model,
                        z=z,
                        image_embeds=image_embeds,
                        indexes_image=indexes_image_uncond,
                        attn_mask=attn_mask,
                        past_key_values=kv_uncond,
                        t=t_cur.to(device),
                        image_token_num=image_seq_len,
                        image_size=(W, H),
                    )
                    v_pred = v_uncond + params.guidance_scale * (v_cond - v_uncond)
                    v_pred = self._renorm_cfg(v_pred, v_cond, params.cfg_norm)
                else:
                    v_pred = v_cond

                # SDE denoise or Euler step
                if is_sde and eta > 1e-7:
                    next_z, log_prob, prev_mean = self.step.denoise(
                        self.strategy,
                        v_t=v_pred,
                        x_t=z,
                        sigma=sigma.to(device),
                        sigma_next=sigma_next.to(device),
                        sigma_max=0.99,
                        eta=eta,
                        prev_sample=None,
                        step_index=step_i,
                    )
                    if log_prob is not None:
                        # SDEStrategy._finalize_logp already reduces log_prob
                        # to [B]; another mean over dim=() would collapse it to
                        # 0-D and break torch.stack(dim=1) below.
                        stored_sde_logp.append(log_prob.to(self._logp_dtype))
                    if prev_mean is not None:
                        stored_sde_means.append(prev_mean.to(self._traj_dtype))
                else:
                    # Deterministic Euler step
                    dt = t_next - t_cur
                    next_z = z + dt * v_pred

                # Unpatchify back to pixel space
                image_pred = rl_ops.unpatchify(model, next_z, pm, H, W)

            # Store final clean frame
            final_z = rl_ops.patchify(model, image_pred, pm)
            stored_latents.append(final_z.to(self._traj_dtype).clone())
            stored_indices.append(T)

        # Build LatentSegment
        latents = torch.stack(stored_latents, dim=1) if stored_latents else None  # [1, K, L, pd]
        sde_logp = torch.stack(stored_sde_logp, dim=1) if stored_sde_logp else None  # [1, S]
        sde_means = torch.stack(stored_sde_means, dim=1) if stored_sde_means else None

        return LatentSegment(
            latents=latents,
            sigmas=schedule.to(device),
            indices=torch.tensor(stored_indices, dtype=torch.long, device=device),
            sde_logp=sde_logp,
            sde_means=sde_means,
            sde_indices=(
                torch.tensor(actual_sde_indices or sorted(sde_indices_set),
                             dtype=torch.long, device=device)
                if sde_indices_set else None
            ),
        )

    # ------------------------------------------------------------------
    # MSE interface: velocity prediction at arbitrary (x_t, sigma)
    # ------------------------------------------------------------------

    def predict_velocity_at(
        self,
        forward_kwargs: Dict[str, Any],
        *,
        sample: Tensor,
        sigma: Tensor,
        params: SensenovaU1DiffusionParams,
    ) -> Tensor:
        """Predict velocity at an arbitrary ``(x_t, sigma)`` point.

        ``sample`` is a patchified latent ``[L, pd]`` or ``[1, L, pd]`` (from
        ``segment.latents_at``). Gradient flows through the velocity forward.
        """
        model = forward_kwargs["model"]
        pm = forward_kwargs["pm"]
        H, W = forward_kwargs["H"], forward_kwargs["W"]
        image_seq_len = forward_kwargs["image_seq_len"]
        noise_scale = forward_kwargs["noise_scale"]
        grid_hw = forward_kwargs["grid_hw"]
        device = next(model.parameters()).device

        z = sample.unsqueeze(0) if sample.dim() == 2 else sample
        z = z.to(device)

        image_pred = rl_ops.unpatchify(model, z, pm, H, W)

        sigma_scalar = sigma.float() if sigma.dim() == 0 else sigma.float().squeeze()
        t_raw = (1.0 - sigma_scalar).reshape(1)
        t_cur = rl_ops.apply_time_schedule(
            model, t_raw.to(device), image_seq_len, params.timestep_shift
        )

        with self._autocast_ctx():
            image_embeds = rl_ops.build_image_embeds(
                model,
                image_pred,
                t_cur.to(device),
                grid_hw=grid_hw,
                noise_scale_value=(
                    noise_scale if model.add_noise_scale_embedding else None
                ),
            )

            v_pred = self.step.predict_velocity(
                model,
                z=z,
                image_embeds=image_embeds,
                indexes_image=forward_kwargs["indexes_image"],
                attn_mask={"full_attention": None},
                past_key_values=forward_kwargs["kv_cond"],
                t=t_cur.to(device),
                image_token_num=image_seq_len,
                image_size=(W, H),
            )

        return v_pred

    def build_forward_kwargs_from_kv(
        self,
        past_kv: Any,
        t_idx: int,
        image_shape: Tuple[int, int],
        *,
        params: SensenovaU1DiffusionParams,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Build forward kwargs from a pre-built live KV cache (inline MSE path).

        Takes a KV cache that already has ``grad_fn`` from the AR merged
        forward (``use_cache=True`` with grad enabled). Used by InterleaveRL
        to compute velocity MSE where gradient flows through
        ``K/V → und k_proj/v_proj``. The rollout-time :meth:`diffuse` entry
        point is the only other user of this stage's velocity path, and it
        supplies its own no_grad KV directly.
        """
        bundle = self.model
        model = bundle.model

        H, W = image_shape
        patch_size = bundle.patch_size
        merge_size = bundle.merge_size
        pm = patch_size * merge_size

        token_h = H // pm
        token_w = W // pm
        image_seq_len = token_h * token_w

        noise_scale = self._compute_noise_scale(model, params, image_seq_len)

        indexes_image = rl_ops.build_image_indexes(
            model, token_h, token_w, t_idx, device
        )
        raw_grid_hw = torch.tensor(
            [[H // patch_size, W // patch_size]], device=device, dtype=torch.long
        )

        return {
            "model": model,
            "kv_cond": past_kv,
            "kv_uncond": None,
            "text_len": t_idx,
            "H": H,
            "W": W,
            "pm": pm,
            "token_h": token_h,
            "token_w": token_w,
            "image_seq_len": image_seq_len,
            "noise_scale": noise_scale,
            "indexes_image": indexes_image,
            "grid_hw": raw_grid_hw,
        }

    # ------------------------------------------------------------------
    # Replay (for future image RL)
    # ------------------------------------------------------------------

    def replay(
        self,
        *,
        past_kv_cond: Any,
        text_len: int,
        image_shape: Tuple[int, int],
        segment: LatentSegment,
        params: SensenovaU1DiffusionParams,
        past_kv_uncond: Optional[Any] = None,
        step_indices: Optional[List[int]] = None,
    ) -> Any:
        """Recompute SDE log-probs for stored trajectory.

        Same structure as ``diffuse`` but with ``prev_sample`` set from stored
        trajectory (scoring mode, not sampling mode). KV inputs are supplied
        by the caller — this stage carries no conditions object of its own.
        """
        from unirl.models.types.diffusion import ReplayResult

        bundle = self.model
        model = bundle.model
        device = next(model.parameters()).device

        kv_cond = past_kv_cond
        H, W = image_shape

        patch_size = bundle.patch_size
        merge_size = bundle.merge_size
        pm = patch_size * merge_size

        token_h = H // pm
        token_w = W // pm
        image_seq_len = token_h * token_w

        noise_scale = self._compute_noise_scale(model, params, image_seq_len)

        # Convert schedule
        schedule = segment.sigmas
        timesteps = 1.0 - schedule.float()
        timesteps = rl_ops.apply_time_schedule(
            model, timesteps, image_seq_len, params.timestep_shift
        )

        # Resolve target steps
        if step_indices is not None:
            target_steps = step_indices
        elif segment.sde_indices is not None:
            target_steps = segment.sde_indices.tolist()
        else:
            target_steps = []

        indexes_image = rl_ops.build_image_indexes(
            model, token_h, token_w, text_len, device
        )
        raw_grid_hw = torch.tensor([[H // patch_size, W // patch_size]], device=device, dtype=torch.long)

        replay_logps = []
        replay_means = []

        with self._autocast_ctx():
            for step_i in target_steps:
                t_cur = timesteps[step_i]
                sigma = schedule[step_i]
                sigma_next = schedule[step_i + 1]

                # Get stored x_t and x_{t+1}
                idx_in_stored = (segment.indices == step_i).nonzero(as_tuple=True)[0]
                idx_next = (segment.indices == step_i + 1).nonzero(as_tuple=True)[0]

                if len(idx_in_stored) == 0 or len(idx_next) == 0:
                    continue

                z = segment.latents[:, idx_in_stored[0].item()].to(device)
                prev_sample = segment.latents[:, idx_next[0].item()].to(device)

                # Unpatchify to pixel for embedding
                image_pred = rl_ops.unpatchify(model, z, pm, H, W)

                # Build embeds
                image_embeds = rl_ops.build_image_embeds(
                    model, image_pred, t_cur.unsqueeze(0).to(device),
                    grid_hw=raw_grid_hw,
                    noise_scale_value=noise_scale if model.add_noise_scale_embedding else None,
                )

                # Predict velocity with gradient
                v_pred = self.step.predict_velocity(
                    model,
                    z=z,
                    image_embeds=image_embeds,
                    indexes_image=indexes_image,
                    attn_mask={"full_attention": None},
                    past_key_values=kv_cond,
                    t=t_cur.unsqueeze(0).to(device),
                    image_token_num=image_seq_len,
                    image_size=(W, H),
                )

                # SDE denoise with prev_sample set (scoring mode)
                _, log_prob, prev_mean = self.step.denoise(
                    self.strategy,
                    v_t=v_pred,
                    x_t=z,
                    sigma=sigma.to(device),
                    sigma_next=sigma_next.to(device),
                    sigma_max=0.99,
                    eta=params.eta,
                    prev_sample=prev_sample,
                    step_index=step_i,
                )

                if log_prob is not None:
                    # See note in ``diffuse`` — log_prob already comes in as [B].
                    replay_logps.append(log_prob.to(self._logp_dtype))
                if prev_mean is not None:
                    replay_means.append(prev_mean.to(self._traj_dtype))

        log_probs = torch.stack(replay_logps, dim=1) if replay_logps else None
        prev_sample_means = torch.stack(replay_means, dim=1) if replay_means else None

        return ReplayResult(
            log_probs=log_probs,
            prev_sample_means=prev_sample_means,
        )

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def trainable_module(self) -> nn.Module:
        return self.model.transformer


__all__ = [
    "SensenovaU1DiffusionParams",
    "SensenovaU1DiffusionStep",
    "SensenovaU1DiffusionStage",
]
