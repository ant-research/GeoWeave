"""Interleave-RL with inline velocity MSE at interleave image boundaries.

Subclasses :class:`GRPO` and overrides ``compute_loss_and_backward`` to
integrate velocity-MSE regularization at image boundaries during the AR
replay forward. The MSE gradient flows through the KV cache's ``grad_fn``
into the understanding branch's ``k_proj``/``v_proj`` weights, preventing
the shared backbone from drifting away from the pretrained image-generation
capability during Interleave-RL training.

Designed for GeoWeave's MoT architecture where:
- gen branch params are frozen (``freeze_gen=True``)
- gen branch reads K/V produced by und branch (KV cache is shared)
- Standard FlowMSE would yield MSE=0 because detached KV + frozen gen ⟹ v_θ ≡ v_ref

With inline MSE, the K/V at image boundaries have ``grad_fn`` from the merged
text-segment forward (``use_cache=True, WITH_GRAD``), enabling real gradient flow.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment
from unirl.utils.memory_utils import log_memory_usage

from .base import (
    AlgorithmStepResult,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
    rollout_replay_logp_absdiff,
    typed_conditions,
)
from .flow_mse import _disable_lora
from .grpo import GRPO


class InterleaveRL(GRPO):
    """Interleave-RL with inline velocity MSE at interleave image boundaries.

    When ``mse_weight > 0`` and the sample contains interleave images, this
    algorithm calls ``stage.replay_with_mse(...)`` which produces the standard
    Interleave-RL log-probabilities and velocity-MSE in one merged forward. When ``ar_kl_coef > 0``,
    it also regularizes sampled AR tokens against the pretrained reference:

        L = L_grpo + ar_kl_coef * L_ar_kl + mse_weight * L_mse

    The image path reuses the existing MSE reference context forward for AR
    reference logps. Text-only/no-MSE batches use ``replay_with_reference``.
    """

    supports_multi_update = True

    def __init__(
        self,
        *,
        mse_weight: float = 0.0,
        mse_steps: int = 3,
        diffusion_stage_attr: str = "diffusion",
        diffusion_params: Any = None,
        train_reshard_after_forward: bool = False,
        **grpo_kwargs: Any,
    ) -> None:
        # Capture pipeline before super().__init__ consumes it
        pipeline = grpo_kwargs.get("pipeline")
        super().__init__(**grpo_kwargs)
        self.mse_weight = float(mse_weight)
        self.mse_steps = int(mse_steps)
        self._diffusion_stage_attr = diffusion_stage_attr
        self._diffusion_params = diffusion_params
        self.train_reshard_after_forward = bool(train_reshard_after_forward)
        self._pipeline = pipeline
        # Keyed by parameter FQN (stable across Ray serialization + FSDP
        # rewraps). id()-based keying breaks when the algorithm is pickled to
        # workers or when FSDP replaces underlying tensors.
        self._ref_snapshot: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _has_lora(transformer: Any) -> bool:
        try:
            from peft.tuners.lora import LoraLayer
        except Exception:
            return False
        return any(isinstance(m, LoraLayer) for m in transformer.modules())

    @staticmethod
    def _reshard_transformer(transformer: Any) -> None:
        """Return every FSDP2-wrapped submodule to its stable local-shard state."""
        for module in transformer.modules():
            reshard = getattr(module, "reshard", None)
            if callable(reshard):
                reshard()

    @staticmethod
    def _set_reshard_after_forward(transformer: Any, enabled: bool) -> None:
        """Set the FSDP2 policy on wrapped blocks without changing rollout config."""
        for module in transformer.modules():
            setter = getattr(module, "set_reshard_after_forward", None)
            if callable(setter):
                setter(bool(enabled), recurse=False)

    @contextmanager
    def _reference_weights(self, transformer: Any) -> Iterator[None]:
        """Swap frozen base weights into trainable params for v_ref forward.

        For FSDP2 DTensor Parameters we operate on ``p._local_tensor`` (the
        plain local shard) via ``copy_``. Snapshot values are captured from
        the SAME local shard so shapes/types match. Root-level plain params
        use ``p.data``.

        FSDP2 lazy-shards on first access: a snapshot built while ``p`` was
        still a plain Parameter (shape [N, K]) may later be applied when
        ``p`` has become a DTensor with local shard [N/DP, K]. In that case
        we slice the full snapshot down to this rank's shard.
        """
        def _shard(p: Any) -> torch.Tensor:
            return p._local_tensor if hasattr(p, "_local_tensor") else p.data

        def _match_shard(snap: torch.Tensor, shard: torch.Tensor) -> torch.Tensor:
            if snap.shape != shard.shape:
                raise RuntimeError(
                    "Reference snapshot/local shard mismatch after explicit reshard: "
                    f"snap={tuple(snap.shape)} shard={tuple(shard.shape)}"
                )
            return snap

        self._reshard_transformer(transformer)

        live = [(n, p) for n, p in transformer.named_parameters() if p.requires_grad]
        if not live:
            yield
            return

        if self._ref_snapshot is None:
            import logging
            logging.getLogger(__name__).warning(
                "InterleaveRL._reference_weights: lazy snapshot fallback. If "
                "resuming from checkpoint, the snapshot will capture RESUMED "
                "weights (not pretrained base) → reference regularization degenerates. "
                "Trainer should call prime_reference_snapshot() before checkpoint "
                "restore."
            )
            self._ref_snapshot = {
                n: _shard(p).detach().to(dtype=torch.bfloat16).clone()
                for n, p in live
            }

        stash: List[torch.Tensor] = []
        for n, p in live:
            shard = _shard(p)
            stash.append(shard.detach().clone())
            shard.copy_(_match_shard(self._ref_snapshot[n], shard))
        try:
            yield
        finally:
            # The reference forward leaves reshard_after_forward=false groups
            # unsharded. Restore policy weights only after returning to local
            # shards, matching the representation captured in ``stash``.
            self._reshard_transformer(transformer)
            for (_, p), saved in zip(live, stash):
                _shard(p).copy_(saved)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def prime_reference_snapshot(self) -> None:
        """Eagerly capture pretrained base weights for AR-KL/v_ref.

        Trainer MUST call this after bundle load but BEFORE
        ``maybe_load_checkpoint``, so the snapshot pins to the pretrained base
        regardless of resume state. Idempotent; no-op for LoRA or
        no-trainable-params paths.
        """
        if self.mse_weight <= 0 and self.ar_kl_coef <= 0:
            return
        transformer = self.stage.trainable_module()
        if self._has_lora(transformer):
            return
        if self._ref_snapshot is not None:
            return
        self._reshard_transformer(transformer)
        live = [(n, p) for n, p in transformer.named_parameters() if p.requires_grad]
        if not live:
            return
        self._ref_snapshot = {
            n: (p._local_tensor if hasattr(p, "_local_tensor") else p.data)
            .detach()
            .to(dtype=torch.bfloat16)
            .clone()
            for n, p in live
        }

    def _get_ref_ctx(self, transformer: Any):
        """Return a callable that produces the appropriate ref-weight context manager."""
        if self._has_lora(transformer):
            return lambda: _disable_lora(transformer)
        return lambda: self._reference_weights(transformer)

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
        sample_mask: Optional[torch.Tensor] = None,
        record_policy_entropy: bool = False,
    ) -> AlgorithmStepResult:
        if segment.tokens is None or segment.lengths is None or segment.log_probs is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        needs_reference_logps = self.ar_kl_coef > 0.0

        # Check globally: if any DP rank has a real image/MSE trajectory, every
        # rank must enter replay_with_mse so packed context and padded velocity
        # forwards stay collective-symmetric.
        local_has_images = (
            self.mse_weight > 0
            and hasattr(typed_conds, "generated_images")
            and hasattr(typed_conds, "latent_segments")
            and any(len(imgs) > 0 for imgs in typed_conds.generated_images)
            and any(len(segs) > 0 for segs in typed_conds.latent_segments)
        )
        flag = torch.tensor(
            int(local_has_images), device=segment.tokens.device, dtype=torch.int32
        )
        if self.mse_weight > 0 and torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        has_images = bool(flag.item())

        if not has_images and not needs_reference_logps:
            return super().compute_loss_and_backward(
                conditions=conditions,
                segment=segment,
                advantages=advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
                sample_mask=sample_mask,
                record_policy_entropy=record_policy_entropy,
            )

        transformer = self.stage.trainable_module()
        diffusion_stage = None
        if self._pipeline is not None:
            diffusion_stage = getattr(self._pipeline, self._diffusion_stage_attr, None)
        use_mse_path = has_images and diffusion_stage is not None

        # Preserve the historical fallback when MSE was requested but the
        # diffusion stage is unavailable. AR KL can still run independently.
        if not use_mse_path and not needs_reference_logps:
            return super().compute_loss_and_backward(
                conditions=conditions,
                segment=segment,
                advantages=advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
                sample_mask=sample_mask,
                record_policy_entropy=record_policy_entropy,
            )

        ref_ctx = self._get_ref_ctx(transformer)
        reshard_enabled = False
        if self.train_reshard_after_forward:
            # Ragged rollout stays ZeRO-2; packed replay is call-symmetric and
            # may temporarily enable post-forward resharding.
            self._reshard_transformer(transformer)
            self._set_reshard_after_forward(transformer, True)
            reshard_enabled = True

        try:
            if use_mse_path:
                replay_result = self.stage.replay_with_mse(
                    typed_conds,
                    segment=segment,
                    temperature=self.sampling_temperature,
                    diffusion_stage=diffusion_stage,
                    diffusion_params=self._diffusion_params,
                    ref_weight_ctx=ref_ctx,
                    mse_steps=self.mse_steps,
                    sample_mask=sample_mask,
                    compute_ref_logps=needs_reference_logps,
                    compute_policy_entropy=bool(record_policy_entropy),
                )
            else:
                replay_result = self.stage.replay_with_reference(
                    typed_conds,
                    segment=segment,
                    temperature=self.sampling_temperature,
                    ref_weight_ctx=ref_ctx,
                    sample_mask=sample_mask,
                    compute_policy_entropy=bool(record_policy_entropy),
                )

            policy_entropy = None
            if use_mse_path:
                if record_policy_entropy:
                    new_logp, ref_logp, mse_loss, policy_entropy = replay_result
                else:
                    new_logp, ref_logp, mse_loss = replay_result
            else:
                if record_policy_entropy:
                    new_logp, ref_logp, policy_entropy = replay_result
                else:
                    new_logp, ref_logp = replay_result
                mse_loss = new_logp.sum() * 0.0

            if new_logp.shape[0] == 0:
                return AlgorithmStepResult(
                    loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False
                )

            old_logp = segment.log_probs.to(
                dtype=new_logp.dtype, device=new_logp.device
            )
            adv_per_token = self._resolve_token_advantages(
                advantages,
                segment,
                dtype=new_logp.dtype,
                device=new_logp.device,
            )
            clip_range = _resolve_clip_range_from_schedule(
                self.clip_range, self.clip_schedule, training_progress
            )
            clip_high = (
                None
                if self.clip_range_high is None
                else _resolve_clip_range_from_schedule(
                    self.clip_range_high,
                    self.clip_schedule,
                    training_progress,
                )
            )
            loss_per_elem, ratio_metrics = _grpo_clip_loss(
                new_logp=new_logp,
                old_logp=old_logp,
                advantages=adv_per_token,
                clip_range=clip_range,
                clip_range_high=clip_high,
            )
            grpo_loss = self._reduce_token_loss(
                loss_per_elem,
                segment.lengths,
                sample_mask=sample_mask,
                token_mask=segment.sca_token_mask,
            )
            ar_ref_kl, kl_metrics = self._reference_kl_loss(
                new_logp=new_logp,
                ref_logp=ref_logp,
                lengths=segment.lengths,
                sample_mask=sample_mask,
                token_mask=segment.sca_token_mask,
            )
            ar_kl_loss = self.ar_kl_coef * ar_ref_kl
            total_loss = grpo_loss + ar_kl_loss + self.mse_weight * mse_loss

            profile_memory = os.environ.get("SENSENOVA_MEM_PROFILE", "0") == "1"
            if profile_memory and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            (total_loss * loss_scale).backward()
            if profile_memory and torch.cuda.is_available():
                torch.cuda.synchronize()
                log_memory_usage("geoweave.backward", level=logging.WARNING)
        finally:
            if reshard_enabled:
                self._reshard_transformer(transformer)
                self._set_reshard_after_forward(transformer, False)

        mse_val = float(mse_loss.detach().item())
        grpo_val = float(grpo_loss.detach().item())
        ar_ref_kl_val = float(ar_ref_kl.detach().item())
        metrics: Dict[str, Any] = {
            "policy_loss": grpo_val,
            "velocity_mse": mse_val,
            "mse_weight": self.mse_weight,
            "ar_ref_kl": ar_ref_kl_val,
            "ar_kl_coef": self.ar_kl_coef,
            "ar_kl_loss": float(ar_kl_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "clip_range": float(clip_range),
            "clip_range_high": float(clip_range if clip_high is None else clip_high),
            **rollout_replay_logp_absdiff(new_logp, old_logp),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
            **{k: float(v.item()) for k, v in kl_metrics.items() if k != "ar_ref_kl"},
            **self._masked_policy_metrics(
                new_logp=new_logp,
                old_logp=old_logp,
                lengths=segment.lengths,
                sample_mask=sample_mask,
                token_mask=segment.sca_token_mask,
                clip_range=clip_range,
                clip_range_high=clip_high,
            ),
        }
        if policy_entropy is not None:
            metrics["policy_entropy"] = float(policy_entropy.detach().item())
        return AlgorithmStepResult(
            loss=float(total_loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )


__all__ = ["InterleaveRL"]
