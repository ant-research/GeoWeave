"""Pure velocity-MSE regularization algorithm (no GRPO surrogate).

Pulls the RL-tuned velocity field back toward the frozen pre-trained base::

    L_MSE(theta) = mse_weight * mean_s || v_theta(x_t, t, y) - v_ref(x_t, t, y) ||^2

evaluated at the SDE-recorded timesteps. Unlike :class:`BagelFlowUniGRPO`, this
algorithm has **no** clipped GRPO surrogate — the MSE is the sole loss term —
and therefore does NOT require per-sample advantages.

Designed for the GeoWeave interleave flow where text is trained via Interleave-RL
(reward-based) and image generation is regularized via this MSE constraint
only, preventing the shared backbone from drifting away from the pretrained
image-generation capability.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import AlgorithmStepResult, StageAlgorithm, typed_conditions


@contextmanager
def _disable_lora(module: Any) -> Iterator[bool]:
    """Temporarily disable LoRA adapters so a forward runs the base model."""
    try:
        from peft.tuners.lora import LoraLayer
    except Exception:
        yield False
        return
    layers = [m for m in module.modules() if isinstance(m, LoraLayer)]
    if not layers:
        yield False
        return
    for layer in layers:
        layer.enable_adapters(False)
    try:
        yield True
    finally:
        for layer in layers:
            layer.enable_adapters(True)


class FlowMSE(StageAlgorithm):
    """Pure velocity-MSE regularization for the diffusion track.

    Class attributes:
        requires_advantages: False — this algorithm ignores the advantage
            signal entirely (no policy gradient term).
        supports_multi_update: True — the MSE reference is the frozen base,
            not a per-update anchor; safe across N optimizer steps.
    """

    requires_advantages: bool = False
    supports_multi_update: bool = True

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        mse_weight: float = 1.5e-5,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("FlowMSE: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.params = params
        self.mse_weight = float(mse_weight)
        self.conditions_cls = conditions_cls
        self._ref_snapshot: Optional[Dict[int, torch.Tensor]] = None

    @staticmethod
    def _has_lora(transformer: Any) -> bool:
        try:
            from peft.tuners.lora import LoraLayer
        except Exception:
            return False
        return any(isinstance(m, LoraLayer) for m in transformer.modules())

    @contextmanager
    def _reference_weights(self, transformer: Any) -> Iterator[None]:
        """Swap frozen base weights into trainable params for v_ref forward.

        Full-FT analog of :func:`_disable_lora`. Captures a bf16 snapshot on
        first call; swaps in/out via in-place copy of FSDP local shards.
        """
        from unirl.train.ema import local_view

        live = [p for p in transformer.parameters() if p.requires_grad]
        if not live:
            raise RuntimeError(
                "FlowMSE: mse_weight > 0 with no LoRA and no trainable params — "
                "the transformer is fully frozen. Enable fine-tuning or set mse_weight=0."
            )
        if self._ref_snapshot is None:
            import logging
            logging.getLogger(__name__).warning(
                "FlowMSE._reference_weights: lazy snapshot fallback. If resuming "
                "from checkpoint, the snapshot will capture RESUMED weights (not "
                "pretrained base) → MSE regularizer degenerates. Trainer should "
                "call prime_reference_snapshot() before checkpoint restore."
            )
            self._ref_snapshot = {
                id(p): local_view(p).detach().to(dtype=torch.bfloat16).clone()
                for p in live
            }

        stash: List[torch.Tensor] = []
        for p in live:
            lv = local_view(p)
            stash.append(lv.detach().clone())
            lv.copy_(self._ref_snapshot[id(p)])
        try:
            yield
        finally:
            for p, saved in zip(live, stash):
                local_view(p).copy_(saved)

    def prime_reference_snapshot(self) -> None:
        """Eagerly capture pretrained base weights for v_ref.

        Trainer MUST call this after bundle load but BEFORE
        ``maybe_load_checkpoint``, so the snapshot pins to the pretrained base
        regardless of resume state. Idempotent; no-op for LoRA or
        no-trainable-params paths.
        """
        from unirl.train.ema import local_view

        transformer = self.stage.trainable_module()
        if self._has_lora(transformer):
            return
        if self._ref_snapshot is not None:
            return
        live = [p for p in transformer.parameters() if p.requires_grad]
        if not live:
            return
        self._ref_snapshot = {
            id(p): local_view(p).detach().to(dtype=torch.bfloat16).clone()
            for p in live
        }

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: Optional[torch.Tensor] = None,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        target_steps = self._resolve_target_steps(segment)
        if not target_steps or segment.sigmas is None:
            return AlgorithmStepResult(
                loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False
            )

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        transformer = self.stage.trainable_module()
        device = next(transformer.parameters()).device
        schedule = segment.sigmas.to(device)

        forward_kwargs = self.stage.build_forward_kwargs(
            typed_conds, params=self.params, device=device
        )

        full_ft_ref = not self._has_lora(transformer)

        with torch.no_grad():
            if full_ft_ref:
                ref_ctx = self._reference_weights(transformer)
            else:
                ref_ctx = _disable_lora(transformer)
            with ref_ctx as disabled:
                if not full_ft_ref and not disabled:
                    raise RuntimeError(
                        "FlowMSE: mse_weight > 0 but found neither peft LoRA layers "
                        "to disable nor trainable params to snapshot as v_ref. "
                        "Train with LoRA or full fine-tuning, or set mse_weight=0."
                    )
                v_refs = [
                    self.stage.predict_velocity_at(
                        forward_kwargs,
                        sample=segment.latents_at(s)[0].to(device),
                        sigma=schedule[s],
                        params=self.params,
                    ).detach()
                    for s in target_steps
                ]

        if full_ft_ref and torch.cuda.is_available():
            torch.cuda.empty_cache()

        mse_terms: List[torch.Tensor] = []
        for step_idx, v_ref in zip(target_steps, v_refs):
            x_t = segment.latents_at(step_idx)[0].to(device)
            sigma = schedule[step_idx]
            v_theta = self.stage.predict_velocity_at(
                forward_kwargs, sample=x_t, sigma=sigma, params=self.params
            )
            mse_terms.append(((v_theta - v_ref) ** 2).mean())

        mse = torch.stack(mse_terms).mean()
        (self.mse_weight * mse * loss_scale).backward()

        mse_val = float(mse.detach().item())
        return AlgorithmStepResult(
            loss=self.mse_weight * mse_val,
            metrics={"velocity_mse": mse_val, "mse_weight": self.mse_weight},
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    def _resolve_target_steps(self, segment: "LatentSegment") -> List[int]:
        if segment.sde_indices is None:
            return []
        return [int(i) for i in segment.sde_indices.tolist()]


__all__ = ["FlowMSE"]
