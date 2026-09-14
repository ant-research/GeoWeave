"""Stage-driven ``GRPO`` over a ``TextSegment``.

Implements :class:`StageAlgorithm` and shares the module-level
``_grpo_clip_loss`` / ``_resolve_clip_range_from_schedule`` helpers (in
:mod:`unirl.algorithms.base`) with :class:`FlowGRPO` so their loss
math stays identical. The teacher-forced forward and per-token log-prob
recompute are owned by ``stage.replay(...)``; the algorithm is ~20 lines of
ratio-clip math.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
    rollout_replay_logp_absdiff,
    typed_conditions,
)


@dataclass
class GRPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    clip_range: float = 1e-4
    clip_range_high: Optional[float] = None
    policy_entropy_interval: int = 0
    clip_schedule: str = "constant"
    ar_kl_coef: float = 0.0


class GRPO(StageAlgorithm):
    """GRPO over an AR ``TextSegment`` via ``ARStage.replay``.

    The teacher-forced forward and per-token log-prob recompute is owned by
    :meth:`ARStage.replay`; this class expands per-sample advantages to per-
    token via ``cu_seqlens`` and runs the same PPO clip math.

    Args:
        stage: The :class:`ARStage` whose ``replay`` produces packed-varlen
            new log-probs aligned with ``segment.log_probs``.
        clip_range: PPO clip range epsilon.
        clip_schedule: ``"constant"``, ``"linear_decay"``, or
            ``"cosine_decay"``.
        conditions_cls: Stage-typed conditions container with
            ``from_dict(Mapping[str, Condition])``.
        sampling_temperature: AR rollout temperature, applied as a
            ``logits / T`` scaling inside :meth:`ARStage.replay` so
            replay's log-softmax matches SGLang's sampling distribution
            (``log_softmax(logits / T)``). Injected at construction time
            from the rollout engine config; falls back to
            :class:`ARSamplingParams` default when no engine is configured.
    """

    # old_logp is the rollout (SGLang) log-prob, which is frozen on the segment
    # and does NOT change across mini-batch updates — so reusing it across
    # num_updates_per_batch>1 is the deliberate rollout-anchored PPO ratio
    # (verl bypass_mode=True parity), matching DRPO. The ratio then absorbs the
    # rollout-vs-train engine gap on later mini-batches (accepted for parity).
    supports_multi_update = True

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        clip_range_high: Optional[float] = None,
        policy_entropy_interval: int = 0,
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        ar_kl_coef: float = 0.0,
        conditions_cls: Optional[Type[Any]] = None,
        sampling_temperature: Optional[float] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("GRPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.clip_range = float(clip_range)
        self.clip_range_high = None if clip_range_high is None else float(clip_range_high)
        self.policy_entropy_interval = int(policy_entropy_interval)
        if self.policy_entropy_interval < 0:
            raise ValueError(
                "GRPO: policy_entropy_interval must be >= 0 "
                f"(0 disables), got {self.policy_entropy_interval}"
            )
        self.clip_schedule = str(clip_schedule)
        self.loss_agg_mode = str(loss_agg_mode)
        self.horizon = int(horizon)
        self.ar_kl_coef = float(ar_kl_coef)
        if self.ar_kl_coef < 0.0:
            raise ValueError(f"GRPO: ar_kl_coef must be >= 0, got {self.ar_kl_coef}")
        self.conditions_cls = conditions_cls
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)

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
        if self.ar_kl_coef > 0.0:
            raise RuntimeError(
                "GRPO: ar_kl_coef > 0 requires a reference-aware algorithm "
                "such as InterleaveRL"
            )

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        _log = logging.getLogger(__name__)
        _debug_timing = _log.isEnabledFor(logging.DEBUG)
        _rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        _t_replay = time.perf_counter() if _debug_timing else 0.0
        replay_result = self.stage.replay(
            typed_conds,
            segment=segment,
            temperature=self.sampling_temperature,
            compute_policy_entropy=bool(record_policy_entropy),
        )
        policy_entropy = None
        if record_policy_entropy:
            new_logp, policy_entropy = replay_result
        else:
            new_logp = replay_result
        # new_logp: [total_tokens]
        if _debug_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        if _debug_timing:
            _log.debug(
                "[DEBUG][TRAIN] rank=%d phase=replay tokens=%d elapsed_s=%.1f",
                _rank,
                int(new_logp.shape[0]),
                time.perf_counter() - _t_replay,
            )
        # old_logp = the rollout log-prob, frozen on the segment — the deliberate
        # rollout-anchored ratio across num_updates_per_batch steps (see the
        # supports_multi_update class comment; verl bypass_mode=True parity).
        old_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)
        adv_per_token = self._resolve_token_advantages(
            advantages, segment, dtype=new_logp.dtype, device=new_logp.device
        )

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        clip_high = (
            None
            if self.clip_range_high is None
            else _resolve_clip_range_from_schedule(self.clip_range_high, self.clip_schedule, training_progress)
        )
        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_per_token,
            clip_range=clip_range,
            clip_range_high=clip_high,
        )
        loss = self._reduce_token_loss(
            loss_per_elem,
            segment.lengths,
            sample_mask=sample_mask,
            token_mask=segment.sca_token_mask,
        )
        _t_bwd = time.perf_counter() if _debug_timing else 0.0
        (loss * loss_scale).backward()
        if _debug_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        if _debug_timing:
            _log.debug(
                "[DEBUG][TRAIN] rank=%d phase=backward elapsed_s=%.1f",
                _rank,
                time.perf_counter() - _t_bwd,
            )

        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            "clip_range_high": float(clip_range if clip_high is None else clip_high),
            **rollout_replay_logp_absdiff(new_logp, old_logp),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
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
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )

    @staticmethod
    def _masked_policy_metrics(
        *,
        new_logp: torch.Tensor,
        old_logp: torch.Tensor,
        lengths: torch.Tensor,
        sample_mask: Optional[torch.Tensor],
        clip_range: float,
        clip_range_high: Optional[float],
        token_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Policy diagnostics over samples that actually contribute gradients."""
        if sample_mask is None and token_mask is None:
            return {}
        lengths_list = [int(n) for n in lengths.tolist()]
        valid = (
            torch.ones(len(lengths_list), dtype=torch.bool, device=new_logp.device)
            if sample_mask is None
            else sample_mask.detach().to(device=new_logp.device).reshape(-1) > 0.5
        )
        if int(valid.shape[0]) != len(lengths_list):
            raise ValueError(
                f"GRPO masked metrics: sample_mask={int(valid.shape[0])} "
                f"!= batch={len(lengths_list)}"
            )
        effective_token_mask = torch.repeat_interleave(
            valid, torch.tensor(lengths_list, dtype=torch.long, device=new_logp.device)
        )
        if token_mask is not None:
            packed = token_mask.detach().to(device=new_logp.device).reshape(-1) > 0.5
            if packed.numel() != effective_token_mask.numel():
                raise ValueError("GRPO masked metrics token_mask must align with tokens")
            effective_token_mask = effective_token_mask & packed
        selected_new = new_logp[effective_token_mask]
        selected_old = old_logp[effective_token_mask]
        sample_has_tokens = []
        for part in torch.split(effective_token_mask, lengths_list):
            sample_has_tokens.append(bool(part.any().item()))
        effective_samples = sum(sample_has_tokens)
        metrics = {
            "effective_train_samples": float(effective_samples),
            "effective_train_tokens": float(effective_token_mask.sum().item()),
            "train_sample_fraction": float(effective_samples / len(lengths_list)) if lengths_list else 0.0,
            "train_token_fraction": float(effective_token_mask.float().mean().item()) if effective_token_mask.numel() else 0.0,
        }
        if selected_new.numel() == 0:
            return metrics
        log_diff = selected_new - selected_old
        ratio = torch.exp(log_diff)
        high = clip_range if clip_range_high is None else clip_range_high
        clipped = torch.maximum(
            (ratio - 1.0 > high).float(),
            (1.0 - ratio > clip_range).float(),
        )
        absdiff = log_diff.abs()
        metrics.update(
            {
                "masked_ratio_mean": float(ratio.mean().item()),
                "masked_ratio_std": float(ratio.std().item()) if ratio.numel() > 1 else 0.0,
                "masked_clip_fraction": float(clipped.mean().item()),
                "masked_approx_kl": float((0.5 * log_diff.pow(2)).mean().item()),
                "masked_rollout_replay_logp_absdiff_mean": float(absdiff.mean().item()),
                "masked_rollout_replay_logp_absdiff_max": float(absdiff.max().item()),
            }
        )
        return metrics

    def _reduce_token_loss(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
        *,
        sample_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reduce packed per-token values with the configured GRPO semantics.

        ``sample_mask`` is intentionally optional. Existing policy-loss callers
        omit it to preserve historical behavior; reference-KL callers pass it so
        DP padding/invalid samples do not contribute to the regularizer.
        """
        lengths_list = [int(n) for n in lengths.tolist()]
        if sum(lengths_list) != int(values.shape[0]):
            raise ValueError(
                "GRPO token reduction: packed values="
                f"{int(values.shape[0])} != sum(lengths)={sum(lengths_list)}"
            )
        valid = None
        if sample_mask is not None:
            valid = sample_mask.detach().to(device=values.device).reshape(-1) > 0.5
            if int(valid.shape[0]) != len(lengths_list):
                raise ValueError(
                    "GRPO token reduction: sample_mask="
                    f"{int(valid.shape[0])} != batch={len(lengths_list)}"
                )

        packed_mask = None
        if token_mask is not None:
            packed_mask = token_mask.detach().to(device=values.device).reshape(-1) > 0.5
            if int(packed_mask.numel()) != int(values.shape[0]):
                raise ValueError(
                    "GRPO token reduction: token_mask="
                    f"{int(packed_mask.numel())} != packed values={int(values.shape[0])}"
                )

        parts = torch.split(values, lengths_list)
        mask_parts = (
            torch.split(packed_mask, lengths_list) if packed_mask is not None else [None] * len(parts)
        )
        if self.loss_agg_mode in ("seq-mean-token-sum-norm", "seq-mean-token-mean"):
            reduced: List[torch.Tensor] = []
            for i, (part, part_mask) in enumerate(zip(parts, mask_parts)):
                if valid is not None and not bool(valid[i].item()):
                    continue
                selected_part = part if part_mask is None else part[part_mask]
                if part_mask is not None and selected_part.numel() == 0:
                    continue
                if self.loss_agg_mode == "seq-mean-token-sum-norm":
                    reduced.append(selected_part.sum() / float(self.horizon))
                else:
                    reduced.append(
                        selected_part.mean() if selected_part.numel() else part.sum() * 0.0
                    )
            if not reduced:
                return values.sum() * 0.0
            return torch.stack(reduced).mean()

        effective_mask = packed_mask
        if valid is not None:
            sample_token_mask = torch.repeat_interleave(
                valid,
                torch.tensor(lengths_list, dtype=torch.long, device=values.device),
            )
            effective_mask = (
                sample_token_mask
                if effective_mask is None
                else effective_mask & sample_token_mask
            )
        if effective_mask is None:
            return values.mean()
        selected = values[effective_mask]
        return selected.mean() if selected.numel() else values.sum() * 0.0

    def _reference_kl_loss(
        self,
        *,
        new_logp: torch.Tensor,
        ref_logp: Optional[torch.Tensor],
        lengths: torch.Tensor,
        sample_mask: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Sampled-token k3 estimate of ``KL(policy || reference)``.

        Tokens come from the rollout behavior policy, so after policy updates
        this is the standard near-on-policy approximation rather than an exact
        full-vocabulary categorical KL. Reference log-probs are always detached;
        gradients flow only through ``new_logp``.
        """
        zero = new_logp.sum() * 0.0
        if ref_logp is None:
            return zero, {}
        if ref_logp.shape != new_logp.shape:
            raise ValueError(
                "GRPO reference KL: ref_logp shape="
                f"{tuple(ref_logp.shape)} != new_logp shape={tuple(new_logp.shape)}"
            )
        ref = ref_logp.detach().to(dtype=new_logp.dtype, device=new_logp.device)
        ref_minus_policy = ref - new_logp
        kl_per_token = torch.exp(ref_minus_policy) - ref_minus_policy - 1.0
        kl = self._reduce_token_loss(
            kl_per_token, lengths, sample_mask=sample_mask, token_mask=token_mask
        )
        with torch.no_grad():
            policy_ref_log_ratio = new_logp - ref
            metric_values = policy_ref_log_ratio
            effective_mask = None
            lengths_list = [int(n) for n in lengths.tolist()]
            if sample_mask is not None:
                valid = sample_mask.detach().to(device=new_logp.device).reshape(-1) > 0.5
                effective_mask = torch.repeat_interleave(
                    valid,
                    torch.tensor(lengths_list, dtype=torch.long, device=new_logp.device),
                )
            if token_mask is not None:
                packed = token_mask.detach().to(device=new_logp.device).reshape(-1) > 0.5
                effective_mask = packed if effective_mask is None else effective_mask & packed
            if effective_mask is not None:
                metric_values = policy_ref_log_ratio[effective_mask]
            if metric_values.numel():
                ratio_mean = metric_values.mean()
                ratio_max = metric_values.max()
            else:
                ratio_mean = policy_ref_log_ratio.sum() * 0.0
                ratio_max = policy_ref_log_ratio.sum() * 0.0
            metrics = {
                "ar_ref_kl": kl.detach(),
                "policy_ref_log_ratio_mean": ratio_mean,
                "policy_ref_log_ratio_max": ratio_max,
            }
        return kl, metrics

    @staticmethod
    def _resolve_token_advantages(
        advantages: torch.Tensor,
        segment: TextSegment,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Use packed SCA advantages when present, else expand sequence GRPO."""
        if segment.token_advantages is not None:
            packed = segment.token_advantages.detach().to(dtype=dtype, device=device)
            if segment.tokens is None or int(packed.shape[0]) != int(segment.tokens.shape[0]):
                raise ValueError("SCA token_advantages must align 1:1 with segment.tokens")
            return packed
        if segment.lengths is None:
            raise ValueError("GRPO requires segment lengths to expand sample advantages")
        return GRPO._expand_advantages_to_tokens(
            advantages, segment.lengths, dtype=dtype, device=device
        )

    @staticmethod
    def _expand_advantages_to_tokens(
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand per-sample ``advantages [B]`` to per-token ``[total_tokens]``.

        Each sample's advantage is repeated across its ``lengths``-defined
        token span so that token positions in segment ``k`` all see
        ``advantages[k]``. ``lengths`` comes from
        :attr:`Batch.lengths` on the segment (derived from the framework-
        managed cu_seqlens).
        """
        bs = int(advantages.shape[0])
        if int(lengths.shape[0]) != bs:
            raise ValueError(f"GRPO advantage expansion: advantages batch={bs} != lengths={int(lengths.shape[0])}")
        chunks: List[torch.Tensor] = []
        adv_cast = advantages.detach().to(dtype=dtype, device=device)
        for k in range(bs):
            n = int(lengths[k].item())
            if n > 0:
                chunks.append(adv_cast[k].expand(n))
        if not chunks:
            return torch.zeros(0, dtype=dtype, device=device)
        return torch.cat(chunks, dim=0)


__all__ = ["GRPO", "GRPOConfig"]
