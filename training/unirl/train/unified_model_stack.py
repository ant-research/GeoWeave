"""Unified-backbone multi-algorithm train stack (HunyuanImage3).

Wraps ONE :class:`FSDPBackend` (a single shared transformer + optimizer +
scheduler + EMA) and TWO :class:`StageAlgorithm` siblings — an ``ar`` algorithm
over the ``TextSegment`` and an ``image`` algorithm over the ``LatentSegment`` —
into a single training driver.  Both algorithms run forward/backward against the
*same* shared backbone (HunyuanImage3 operates in ``mode="gen_text"`` for AR and
``mode="gen_image"`` for DiT on one set of weights), so their gradients
accumulate into one LoRA adapter and a single optimizer step applies both.

Mirrors :class:`unirl.train.stack.TrainStack` but for the unified-backbone
two-algorithm case.  Sequencing per :meth:`train` call::

    prepare_segment(ar); prepare_segment(image)              # once: freeze both π_old anchors
    for u in range(num_updates_per_batch):                   # PPO-style mini-batches
        backend.zero_grad()
        for name in ("ar", "image"):
            for (start, end) in micro_slices(mini_batch_u):
                algorithm[name].compute_loss_and_backward(loss_scale=1/N, ...)  # grads accumulate
        backend.optimizer_step(max_grad_norm=...)            # ONE step per mini-batch
    on_rollout_end()
    return {name: TrainStepResult, ...}                      # reduced across updates

With ``gradient_accumulation_steps > 1`` (and necessarily
``num_updates_per_batch == 1``), ``zero_grad`` runs at the start of a rollout
window, each rollout contributes one averaged mini-batch gradient, and gradient
clipping plus optimizer/scheduler/EMA advancement happens only when the window
closes. The final short window can be closed with ``force_optimizer_step=True``.

``num_updates_per_batch`` (default 1) splits each rollout shard into that many
disjoint mini-batches and runs one optimizer step per mini-batch, with each track's
π_old anchor frozen once across all of them — so the 2nd+ step is off-policy and the
clip / ratio trust region actually engages (the UniGRPO / FlowGRPO PPO schedule).
Mirrors :class:`~unirl.train.stack.TrainStack` but for the two-algorithm backbone.

This is the multi-stage train stack — several stage algorithms share one
optimizer step, in contrast to the single-stage ``TrainStack``.
"""

from __future__ import annotations

import inspect
import logging
from contextlib import nullcontext
from dataclasses import replace
from typing import Dict, List, Mapping, Optional, Tuple

import torch

from unirl.algorithms import AlgorithmStepResult, StageAlgorithm
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.train.backend.fsdp import FSDPBackend
from unirl.train.stack import TrainStepResult, _build_micro_batch_slices
from unirl.train.stack.base import _aggregate_update_results
from unirl.train.stack.planner.types import _positive_int, _update_ranges
from unirl.types.rollout_resp import RolloutTrack
from unirl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


class UnifiedModelTrainStack(Remote):
    """Single-backbone, multi-algorithm train stack.

    Holds one shared :class:`FSDPBackend` and a dict of named
    :class:`StageAlgorithm` siblings (``{"ar": GRPO, "image": FlowGRPO}``).
    Each algorithm trains its own track but backward-accumulates into the same
    shared transformer; one optimizer step applies all algorithms' gradients.

    Created as a sibling ``Remote`` inside a placement block; takes handles to
    its ``FSDPBackend`` and ``StageAlgorithm`` siblings via sibling-handle
    auto-resolve (same pattern as :class:`TrainStack`).
    """

    def __init__(
        self,
        *,
        fsdp_backend: FSDPBackend,
        ar_algorithm: StageAlgorithm,
        image_algorithm: StageAlgorithm = None,
        micro_batch_size: int,
        max_grad_norm: float,
        num_updates_per_batch: int = 1,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        super().__init__()
        if int(micro_batch_size) < 1:
            raise ValueError(f"UnifiedModelTrainStack.micro_batch_size must be >= 1; got {micro_batch_size}.")
        if float(max_grad_norm) <= 0.0:
            raise ValueError(f"UnifiedModelTrainStack.max_grad_norm must be > 0; got {max_grad_norm}.")
        self.fsdp_backend = fsdp_backend
        if self.fsdp_backend.grad_sync_deferred:
            raise ValueError(
                "UnifiedModelTrainStack does not yet support fsdp.defer_grad_sync: "
                "AR/image tracks and optional cross-rollout accumulation need one "
                "globally-last backward to trigger the deferred FSDP reduce-scatter. "
                "Set backend.fsdp_cfg.defer_grad_sync=false."
            )
        # Order matters only for logging; gradients accumulate regardless.
        self.algorithms: Dict[str, StageAlgorithm] = {"ar": ar_algorithm}
        if image_algorithm is not None:
            self.algorithms["image"] = image_algorithm
        self.micro_batch_size = int(micro_batch_size)
        self.max_grad_norm = float(max_grad_norm)
        # PPO-style multi-update: split each rollout shard into this many disjoint
        # mini-batches and run ONE optimizer step per mini-batch, with the π_old
        # anchor frozen once across all of them (prepare_segment). >1 makes the
        # clip / ratio trust region actually engage (the 2nd+ step is off-policy);
        # 1 (default) keeps the prior single-step behavior. BOTH algorithms must
        # keep their anchor frozen across the N steps (supports_multi_update).
        self.num_updates_per_batch = _positive_int(
            name="UnifiedModelTrainStack.num_updates_per_batch", value=num_updates_per_batch
        )
        if self.num_updates_per_batch > 1:
            for name, algo in self.algorithms.items():
                if not getattr(algo, "supports_multi_update", False):
                    raise ValueError(
                        f"num_updates_per_batch={self.num_updates_per_batch} requires every algorithm's "
                        f"π_old anchor to stay frozen across the N optimizer steps, but the {name!r} "
                        f"algorithm ({type(algo).__name__}) sets supports_multi_update=False. Set "
                        f"num_updates_per_batch=1."
                    )
        self.gradient_accumulation_steps = _positive_int(
            name="UnifiedModelTrainStack.gradient_accumulation_steps",
            value=gradient_accumulation_steps,
        )
        if self.gradient_accumulation_steps > 1 and self.num_updates_per_batch > 1:
            raise ValueError(
                "gradient_accumulation_steps>1 currently requires "
                "num_updates_per_batch=1: cross-rollout accumulation and PPO-style "
                "multiple optimizer updates have different step/anchor semantics."
            )
        self._gradient_accumulation_count = 0
        self._gradient_accumulation_has_backward = False
        self._gradient_accumulation_results: Dict[str, List[TrainStepResult]] = {}
        self._gradient_accumulation_track_batch_sizes: Dict[str, int] = {}

    def _optimizer_step_slices(self, total: int) -> List[List[Tuple[int, int]]]:
        """Per-optimizer-step lists of absolute ``(start, end)`` micro-batch slices.

        One inner list per ``num_updates_per_batch`` mini-batch (one optimizer step),
        each split into ``micro_batch_size`` micro-batches. Shared by
        :meth:`prepare_segment` (to freeze the anchor at the exact geometry) and the
        train loop. Mirrors :meth:`unirl.train.stack.TrainStack._optimizer_step_slices`.
        """
        steps: List[List[Tuple[int, int]]] = []
        for mini_start, mini_end in _update_ranges(total_size=total, num_updates=self.num_updates_per_batch):
            steps.append(
                [
                    (mini_start + ms, mini_start + me)
                    for ms, me in _build_micro_batch_slices(
                        total_size=mini_end - mini_start, micro_batch_size=self.micro_batch_size
                    )
                ]
            )
        return steps

    def prepare_segment(self, name: str, resp_track: RolloutTrack) -> None:
        """Freeze one algorithm's π_old anchor once, before the multi-update loop.

        No-op if ``segment`` is None or the algorithm has no ``prepare_segment``. If
        the algorithm recomputes its anchor at train geometry (``recomputes_anchor()``
        — e.g. FlowGRPO under ``old_logp_source='replay'``), the declared
        ``anchor_fields`` are recomputed at the SAME (mini, micro) slices training will
        use, so the on-policy ratio is exactly 1 (mirrors
        :meth:`TrainStack.prepare_segment`). Rollout-anchored algorithms (the BAGEL
        UniGRPO recipe: AR GRPO + image ``old_logp_source='rollout'``) take the
        one-shot path — the anchor is the rollout emission, geometry-independent.
        """
        if resp_track.segment is None:
            return
        algorithm = self.algorithms[name]
        prepare = getattr(algorithm, "prepare_segment", None)
        if prepare is None:
            return
        recomputes = getattr(algorithm, "recomputes_anchor", None)
        if recomputes is None or not recomputes():
            prepare(conditions=resp_track.conditions, segment=resp_track.segment)
            return
        micro_slices = [sl for step in self._optimizer_step_slices(int(resp_track.batch_size)) for sl in step]
        if len(micro_slices) == 1:
            prepare(conditions=resp_track.conditions, segment=resp_track.segment)
            return
        anchor_fields = getattr(algorithm, "anchor_fields", ())
        collected: Dict[str, List[torch.Tensor]] = {field: [] for field in anchor_fields}
        for start, end in micro_slices:
            micro = resp_track.slice(start, end)
            prepare(conditions=micro.conditions, segment=micro.segment)
            for field in collected:
                value = getattr(micro.segment, field, None)
                if value is None:
                    raise RuntimeError(
                        f"UnifiedModelTrainStack.prepare_segment: {type(algorithm).__name__} declares "
                        f"anchor field {field!r} but a micro-slice produced None."
                    )
                collected[field].append(value)
        for field, parts in collected.items():
            setattr(resp_track.segment, field, torch.cat(parts, dim=0))

    def _record_policy_entropy(
        self, name: str, *, optimizer_step: int, will_step: bool
    ) -> bool:
        """Whether this optimizer step should run the exact AR entropy monitor."""
        interval = int(
            getattr(self.algorithms[name], "policy_entropy_interval", 0)
        )
        return bool(will_step and interval > 0 and optimizer_step % interval == 0)

    def _backward_track(
        self,
        name: str,
        resp_track: RolloutTrack,
        micro_slices: List[Tuple[int, int]],
        *,
        training_progress: float,
        record_policy_entropy: bool = False,
    ) -> tuple[TrainStepResult, bool]:
        """Backward one algorithm's track over the given absolute ``micro_slices``
        (no zero_grad / no optimizer step).

        Returns ``(per_algorithm_result, has_backward)``. ``zero_grad`` and the shared
        ``optimizer_step`` are owned by :meth:`_train_one_step` so both algorithms
        accumulate into one step. ``micro_slices`` are absolute ranges into
        ``resp_track`` for ONE optimizer step (one ``num_updates_per_batch``
        mini-batch), produced by :meth:`_optimizer_step_slices`.
        """
        algorithm = self.algorithms[name]
        if resp_track.advantages is None and getattr(algorithm, "requires_advantages", True):
            raise ValueError(
                f"UnifiedModelTrainStack.train: track {name!r} has advantages=None; "
                "upstream advantage pipeline must populate it before training."
            )
        if not micro_slices:
            raise ValueError(f"UnifiedModelTrainStack.train: empty micro_slices for track {name!r}.")

        bs = int(resp_track.batch_size)
        update_total = sum(end - start for start, end in micro_slices)
        micros: List[AlgorithmStepResult] = []
        total_loss = 0.0
        has_backward = False

        single_micro = len(micro_slices) == 1 and micro_slices[0] == (0, bs)
        for start, end in micro_slices:
            micro_track = resp_track if single_micro else resp_track.slice(start, end)
            # Algorithms report a mean loss over the micro-batch. Weight each
            # backward by its sample share so a short tail micro-batch is not
            # over-represented (1/len(micros) is only correct for equal sizes).
            loss_scale = (end - start) / float(update_total)
            backward_kwargs = dict(
                conditions=micro_track.conditions,
                segment=micro_track.segment,
                advantages=micro_track.advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
            signature = inspect.signature(algorithm.compute_loss_and_backward).parameters
            if micro_track.status is not None and "sample_mask" in signature:
                backward_kwargs["sample_mask"] = micro_track.status
            if "record_policy_entropy" in signature:
                backward_kwargs["record_policy_entropy"] = bool(record_policy_entropy)
            result = algorithm.compute_loss_and_backward(**backward_kwargs)
            micros.append(result)
            total_loss += result.loss
            has_backward = has_backward or result.has_backward

        aggregated: Mapping[str, object] = aggregate_numeric_metrics([r.metrics for r in micros if r.metrics])
        # grad_norm / lr are filled by ``_train_one_step`` after the shared optimizer step.
        partial = TrainStepResult(
            loss=total_loss,
            grad_norm=0.0,
            lr=0.0,
            has_backward=has_backward,
            micros=micros,
            metrics=aggregated,
        )
        return partial, has_backward

    @torch.no_grad()
    def _scale_accumulated_gradients(self, scale: float) -> None:
        """Scale every materialized gradient before the accumulated optimizer step."""
        for parameter in self.fsdp_backend.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(float(scale))

    def _stamp_accumulation_metrics(
        self,
        result: TrainStepResult,
        *,
        optimizer_stepped: bool,
        accumulation_count: int,
        window_closed: bool,
    ) -> TrainStepResult:
        return replace(
            result,
            metrics={
                **dict(result.metrics),
                "optimizer_stepped": float(optimizer_stepped),
                "gradient_accumulation_window_closed": float(window_closed),
                "gradient_accumulation_count": float(accumulation_count),
                "gradient_accumulation_steps": float(self.gradient_accumulation_steps),
            },
        )

    def _accumulate_one_rollout(
        self,
        tracks: Dict[str, RolloutTrack],
        slices_by_track: Dict[str, List[Tuple[int, int]]],
        *,
        training_progress: float,
        force_optimizer_step: bool,
    ) -> Dict[str, TrainStepResult]:
        """Accumulate one rollout's gradients and optionally close the window.

        Gradients are reduced normally by FSDP on every backward and may be
        offloaded with the model between rollouts.  We average the completed
        window immediately before clipping/stepping, which also gives the final
        short window the correct divisor without knowing its size in advance.
        """
        present_batch_sizes = {
            name: int(tracks[name].batch_size)
            for name in self.algorithms
            if name in tracks and name in slices_by_track
        }
        if self._gradient_accumulation_count == 0:
            self.fsdp_backend.zero_grad()
            self._gradient_accumulation_has_backward = False
            self._gradient_accumulation_results = {}
            self._gradient_accumulation_track_batch_sizes = present_batch_sizes
        elif present_batch_sizes != self._gradient_accumulation_track_batch_sizes:
            raise ValueError(
                "UnifiedModelTrainStack cross-rollout accumulation requires each "
                "track to keep a fixed presence and per-rank batch size within a "
                "window; otherwise averaging the shared AR/image gradients by one "
                "rollout count changes their relative objective weights. Got "
                f"expected={self._gradient_accumulation_track_batch_sizes}, "
                f"current={present_batch_sizes}. Use gradient_accumulation_steps=1 "
                "for variable tracks or implement per-track gradient buffers."
            )

        will_step = (
            self._gradient_accumulation_count + 1 >= self.gradient_accumulation_steps
            or force_optimizer_step
        )
        optimizer_step = int(self.fsdp_backend._optimizer_step_count) + 1
        current: Dict[str, TrainStepResult] = {}
        for name in self.algorithms:
            if name not in tracks or name not in slices_by_track:
                continue
            partial, has_backward = self._backward_track(
                name,
                tracks[name],
                slices_by_track[name],
                training_progress=training_progress,
                record_policy_entropy=self._record_policy_entropy(
                    name, optimizer_step=optimizer_step, will_step=will_step
                ),
            )
            current[name] = partial
            self._gradient_accumulation_has_backward = (
                self._gradient_accumulation_has_backward or has_backward
            )
            self._gradient_accumulation_results.setdefault(name, []).append(partial)

        self._gradient_accumulation_count += 1
        accumulation_count = self._gradient_accumulation_count
        should_step = (
            accumulation_count >= self.gradient_accumulation_steps
            or force_optimizer_step
        )
        if not should_step:
            current_lr = self._current_lr()
            return {
                name: self._stamp_accumulation_metrics(
                    replace(result, lr=current_lr),
                    optimizer_stepped=False,
                    accumulation_count=accumulation_count,
                    window_closed=False,
                )
                for name, result in current.items()
            }

        if self._gradient_accumulation_has_backward:
            self._scale_accumulated_gradients(1.0 / float(accumulation_count))
            grad_norm = float(
                self.fsdp_backend.optimizer_step(
                    max_grad_norm=float(self.max_grad_norm)
                )
            )
            optimizer_stepped = True
            self.on_rollout_end()
        else:
            grad_norm = 0.0
            optimizer_stepped = False
            logger.warning(
                "UnifiedModelTrainStack._accumulate_one_rollout: accumulation "
                "window reported no backward; skipping optimizer step."
            )

        lr = self._current_lr()
        completed: Dict[str, TrainStepResult] = {}
        for name, window_results in self._gradient_accumulation_results.items():
            n = len(window_results)
            aggregated = TrainStepResult(
                loss=sum(result.loss for result in window_results) / n,
                grad_norm=grad_norm,
                lr=lr,
                has_backward=any(result.has_backward for result in window_results),
                micros=[micro for result in window_results for micro in result.micros],
                metrics=aggregate_numeric_metrics(
                    [dict(result.metrics) for result in window_results if result.metrics]
                ),
            )
            completed[name] = self._stamp_accumulation_metrics(
                aggregated,
                optimizer_stepped=optimizer_stepped,
                accumulation_count=accumulation_count,
                window_closed=True,
            )

        self._gradient_accumulation_count = 0
        self._gradient_accumulation_has_backward = False
        self._gradient_accumulation_results = {}
        self._gradient_accumulation_track_batch_sizes = {}
        return completed

    def _train_one_step(
        self,
        tracks: Dict[str, RolloutTrack],
        slices_by_track: Dict[str, List[Tuple[int, int]]],
        *,
        training_progress: float,
    ) -> Dict[str, TrainStepResult]:
        """One optimizer step: zero_grad → backward BOTH tracks over their mini-batch
        slices → shared optimizer_step → stamp grad_norm / lr onto each track's result.
        """
        self.fsdp_backend.zero_grad()
        results: Dict[str, TrainStepResult] = {}
        any_backward = False
        optimizer_step = int(self.fsdp_backend._optimizer_step_count) + 1
        for name in self.algorithms:
            if name not in tracks or name not in slices_by_track:
                continue
            partial, has_backward = self._backward_track(
                name,
                tracks[name],
                slices_by_track[name],
                training_progress=training_progress,
                record_policy_entropy=self._record_policy_entropy(
                    name, optimizer_step=optimizer_step, will_step=True
                ),
            )
            results[name] = partial
            any_backward = any_backward or has_backward

        if any_backward:
            # Multi-update only: the prior update's forward/backward churn fragments the
            # CUDA pool, so this step's clip_grad_norm NCCL all_reduce can fail to find a
            # contiguous buffer (OOM with free-but-fragmented memory — exactly the
            # num_updates_per_batch>1 optimizer-step OOM). Returning the freed activation
            # blocks to the driver first defragments. Gated on >1 so the single-update
            # path (and the LoRA recipe) pays nothing.
            if self.num_updates_per_batch > 1 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            grad_norm = float(self.fsdp_backend.optimizer_step(max_grad_norm=float(self.max_grad_norm)))
        else:
            grad_norm = 0.0
            logger.warning("UnifiedModelTrainStack._train_one_step: no algorithm reported backward; skipping step.")

        lr = self._current_lr()
        for name, r in list(results.items()):
            results[name] = TrainStepResult(
                loss=r.loss, grad_norm=grad_norm, lr=lr, has_backward=r.has_backward, micros=r.micros, metrics=r.metrics
            )
        return results

    def on_rollout_end(self) -> None:
        """Per-rollout-boundary hook — delegates to the FSDPBackend's EMA."""
        self.fsdp_backend.on_rollout_end()

    def _train_step_profiler(self):
        """Lazily build the per-worker train-step profiler (None unless UNIRL_PROFILE)."""
        cached = getattr(self, "_profiler_cache", "unset")
        if cached == "unset":
            from unirl.utils.profiling import maybe_build_train_profiler

            cached = maybe_build_train_profiler(int(getattr(self.fsdp_backend, "_rank", 0)))
            self._profiler_cache = cached
        return cached

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def train_track(
        self,
        ar_track: RolloutTrack,
        image_track: Optional[RolloutTrack] = None,
        *,
        training_progress: float,
        force_optimizer_step: bool = False,
    ) -> Dict[str, TrainStepResult]:
        """Driver-callable: prepare and backward one rollout's train tracks.

        With ``gradient_accumulation_steps=1`` this preserves the historical
        one-or-more optimizer updates per rollout. With a larger value, gradients
        survive across calls and only the closing call performs the shared optimizer
        step; ``force_optimizer_step`` closes a final short window.

        Both tracks arrive DP_SCATTER-sharded (each DP worker gets its shard of
        both). ``prepare_segment`` freezes each track's π_old anchor ONCE; then the
        shard is split into ``num_updates_per_batch`` disjoint mini-batches and one
        optimizer step runs per mini-batch (each: backward ar + image over its
        mini-batch → one shared step). The 2nd+ step is off-policy, so the clip /
        ratio trust region engages; ``num_updates_per_batch=1`` is the prior
        single-step behavior. Per-track results are reduced across the updates;
        per-shard results merge back via ``pytree_cat`` on collect.

        When ``image_algorithm`` is None (text-only GRPO), the image track is
        accepted but not trained — only the AR track participates in backward.
        """
        device = self.fsdp_backend._device
        ar_track = ar_track.to_device(device)

        from unirl.utils.profiling import profile_scope

        scope = profile_scope()
        if scope == "one-update" and not getattr(self, "_warned_one_update", False):
            self._warned_one_update = True
            logger.warning(
                "UNIRL_PROFILE=one-update is not supported on the unified-model stack "
                "(no _run_updates loop); use UNIRL_PROFILE=train. No trace produced."
            )
        profiler = self._train_step_profiler() if scope == "train" else None
        with profiler.record("train_track") if profiler is not None else nullcontext():
            tracks: Dict[str, RolloutTrack] = {"ar": ar_track}
            if image_track is not None:
                image_track = image_track.to_device(device)
                tracks["image"] = image_track
            # Freeze each track's π_old anchor once, before the multi-update loop.
            for name in self.algorithms:
                if name in tracks:
                    self.prepare_segment(name, tracks[name])

            steps_by_track = {
                name: self._optimizer_step_slices(int(tracks[name].batch_size))
                for name in self.algorithms if name in tracks
            }
            if self.gradient_accumulation_steps > 1:
                slices_by_track = {
                    name: steps_by_track[name][0]
                    for name in self.algorithms if name in steps_by_track
                }
                results = self._accumulate_one_rollout(
                    tracks,
                    slices_by_track,
                    training_progress=float(training_progress),
                    force_optimizer_step=bool(force_optimizer_step),
                )
                per_update: List[Dict[str, TrainStepResult]] = []
            else:
                per_update = []
                for u in range(self.num_updates_per_batch):
                    slices_by_track = {
                        name: steps_by_track[name][u]
                        for name in self.algorithms if name in steps_by_track
                    }
                    per_update.append(
                        self._train_one_step(
                            tracks,
                            slices_by_track,
                            training_progress=float(training_progress),
                        )
                    )
        if profiler is not None:
            profiler.step()

        if self.gradient_accumulation_steps > 1:
            return results

        self.on_rollout_end()

        # Reduce each track's per-optimizer-step results into one summary, attaching
        # each optimizer step's own metrics on ``per_update`` so the logger emits ONE
        # wandb point per optimizer update (on-policy update0 vs off-policy update1+
        # stay distinct series instead of being averaged into one misleading
        # ratio_mean). Mirrors TrainStack.train_track; passthrough at num_updates==1.
        #
        # Only aggregate tracks that were actually trained THIS call: an image
        # algorithm configured but receiving ``image_track=None`` (base ckpt that
        # doesn't emit ``<img>`` yet) produces per_update dicts with only "ar" —
        # iterating self.algorithms would KeyError on "image".
        results: Dict[str, TrainStepResult] = {}
        trained_names = [n for n in self.algorithms if n in tracks]
        for name in trained_names:
            updates = [upd[name] for upd in per_update]
            aggregated = _aggregate_update_results(updates)
            if len(updates) > 1:
                aggregated = replace(
                    aggregated,
                    per_update=tuple(
                        {**dict(r.metrics), "loss": float(r.loss), "grad_norm": float(r.grad_norm), "lr": float(r.lr)}
                        for r in updates
                    ),
                )
            results[name] = aggregated
        return results

    def _current_lr(self) -> float:
        optimizer = self.fsdp_backend.optimizer
        param_groups = getattr(optimizer, "param_groups", None)
        if isinstance(param_groups, list) and param_groups:
            return float(param_groups[0]["lr"])
        scheduler = self.fsdp_backend.scheduler
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            last = scheduler.get_last_lr()
            if isinstance(last, list) and last:
                return float(last[0])
        return 0.0


__all__ = ["UnifiedModelTrainStack"]
