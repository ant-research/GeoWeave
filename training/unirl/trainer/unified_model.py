"""UniRL v2 HunyuanImage3 unified-backbone trainer.

One shared HunyuanImage3 backbone (a single MoE transformer that operates in
``mode="gen_text"`` for AR and ``mode="gen_image"`` for DiT) trained jointly by
two algorithms — ``GRPO`` over the AR ``TextSegment`` and ``FlowGRPO``
over the DiT ``LatentSegment`` — both backward-accumulating into ONE LoRA
adapter and sharing each optimizer update (see :class:`UnifiedModelTrainStack`).

Two-engine design (mirrors :class:`~unirl.models.pe.pipeline.PEPipeline`'s
two-level fan-out but with the backbone shared). PE composes two in-process
child pipelines (SD3 + Qwen3, two LoRAs); HI3 instead drives TWO standalone
vLLM-Omni engine Remotes that share ONE backbone / ONE LoRA:

- ``ar_rollout`` (modality ``hi3_ar_recaption``, GPUs 0-3): original prompt → ``N``
  think/recaption texts (group-by-prompt → AR GRPO).
- ``dit_rollout`` (modality ``hi3_dit_recaption``, GPUs 4-7): each recaption → ``M``
  images of distinct noise (group-by-recaption → FlowGRPO).

The trainer assembles the lineage itself (``make_root_track(N)`` /
``fork_track(M)``, exactly like ``PEPipeline.generate``) because the two engines
are independent Remotes, not a composed pipeline. Reward routing then matches
:class:`~unirl.trainer.pe.PETrainer`: score the image track, credit-assign
the mean image reward up to the AR track, per-track GRPO advantages, then one
:class:`UnifiedModelTrainStack` call. AR and image gradients share the same LoRA;
the optimizer advances immediately or at a cross-rollout accumulation boundary.

GPU partition: each engine is ONE multi-GPU actor anchored on a distinct worker
via ``pool.create_remote(device_ids=[0])`` / ``[4]`` (NOT plain ``remote()``,
which would bind it to the whole fraction=1.0 scope and collide both engines'
device-env in one process). Each engine clears ``CUDA_VISIBLE_DEVICES`` for its
multi-GPU HI3 modality (see ``engine._HI3_MULTI_GPU_MODALITIES``) and its stage
YAML's ``runtime.devices`` pins AR→0-3 / DiT→4-7 — disjoint physical cards. The
boot-smoke anchor was unsafe only because nothing time-shared the cards; here
the colocate dance (base offloaded during rollout, engines asleep during train)
makes anchoring correct — see ``train_step`` and ``_wire_engine``.

One ``train_step``::

    wake ar+dit; [sync → both]; ar_resp = ar_rollout.generate(ar_req)
    img_shell = ar_track.fork_track(M); dit_resp = dit_rollout.generate(dit_req)
    sleep ar+dit
    reward.score_and_attach(image track)         # only the image track is scorable
    resp.propagate_rewards("mean")               # image reward → ar track
    track.compute_advantages() per track         # ar groups by prompt, image by ar-sample
    unified_model_stack.train_track(ar_track, image_track) # 2 backward → shared update/window

Pairs with ``examples/unified_model/hi3_vllmomni.yaml`` and ``unirl/train_unified_model.py``.
Deferred (same as the reference trainers): multi-epoch replay, checkpoint /
eval cadence, structured logging.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import shutil
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import TensorRef, hydrate
from unirl.distributed.tensor.batch import Batch
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.primitives import Texts
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack, _track_with_field
from unirl.types.sampling import BaseSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)

# Track names produced by the vLLM-Omni HI3 rollout (see
# ``rollout/engine/vllm_omni/response.py``): "ar" is the root (TextSegment,
# groups by prompt), "image" is its 1:1 child (LatentSegment).
AR_TRACK = "ar"
IMAGE_TRACK = "image"


@dataclasses.dataclass(frozen=True)
class _DumpImage:
    filename: str
    pixels: torch.Tensor


@dataclasses.dataclass(frozen=True)
class _InterleaveDumpSnapshot:
    rollout_id: int
    out_dir: str
    records: list[dict[str, Any]]
    images: list[_DumpImage]
    histogram: dict[int, int]
    boundary_mismatches: int
    reward_nonzero: int
    save_results: bool = True


def _save_dump_image(image: _DumpImage, out_dir: str, quality: int) -> None:
    from PIL import Image

    pixels = image.pixels
    if pixels.device.type != "cpu" or pixels.dtype != torch.uint8:
        raise TypeError("Dump JPEG writer requires a CPU uint8 tensor")
    if pixels.ndim != 3 or pixels.shape[0] not in (1, 3):
        raise ValueError(
            f"Dump JPEG writer expected CHW with 1 or 3 channels, got {tuple(pixels.shape)}"
        )
    array = pixels.permute(1, 2, 0).contiguous().numpy()
    if array.shape[2] == 1:
        array = array[:, :, 0]
    output_path = os.path.join(out_dir, image.filename)
    if image.filename.lower().endswith(".png"):
        Image.fromarray(array).save(output_path, format="PNG")
    else:
        Image.fromarray(array).save(
            output_path, format="JPEG", quality=quality, optimize=False
        )


def _write_interleave_dump_snapshot(
    snapshot: _InterleaveDumpSnapshot,
    *,
    image_workers: int,
    jpeg_quality: int,
) -> None:
    started = time.perf_counter()
    tmp_dir = snapshot.out_dir + ".tmp"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        with ThreadPoolExecutor(
            max_workers=image_workers,
            thread_name_prefix="unirl-dump-image",
        ) as executor:
            list(
                executor.map(
                    lambda image: _save_dump_image(image, tmp_dir, jpeg_quality),
                    snapshot.images,
                )
            )
        if snapshot.save_results:
            with open(os.path.join(tmp_dir, "samples.jsonl"), "w") as f:
                for record in snapshot.records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(os.path.join(tmp_dir, "_SUCCESS"), "w") as f:
            f.write("ok\n")
        if os.path.isdir(snapshot.out_dir):
            shutil.rmtree(snapshot.out_dir)
        os.replace(tmp_dir, snapshot.out_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    elapsed = time.perf_counter() - started
    logger.info(
        "[RUN][DUMP] rollout=%d samples=%d images=%d elapsed_s=%.2f path=%s",
        snapshot.rollout_id,
        len(snapshot.records),
        len(snapshot.images),
        elapsed,
        snapshot.out_dir,
    )
    logger.debug(
        "[DEBUG][DUMP] rollout=%d histogram=%s boundary_mismatch=%d "
        "reward_nonzero=%d/%d jpeg_quality=%d image_workers=%d",
        snapshot.rollout_id,
        dict(sorted(snapshot.histogram.items())),
        snapshot.boundary_mismatches,
        snapshot.reward_nonzero,
        len(snapshot.records),
        jpeg_quality,
        image_workers,
    )


def deep_hydrate(obj: Any) -> Any:
    """Materialize every ``TensorRef`` leaf in ``obj`` to a real tensor, in place.

    The anchored single-actor engines return each track as ONE transport handle
    (a single ref spanning all samples), but the train side is num_devices-way DP and
    slices each track into per-rank shards — a single ref can't be intra-handle
    sliced. Hydrating on the driver fixes the mismatch (the DP dispatch then
    re-shards real tensors), but the driver has no ``TensorTransportRuntime``
    installed, so the runtime-backed ``TensorTransport.hydrate`` is
    unavailable here. ``hydrate`` instead pulls each leaf through
    its ref's ``.materialize(backend=None)`` (a plain ``ray.get`` from the owning worker's store),
    which works from the driver — we walk the nested Batch/dict/list/TUPLE
    structure and apply it to every ``TensorRef``.

    NB: this walks TUPLES too (rebuilding them), unlike ``_collect_leaves``
    which skips them. HunyuanImage3's fused condition stores ``rope_cache`` as a
    ``tuple`` of two TensorRef; the DP scatter's driver-side
    ``RolloutTrack.concat`` pads that rope (``conditions.concat`` → ``_pad_seq``
    → ``t.ndim``), so the rope MUST be real tensors here. (dp=1 never concats on
    the driver, so it never tripped on this.)
    """
    if isinstance(obj, TensorRef):
        return hydrate(obj)
    if isinstance(obj, Batch):
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if v is not None:
                new = deep_hydrate(v)
                if new is not v:
                    setattr(obj, f.name, new)
        return obj
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = deep_hydrate(obj[k])
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = deep_hydrate(obj[i])
        return obj
    if isinstance(obj, tuple):
        return tuple(deep_hydrate(x) for x in obj)
    return obj


class UnifiedModelTrainer(BaseTrainer):
    """HunyuanImage3 unified-backbone joint trainer (AR + DiT, one LoRA)."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        reward_cfg: DictConfig,
        ar_algorithm_cfg: DictConfig,
        image_algorithm_cfg: Optional[DictConfig] = None,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        ar_rollout_cfg: Optional[DictConfig] = None,
        dit_rollout_cfg: Optional[DictConfig] = None,
        rollout_cfg: Optional[DictConfig] = None,
        sync_cfg: Optional[DictConfig] = None,
        dump_dir: Optional[str] = None,
        dump_async: bool = True,
        dump_image_workers: int = 8,
        dump_jpeg_quality: int = 90,
        dump_max_pending: int = 1,
        logging_cfg: Optional[DictConfig] = None,
        enable_fsdp_offload: bool = True,
        stage_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        stage_values: Dict[str, Any] = dict(stage_config) if stage_config else {}
        dynamic_trajectory_scheduling = bool(
            stage_values.get("dynamic_trajectory_scheduling", False)
        )
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        # Colocate memory dance: offload the FSDP train state (base + grads +
        # optimizer) to CPU during rollout so the awake engines fit, onload
        # before the train backward. HI3's ~150GB base needs this → default True.
        self._enable_fsdp_offload = bool(enable_fsdp_offload)
        # Per-request routing metadata pinned by the recipe (e.g.
        # {"system_message": ""} for GeoWeave interleave), forwarded onto
        # every RolloutReq via _build_req. Empty ⇒ pipelines see an empty dict
        # and fall back to their per-key defaults.
        self._stage_config = stage_values
        self._dynamic_trajectory_scheduling = dynamic_trajectory_scheduling
        self._dynamic_reward_workers = int(
            self._stage_config.get("dynamic_reward_workers", 1)
        )
        self._dynamic_prompt_group_refill = bool(
            self._stage_config.get("dynamic_prompt_group_refill", False)
        )
        self._exclude_truncated_from_advantage_stats = bool(
            self._stage_config.get("exclude_truncated_from_advantage_stats", False)
        )
        if self._exclude_truncated_from_advantage_stats and not bool(
            self._stage_config.get("ignore_truncated_samples", False)
        ):
            raise ValueError(
                "stage_config.exclude_truncated_from_advantage_stats=true requires "
                "stage_config.ignore_truncated_samples=true"
            )
        self._reserve_prompt_count = int(
            self._stage_config.get(
                "reserve_prompt_count",
                self.batch_size if self._dynamic_prompt_group_refill else 0,
            )
        )
        if self._reserve_prompt_count < 0:
            raise ValueError("stage_config.reserve_prompt_count must be >= 0")
        self._last_dynamic_consumed_prompt_indices: List[int] = []
        self._last_dynamic_scheduler_metrics: Dict[str, float] = {}
        if self._dynamic_reward_workers < 1:
            raise ValueError("stage_config.dynamic_reward_workers must be >= 1")
        self._driver_reward = None
        if self._dynamic_trajectory_scheduling:
            fsdp_cfg = backend_cfg.get("fsdp_cfg")
            rollout_reshard = (
                bool(fsdp_cfg.get("reshard_after_forward", False))
                if fsdp_cfg is not None
                else False
            )
            if rollout_reshard:
                raise ValueError(
                    "dynamic_trajectory_scheduling requires backend.fsdp_cfg."
                    "reshard_after_forward=false. Note that this setting alone "
                    "does not make partial-rank rollout safe for shared FSDP; "
                    "multi-DP trainside layouts are rejected separately."
                )

        # Driver-side alignment for a variable-length image track (GeoWeave
        # interleave emits 0..K images per AR sample). The train stack's
        # ``_update_ranges`` requires per-worker batch_size to be a positive
        # multiple of ``num_updates_per_batch``; DP_SCATTER shards the image
        # track over ``train_dp_size`` workers. So the WHOLE image track must
        # be sliced down to a multiple of ``alignment = train_dp_size *
        # num_updates_per_batch`` before ``stack.train_track``. If the
        # remainder truncates to zero, we drop the image track entirely so
        # every worker uniformly skips image backward (keeping FSDP grad sync
        # in step). Read the divisor from stack_cfg — must match the stack's
        # constructor arg.
        self._stack_updates = int(stack_cfg.get("num_updates_per_batch", 1) or 1)
        if self._stack_updates < 1:
            self._stack_updates = 1
        self._gradient_accumulation_steps = int(
            stack_cfg.get("gradient_accumulation_steps", 1) or 1
        )
        if self._gradient_accumulation_steps < 1:
            raise ValueError("stack.gradient_accumulation_steps must be >= 1")
        if self._gradient_accumulation_steps > 1 and self._stack_updates > 1:
            raise ValueError(
                "stack.gradient_accumulation_steps>1 currently requires "
                "stack.num_updates_per_batch=1"
            )

        # Filled from the constructed stack Handle below. It may be smaller than
        # pool.num_devices when sequence parallelism groups multiple ranks into
        # one DP rank, so using the raw GPU count would over-truncate image data.
        self._train_dp_size = 0
        self._image_track_alignment = 0

        # W&B logging (logging_cfg, wandb_logger, optimizer-step counter) is owned
        # by BaseTrainer + UniRLWandBLogger now — see super().__init__ above.

        # Intrusive debug dump: per rollout, write original prompt + AR output
        # text (= the think/recaption that conditions DiT) + decoded images +
        # rewards under ``dump_dir/rollout_<id>/``. None disables. Best-effort —
        # never breaks training (see :meth:`_dump_rollout`).
        self.dump_dir = str(dump_dir) if dump_dir else None
        self._dump_rollout_id = 0
        self._dump_async = bool(dump_async)
        self._dump_image_workers = int(dump_image_workers)
        self._dump_jpeg_quality = int(dump_jpeg_quality)
        self._dump_max_pending = int(dump_max_pending)
        if self._dump_image_workers < 1:
            raise ValueError("dump_image_workers must be >= 1")
        if not 1 <= self._dump_jpeg_quality <= 100:
            raise ValueError("dump_jpeg_quality must be in [1, 100]")
        if self._dump_max_pending < 1:
            raise ValueError("dump_max_pending must be >= 1")
        self._dump_executor: Optional[ThreadPoolExecutor] = None
        self._dump_futures: Deque[Future] = deque()
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
            if self._dump_async:
                self._dump_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="unirl-rollout-dump",
                )

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)

        # Set below from the `sync` block; None means no sync (e.g. trainside).
        self.weight_sync = None

        # Single shared slab: train backbone + both algorithms + rollout +
        # reward are siblings on one Worker (colocate; mirrors DiffusionTrainer's
        # non-separate branch).
        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.reward = remote_hydra(reward_cfg)

            # Two algorithms over the SAME shared pipeline (each resolves its
            # own stage via ``stage_attr``: ar→pipeline.ar, image→pipeline.diffusion).
            self.ar_algorithm = remote_hydra(ar_algorithm_cfg, pipeline=self.pipeline)
            self.image_algorithm = (
                remote_hydra(image_algorithm_cfg, pipeline=self.pipeline)
                if image_algorithm_cfg is not None
                else None
            )

            self._image_algo_needs_advantages = (
                image_algorithm_cfg is not None
                and image_algorithm_cfg.get("requires_advantages", True)
            )

            # One stack owns the single backend + both algorithms → one step.
            self.stack = remote_hydra(
                stack_cfg,
                fsdp_backend=self.backend,
                ar_algorithm=self.ar_algorithm,
                image_algorithm=self.image_algorithm,
            )
            self._train_dp_size = int(self.stack.dp_size)
            self._image_track_alignment = (
                self._train_dp_size * self._stack_updates
            )
            self._validate_train_batch_geometry()

            # Rollout wiring. Single-engine (M=1 / UniGRPO — a trainside or single
            # engine on the SHARED pipeline) short-circuits the two-engine HI3 path
            # below: no GPU partition, no weight sync, and (trainside) no base
            # offload since it samples the live FSDP modules. ``_shared_advantage``
            # makes train_step copy the AR's prompt-level advantage onto the 1:1
            # image track (M=1) instead of the degenerate per-rewrite grouping.
            self._single_engine = rollout_cfg is not None
            self._shared_advantage = self._single_engine
            self._rollout_is_trainside = False
            if self._single_engine:
                self.dp = 1
                self.ar_rollouts = []
                self.dit_rollouts = []
                self.ar_rollout = None
                self.dit_rollout = None
                rollout_parsed = parse_hydra_cfg(rollout_cfg)
                self._rollout_is_trainside = "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters
                if self._rollout_is_trainside:
                    self.rollout = remote(**rollout_parsed, pipeline=self.pipeline)
                    self._enable_fsdp_offload = False  # shares live FSDP modules
                else:
                    self.rollout = remote(**rollout_parsed)
                if getattr(self, "_dynamic_trajectory_scheduling", False):
                    if not self._rollout_is_trainside:
                        raise ValueError(
                            "dynamic_trajectory_scheduling currently requires a trainside rollout engine"
                        )
                    if self._image_algo_needs_advantages:
                        raise ValueError(
                            "dynamic_trajectory_scheduling currently supports the GeoWeave text-reward path only"
                        )
                    driver_reward_cfg = OmegaConf.create(
                        OmegaConf.to_container(reward_cfg, resolve=True)
                    )
                    backend_cfg = driver_reward_cfg.get("backend")
                    if backend_cfg is not None and "base_device" in backend_cfg:
                        backend_cfg.base_device = "cpu"
                    self._driver_reward = instantiate(driver_reward_cfg)
                return

            if ar_rollout_cfg is None or dit_rollout_cfg is None:
                raise ValueError(
                    "UnifiedModelTrainer: two-engine mode needs ar_rollout_cfg + dit_rollout_cfg; "
                    "pass a single rollout_cfg for single-engine (M=1 / UniGRPO) mode."
                )

            # COLOCATE MEMORY: offload the ~150GB frozen base to CPU BEFORE
            # booting the engines. Each engine grabs ~70GB (AR) / ~45GB (DiT) on
            # its 4 cards at boot; with the FSDP base still resident (~19GB/card)
            # that overlaps to >78GB and OOMs. With the base on CPU the engines
            # boot on their disjoint cards (AR 0-3, DiT 4-7) with room to spare.
            if self._enable_fsdp_offload:
                self.backend.offload()

            # Two standalone vLLM-Omni engines, each ONE multi-GPU actor anchored
            # on a DISTINCT worker (AR→device 0, DiT→device 4). The anchor is
            # load-bearing: plain remote() binds the engine to the whole
            # fraction=1.0 scope (all 8 devices, shared base worker), so BOTH
            # engines land in the same worker process and their device-env setup
            # collides — vllm-omni's set_stage_devices then remaps DiT's yaml
            # "4,5,6,7" back onto physical 0-3, overlapping AR → OOM. Anchoring on
            # separate workers keeps them in separate processes: each pops
            # CUDA_VISIBLE_DEVICES and its stage YAML's runtime.devices pins the
            # TP group to disjoint physical cards (AR 0-3, DiT 4-7) — the layout
            # boot smoke gotcha C verified. Colocate-safe because the train base is
            # offloaded during rollout and the engines sleep during train (the
            # memory dance in train_step time-shares the cards — so this is NOT
            # the boot-smoke landmine of engine+FSDP residing simultaneously).
            # DP over engine REPLICAS, one (AR, DiT) pair per node. dp = nodes
            # (16 devices / 8 per node → dp=2; single node → dp=1, fully
            # backward-compatible: range(1), anchors 0/4 = the original path).
            # Replica r is anchored on node r (DevicePool is node-aware,
            # node = device_id // devices_per_node): AR host-worker on device
            # r*8+1, DiT on r*8+4; each engine still spans cards r*8..r*8+3 /
            # r*8+4..r*8+7 via its stage YAML. AR is +1 (not r*8) to keep its host
            # worker off the train rank-0 worker (device 0) — see the push
            # self-deadlock note at the _wire_engine call below.
            per_node = self.pool.devices_per_node
            # Each replica pins ONE (AR 0-3, DiT 4-7) engine pair to a single
            # node, anchored at base+1 / base+4 with base = r*per_node. That
            # layout needs >= 8 cards on the node; with fewer, base+4 spills onto
            # the next node and silently splits the pair cross-node. Fail loud.
            if per_node < 8:
                raise ValueError(
                    "UnifiedModelTrainer: HI3 needs >= 8 devices/node for one "
                    "(AR 0-3, DiT 4-7) engine pair per node; got "
                    f"devices_per_node={per_node}."
                )
            self.dp = max(1, self.pool.num_devices // per_node)
            self.ar_rollouts = []
            self.dit_rollouts = []
            for r in range(self.dp):
                base = r * per_node
                # SERIALIZE engine boot: build one engine, then immediately
                # .sleep() it before building the next. Every @distributed Handle
                # call is synchronous (ray.get) and the heavy boot is Omni(...) in
                # the engine's __init__, so .sleep() blocks until THIS engine has
                # finished booting. Booting all dp*2 engines concurrently deadlocks
                # in the DiT warmup's kv_transfer_manager handshake (the 4-way-boot
                # blocker), so the per-engine quiesce is load-bearing — and it also
                # leaves every engine asleep, the steady state train_step expects.
                # AR anchor is base+1, NOT base: weight_sync rank 0 lives on the
                # train DP rank-0 worker = global device 0. If the AR engine were
                # anchored there too (base==0 for replica 0), it shares that one
                # worker PROCESS, and RemoteLoraWeightSync.push() — which runs on
                # rank 0 and does ray.get([... set_lora on the AR engine ...]) —
                # would block-call its own actor (the set_lora task queues behind
                # the in-flight push) → self-deadlock (push never returns, AR
                # set_lora never runs; DiT on device 4 is a separate process so it
                # loads fine). base+1 keeps the AR host worker off device 0 while
                # the engine still uses cards 0-3 via its stage YAML's runtime.devices.
                ar = self._wire_engine(ar_rollout_cfg, anchor_device=base + 1)
                ar.sleep()
                self.ar_rollouts.append(ar)
                dit = self._wire_engine(dit_rollout_cfg, anchor_device=base + 4)
                dit.sleep()
                self.dit_rollouts.append(dit)
            # Back-compat aliases for replica 0 (single-node code paths, dump,
            # debug, and any single-engine references still use these).
            self.ar_rollout = self.ar_rollouts[0]
            self.dit_rollout = self.dit_rollouts[0]

            if sync_cfg is not None:
                # LoRA sync gets ONLY the backend (a same-worker sibling); the
                # engines are cross-slab. RemoteLoraWeightSync.sync() extracts on
                # the train workers and pushes from rank 0 to EACH engine via a
                # plain Ray RPC, so hand it every replica's (role, workers) here.
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
                self.weight_sync.set_rollout_targets(
                    [(eng.role_name, eng.workers) for eng in self.ar_rollouts + self.dit_rollouts]
                )

    def _validate_train_batch_geometry(self) -> None:
        """Fail before model rollout when the fixed AR batch cannot DP-shard.

        ``DP_SCATTER`` splits over the stack Handle's real ``dp_size`` (which is
        ``world_size / sp_size`` under Ulysses), and each local shard is further
        partitioned by ``num_updates_per_batch``. The AR fanout is fixed by
        configuration, so validate it eagerly instead of failing after an
        expensive first rollout. Variable image tracks remain runtime-aligned by
        :meth:`_align_image_track_for_train`.
        """
        if self._dynamic_trajectory_scheduling:
            return
        ar_params = self.sampling_params.get("ar")
        ar_samples = int(ar_params.samples_per_prompt) if ar_params is not None else 1
        global_ar_batch = int(self.batch_size) * ar_samples
        required_multiple = self._train_dp_size * self._stack_updates
        if global_ar_batch % required_multiple != 0:
            raise ValueError(
                "UnifiedModelTrainer: fixed AR train batch must be divisible by "
                "stack.dp_size * num_updates_per_batch; got "
                f"batch_size={self.batch_size}, ar.samples_per_prompt={ar_samples}, "
                f"global_ar_batch={global_ar_batch}, stack.dp_size={self._train_dp_size}, "
                f"num_updates_per_batch={self._stack_updates}, "
                f"required_multiple={required_multiple}. Increase batch_size or "
                "samples_per_prompt, reduce num_devices/SP-adjusted DP size, or "
                "set num_updates_per_batch=1."
            )

    def _wire_engine(self, cfg: DictConfig, *, anchor_device: int) -> Any:
        """Build ONE multi-GPU vLLM-Omni engine actor anchored on one worker.

        ``device_ids=[anchor_device]`` pins the actor to a SINGLE worker (one
        process), not the whole placement scope — the engine is one TP-parallel
        Omni server, not a per-device DP replica. Inside the Omni subprocess the
        engine clears ``CUDA_VISIBLE_DEVICES`` and its stage YAML's
        ``runtime.devices`` spreads the TP group across its physical cards; using
        a distinct anchor per engine keeps the two engines' device-env setup in
        separate processes so they pin to disjoint cards (see the call site).
        The standalone HI3 engines take no ``pipeline`` (they boot their own
        Omni), so nothing sibling-handle-resolved is forwarded.
        """
        parsed = parse_hydra_cfg(cfg)
        role_cls = parsed.pop("role_cls")
        return self.pool.create_remote(role_cls, device_ids=[anchor_device], init_kwargs=parsed)

    def _build_req(self, inputs: RolloutInputs, rollout_id: int) -> RolloutReq:
        """Turn a data-source batch of ``P`` prompts into a typed ``RolloutReq``.

        Like :meth:`PETrainer._build_req`, NO pre-expansion: ``train_step`` fans
        out ``P → P*N → P*N*M`` itself (make_root_track / fork_track), and the
        reward expands ``req.primitives`` by ``N*M`` to align prompts to images.
        Pre-expanding here would double-count. The composed sampling params are
        kept whole (the reward reads ``ar.samples_per_prompt * diffusion.``
        ``samples_per_prompt`` to validate the expansion factor); the SDE step
        schedule is resolved off the diffusion sub-block per rollout and stamped
        back onto a per-request copy.
        """
        diff_params = self.sampling_params.get("diffusion")
        if diff_params is None:
            sampling_params = dict(self.sampling_params)
        else:
            sde_indices = diff_params.resolve_sde_indices(rollout_id)
            diffusion = dataclasses.replace(
                diff_params, sde_indices=sde_indices, scheduler=None
            )
            sampling_params = {**self.sampling_params, "diffusion": diffusion}
        return RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions={},
            sampling_params=sampling_params,
            stage_config=dict(self._stage_config),
            metadata=list(inputs.metadata) if inputs.metadata else [],
        )

    def run_rollout(self, req: RolloutReq) -> RolloutResp:
        """DP rollout: scatter the P prompts across the ``dp`` engine replicas
        (one (AR, DiT) pair per node), run each sub-batch on its replica, then
        ``RolloutTrack.concat`` the per-replica tracks. ``dp<=1`` or ``P<=1``
        falls back to the single-replica path (the original single-node rollout),
        so this is a transparent wrapper when not multinode.

        Replica calls are issued concurrently from driver threads. Each call
        still blocks on its own Ray Handle, but independent per-node engine actors
        overlap instead of serializing the rollout across nodes.
        """
        # Single-engine (M=1 / UniGRPO): the shared pipeline returns the 2-track
        # {"ar","image"} resp directly (DP_SCATTER-sharded like the train stack —
        # no anchored-engine single-handle hydration needed).
        if self._single_engine:
            return self.rollout.generate(req)
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError("UnifiedModelTrainer.run_rollout: req.primitives['text'] must be a Texts primitive.")
        prompts = list(texts.texts)
        n = len(prompts)
        if self.dp <= 1 or n <= 1:
            return self._run_rollout_one(self.ar_rollouts[0], self.dit_rollouts[0], req)

        # Contiguous near-equal prompt bounds across the dp replicas.
        bounds = [(n * r) // self.dp for r in range(self.dp + 1)]
        jobs: list[tuple[Any, Any, RolloutReq]] = []
        for r in range(self.dp):
            lo, hi = bounds[r], bounds[r + 1]
            if lo >= hi:
                continue
            # Only "text" is consumed downstream by _run_rollout_one; slice it
            # to this replica's prompt range and rebuild a standalone sub-req.
            sub_req = RolloutReq(
                sample_ids=list(req.sample_ids[lo:hi]),
                group_ids=list(req.group_ids[lo:hi]),
                primitives={"text": Texts(texts=prompts[lo:hi])},
                request_conditions=dict(req.request_conditions),
                sampling_params=req.sampling_params,
                stage_config=dict(req.stage_config),
                metadata=list(req.metadata[lo:hi]) if req.metadata else [],
            )
            jobs.append((self.ar_rollouts[r], self.dit_rollouts[r], sub_req))

        def _run_replica(job: tuple[Any, Any, RolloutReq]) -> RolloutResp:
            ar_engine, dit_engine, sub_req = job
            return self._run_rollout_one(ar_engine, dit_engine, sub_req)

        with ThreadPoolExecutor(
            max_workers=len(jobs),
            thread_name_prefix="unirl-rollout-replica",
        ) as executor:
            # executor.map preserves replica order, so concat keeps the original
            # prompt order even when nodes finish at different times.
            shards = list(executor.map(_run_replica, jobs))
        # Merge per-replica tracks via the default Batch.concat per-field
        # merge; segment rows are 1:1 with track samples, so the AR/image
        # segments stay globally consistent across replicas.
        #
        # CAVEAT — the fused condition's rope_cache is a ``shared_field``
        # (FusedMultimodalCondition), so this concat keeps replica-0's tensor
        # verbatim: the merged condition carries a rope_cache whose batch dim is
        # replica-0's sample count, NOT the global P*N*M. Harmless TODAY because
        # HI3 replay rebuilds rope from gen_image_mask + the real latent shape
        # (diffusion.py ``predict_noise`` [ROPE-FIX]; ar.py likewise) and never
        # reads the track's rope_cache — it only rides along in the KV-propagation
        # kwargs. If a future change makes replay consume ``fused.rope_cache``,
        # dp>1 would SILENTLY feed replica-0 rope to every sample (wrong gradient,
        # no crash, reward unaffected); make rope_cache a tuple-aware CONCAT field
        # before relying on it.
        return RolloutResp(
            tracks={name: RolloutTrack.concat([s.tracks[name] for s in shards]) for name in (AR_TRACK, IMAGE_TRACK)}
        )

    def _run_rollout_one(self, ar_engine: Any, dit_engine: Any, req: RolloutReq) -> RolloutResp:
        """One (AR, DiT) engine pair: PE-style fan-out → 2-track ``RolloutResp`` {"ar","image"}.

        Drives the given ``ar_engine`` / ``dit_engine`` pair (one replica). The
        DP wrapper :meth:`run_rollout` calls this once per replica with that
        node's engines; ``dp=1`` calls it once with replica 0.

        ::

            P prompts ─make_root_track(N)─▶ P*N recaptions  (AR engine, root "ar")
                      ─fork_track(M)──────▶ P*N*M images     (DiT engine, "image")

        Mirrors :meth:`PEPipeline.generate`, but the two 1:1 child generators are
        independent vLLM-Omni engine Remotes sharing one backbone/LoRA — so this
        assembles the lineage explicitly and grafts each engine's
        segment/decoded/conditions onto the lineage shell. The DiT engine reads
        the ORIGINAL prompt (``primitives['text']``) plus the recaption
        (``primitives['cot_text']``); each image's unique ``sample_id`` drives
        ``engine.seed_from_sample_id`` so the M images of a recaption differ.
        """
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError("UnifiedModelTrainer.run_rollout: req.primitives['text'] must be a Texts primitive.")
        prompts = list(texts.texts)

        ar_params = req.sampling_params.get("ar")
        diff_params = req.sampling_params.get("diffusion")
        n_recaptions = int(ar_params.samples_per_prompt) if ar_params is not None else 1
        n_images = int(diff_params.samples_per_prompt)

        # ── Level 1: P → P*N recaptions. Root "ar" track groups by prompt.
        ar_shell = req.make_root_track(track_name=AR_TRACK, branch=n_recaptions)
        ar_texts = Texts(texts=[t for t in prompts for _ in range(n_recaptions)])
        # Ship the WHOLE composed params: the hi3_ar_recaption adapter reads its AR
        # slice for sampling AND the diffusion slice's height/width for the
        # recaption prompt (the engine keeps no sampling defaults).
        ar_req = RolloutReq(
            sample_ids=list(ar_shell.sample_ids),
            group_ids=list(ar_shell.parent_ids),
            primitives={"text": ar_texts},
            request_conditions={},
            sampling_params=req.sampling_params,
        )
        ar_resp = ar_engine.generate(ar_req)
        ar_inner = ar_resp.tracks.get(AR_TRACK)
        recaptions = ar_inner.decoded if ar_inner is not None else None
        if not isinstance(recaptions, Texts):
            raise RuntimeError("UnifiedModelTrainer.run_rollout: AR engine returned no decoded Texts on tracks['ar'].")
        if len(recaptions.texts) != len(ar_shell.sample_ids):
            raise RuntimeError(
                f"UnifiedModelTrainer.run_rollout: AR engine returned {len(recaptions.texts)} recaption(s) "
                f"but the AR track expects {len(ar_shell.sample_ids)} (= P*N). The AR engine must be 1:1."
            )
        ar_track = _track_with_field(ar_shell, "segment", ar_inner.segment)
        ar_track = _track_with_field(ar_track, "decoded", recaptions)
        ar_track = _track_with_field(ar_track, "conditions", dict(ar_inner.conditions))

        # ── Level 2: P*N → P*N*M images. Fork "image" from "ar". For AR sample i
        # (0..P*N-1) the original prompt is prompts[i // N] and the recaption is
        # recaptions[i]; replicate each M× for the 1:1 DiT engine.
        img_shell = ar_track.fork_track(parent_name=AR_TRACK, child_name=IMAGE_TRACK, branch=n_images)
        n_ar = len(ar_shell.sample_ids)
        dit_prompts = Texts(texts=[prompts[i // n_recaptions] for i in range(n_ar) for _ in range(n_images)])
        dit_cot = Texts(texts=[recaptions.texts[i] for i in range(n_ar) for _ in range(n_images)])
        # Driver-authoritative x_T RECIPE (per-IMAGE, ROLLOUT-keyed gids). HI3's
        # DiT latent shape is AR-dynamic, so we ship only the recipe (no shape);
        # the worker's prepare_latents hook fills the shape post-AR and regenerates
        # the byte-identical x_T (NoiseRecipe). Keying on (rollout_id, image
        # sample_id) makes x_T per-rollout-VARYING — overriding the engine's
        # seed_from_sample_id, which is keyed on the rollout-STABLE sample_id alone
        # and so reused the SAME x_T every rollout (frozen-noise overfit, the bug
        # this fixes). ``_dump_rollout_id`` is set to the current rollout_id by the
        # train loop just before train_step. Opt out via DISABLE_DRIVER_XT.
        dit_noise_gids = (
            []
            if os.environ.get("DISABLE_DRIVER_XT")
            else [f"r{int(self._dump_rollout_id)}:{sid}" for sid in img_shell.sample_ids]
        )
        dit_req = RolloutReq(
            sample_ids=list(img_shell.sample_ids),
            group_ids=list(img_shell.parent_ids),
            primitives={"text": dit_prompts, "cot_text": dit_cot},
            request_conditions={},
            sampling_params={"diffusion": diff_params},
            init_noise_group_ids=dit_noise_gids,
        )
        dit_resp = dit_engine.generate(dit_req)
        img_inner = dit_resp.tracks.get(IMAGE_TRACK)
        if img_inner is None:
            raise RuntimeError(
                f"UnifiedModelTrainer.run_rollout: DiT engine returned no 'image' track (got {sorted(dit_resp.tracks.keys())})."
            )
        if len(img_inner.sample_ids) != len(img_shell.sample_ids):
            raise RuntimeError(
                f"UnifiedModelTrainer.run_rollout: DiT engine returned {len(img_inner.sample_ids)} image(s) "
                f"but the image track expects {len(img_shell.sample_ids)} (= P*N*M). The DiT engine must be 1:1."
            )
        img_track = _track_with_field(img_shell, "segment", img_inner.segment)
        img_track = _track_with_field(img_track, "decoded", img_inner.decoded)
        img_track = _track_with_field(img_track, "conditions", dict(img_inner.conditions))
        img_track = _track_with_field(img_track, "media_preview", img_inner.media_preview)

        # Each anchored engine returns its track as ONE transport handle (a single
        # ref spanning all P*N / P*N*M samples). The train side is num_devices-way DP and
        # slices each track into per-rank shards — but a single ref can't be
        # intra-handle-sliced ("does not align to ref boundaries"). Materialize
        # the tracks to real tensors on the driver here; the reward / advantage /
        # train DP dispatch then re-shards real tensors. (DiffusionTrainer dodges
        # this because its per-worker DP engine already emits one ref per rank,
        # aligned to the train DP boundaries — our single-actor TP engines don't.)
        deep_hydrate(ar_track)
        deep_hydrate(img_track)

        return RolloutResp(tracks={AR_TRACK: ar_track, IMAGE_TRACK: img_track})

    def train_step(
        self,
        req: RolloutReq,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
        force_optimizer_step: bool = False,
    ) -> Tuple[Dict[str, TrainStepResult], float]:
        """One ``rollout → reward → credit-assign → advantage → step`` pass.

        Returns ``(per_track_results, mean_reward)`` — ``mean_reward`` is the
        mean unnormalized image reward (for the log line). ``rollout_id`` keys
        the wandb panels (see :meth:`UniRLWandBLogger.log_rollout_step`).
        """
        t0 = time.perf_counter()
        rewards_precomputed = False
        dynamic_owner_ranks = None
        if self._single_engine:
            # Trainside / single-engine (M=1): the rollout shares the live FSDP
            # modules — no engine wake/sleep, no base offload, no weight sync.
            if getattr(self, "_dynamic_trajectory_scheduling", False):
                from unirl.trainer.dynamic_trainside import (
                    run_dynamic_trainside_rollout,
                )

                dynamic_result = run_dynamic_trainside_rollout(
                    rollout_handle=self.rollout,
                    driver_reward=self._driver_reward,
                    req=req,
                    rollout_id=rollout_id,
                    reward_workers=self._dynamic_reward_workers,
                    target_prompt_count=self.batch_size,
                )
                resp = dynamic_result.resp
                dynamic_owner_ranks = dynamic_result.owner_ranks
                self._last_dynamic_consumed_prompt_indices = list(
                    dynamic_result.consumed_prompt_indices
                )
                self._last_dynamic_scheduler_metrics = dict(dynamic_result.metrics)
                req = req.select(
                    torch.tensor(
                        dynamic_result.selected_prompt_indices, dtype=torch.long
                    )
                )
                rewards_precomputed = True
            else:
                resp = self.run_rollout(req)
        else:
            # Colocate memory dance (150GB base can't coexist with an awake engine
            # on the same card). Steady state on entry: base offloaded, engines
            # asleep. EXTRACT (base onloaded) -> wake engines -> PUSH adapter ->
            # rollout (base offloaded) -> sleep engines -> onload base for backward.
            if sync_weights and self.weight_sync is not None:
                if self._enable_fsdp_offload:
                    self.backend.onload()
                self.weight_sync.extract()
                if self._enable_fsdp_offload:
                    self.backend.offload()
            for _eng in self.ar_rollouts + self.dit_rollouts:
                _eng.wake_up()
            if sync_weights and self.weight_sync is not None:
                self.weight_sync.push()
            resp = self.run_rollout(req)
            for _eng in self.ar_rollouts + self.dit_rollouts:
                _eng.sleep()
            if self._enable_fsdp_offload:
                self.backend.onload()
        rollout_elapsed = time.perf_counter() - t0
        prepare_started = time.perf_counter()

        # 1. Score and compute advantages.
        #    Branch on whether the image algorithm needs advantages:
        #    - HI3/BAGEL path: score IMAGE track → propagate → advantages for both
        #    - Text-only reward + MSE path: score AR track → advantages for AR only
        image_algo_needs_advantages = getattr(
            self, "_image_algo_needs_advantages", False
        )

        if image_algo_needs_advantages:
            # ── HI3 path: score image track, propagate to AR ──
            img_track = resp.tracks[IMAGE_TRACK]
            ar_params = req.sampling_params.get("ar")
            diff_params = req.sampling_params.get("diffusion")
            n_rec = int(ar_params.samples_per_prompt) if ar_params is not None else 1
            n_img = int(diff_params.samples_per_prompt)
            orig_texts = req.primitives.get("text")
            reward_texts = Texts(texts=[orig_texts.texts[i // (n_rec * n_img)] for i in range(len(img_track.sample_ids))])
            reward_metadata = (
                [req.metadata[i // (n_rec * n_img)] for i in range(len(img_track.sample_ids))] if req.metadata else []
            )
            reward_req = RolloutReq(
                sample_ids=list(img_track.sample_ids),
                group_ids=list(img_track.parent_ids) if img_track.parent_ids else list(img_track.sample_ids),
                primitives={"text": reward_texts},
                request_conditions={},
                sampling_params=req.sampling_params,
                metadata=reward_metadata,
            )
            scored = self.reward.score_and_attach(req=reward_req, track=img_track)
            if scored.rewards is not None:
                scored.rewards = hydrate(scored.rewards)
            resp.tracks[IMAGE_TRACK] = scored

            # 2. Credit-assign image reward up the lineage → fills the "ar" track.
            resp = resp.propagate_rewards(op="mean")

            # 3. Mean image reward for the log line.
            mean_reward = 0.0
            di_rewards = resp.tracks[IMAGE_TRACK].rewards
            if di_rewards is not None:
                mean_reward = float(hydrate(di_rewards).to(torch.float32).mean().item())

            # 3b. Intrusive debug dump (best-effort).
            if self.dump_dir:
                self._dump_rollout(self._dump_rollout_id, req, resp)

            # 4. GRPO advantages for both tracks.
            resp.tracks[AR_TRACK] = resp.tracks[AR_TRACK].compute_advantages(normalize=True)
            if self._shared_advantage:
                resp.tracks[IMAGE_TRACK] = _track_with_field(
                    resp.tracks[IMAGE_TRACK], "advantages", resp.tracks[AR_TRACK].advantages
                )
            else:
                resp.tracks[IMAGE_TRACK] = resp.tracks[IMAGE_TRACK].compute_advantages(normalize=True)

            self._drop_decoded(
                req,
                resp,
                rollout_id=rollout_id,
                media_prompts={IMAGE_TRACK: list(reward_texts.texts)},
            )
        else:
            # ── Text-only reward path (GeoWeave interleave): score AR track ──
            ar_track = resp.tracks[AR_TRACK]
            ar_params = req.sampling_params.get("ar")
            n_rec = int(ar_params.samples_per_prompt) if ar_params is not None else 1
            orig_texts = req.primitives.get("text")
            reward_texts = Texts(
                texts=[orig_texts.texts[i // n_rec] for i in range(len(ar_track.sample_ids))]
            )
            reward_metadata = (
                [req.metadata[i // n_rec] for i in range(len(ar_track.sample_ids))]
                if req.metadata
                else []
            )
            reward_primitives = {"text": reward_texts}
            for primitive_name, primitive in req.primitives.items():
                if primitive_name == "text" or primitive is None:
                    continue
                reward_primitives[primitive_name] = primitive.repeat_interleave(n_rec)
            reward_req = RolloutReq(
                sample_ids=list(ar_track.sample_ids),
                group_ids=(
                    list(ar_track.parent_ids)
                    if ar_track.parent_ids
                    else list(ar_track.sample_ids)
                ),
                primitives=reward_primitives,
                request_conditions={},
                sampling_params=req.sampling_params,
                metadata=reward_metadata,
            )
            if rewards_precomputed:
                scored = ar_track
            else:
                scored = self.reward.score_and_attach(req=reward_req, track=ar_track)
            if scored.rewards is not None:
                scored.rewards = hydrate(scored.rewards)
            resp.tracks[AR_TRACK] = scored

            # Mean AR reward for the log line.
            mean_reward = 0.0
            ar_rewards = resp.tracks[AR_TRACK].rewards
            if ar_rewards is not None:
                mean_reward = float(hydrate(ar_rewards).to(torch.float32).mean().item())

            # SCA carries sample-aligned process judgments and computes one
            # normalized advantage per generated token across each prompt group.
            # Scalar-only backends keep the historical sequence-reward GRPO path.
            if resp.tracks[AR_TRACK].process_annotations is not None:
                from unirl.reward.sca import compute_sca_token_advantages

                ar_with_advantages = compute_sca_token_advantages(
                    resp.tracks[AR_TRACK],
                    outcome_correct_credit=float(
                        self._stage_config.get("sca_outcome_correct_credit", 2.0)
                    ),
                    process_error_penalty=float(
                        self._stage_config.get("sca_process_error_penalty", 1.0)
                    ),
                    eps=float(self._stage_config.get("sca_advantage_eps", 1e-6)),
                )
            else:
                # In exclude mode, status=0 rows do not enter group mean/std and
                # receive zero advantage: 8 candidates with 1 truncation => 7-way GRPO.
                advantage_mask = (
                    resp.tracks[AR_TRACK].status
                    if getattr(self, "_exclude_truncated_from_advantage_stats", False)
                    else None
                )
                ar_with_advantages = resp.tracks[AR_TRACK].compute_advantages(
                    normalize=True, valid_mask=advantage_mask
                )
            if ar_with_advantages.status is not None:
                train_mask = hydrate(ar_with_advantages.status).to(
                    dtype=torch.float32, device=ar_with_advantages.advantages.device
                )
                ar_with_advantages = _track_with_field(
                    ar_with_advantages,
                    "advantages",
                    ar_with_advantages.advantages * train_mask,
                )
            resp.tracks[AR_TRACK] = ar_with_advantages

            # Dump after advantage computation so samples.jsonl captures the
            # exact status/advantage pair used by the optimizer.  This makes
            # truncated-sample exclusion auditable per prompt group.
            if self.dump_dir:
                self._dump_rollout(self._dump_rollout_id, req, resp)

            self._drop_decoded(req, resp, rollout_id=rollout_id)
        if dynamic_owner_ranks is not None:
            from unirl.trainer.dynamic_trainside import (
                post_advantage_owner_affine_train_permutation,
            )

            ar_track = resp.tracks[AR_TRACK]
            if ar_track.advantages is None:
                raise RuntimeError(
                    "Owner-affine train routing is only valid after GRPO advantages are computed."
                )
            permutation = post_advantage_owner_affine_train_permutation(
                dynamic_owner_ranks,
                self.rollout.dp_size,
            )
            shard_size = len(permutation) // self.rollout.dp_size
            local_samples = sum(
                dynamic_owner_ranks[sample_index] == train_rank
                for train_rank in range(self.rollout.dp_size)
                for sample_index in permutation[
                    train_rank * shard_size : (train_rank + 1) * shard_size
                ]
            )
            logger.debug(
                "[DEBUG][TRAIN][ROUTING] local_samples=%d/%d dp_size=%d",
                local_samples,
                len(permutation),
                self.rollout.dp_size,
            )
            resp.tracks[AR_TRACK] = ar_track.select(permutation)
        # 5. Two backward on the shared backbone; step now or accumulate.
        image_track_for_train = resp.tracks.get(IMAGE_TRACK)
        image_track_for_train = self._align_image_track_for_train(image_track_for_train)
        prepare_elapsed = time.perf_counter() - prepare_started
        train_started = time.perf_counter()
        results: Dict[str, TrainStepResult] = self.stack.train_track(
            resp.tracks[AR_TRACK],
            image_track_for_train,
            training_progress=float(training_progress),
            force_optimizer_step=bool(force_optimizer_step),
        )
        train_elapsed = time.perf_counter() - train_started
        step_elapsed = time.perf_counter() - t0
        result_metrics = (
            dict(next(iter(results.values())).metrics) if results else {}
        )
        accumulation_count = int(
            result_metrics.get("gradient_accumulation_count", 1)
        )
        accumulation_steps = int(
            result_metrics.get("gradient_accumulation_steps", 1)
        )
        optimizer_stepped = bool(result_metrics.get("optimizer_stepped", 1.0))
        self._last_progress_extra = {
            "rollout_s": f"{rollout_elapsed:.1f}",
            "prepare_s": f"{prepare_elapsed:.1f}",
            "train_s": f"{train_elapsed:.1f}",
            "step_s": f"{step_elapsed:.1f}",
            "grad_accum": f"{accumulation_count}/{accumulation_steps}",
            "optimizer_stepped": int(optimizer_stepped),
        }
        metric_logger = getattr(
            self, "metric_logger", getattr(self, "wandb_logger", None)
        )
        if metric_logger is not None:
            metric_logger.log_rollout_step(
                rollout_id,
                results,
                resp,
                step_time_s=step_elapsed,
                extra_metrics={
                    "sync_weights": float(bool(sync_weights)),
                    **{
                        f"scheduler/{key}": value
                        for key, value in getattr(
                            self, "_last_dynamic_scheduler_metrics", {}
                        ).items()
                    },
                },
            )

        # 6. Back to steady state (base on CPU) so the next rollout's engines
        #    have room to wake.
        if self._enable_fsdp_offload:
            self.backend.offload()
        return results, mean_reward

    def _align_image_track_for_train(self, image_track: Optional[RolloutTrack]) -> Optional[RolloutTrack]:
        """Truncate the variable-length image track to a divisor-safe size.

        The train stack requires per-worker ``batch_size`` to be a positive
        multiple of ``num_updates_per_batch`` (see
        :func:`unirl.train.stack.planner.types._update_ranges`), and DP_SCATTER
        shards uniformly over ``train_dp_size``. So the global image track must
        be a multiple of ``train_dp_size * num_updates_per_batch`` before
        dispatch. When the remainder truncates to zero we drop the image track
        entirely — returning ``None`` makes every worker uniformly skip image
        backward, which keeps FSDP grad sync in step (a per-worker skip would
        desync). Any trimmed image samples are simply not trained this rollout.
        """
        if image_track is None:
            return None
        alignment = self._image_track_alignment
        bs = int(image_track.batch_size)
        if bs == 0:
            return None
        if alignment <= 1:
            return image_track
        keep_n = (bs // alignment) * alignment
        if keep_n == 0:
            logger.warning(
                "UnifiedModelTrainer: dropping image_track (bs=%d < alignment=%d = train_dp %d × num_updates %d).",
                bs, alignment, self._train_dp_size, alignment // self._train_dp_size,
            )
            return None
        if keep_n < bs:
            logger.warning(
                "UnifiedModelTrainer: truncating image_track %d → %d to satisfy alignment=%d.",
                bs, keep_n, alignment,
            )
            return image_track.slice(0, keep_n)
        return image_track

    @staticmethod
    def _image_to_cpu_uint8(image: torch.Tensor) -> torch.Tensor:
        image = image.detach()
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"Expected CHW generated image, got {tuple(image.shape)}")
        return (
            image.mul(0.5)
            .add(0.5)
            .mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .to(torch.uint8)
            .cpu()
        )

    def _wait_dump_future(self, future: Future) -> None:
        try:
            future.result()
        except Exception as exc:  # noqa: BLE001 - dump is best-effort
            logger.warning("[DUMP] asynchronous dump failed (non-fatal): %s", exc)

    def _wait_for_dump_capacity(self) -> None:
        if not hasattr(self, "_dump_futures"):
            self._dump_futures = deque()
        max_pending = int(getattr(self, "_dump_max_pending", 1))
        while len(self._dump_futures) >= max_pending:
            self._wait_dump_future(self._dump_futures.popleft())

    def _submit_interleave_dump(self, snapshot: _InterleaveDumpSnapshot) -> None:
        self._wait_for_dump_capacity()
        if not bool(getattr(self, "_dump_async", False)):
            _write_interleave_dump_snapshot(
                snapshot,
                image_workers=int(getattr(self, "_dump_image_workers", 1)),
                jpeg_quality=int(getattr(self, "_dump_jpeg_quality", 90)),
            )
            return
        if self._dump_executor is None:
            raise RuntimeError("Async dump executor is not initialized")
        self._dump_futures.append(
            self._dump_executor.submit(
                _write_interleave_dump_snapshot,
                snapshot,
                image_workers=int(getattr(self, "_dump_image_workers", 1)),
                jpeg_quality=int(getattr(self, "_dump_jpeg_quality", 90)),
            )
        )

    def _flush_dump_tasks(self) -> None:
        while self._dump_futures:
            self._wait_dump_future(self._dump_futures.popleft())
        if self._dump_executor is not None:
            self._dump_executor.shutdown(wait=True, cancel_futures=False)
            self._dump_executor = None

    def _snapshot_interleave_dump(
        self,
        rollout_id: int,
        interleave_cond: Any,
        prompts: list[str],
        prompt_ids: list[str],
        source_ids: list[Any],
        ar_texts: list[str],
        ar_rewards: Optional[list[float]],
        ar_advantages: Optional[list[float]],
        ar_status: Optional[list[float]],
        ar_component_rewards: dict[str, list[float]],
        ar_process_annotations: Optional[list[Optional[dict[str, Any]]]],
        ar_token_signals: dict[str, torch.Tensor],
        ar_tokens: Optional[torch.Tensor],
        ar_log_probs: Optional[torch.Tensor],
        ar_cu_seqlens: Optional[list[int]],
        n_rec: int,
        *,
        ar_sample_ids: Optional[list[str]] = None,
        metadata: Optional[list[Any]] = None,
    ) -> _InterleaveDumpSnapshot:
        snapshot_started = time.perf_counter()
        records: list[dict[str, Any]] = []
        images: list[_DumpImage] = []
        histogram: Dict[int, int] = {}
        boundary_mismatches = 0
        save_images = bool(getattr(self, "save_rollout_images", True))
        save_results = bool(getattr(self, "save_rollout_results", True))
        # Preserve the current JPEG dump path by default.  The explicit legacy
        # save_rollout_* controls historically promised PNG filenames.
        image_suffix = (
            ".png"
            if hasattr(self, "save_rollout_images")
            or hasattr(self, "save_rollout_results")
            else ".jpg"
        )
        ar_sample_ids = ar_sample_ids or []
        metadata = metadata or []

        for k, response in enumerate(ar_texts):
            p_idx = k // n_rec
            gen_imgs = (
                interleave_cond.generated_images[k]
                if k < len(interleave_cond.generated_images)
                else []
            )
            boundaries = (
                interleave_cond.text_segment_boundaries[k]
                if k < len(interleave_cond.text_segment_boundaries)
                else []
            )
            n_aux = len(gen_imgs)
            histogram[n_aux] = histogram.get(n_aux, 0) + 1
            expected_boundaries = n_aux + 1 if n_aux > 0 else 1
            if len(boundaries) != expected_boundaries:
                boundary_mismatches += 1

            aux_files: list[Optional[str]] = []
            aux_shapes: list[list[int]] = []
            for j, image_ref in enumerate(gen_imgs):
                filename = f"sample_{k}_aux_{j}{image_suffix}"
                try:
                    shape = list(image_ref.shape)
                    image = hydrate(image_ref)
                    if image is None:
                        raise ValueError("generated image TensorRef has no spans")
                    if save_images:
                        images.append(
                            _DumpImage(
                                filename=filename,
                                pixels=self._image_to_cpu_uint8(image),
                            )
                        )
                        aux_files.append(filename)
                    aux_shapes.append(shape)
                except Exception as exc:  # noqa: BLE001 - per-image best-effort
                    logger.warning(
                        "[DUMP] rollout=%d sample=%d image=%d snapshot failed: %s",
                        rollout_id,
                        k,
                        j,
                        exc,
                    )
                    aux_files.append(None)
                    aux_shapes.append(list(getattr(image_ref, "shape", ())))

            token_ids = None
            rollout_log_probs = None
            token_signals: dict[str, list[float]] = {}
            if ar_cu_seqlens is not None and k + 1 < len(ar_cu_seqlens):
                tok_start = int(ar_cu_seqlens[k])
                tok_end = int(ar_cu_seqlens[k + 1])
                if ar_tokens is not None:
                    token_ids = ar_tokens[tok_start:tok_end].tolist()
                if ar_log_probs is not None:
                    rollout_log_probs = ar_log_probs[tok_start:tok_end].tolist()
                token_signals = {
                    name: values[tok_start:tok_end].tolist()
                    for name, values in ar_token_signals.items()
                }
            generated_token_count = len(token_ids) if token_ids is not None else 0
            stop_reasons = getattr(interleave_cond, "stop_reasons", [])
            stop_reason = stop_reasons[k] if k < len(stop_reasons) else "unknown"
            image_context_token_counts = getattr(
                interleave_cond, "image_context_token_counts", []
            )
            image_context_token_count = (
                int(image_context_token_counts[k])
                if k < len(image_context_token_counts)
                else 0
            )
            sample_components = {
                name: values[k] if k < len(values) else None
                for name, values in ar_component_rewards.items()
            }
            records.append(
                {
                    "sample_idx": k,
                    "sample_id": ar_sample_ids[k] if k < len(ar_sample_ids) else None,
                    "source_id": (
                        source_ids[p_idx] if p_idx < len(source_ids) else None
                    ),
                    "prompt_id": (
                        prompt_ids[p_idx] if p_idx < len(prompt_ids) else None
                    ),
                    "prompt": prompts[p_idx] if p_idx < len(prompts) else None,
                    "gold_answer": (
                        metadata[p_idx].get("answer")
                        if p_idx < len(metadata) and isinstance(metadata[p_idx], dict)
                        else None
                    ),
                    "response": response,
                    "generated_text": response,
                    "response_length": generated_token_count,
                    "response_char_count": len(response),
                    "generated_token_count": generated_token_count,
                    "token_ids": token_ids,
                    "rollout_log_probs": rollout_log_probs,
                    "sca_token_signals": token_signals or None,
                    "n_aux_images": n_aux,
                    "generated_image_count": n_aux,
                    "stop_reason": stop_reason,
                    "hit_max_new_tokens": stop_reason == "max_new_tokens",
                    "truncated": stop_reason == "max_new_tokens",
                    "hit_max_images": stop_reason == "max_images",
                    "kv_image_token_count": image_context_token_count,
                    "total_context_advance": (
                        generated_token_count + image_context_token_count
                    ),
                    "aux_image_shapes": aux_shapes,
                    "generated_image_shapes": aux_shapes,
                    "text_boundaries": boundaries,
                    "text_segment_boundaries": boundaries,
                    "boundary_count": len(boundaries),
                    "reward": (
                        ar_rewards[k]
                        if ar_rewards is not None and k < len(ar_rewards)
                        else None
                    ),
                    "advantage": (
                        ar_advantages[k]
                        if ar_advantages is not None and k < len(ar_advantages)
                        else None
                    ),
                    "status": (
                        ar_status[k]
                        if ar_status is not None and k < len(ar_status)
                        else None
                    ),
                    "reward_components": sample_components,
                    "process_annotation": (
                        ar_process_annotations[k]
                        if ar_process_annotations is not None
                        and k < len(ar_process_annotations)
                        else None
                    ),
                    "judge_failed": sample_components.get(
                        "judge_failed", sample_components.get("sca_judge_failed")
                    ),
                    "aux_files": aux_files,
                    "generated_image_files": [
                        filename for filename in aux_files if filename is not None
                    ],
                }
            )

        snapshot = _InterleaveDumpSnapshot(
            rollout_id=rollout_id,
            out_dir=os.path.join(self.dump_dir, f"rollout_{rollout_id}"),
            records=records,
            images=images,
            histogram=histogram,
            boundary_mismatches=boundary_mismatches,
            reward_nonzero=sum(1 for reward in (ar_rewards or []) if reward != 0),
            save_results=save_results,
        )
        logger.debug(
            "[DEBUG][DUMP] rollout=%d snapshot_samples=%d snapshot_images=%d "
            "snapshot_s=%.2f async=%s",
            rollout_id,
            len(records),
            len(images),
            time.perf_counter() - snapshot_started,
            bool(getattr(self, "_dump_async", False)),
        )
        return snapshot

    def _dump_rollout(self, rollout_id: int, req: RolloutReq, resp: Any) -> None:
        """Snapshot a rollout and write debug data without blocking GPU train."""
        try:
            prompts_obj = req.primitives.get("text")
            prompts = list(prompts_obj.texts) if prompts_obj is not None else []
            prompt_ids = list(req.group_ids)
            source_ids = [
                metadata.get("source_id") if isinstance(metadata, dict) else None
                for metadata in req.metadata
            ]
            ar_track = resp.tracks.get(AR_TRACK)
            ar_decoded = (
                getattr(ar_track, "decoded", None) if ar_track is not None else None
            )
            ar_texts = list(ar_decoded.texts) if ar_decoded is not None else []
            ar_rewards_raw = ar_track.rewards if ar_track is not None else None
            ar_rewards = (
                hydrate(ar_rewards_raw).to(torch.float32).cpu().tolist()
                if ar_rewards_raw is not None
                else None
            )
            ar_advantages_raw = ar_track.advantages if ar_track is not None else None
            ar_advantages = (
                hydrate(ar_advantages_raw).to(torch.float32).cpu().tolist()
                if ar_advantages_raw is not None
                else None
            )
            ar_status_raw = ar_track.status if ar_track is not None else None
            ar_status = (
                hydrate(ar_status_raw).to(torch.float32).cpu().tolist()
                if ar_status_raw is not None
                else None
            )
            ar_component_rewards = {}
            if ar_track is not None and ar_track.component_rewards:
                ar_component_rewards = {
                    str(name): hydrate(values).to(torch.float32).cpu().tolist()
                    for name, values in ar_track.component_rewards.items()
                }
            ar_process_annotations = (
                list(ar_track.process_annotations)
                if ar_track is not None and ar_track.process_annotations is not None
                else None
            )
            ar_segment = ar_track.segment if ar_track is not None else None
            ar_token_signals: dict[str, torch.Tensor] = {}
            if ar_segment is not None:
                for signal_name in (
                    "process_error_mask",
                    "raw_token_credit",
                    "sca_token_mask",
                    "token_advantages",
                ):
                    values = getattr(ar_segment, signal_name, None)
                    if values is not None:
                        ar_token_signals[signal_name] = (
                            hydrate(values).to(torch.float32).cpu()
                        )
            ar_tokens = (
                hydrate(ar_segment.tokens).cpu()
                if ar_segment is not None and ar_segment.tokens is not None
                else None
            )
            ar_log_probs = (
                hydrate(ar_segment.log_probs).to(torch.float32).cpu()
                if ar_segment is not None and ar_segment.log_probs is not None
                else None
            )
            ar_cu_seqlens = (
                hydrate(ar_segment.cu_seqlens).cpu().tolist()
                if ar_segment is not None and ar_segment.cu_seqlens is not None
                else None
            )
            ar_cond = ar_track.conditions if ar_track is not None else None
            interleave_cond = (
                ar_cond.get("sensenova_u1_ar") if isinstance(ar_cond, dict) else None
            )
            ar_params = self.sampling_params.get("ar")
            n_rec = int(ar_params.samples_per_prompt) if ar_params is not None else 1

            if interleave_cond is not None and hasattr(
                interleave_cond, "generated_images"
            ):
                # Bound CPU snapshot memory as well as queued writer tasks.
                self._wait_for_dump_capacity()
                snapshot = self._snapshot_interleave_dump(
                    rollout_id,
                    interleave_cond,
                    prompts,
                    prompt_ids,
                    source_ids,
                    ar_texts,
                    ar_rewards,
                    ar_advantages,
                    ar_status,
                    ar_component_rewards,
                    ar_process_annotations,
                    ar_token_signals,
                    ar_tokens,
                    ar_log_probs,
                    ar_cu_seqlens,
                    n_rec,
                    ar_sample_ids=list(ar_track.sample_ids),
                    metadata=list(req.metadata),
                )
                self._submit_interleave_dump(snapshot)
                return

            if ar_track is not None and ar_texts:
                # AR-only rollouts have no interleave condition object, but use
                # the same schema/writer so SCA diagnostics and legacy dump
                # consumers remain aligned.
                token_counts = []
                if ar_cu_seqlens is not None:
                    token_counts = [
                        int(ar_cu_seqlens[i + 1]) - int(ar_cu_seqlens[i])
                        for i in range(len(ar_cu_seqlens) - 1)
                    ]
                max_new_tokens = int(
                    getattr(req.sampling_params.get("ar"), "max_new_tokens", 0)
                    or 0
                )
                text_only_cond = type("_TextOnlyDumpCondition", (), {})()
                text_only_cond.generated_images = [[] for _ in ar_texts]
                text_only_cond.text_segment_boundaries = [
                    [(0, count)] for count in token_counts
                ]
                text_only_cond.stop_reasons = [
                    "max_new_tokens"
                    if max_new_tokens > 0 and count >= max_new_tokens
                    else "unknown"
                    for count in token_counts
                ]
                text_only_cond.image_context_token_counts = [0 for _ in ar_texts]
                self._wait_for_dump_capacity()
                snapshot = self._snapshot_interleave_dump(
                    rollout_id,
                    text_only_cond,
                    prompts,
                    prompt_ids,
                    source_ids,
                    ar_texts,
                    ar_rewards,
                    ar_advantages,
                    ar_status,
                    ar_component_rewards,
                    ar_process_annotations,
                    ar_token_signals,
                    ar_tokens,
                    ar_log_probs,
                    ar_cu_seqlens,
                    n_rec,
                    ar_sample_ids=list(ar_track.sample_ids),
                    metadata=list(req.metadata),
                )
                self._submit_interleave_dump(snapshot)
                return

            # Keep the legacy HI3 path synchronous, but store previews as JPEG.
            out_dir = os.path.join(self.dump_dir, f"rollout_{rollout_id}")
            os.makedirs(out_dir, exist_ok=True)
            image_track = resp.tracks.get(IMAGE_TRACK)
            img_decoded = (
                getattr(image_track, "decoded", None)
                if image_track is not None
                else None
            )
            sample_ids = list(image_track.sample_ids) if image_track is not None else []
            parent_ids = (
                list(image_track.parent_ids)
                if image_track is not None and image_track.parent_ids
                else []
            )
            rewards = None
            if image_track is not None and image_track.rewards is not None:
                rewards = hydrate(image_track.rewards).to(torch.float32).cpu().tolist()

            n_imgs = 0
            if img_decoded is not None and getattr(img_decoded, "pixels", None) is not None:
                pixels = hydrate(img_decoded.pixels).detach().to(torch.float32).clamp(0, 1)
                n_imgs = int(pixels.shape[0])
                dump_images = [
                    _DumpImage(
                        filename=f"img_{k}.jpg",
                        pixels=pixels[k].mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8).cpu(),
                    )
                    for k in range(n_imgs)
                ]
                with ThreadPoolExecutor(
                    max_workers=int(getattr(self, "_dump_image_workers", 1))
                ) as executor:
                    list(
                        executor.map(
                            lambda image: _save_dump_image(
                                image,
                                out_dir,
                                int(getattr(self, "_dump_jpeg_quality", 90)),
                            ),
                            dump_images,
                        )
                    )

            diff_params = self.sampling_params.get("diffusion")
            n_img = (
                max(1, int(diff_params.samples_per_prompt))
                if diff_params is not None
                else 1
            )
            n = max(len(sample_ids), n_imgs)
            with open(os.path.join(out_dir, "samples.jsonl"), "w") as f:
                for k in range(n):
                    p_idx = k // (n_rec * n_img)
                    a_idx = k // n_img
                    f.write(
                        json.dumps(
                            {
                                "sample_id": (
                                    sample_ids[k] if k < len(sample_ids) else None
                                ),
                                "parent_id": (
                                    parent_ids[k] if k < len(parent_ids) else None
                                ),
                                "source_id": (
                                    source_ids[p_idx]
                                    if p_idx < len(source_ids)
                                    else None
                                ),
                                "prompt_id": (
                                    prompt_ids[p_idx]
                                    if p_idx < len(prompt_ids)
                                    else None
                                ),
                                "prompt": (
                                    prompts[p_idx] if p_idx < len(prompts) else None
                                ),
                                "ar_text_fed_to_dit": (
                                    ar_texts[a_idx] if a_idx < len(ar_texts) else None
                                ),
                                "image_reward": (
                                    rewards[k]
                                    if rewards is not None and k < len(rewards)
                                    else None
                                ),
                                "image_file": (
                                    f"img_{k}.jpg" if k < n_imgs else None
                                ),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            logger.info(
                "[RUN][DUMP] rollout=%d samples=%d images=%d path=%s",
                rollout_id,
                n,
                n_imgs,
                out_dir,
            )
        except Exception as exc:  # noqa: BLE001 - dump must not break training
            logger.warning(
                "[DUMP] rollout %d snapshot/dump failed (non-fatal): %s",
                rollout_id,
                exc,
            )

    def _prime_ref_snapshots(self) -> None:
        """Eagerly snapshot pretrained base weights for velocity-MSE reference.

        Must run AFTER bundle load (so trainable params exist and hold the
        pretrained base) and BEFORE ``maybe_load_checkpoint`` (so resume does
        not overwrite the snapshot target). Called from :meth:`train` for
        exactly this reason. Algorithms without ``prime_reference_snapshot``
        are silently skipped.
        """
        for algo in (self.ar_algorithm, self.image_algorithm):
            if algo is not None and hasattr(algo, "prime_reference_snapshot"):
                algo.prime_reference_snapshot()

    @staticmethod
    def _optimizer_stepped(results: Dict[str, TrainStepResult]) -> bool:
        for result in results.values():
            metrics = dict(result.metrics)
            if "optimizer_stepped" in metrics:
                return bool(metrics["optimizer_stepped"])
            if result.has_backward:
                return True
        return False

    @staticmethod
    def _gradient_accumulation_window_closed(
        results: Dict[str, TrainStepResult],
    ) -> bool:
        if not results:
            return True
        first = next(iter(results.values()))
        metrics = dict(first.metrics)
        if "gradient_accumulation_window_closed" not in metrics:
            return True
        return bool(metrics["gradient_accumulation_window_closed"])

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        """Run exactly ``num_rollouts`` rollout/reward/backward iterations.

        ``num_rollouts`` remains the total rollout budget even when
        ``stack.gradient_accumulation_steps > 1``; it does not become an optimizer
        step count. Accumulation only reduces how often the optimizer/scheduler/EMA
        advance, and the final short accumulation window is stepped on the last
        rollout.

        ``save_interval``: write a checkpoint every N rollouts (and on the last
        one); ``0`` disables it. ``save_dir`` defaults to ``./checkpoints``;
        ``save_mode="auto"`` writes LoRA-only checkpoints when LoRA is active
        and full checkpoints otherwise. ``load_dir``: restore from a checkpoint
        directory and RESUME from its saved step — ``num_rollouts`` is the TOTAL
        budget.
        """
        interval = max(1, weight_sync_interval)
        # Pin velocity-MSE reference weights to the pretrained base BEFORE any
        # checkpoint restore. Without this, resume-from-checkpoint captures the
        # resumed weights on the first _reference_weights() call → MSE ≡ 0 and
        # the regularizer degenerates silently.
        self._prime_ref_snapshots()
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        # Fast-forward the data stream to the resume point — exact when
        # run.seed is set (deterministic shuffle); with seed=null the stream
        # is non-reproducible anyway.
        prompt_pool_size = self.batch_size + (
            self._reserve_prompt_count if self._dynamic_trajectory_scheduling else 0
        )
        for _ in range(start_rollout):
            # Dynamic refill consumes a variable number of reserve prompts and
            # old checkpoints do not serialize that driver buffer. Preserve the
            # historical one-primary-batch fast-forward contract here.
            self.data_source.get_samples(self.batch_size)
        self._init_logging(num_rollouts=num_rollouts)
        checkpoint_pending = False
        weights_dirty = False
        try:
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                self._dump_rollout_id = rollout_id  # picked up by train_step's dump
                inputs = self.data_source.get_samples(prompt_pool_size)
                self._last_dynamic_consumed_prompt_indices = []
                self._last_dynamic_scheduler_metrics = {}
                req = self._build_req(inputs, rollout_id)
                # Sync before generate; skip step 0 (nothing trained yet). On
                # resume, force the first sync — the engine booted with fresh
                # weights and needs the restored adapter before generate. The
                # HI3_SYNC_FIRST env forces a sync on rollout 0 too — a debug knob
                # to exercise the LoRA-sync path early (cheaply) without a full
                # extra rollout; the rollout-0 adapter is ~0 but that's fine for
                # testing the register→activate mechanism.
                force_sync = (resumed and rollout_id == start_rollout) or (
                    rollout_id == 0 and bool(os.environ.get("HI3_SYNC_FIRST"))
                )
                sync_due = rollout_id > 0 and rollout_id % interval == 0
                sync_weights = force_sync or (weights_dirty and sync_due)
                if sync_weights:
                    weights_dirty = False
                results, mean_reward = self.train_step(
                    req,
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                    force_optimizer_step=(rollout_id + 1 == num_rollouts),
                )
                if self._dynamic_trajectory_scheduling and inputs.batch_size > self.batch_size:
                    consumed = set(self._last_dynamic_consumed_prompt_indices)
                    unused = [i for i in range(inputs.batch_size) if i not in consumed]
                    if unused:
                        self.data_source.return_samples(
                            inputs.select(torch.tensor(unused, dtype=torch.long))
                        )
                # Per-track console line (ar / image) with the step-0 ratio probe
                # (π_old vs π_θ alignment): on rollout 0 the LoRA is ~0 so a correct
                # replay should give ratio≈1, std≈0; a systematic offset means the
                # logp convention (temperature / top-k-p filtering / full-vs-renorm
                # softmax) doesn't match vLLM's sampler.
                self.metric_logger.log_progress(
                    rollout_id,
                    num_rollouts,
                    results,
                    mean_reward,
                    extra=getattr(self, "_last_progress_extra", None),
                    logger=logger,
                )
                optimizer_stepped = self._optimizer_stepped(results)
                if optimizer_stepped:
                    weights_dirty = True

                step = rollout_id + 1
                checkpoint_due = save_interval > 0 and (
                    step % save_interval == 0 or step >= num_rollouts
                )
                checkpoint_pending = checkpoint_pending or checkpoint_due
                window_closed = self._gradient_accumulation_window_closed(results)
                if checkpoint_pending and window_closed:
                    # A checkpoint does not serialize live gradients. Defer an
                    # interval that lands inside an accumulation window to its
                    # next safe optimizer boundary.
                    self.maybe_save_checkpoint(
                        rollout_id,
                        num_rollouts,
                        save_interval=1,
                        save_dir=save_dir,
                        save_mode=save_mode,
                    )
                    checkpoint_pending = False
                elif checkpoint_due and not window_closed:
                    logger.info(
                        "[RUN][CHECKPOINT] rollout=%d status=deferred "
                        "reason=gradient_accumulation",
                        step,
                    )
        finally:
            self._flush_dump_tasks()
            self._finish_logging()


__all__ = ["UnifiedModelTrainer"]
