"""TensorBoard metrics and media logger for UniRL training.

The public methods intentionally mirror :class:`UniRLWandBLogger` so trainers
can select either backend without backend-specific logging code.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .wandb_logger import UniRLWandBLogger

module_logger = logging.getLogger(__name__)


class UniRLTensorBoardLogger(UniRLWandBLogger):
    """TensorBoard backend with the same training-facing API as W&B."""

    def __init__(
        self,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Any] = None,
        log_dir: Optional[str] = None,
        rank: int = 0,
        media_log_interval: int = 1,
        media_max_items: int = 8,
        log_media: bool = False,
        enabled: bool = True,
        tags: Optional[List[str]] = None,
        entity: Optional[str] = None,
        run_id: Optional[str] = None,
        optimizer_step: int = 0,
    ) -> None:
        # Initialize the shared metric aggregation/progress state without
        # starting a W&B run. Backend-specific writing is implemented below.
        super().__init__(
            project=project,
            run_name=run_name,
            config=None,
            log_dir=log_dir,
            rank=rank,
            media_log_interval=media_log_interval,
            media_max_items=media_max_items,
            log_media=log_media,
            enabled=False,
            tags=tags,
            entity=entity,
            run_id=run_id,
            optimizer_step=optimizer_step,
        )
        self.enabled = bool(enabled and rank == 0)
        self.writer = None
        if self.enabled:
            self._init_tensorboard(config=config, resume_log_dir=run_id)

    def _init_tensorboard(
        self,
        *,
        config: Optional[Any],
        resume_log_dir: Optional[str],
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "TensorBoard reporting was requested but tensorboard is not installed"
            ) from exc

        if resume_log_dir:
            run_dir = Path(resume_log_dir)
        elif self.log_dir:
            # An explicit logging_dir is already scoped to the experiment and
            # is therefore the final TensorBoard event directory.
            run_dir = Path(self.log_dir)
        else:
            base_dir = Path("runs")
            if self.project:
                base_dir /= str(self.project)
            run_component = self.run_name or time.strftime("run-%Y%m%d-%H%M%S")
            run_dir = base_dir / run_component
        run_dir = run_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.writer = SummaryWriter(log_dir=str(run_dir))
            self.log_dir = str(run_dir)
            # ``run_id`` is backend-neutral checkpoint state in BaseTrainer. For
            # TensorBoard it is the exact event directory to append to on resume.
            self.run_id = str(run_dir)
            self._initialized = True
            self._log_run_metadata(config)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize TensorBoard: {exc}") from exc

    def _log_run_metadata(self, config: Optional[Any]) -> None:
        if self.writer is None:
            return
        if isinstance(config, dict):
            config_dict = config
        elif config is not None and hasattr(config, "__dict__"):
            config_dict = vars(config)
        else:
            config_dict = None
        metadata = {
            "project": self.project,
            "run_name": self.run_name,
            "tags": self.tags,
            "config": config_dict,
        }
        self.writer.add_text(
            "run/metadata",
            "```json\n" + json.dumps(metadata, indent=2, default=str) + "\n```",
            global_step=0,
        )

    def log_with_step(
        self,
        *,
        step_key: str,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "",
    ) -> None:
        """Write scalar metrics using TensorBoard's native per-series step."""
        if not self.enabled or not self._initialized or self.writer is None:
            return
        try:
            for key, value in metrics.items():
                metric_key = self._apply_prefix(str(key), prefix)
                if metric_key == step_key:
                    continue
                scalar = self._coerce_metric_value(value)
                if scalar is not None:
                    self.writer.add_scalar(metric_key, scalar, global_step=int(step))
        except Exception as exc:  # logging must not terminate training
            module_logger.warning(
                "TensorBoard metric logging failed (%s): %s", step_key, exc
            )

    @staticmethod
    def _image_tensor(image: Any) -> torch.Tensor:
        if torch.is_tensor(image):
            tensor = image.detach().cpu()
        else:
            try:
                import numpy as np

                tensor = torch.as_tensor(np.asarray(image))
            except Exception as exc:
                raise TypeError(f"Unsupported TensorBoard image type: {type(image)!r}") from exc
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim != 3 or tensor.shape[0] not in (1, 3, 4):
            raise ValueError(f"Expected CHW/HWC image, got {tuple(tensor.shape)}")
        if tensor.shape[0] == 4:
            tensor = tensor[:3]
        if tensor.dtype == torch.uint8:
            return tensor
        return tensor.to(torch.float32).clamp(0.0, 1.0)

    def log_generated_media(
        self,
        rollout_id: int,
        media_preview: Any,
        *,
        key: str = "rollout/generated_media",
        video_key: Optional[str] = None,
        video_fps: int = 8,
    ) -> None:
        """Write images, videos, optional audio, and captions to TensorBoard."""
        if (
            media_preview is None
            or not self.enabled
            or not self._initialized
            or self.writer is None
        ):
            return
        if isinstance(media_preview, dict):
            images = media_preview.get("images") or []
            videos = media_preview.get("videos") or []
            prompts = media_preview.get("prompts") or []
            rewards = media_preview.get("rewards")
            audios = media_preview.get("audios") or []
            audio_sample_rate = media_preview.get("audio_sample_rate")
        else:
            images = getattr(media_preview, "images", None) or []
            videos = getattr(media_preview, "videos", None) or []
            prompts = getattr(media_preview, "prompts", None) or []
            rewards = getattr(media_preview, "rewards", None)
            audios = getattr(media_preview, "audios", None) or []
            audio_sample_rate = getattr(media_preview, "audio_sample_rate", None)

        reward_values: Optional[List[float]] = None
        if isinstance(rewards, dict):
            rewards = rewards.get("avg", rewards.get("rewards"))
        if torch.is_tensor(rewards):
            reward_values = rewards.detach().cpu().reshape(-1).tolist()
        elif rewards is not None:
            try:
                reward_values = [float(value) for value in rewards]
            except (TypeError, ValueError):
                reward_values = None

        if video_key is None:
            video_key = "rollout/generated_videos" if key == "rollout/generated_media" else f"{key}/videos"
        try:
            for idx, image in enumerate(images[: self.media_max_items]):
                self.writer.add_image(
                    f"{key}/{idx}",
                    self._image_tensor(image),
                    global_step=int(rollout_id),
                )
            for idx, video in enumerate(videos[: self.media_max_items]):
                if not torch.is_tensor(video) or video.ndim != 4:
                    continue
                # MediaPreview stores [C, T, H, W]; SummaryWriter expects
                # [N, T, C, H, W] in [0, 1].
                tb_video = (
                    video.detach()
                    .cpu()
                    .to(torch.float32)
                    .clamp(0.0, 1.0)
                    .permute(1, 0, 2, 3)
                    .unsqueeze(0)
                )
                self.writer.add_video(
                    f"{video_key}/{idx}",
                    tb_video,
                    global_step=int(rollout_id),
                    fps=int(video_fps),
                )
            if audio_sample_rate is not None:
                for idx, audio in enumerate(audios[: self.media_max_items]):
                    if not torch.is_tensor(audio):
                        continue
                    waveform = audio.detach().cpu().to(torch.float32)
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0)
                    self.writer.add_audio(
                        f"{video_key}/{idx}/audio",
                        waveform[:1],
                        global_step=int(rollout_id),
                        sample_rate=int(audio_sample_rate),
                    )
            for idx in range(min(max(len(images), len(videos)), self.media_max_items)):
                prompt = str(prompts[idx]) if idx < len(prompts) else ""
                reward = (
                    f"\n\nreward: {reward_values[idx]:.4f}"
                    if reward_values is not None and idx < len(reward_values)
                    else ""
                )
                self.writer.add_text(
                    f"{key}/{idx}/caption",
                    prompt[:1000] + reward,
                    global_step=int(rollout_id),
                )
        except Exception as exc:  # media logging is best-effort
            module_logger.warning("TensorBoard media logging failed: %s", exc)

    def finish(self) -> None:
        if self.writer is None:
            return
        try:
            self.writer.flush()
            self.writer.close()
        except Exception as exc:
            module_logger.warning("Failed to close TensorBoard writer: %s", exc)
        finally:
            self.writer = None
            self._initialized = False


def init_tensorboard_logger(**kwargs: Any) -> UniRLTensorBoardLogger:
    """Construct a TensorBoard logger or disabled null-object."""
    return UniRLTensorBoardLogger(**kwargs)


__all__ = ["UniRLTensorBoardLogger", "init_tensorboard_logger"]
