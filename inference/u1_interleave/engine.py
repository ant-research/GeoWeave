"""Thin wrapper around ``model.interleave_gen`` for one-shot end-to-end
text+image generation. Mirrors the structure of
the upstream interleave inference implementation but keeps only what the
geo-aux bench needs (no JSONL batching, no profiler, no LoRA — those live in
the worker/launcher).

Importing this module requires the GeoWeave-local inference package to be
installed (for example, from the GeoWeave workspace root). No separate
upstream model source checkout is required.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import GenerationConfig

import sensenova_u1
from sensenova_u1.utils import (
    DEFAULT_VRAM_MODE,
    load_and_merge_lora_weight_from_safetensors,
    load_model_and_tokenizer,
    make_offload_ctx,
    vram_mode_to_prefetch_count,
)

NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


# Aspect-ratio buckets the official inference script uses when there is no
# input image. The geo-aux bench always has an input image (``images[0]``), so
# these are only a safety fallback.
SUPPORTED_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "1:1": (1536, 1536),
    "16:9": (2048, 1152),
    "9:16": (1152, 2048),
    "3:2": (1888, 1248),
    "2:3": (1248, 1888),
    "4:3": (1760, 1312),
    "3:4": (1312, 1760),
    "1:2": (1088, 2144),
    "2:1": (2144, 1088),
    "1:3": (864, 2592),
    "3:1": (2592, 864),
}
DEFAULT_RESOLUTION = "16:9"


def _round_by(n: int, factor: int) -> int:
    return round(n / factor) * factor


def _ceil_by(n: int, factor: int) -> int:
    return math.ceil(n / factor) * factor


def _floor_by(n: int, factor: int) -> int:
    return math.floor(n / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 512 * 512,
    max_pixels: int = (4 * 2048 * 2048) // 8,
) -> tuple[int, int]:
    """Verbatim from the official interleave inference script: round (H, W) to
    a 32-aligned bucket within the model's trained pixel range while
    preserving aspect ratio."""
    if max(height, width) / max(1, min(height, width)) > 200:
        raise ValueError(
            f"absolute aspect ratio must be < 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, _round_by(height, factor))
    w_bar = max(factor, _round_by(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by(height / beta, factor))
        w_bar = max(factor, _floor_by(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by(height * beta, factor)
        w_bar = _ceil_by(width * beta, factor)
    return h_bar, w_bar


def _denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(NORM_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def _to_pil(batch: torch.Tensor) -> Image.Image:
    arr = _denorm(batch.float()).permute(0, 2, 3, 1).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr[0])


def resolve_image_size(
    input_images: Sequence[Image.Image],
    fallback: tuple[int, int] = SUPPORTED_RESOLUTIONS[DEFAULT_RESOLUTION],
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    """Pick output (W, H). With an input image, follow its size via
    smart_resize; otherwise use the fallback bucket."""
    if input_images:
        w, h = input_images[0].size
        kwargs = {}
        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
        rh, rw = smart_resize(h, w, **kwargs)
        return rw, rh
    if max_pixels is not None and fallback[0] * fallback[1] > max_pixels:
        kwargs = {"max_pixels": max_pixels}
        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels
        rh, rw = smart_resize(fallback[1], fallback[0], **kwargs)
        return rw, rh
    return fallback


class SenseNovaU1Interleave:
    """One-engine-per-process wrapper. Loads the full model onto a single
    GPU (data-parallel workers); offload modes are exposed but unused by the
    default geo-aux launcher since each L20X has enough VRAM."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_backend: str = "auto",
        vram_mode: str = DEFAULT_VRAM_MODE,
        device_map: str | None = None,
        max_memory: str | None = None,
        lora_path: str | None = None,
    ) -> None:
        sensenova_u1.set_attn_backend(attn_backend)
        self.device = device
        self.vram_mode = vram_mode
        self.prefetch_count = vram_mode_to_prefetch_count(vram_mode)
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path,
            dtype=dtype,
            device=device,
            for_offload=self.prefetch_count > 0,
            device_map=device_map,
            max_memory=max_memory,
        )
        if lora_path:
            self.model = load_and_merge_lora_weight_from_safetensors(self.model, lora_path)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        input_images: Sequence[Image.Image] = (),
        image_size: tuple[int, int] | None = None,
        cfg_scale: float = 4.0,
        img_cfg_scale: float = 1.0,
        timestep_shift: float = 3.0,
        cfg_interval: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 50,
        max_new_tokens: int = 8192,
        repetition_penalty: float | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        think_mode: bool = True,
        system_message: str = "",
        seed: int = 42,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> tuple[str, list[Image.Image], dict[str, float | int | None]]:
        if image_size is None:
            image_size = resolve_image_size(
                input_images,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        if repetition_penalty is not None and repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be positive or null, got {repetition_penalty}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if not 0 < top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")
        generation_config = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        )
        if repetition_penalty is not None:
            generation_config.repetition_penalty = float(repetition_penalty)

        with make_offload_ctx(self.model, self.prefetch_count, self.device) as offloaded:
            text, image_tensors, generation_stats = offloaded.interleave_gen(
                self.tokenizer,
                prompt,
                images=list(input_images),
                generation_config=generation_config,
                image_size=image_size,
                cfg_scale=cfg_scale,
                img_cfg_scale=img_cfg_scale,
                timestep_shift=timestep_shift,
                cfg_interval=cfg_interval,
                num_steps=num_steps,
                system_message=system_message,
                think_mode=think_mode,
                seed=seed,
                verbose=False,
                record_ar_entropy=True,
            )
        return text, [_to_pil(t) for t in image_tensors], generation_stats
