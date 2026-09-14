#!/usr/bin/env python3
"""Strict runtime preflight for the GeoWeave Torch 2.8 environment."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import os
import sys
from importlib import metadata
from pathlib import Path

EXPECTED_TORCH_PREFIX = "2.8.0"
EXPECTED_CUDA = "12.8"
EXPECTED_FLASH_ATTN = "2.8.3.post1"
EXPECTED_CUDNN_PACKAGE = "9.10.2.21"
EXPECTED_CUDNN_RUNTIME = 91002


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as exc:
        raise SystemExit(f"Required package is not installed: {name}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-gpus",
        type=int,
        default=int(os.environ.get("GPUS_PER_NODE", "0")),
        help="Minimum number of CUDA devices required (default: GPUS_PER_NODE or 0)",
    )
    parser.add_argument(
        "--skip-gpu-kernels",
        action="store_true",
        help="Check package/library versions without executing CUDA kernels",
    )
    args = parser.parse_args()

    if args.expected_gpus < 0:
        raise SystemExit("--expected-gpus must be non-negative")

    cudnn_package_version = package_version("nvidia-cudnn-cu12")
    if cudnn_package_version != EXPECTED_CUDNN_PACKAGE:
        raise SystemExit(
            "Expected nvidia-cudnn-cu12 "
            f"{EXPECTED_CUDNN_PACKAGE}, got {cudnn_package_version}"
        )

    # Load by SONAME, not by absolute path. This verifies that LD_LIBRARY_PATH
    # selects the virtualenv wheel before any host/image-level cuDNN copy.
    try:
        cudnn = ctypes.CDLL("libcudnn.so.9")
    except OSError as exc:
        raise SystemExit(f"Unable to load libcuDNN.so.9: {exc}") from exc
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    linked_cudnn_version = int(cudnn.cudnnGetVersion())
    if linked_cudnn_version != EXPECTED_CUDNN_RUNTIME:
        raise SystemExit(
            f"Expected cuDNN runtime {EXPECTED_CUDNN_RUNTIME}, "
            f"got {linked_cudnn_version}; check LD_LIBRARY_PATH ordering"
        )

    for module_name in ("dashscope", "hydra", "tensorboard", "unirl"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            raise SystemExit(
                f"Required module import failed: {module_name}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    import flash_attn
    import ray
    import torch
    from flash_attn import flash_attn_func

    from unirl.models.sensenova_u1.vendor.modeling_qwen3 import (
        effective_attn_backend,
    )

    if not torch.__version__.startswith(EXPECTED_TORCH_PREFIX):
        raise SystemExit(
            f"Expected torch {EXPECTED_TORCH_PREFIX}, got {torch.__version__}"
        )
    if torch.version.cuda != EXPECTED_CUDA:
        raise SystemExit(
            f"Expected torch CUDA runtime {EXPECTED_CUDA}, got {torch.version.cuda}"
        )
    if not torch.compiled_with_cxx11_abi():
        raise SystemExit("FlashAttention wheel requires torch CXX11 ABI=True")
    if flash_attn.__version__ != EXPECTED_FLASH_ATTN:
        raise SystemExit(
            f"Expected flash-attn {EXPECTED_FLASH_ATTN}, got {flash_attn.__version__}"
        )
    if torch.backends.cudnn.version() != EXPECTED_CUDNN_RUNTIME:
        raise SystemExit(
            f"PyTorch loaded cuDNN {torch.backends.cudnn.version()}, "
            f"expected {EXPECTED_CUDNN_RUNTIME}"
        )

    attention_backend = effective_attn_backend()
    if attention_backend != "flash":
        raise SystemExit(
            f"GeoWeave selected attention backend {attention_backend!r}, expected 'flash'"
        )

    gpu_count = torch.cuda.device_count()
    if gpu_count < args.expected_gpus:
        raise SystemExit(
            f"Expected at least {args.expected_gpus} CUDA devices, PyTorch sees {gpu_count}"
        )

    if not args.skip_gpu_kernels:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable")
        try:
            with torch.inference_mode():
                probe = torch.randn(
                    2, 3, 16, 16, device="cuda", dtype=torch.bfloat16
                )
                conv = torch.nn.Conv2d(
                    3, 32, 3, device="cuda", dtype=torch.bfloat16
                )
                conv_out = conv(probe)

                query = torch.randn(
                    1, 16, 2, 64, device="cuda", dtype=torch.bfloat16
                )
                flash_out = flash_attn_func(
                    query, query, query, dropout_p=0.0, causal=True
                )
                torch.cuda.synchronize()
            conv_is_finite = bool(torch.isfinite(conv_out).all().item())
            flash_is_finite = bool(torch.isfinite(flash_out).all().item())
            if not conv_is_finite or not flash_is_finite:
                raise RuntimeError("CUDA preflight produced non-finite output")
        except Exception as exc:
            raise SystemExit(
                f"cuDNN/FlashAttention CUDA kernel preflight failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    cudnn_path = Path(os.environ.get("CUDNN_LIB_DIR", "")) / "libcudnn.so.9"
    print(
        "[geoweave-rl-env] preflight passed: "
        f"python={sys.executable} torch={torch.__version__} "
        f"cuda={torch.version.cuda} cudnn_package={cudnn_package_version} "
        f"cudnn_runtime={torch.backends.cudnn.version()} "
        f"flash_attn={flash_attn.__version__} attention={attention_backend} "
        f"ray={ray.__version__} gpus={gpu_count} cudnn_library={cudnn_path}"
    )
    print(f"[geoweave-rl-env] torch module: {torch.__file__}")
    print(f"[geoweave-rl-env] flash-attn module: {flash_attn.__file__}")


if __name__ == "__main__":
    main()
