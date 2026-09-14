#!/usr/bin/env bash
# Build an isolated GeoWeave trainside environment on PyTorch 2.8 / CUDA 12.8.
#
# This intentionally does NOT install UniRL's `trainside` extra: that extra pins
# the repository's default torch 2.11 stack. Instead, this script installs the
# torch 2.8 stack explicitly, then installs the engine-free `train,infer` extras.
#
# Usage:
#   bash scripts/setup_geoweave_env.sh
#   WITH_DEV=1 bash scripts/setup_geoweave_env.sh
#   CHECK_ONLY=1 bash scripts/setup_geoweave_env.sh
#
# Overrides:
#   UV_BIN=/path/to/uv ENV_DIR=/path/to/.venv-RL PYTHON_VERSION=3.12
#   PYPI_INDEX_URL=https://example.com/simple FLASH_ATTN_WHEEL=/path/to/wheel

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
ENV_DIR="${ENV_DIR:-${REPO_ROOT}/.venv-RL}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-/path/to/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
WITH_DEV="${WITH_DEV:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
CUDNN_VERSION="${CUDNN_VERSION:-9.10.2.21}"

fail() {
    echo "[geoweave-rl-env] ERROR: $*" >&2
    exit 2
}

run() {
    printf '[geoweave-rl-env] run:'
    printf ' %q' "$@"
    echo
    if [ "${DRY_RUN}" != "1" ]; then
        "$@"
    fi
}

require_bool() {
    local name="$1" value="$2"
    [[ "${value}" == "0" || "${value}" == "1" ]] || fail "${name} must be 0 or 1, got ${value}"
}

[ -n "${UV_BIN}" ] || fail "uv is not installed or not on PATH"
[ -x "${UV_BIN}" ] || fail "uv is not executable: ${UV_BIN}"
require_bool WITH_DEV "${WITH_DEV}"
require_bool CHECK_ONLY "${CHECK_ONLY}"
require_bool DRY_RUN "${DRY_RUN}"

PYTHON_BIN="${ENV_DIR}/bin/python"
UV_COMMON=(
    --no-config
    --python "${PYTHON_BIN}"
    --default-index "${PYPI_INDEX_URL}"
)

# The uv cache and ENV_DIR are commonly on different filesystems on some systems. Explicit copy mode avoids a noisy hardlink failure before fallback.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-500}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"

if [ "${CHECK_ONLY}" != "1" ]; then
    if [ ! -x "${PYTHON_BIN}" ]; then
        run "${UV_BIN}" venv --python "${PYTHON_VERSION}" "${ENV_DIR}"
    fi

    if [ "${DRY_RUN}" != "1" ]; then
        [ -x "${PYTHON_BIN}" ] || fail "Python executable not found after uv venv: ${PYTHON_BIN}"
        [ -r "${FLASH_ATTN_WHEEL}" ] || fail "FlashAttention wheel is not readable: ${FLASH_ATTN_WHEEL}"
    fi

    run "${UV_BIN}" pip install "${UV_COMMON[@]}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        "nvidia-cudnn-cu12==${CUDNN_VERSION}"

    PROJECT_EXTRAS="train,infer"
    if [ "${WITH_DEV}" = "1" ]; then
        PROJECT_EXTRAS+=",dev"
    fi
    # Pin the key Python packages to the validated Torch 2.8 environment while
    # leaving transitive CUDA library pins to the official torch wheel metadata.
    run "${UV_BIN}" pip install "${UV_COMMON[@]}" --editable ".[${PROJECT_EXTRAS}]" \
        "numpy==2.5.1" \
        "transformers==5.6.2" \
        "tokenizers==0.22.2" \
        "accelerate==1.14.0" \
        "diffusers==0.39.0" \
        "peft==0.19.1" \
        "ray[default]==2.56.1" \
        "tensorboard==2.21.0" \
        "wandb==0.19.11"

    run "${UV_BIN}" pip install "${UV_COMMON[@]}" "${FLASH_ATTN_WHEEL}"
fi

if [ "${DRY_RUN}" = "1" ] && [ ! -x "${PYTHON_BIN}" ]; then
    echo "[geoweave-rl-env] dry-run complete; environment would be created at ${ENV_DIR}"
    exit 0
fi
[ -x "${PYTHON_BIN}" ] || fail "Python executable not found: ${PYTHON_BIN}"

PYTHON_SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUDNN_LIB_DIR="${PYTHON_SITE_PACKAGES}/nvidia/cudnn/lib"
if [ -f "${CUDNN_LIB_DIR}/libcudnn.so.9" ]; then
    export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

"${UV_BIN}" pip check --python "${PYTHON_BIN}"

"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

required = (
    "torch",
    "torchvision",
    "torchaudio",
    "flash_attn",
    "ray",
    "hydra",
    "omegaconf",
    "transformers",
    "diffusers",
    "peft",
    "dashscope",
    "tensorboard",
    "wandb",
    "unirl",
)
failed = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(f"{name}: {type(exc).__name__}: {exc}")
if failed:
    raise SystemExit("Environment import check failed:\n  " + "\n  ".join(failed))

import flash_attn
import torch
from unirl.models.sensenova_u1.vendor.modeling_qwen3 import effective_attn_backend

if not torch.__version__.startswith("2.8.0"):
    raise SystemExit(f"Expected torch 2.8.0, got {torch.__version__}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"Expected torch CUDA runtime 12.8, got {torch.version.cuda}")
if not torch.compiled_with_cxx11_abi():
    raise SystemExit("The provided FlashAttention wheel requires CXX11 ABI=True")
if flash_attn.__version__ != "2.8.3.post1":
    raise SystemExit(f"Expected flash-attn 2.8.3.post1, got {flash_attn.__version__}")
if effective_attn_backend() != "flash":
    raise SystemExit(f"GeoWeave did not select FlashAttention: {effective_attn_backend()}")
if torch.cuda.is_available():
    try:
        with torch.inference_mode():
            probe = torch.randn(2, 3, 16, 16, device="cuda", dtype=torch.bfloat16)
            conv = torch.nn.Conv2d(3, 32, 3, device="cuda", dtype=torch.bfloat16)
            conv(probe)
            torch.cuda.synchronize()
    except RuntimeError as exc:
        raise SystemExit(f"cuDNN Conv2d preflight failed: {exc}") from exc

print(f"[geoweave-rl-env] python:      {sys.version.split()[0]}")
print(f"[geoweave-rl-env] torch:       {torch.__version__}")
print(f"[geoweave-rl-env] cuda:        runtime={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"[geoweave-rl-env] cudnn/nccl:  {torch.backends.cudnn.version()} / {torch.cuda.nccl.version()}")
print(f"[geoweave-rl-env] flash-attn:  {flash_attn.__version__}")
print(f"[geoweave-rl-env] attention:   {effective_attn_backend()}")
print(f"[geoweave-rl-env] devices:     {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"[geoweave-rl-env] gpu0:        {torch.cuda.get_device_name(0)}")
PY

if [ "${DRY_RUN}" != "1" ] && [ "${CHECK_ONLY}" != "1" ]; then
    "${UV_BIN}" pip freeze --python "${PYTHON_BIN}" > "${ENV_DIR}/requirements.freeze.txt"
    echo "[geoweave-rl-env] freeze:      ${ENV_DIR}/requirements.freeze.txt"
fi

if [ -n "${DASHSCOPE_API_KEY:-}" ]; then
    echo "[geoweave-rl-env] DashScope API key: configured"
else
    echo "[geoweave-rl-env] DashScope API key: not set (export DASHSCOPE_API_KEY before training)"
fi

echo "[geoweave-rl-env] environment ready: ${ENV_DIR}"
echo "[geoweave-rl-env] train with: bash scripts/train_geoweave.sh"
