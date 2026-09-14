#!/usr/bin/env bash
#
# Single-node GeoWeave Interleave-RL training launcher.
#
# Validated defaults:
#   - 8 GPUs, batch_size=8, samples_per_prompt=8 => 64 trajectories/rollout
#   - 64 is divisible by 8 DP ranks * 2 updates/batch, so every rank receives
#     a non-empty, evenly sized train shard. It also provides enough independent
#     rollout work for the dynamic scheduler; very small smoke-test geometry can
#     leave some FSDP ranks without rollout work and may stall collective calls.
#   - 2560 generated text tokens with configurable truncation handling
#   - 8 samples per prompt
#   - 512^2 pixel budget (aspect ratio preserved, dimensions aligned to 32)
#   - 30 deterministic diffusion steps
#   - base_shift=3.0, runtime timestep_shift=1.0, eta=0.0
#   - Interleave-RL text objective + inline velocity MSE, activation checkpointing enabled
#
# The launcher deliberately uses base_shift=3.0, runtime
# timestep_shift=1.0, and eta=0.0. Override them explicitly when testing a
# different diffusion recipe.
#
# Basic usage:
#   bash scripts/train_geoweave.sh
#
# Common overrides:
#   RUN_NAME=my_smoke NUM_ROLLOUTS=10 bash scripts/train_geoweave.sh
#   LOAD_DIR=/path/to/checkpoint-100 RUN_NAME=existing_run NUM_ROLLOUTS=1000 \
#     bash scripts/train_geoweave.sh
#
#   MODEL_PATH=/path/to/model DATA_PATH=/path/to/data.jsonl \
#   IMAGE_SIZE=512 SAMPLES_PER_PROMPT=4 \
#     bash scripts/train_geoweave.sh
#
# Extra positional arguments are forwarded as Hydra overrides and win over the
# defaults assembled below:
#   bash scripts/train_geoweave.sh backend.optimizer_cfg.weight_decay=0.01 sampling.ar.temperature=0.8

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/train_geoweave.sh [Hydra overrides...]

Important environment variables:
  RUN_NAME                 Unique experiment name (timestamped by default)
  MODEL_PATH / DATA_PATH   Model checkpoint and JSONL training data
  NUM_ROLLOUTS             Total rollout budget, including resumed steps
  BATCH_SIZE / SAMPLES_PER_PROMPT
                            Their product is trajectories per rollout. Keep the
                            product divisible by NUM_DEVICES *
                            NUM_UPDATES_PER_BATCH. For the dynamic scheduler,
                            avoid tiny settings that cannot feed every DP rank;
                            the validated 8-GPU default is 8 * 8 = 64.
  GRADIENT_ACCUMULATION_STEPS  Rollouts per optimizer step (default: 1)
  LOAD_DIR                 checkpoint-N directory to resume from
  SAVE_INTERVAL            Checkpoint interval; 0 disables checkpointing
  CHECKPOINT_FORMAT        dcp (default) or torch
  ENABLE_DUMP              true (default) or false
  LOGGING_BACKEND          tensorboard (default) or wandb
  MAX_NEW_TOKENS           Text generation cap / Lmax (default: 2560)
  TRUNCATED_REWARD         zero, keep, or soft (default: keep)
  RETRY_TRUNCATED_TRAJECTORIES  Retry text truncations (default: false)
  IGNORE_TRUNCATED_SAMPLES      Exclude text-truncated rows from updates (default: false)
  EXCLUDE_TRUNCATED_FROM_ADVANTAGE_STATS
                              Also exclude them from group mean/std (default: false)
  RECOVER_REPETITION_TRUNCATIONS
                              Recover exact repeated tails for SCA (default: true)
  OVERLONG_BUFFER_LEN      Soft overlong buffer / Lcache (default: 512)
  OVERLONG_PENALTY_FACTOR  Maximum soft penalty magnitude (default: 1.0)
  CLIP_RANGE               Interleave-RL policy-ratio clip range (default: 0.1)
  LOSS_AGG_MODE            AR loss reduction: token-mean, seq-mean-token-mean,
                           or seq-mean-token-sum-norm (default: token-mean)
  AR_KL_COEF               Sampled-token reference-policy KL coefficient (default: 1e-3)
  LR_SCHEDULER_TYPE         constant (default), linear, or cosine
  WARMUP_STEPS             Optimizer-step warmup length (default: 0, disabled)
  REQUIRE_IDLE_GPUS        false (default); set true to enable the busy-GPU guard
  ALLOW_EXISTING_RUN       false (default); protects an existing experiment
  DRY_RUN                  1 prints the resolved command without launching
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_NAME="${CONFIG_NAME:-unified_model/geoweave_interleave_rl}"
ENV_DIR="${ENV_DIR:-${REPO_ROOT}/.venv-RL}"
PYTHON_BIN="${PYTHON_BIN:-${ENV_DIR}/bin/python}"

MODEL_PATH="${MODEL_PATH:-/path/to/GeoWeave-HF}"
DATA_PATH="${DATA_PATH:-/path/to/train.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"

NUM_DEVICES="${NUM_DEVICES:-8}"
# Rollout/train geometry (important for distributed FSDP execution):
#
#   trajectories_per_rollout = BATCH_SIZE * SAMPLES_PER_PROMPT
#
# Keep that product divisible by NUM_DEVICES * NUM_UPDATES_PER_BATCH so each
# rank/update receives an equal, non-empty shard. With dynamic trajectory
# scheduling, do not use a tiny smoke-test geometry that leaves DP ranks without
# independent rollout jobs: partial-rank entry into shared FSDP collectives can
# stall. The defaults below produce 8 * 8 = 64 trajectories; 64 is divisible by
# 8 * 2 = 16 and has been chosen as the supported 8-GPU training geometry.
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1000}"
NUM_UPDATES_PER_BATCH="${NUM_UPDATES_PER_BATCH:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2560}"
TRUNCATED_REWARD="${TRUNCATED_REWARD:-keep}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-512}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-30}"
MAX_IMAGES="${MAX_IMAGES:-4}"
ROLLOUT_TEXT_BATCH_SIZE="${ROLLOUT_TEXT_BATCH_SIZE:-1}"
ROLLOUT_DIFFUSION_BATCH_SIZE="${ROLLOUT_DIFFUSION_BATCH_SIZE:-2}"
ROLLOUT_REENCODE_BATCH_SIZE="${ROLLOUT_REENCODE_BATCH_SIZE:-2}"
DYNAMIC_TRAJECTORY_SCHEDULING="${DYNAMIC_TRAJECTORY_SCHEDULING:-true}"
DYNAMIC_ROLLOUT_CHUNK_SIZE="${DYNAMIC_ROLLOUT_CHUNK_SIZE:-2}"
CONTINUOUS_BATCHING="${CONTINUOUS_BATCHING:-true}"
PERSISTENT_WORKER_SESSION="${PERSISTENT_WORKER_SESSION:-false}"
CONTINUOUS_REQUEST_ADMISSION="${CONTINUOUS_REQUEST_ADMISSION:-false}"
CONTINUOUS_LIVE_ADMISSION="${CONTINUOUS_LIVE_ADMISSION:-false}"
CONTINUOUS_PROMPT_MICROBUNDLE_SIZE="${CONTINUOUS_PROMPT_MICROBUNDLE_SIZE:-1}"
CONTINUOUS_SESSION_WINDOW_SIZE="${CONTINUOUS_SESSION_WINDOW_SIZE:-8}"
CONTINUOUS_ROLLOUT_POOL_SIZE="${CONTINUOUS_ROLLOUT_POOL_SIZE:-4}"
CONTINUOUS_TEXT_BATCH_SIZE="${CONTINUOUS_TEXT_BATCH_SIZE:-2}"
DYNAMIC_REWARD_WORKERS="${DYNAMIC_REWARD_WORKERS:-4}"
RETRY_TRUNCATED_TRAJECTORIES="${RETRY_TRUNCATED_TRAJECTORIES:-false}"
IGNORE_TRUNCATED_SAMPLES="${IGNORE_TRUNCATED_SAMPLES:-false}"
EXCLUDE_TRUNCATED_FROM_ADVANTAGE_STATS="${EXCLUDE_TRUNCATED_FROM_ADVANTAGE_STATS:-false}"
RECOVER_REPETITION_TRUNCATIONS="${RECOVER_REPETITION_TRUNCATIONS:-true}"
REPETITION_MIN_BLOCK_TOKENS="${REPETITION_MIN_BLOCK_TOKENS:-16}"
REPETITION_MAX_BLOCK_TOKENS="${REPETITION_MAX_BLOCK_TOKENS:-256}"
REPETITION_MIN_REPEATS="${REPETITION_MIN_REPEATS:-3}"
REPETITION_MIN_PREFIX_TOKENS="${REPETITION_MIN_PREFIX_TOKENS:-64}"
REPETITION_TAIL_TOLERANCE_TOKENS="${REPETITION_TAIL_TOLERANCE_TOKENS:-32}"
REPETITION_REQUIRE_COMPLETE_STEP="${REPETITION_REQUIRE_COMPLETE_STEP:-true}"
MAX_ATTEMPTS_PER_CANDIDATE="${MAX_ATTEMPTS_PER_CANDIDATE:-2}"
MAX_TRUNCATED_REFILLS_PER_PROMPT="${MAX_TRUNCATED_REFILLS_PER_PROMPT:-4}"
DYNAMIC_PROMPT_GROUP_REFILL="${DYNAMIC_PROMPT_GROUP_REFILL:-true}"
RESERVE_PROMPT_COUNT="${RESERVE_PROMPT_COUNT:-${BATCH_SIZE}}"
MAX_REPLACEMENT_PROMPTS="${MAX_REPLACEMENT_PROMPTS:-4}"
MAX_TRAJECTORY_MULTIPLIER="${MAX_TRAJECTORY_MULTIPLIER:-1.5}"
ANSWER_JUDGE_MODEL="${ANSWER_JUDGE_MODEL:-qwen3.7-max}"
SCA_JUDGE_MODEL="${SCA_JUDGE_MODEL:-gemini-3.5-flash}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
SCA_JUDGE_TEMPERATURE="${SCA_JUDGE_TEMPERATURE:-0.7}"
SCA_JUDGE_TOP_P="${SCA_JUDGE_TOP_P:-0.9}"
SCA_JUDGE_MAX_TOKENS="${SCA_JUDGE_MAX_TOKENS:-16384}"
SCA_JUDGE_REASONING_EFFORT="${SCA_JUDGE_REASONING_EFFORT:-low}"
SCA_JUDGE_RESPONSE_FORMAT="${SCA_JUDGE_RESPONSE_FORMAT:-json_object}"
SCA_NUM_CRITIQUES="${SCA_NUM_CRITIQUES:-1}"
SCA_VOTING="${SCA_VOTING:-majority}"
SCA_JUDGE_QPS="${SCA_JUDGE_QPS:-2.0}"
SCA_JUDGE_TIMEOUT="${SCA_JUDGE_TIMEOUT:-360.0}"
SCA_JUDGE_MAX_RETRIES="${SCA_JUDGE_MAX_RETRIES:-2}"
SCA_JUDGE_MAX_WORKERS="${SCA_JUDGE_MAX_WORKERS:-8}"
SCA_KEEP_RAW_OUTPUTS="${SCA_KEEP_RAW_OUTPUTS:-false}"
SCA_OUTCOME_CORRECT_CREDIT="${SCA_OUTCOME_CORRECT_CREDIT:-2.0}"
SCA_PROCESS_ERROR_PENALTY="${SCA_PROCESS_ERROR_PENALTY:-1.0}"
SCA_ADVANTAGE_EPS="${SCA_ADVANTAGE_EPS:-1e-6}"
DUMP_ASYNC="${DUMP_ASYNC:-true}"
DUMP_IMAGE_WORKERS="${DUMP_IMAGE_WORKERS:-8}"
DUMP_JPEG_QUALITY="${DUMP_JPEG_QUALITY:-90}"
DUMP_MAX_PENDING="${DUMP_MAX_PENDING:-1}"
CLIP_RANGE="${CLIP_RANGE:-0.1}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
AR_KL_COEF="${AR_KL_COEF:-1e-3}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
MSE_STEPS="${MSE_STEPS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
# Warmup is configured but disabled by default. The current scheduler implementation
# only applies warmup_steps when LR_SCHEDULER_TYPE=linear; constant ignores it.
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
CHECKPOINT_FORMAT="${CHECKPOINT_FORMAT:-dcp}"
LOAD_DIR="${LOAD_DIR:-}"
ENABLE_DUMP="${ENABLE_DUMP:-true}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-false}"
ALLOW_EXISTING_RUN="${ALLOW_EXISTING_RUN:-false}"

RUN_NAME="${RUN_NAME:-geoweave_interleave_rl_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${REPO_ROOT}/experiments}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${EXPERIMENTS_DIR}/${RUN_NAME}}"
LOGGING_BACKEND="${LOGGING_BACKEND:-tensorboard}"
REPORT_TO_TENSORBOARD="${REPORT_TO_TENSORBOARD:-true}"
REPORT_TO_WANDB="${REPORT_TO_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-geoweave_interleave_rl}"
SAVE_DIR="${SAVE_DIR:-${EXPERIMENT_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${EXPERIMENT_DIR}/logs}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${EXPERIMENT_DIR}/runs}"
HYDRA_OUTPUT_DIR="${HYDRA_OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"
DUMP_DIR="${DUMP_DIR:-${EXPERIMENT_DIR}/dumps}"
if [ "${ENABLE_DUMP}" = "false" ]; then
    DUMP_DIR=""
fi

export SENSENOVA_U1_PATH="${MODEL_PATH}"
export DATA_PATH
export RUN_NAME EXPERIMENT_DIR LOGGING_BACKEND REPORT_TO_TENSORBOARD REPORT_TO_WANDB
export TENSORBOARD_LOG_DIR WANDB_PROJECT
if [ "${LOGGING_BACKEND}" = "wandb" ]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
else
    export WANDB_MODE="${WANDB_MODE:-disabled}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SENSENOVA_MEM_PROFILE="${SENSENOVA_MEM_PROFILE:-0}"
export ANSWER_JUDGE_MODEL
export SCA_JUDGE_MODEL JUDGE_BASE_URL SCA_JUDGE_TEMPERATURE SCA_JUDGE_TOP_P
export SCA_JUDGE_MAX_TOKENS SCA_JUDGE_REASONING_EFFORT SCA_JUDGE_RESPONSE_FORMAT
export SCA_NUM_CRITIQUES SCA_VOTING SCA_JUDGE_QPS
export SCA_JUDGE_TIMEOUT SCA_JUDGE_MAX_RETRIES SCA_JUDGE_MAX_WORKERS SCA_KEEP_RAW_OUTPUTS
SENSENOVA_ROLLOUT_RUN_ID="${SENSENOVA_ROLLOUT_RUN_ID:-$(date +%s)-$$}"
export SENSENOVA_ROLLOUT_RUN_ID

if [ -n "${EVAL_DATA_PATH}" ]; then
    export EVAL_DATA_PATH
fi

fail() {
    echo "[geoweave-train] ERROR: $*" >&2
    exit 2
}

require_positive_int() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer, got ${value}"
}

require_non_negative_int() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be a non-negative integer, got ${value}"
}

require_bool() {
    local name="$1"
    local value="$2"
    [[ "${value}" == "true" || "${value}" == "false" ]] || fail "${name} must be true or false, got ${value}"
}

[ -x "${PYTHON_BIN}" ] || fail "Python executable not found: ${PYTHON_BIN}; run scripts/setup_geoweave_env.sh"

# Prefer the CUDA-12 cuDNN wheel installed in the selected virtualenv over the
# host-level cuDNN.  The base image may expose a newer cuDNN built for CUDA 13,
# which can be discovered by the dynamic linker even though this PyTorch build
# targets CUDA 12.8.  That mismatch fails at the first Conv2d with
# CUDNN_STATUS_NOT_INITIALIZED.
PYTHON_SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUDNN_LIB_DIR="${CUDNN_LIB_DIR:-${PYTHON_SITE_PACKAGES}/nvidia/cudnn/lib}"
if [ -f "${CUDNN_LIB_DIR}/libcudnn.so.9" ]; then
    export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# This launcher incorporates the environment validation that previously lived
# in a separate Torch 2.8 wrapper. DRY_RUN still performs this check so it can
# be used to validate both the environment and the resolved training command.
"${PYTHON_BIN}" - <<'PYENV'
import flash_attn
import torch
from unirl.models.sensenova_u1.vendor.modeling_qwen3 import effective_attn_backend

if not torch.__version__.startswith("2.8.0"):
    raise SystemExit(f"Expected torch 2.8.0, got {torch.__version__}")
if torch.version.cuda != "12.8":
    raise SystemExit(f"Expected torch CUDA runtime 12.8, got {torch.version.cuda}")
if not torch.compiled_with_cxx11_abi():
    raise SystemExit("The selected environment must use CXX11 ABI=True")
if flash_attn.__version__ != "2.8.3.post1":
    raise SystemExit(
        f"Expected flash-attn 2.8.3.post1, got {flash_attn.__version__}"
    )
backend = effective_attn_backend()
if backend != "flash":
    raise SystemExit(f"GeoWeave did not select FlashAttention: {backend}")

print(
    "[geoweave-train] environment: "
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"flash_attn={flash_attn.__version__} attention={backend}"
)
PYENV

if [ "${DRY_RUN:-0}" != "1" ]; then
    [ -n "${DASHSCOPE_API_KEY:-}" ] || fail "DASHSCOPE_API_KEY is required by the Qwen outcome judge"
    [ -n "${API_KEY_ENV:-}" ] || fail "API_KEY_ENV is required by the SCA process judge"
    [ -n "${JUDGE_BASE_URL}" ] || fail "JUDGE_BASE_URL is required by the process judge"
    "${PYTHON_BIN}" -c 'import dashscope, hydra, ray, tensorboard, torch, unirl' >/dev/null 2>&1 || \
        fail "training dependencies are incomplete; run: bash scripts/setup_geoweave_env.sh"
fi
[ -d "${MODEL_PATH}" ] || fail "MODEL_PATH is not a directory: ${MODEL_PATH}"
[ -f "${DATA_PATH}" ] || fail "DATA_PATH is not a file: ${DATA_PATH}"
if [ -n "${EVAL_DATA_PATH}" ]; then
    [ -f "${EVAL_DATA_PATH}" ] || fail "EVAL_DATA_PATH is not a file: ${EVAL_DATA_PATH}"
fi

require_positive_int NUM_DEVICES "${NUM_DEVICES}"
require_positive_int BATCH_SIZE "${BATCH_SIZE}"
require_positive_int NUM_ROLLOUTS "${NUM_ROLLOUTS}"
require_positive_int NUM_UPDATES_PER_BATCH "${NUM_UPDATES_PER_BATCH}"
require_positive_int GRADIENT_ACCUMULATION_STEPS "${GRADIENT_ACCUMULATION_STEPS}"
require_positive_int MICRO_BATCH_SIZE "${MICRO_BATCH_SIZE}"
require_positive_int MAX_NEW_TOKENS "${MAX_NEW_TOKENS}"
require_positive_int OVERLONG_BUFFER_LEN "${OVERLONG_BUFFER_LEN}"
require_positive_int SAMPLES_PER_PROMPT "${SAMPLES_PER_PROMPT}"
require_positive_int IMAGE_SIZE "${IMAGE_SIZE}"
require_positive_int DIFFUSION_STEPS "${DIFFUSION_STEPS}"
require_positive_int ROLLOUT_TEXT_BATCH_SIZE "${ROLLOUT_TEXT_BATCH_SIZE}"
require_positive_int ROLLOUT_DIFFUSION_BATCH_SIZE "${ROLLOUT_DIFFUSION_BATCH_SIZE}"
require_positive_int ROLLOUT_REENCODE_BATCH_SIZE "${ROLLOUT_REENCODE_BATCH_SIZE}"
require_positive_int DYNAMIC_REWARD_WORKERS "${DYNAMIC_REWARD_WORKERS}"
require_positive_int SCA_JUDGE_MAX_TOKENS "${SCA_JUDGE_MAX_TOKENS}"
require_positive_int SCA_NUM_CRITIQUES "${SCA_NUM_CRITIQUES}"
require_positive_int SCA_JUDGE_MAX_WORKERS "${SCA_JUDGE_MAX_WORKERS}"
require_bool SCA_KEEP_RAW_OUTPUTS "${SCA_KEEP_RAW_OUTPUTS}"
[[ "${SCA_VOTING}" == "greedy" || "${SCA_VOTING}" == "majority" || "${SCA_VOTING}" == "intersection" || "${SCA_VOTING}" == "union" || "${SCA_VOTING}" == "average" ]] || \
    fail "SCA_VOTING must be greedy, majority, intersection, union, or average; got ${SCA_VOTING}"
require_positive_int DYNAMIC_ROLLOUT_CHUNK_SIZE "${DYNAMIC_ROLLOUT_CHUNK_SIZE}"
require_positive_int CONTINUOUS_ROLLOUT_POOL_SIZE "${CONTINUOUS_ROLLOUT_POOL_SIZE}"
require_positive_int CONTINUOUS_SESSION_WINDOW_SIZE "${CONTINUOUS_SESSION_WINDOW_SIZE}"
require_positive_int CONTINUOUS_TEXT_BATCH_SIZE "${CONTINUOUS_TEXT_BATCH_SIZE}"
require_positive_int CONTINUOUS_PROMPT_MICROBUNDLE_SIZE "${CONTINUOUS_PROMPT_MICROBUNDLE_SIZE}"
require_positive_int MAX_ATTEMPTS_PER_CANDIDATE "${MAX_ATTEMPTS_PER_CANDIDATE}"
require_non_negative_int MAX_TRUNCATED_REFILLS_PER_PROMPT "${MAX_TRUNCATED_REFILLS_PER_PROMPT}"
require_non_negative_int RESERVE_PROMPT_COUNT "${RESERVE_PROMPT_COUNT}"
require_non_negative_int MAX_REPLACEMENT_PROMPTS "${MAX_REPLACEMENT_PROMPTS}"
[[ "${MAX_TRAJECTORY_MULTIPLIER}" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail "MAX_TRAJECTORY_MULTIPLIER must be numeric, got ${MAX_TRAJECTORY_MULTIPLIER}"
require_positive_int DUMP_IMAGE_WORKERS "${DUMP_IMAGE_WORKERS}"
require_positive_int DUMP_MAX_PENDING "${DUMP_MAX_PENDING}"
[[ "${DUMP_JPEG_QUALITY}" =~ ^[1-9][0-9]?$|^100$ ]] || fail "DUMP_JPEG_QUALITY must be in [1, 100], got ${DUMP_JPEG_QUALITY}"
require_bool DUMP_ASYNC "${DUMP_ASYNC}"
require_bool CONTINUOUS_BATCHING "${CONTINUOUS_BATCHING}"
require_bool PERSISTENT_WORKER_SESSION "${PERSISTENT_WORKER_SESSION}"
require_bool CONTINUOUS_REQUEST_ADMISSION "${CONTINUOUS_REQUEST_ADMISSION}"
require_bool CONTINUOUS_LIVE_ADMISSION "${CONTINUOUS_LIVE_ADMISSION}"
require_bool ENABLE_DUMP "${ENABLE_DUMP}"
require_bool REQUIRE_IDLE_GPUS "${REQUIRE_IDLE_GPUS}"
require_bool ALLOW_EXISTING_RUN "${ALLOW_EXISTING_RUN}"
[[ "${LOGGING_BACKEND}" == "tensorboard" || "${LOGGING_BACKEND}" == "wandb" ]] || fail "LOGGING_BACKEND must be tensorboard or wandb, got ${LOGGING_BACKEND}"
[[ "${CHECKPOINT_FORMAT}" == "torch" || "${CHECKPOINT_FORMAT}" == "dcp" ]] || fail "CHECKPOINT_FORMAT must be torch or dcp, got ${CHECKPOINT_FORMAT}"
[[ "${LOSS_AGG_MODE}" == "token-mean" || "${LOSS_AGG_MODE}" == "seq-mean-token-mean" || "${LOSS_AGG_MODE}" == "seq-mean-token-sum-norm" ]] || \
    fail "LOSS_AGG_MODE must be token-mean, seq-mean-token-mean, or seq-mean-token-sum-norm; got ${LOSS_AGG_MODE}"
require_bool REPORT_TO_TENSORBOARD "${REPORT_TO_TENSORBOARD}"
require_bool REPORT_TO_WANDB "${REPORT_TO_WANDB}"
require_bool DYNAMIC_TRAJECTORY_SCHEDULING "${DYNAMIC_TRAJECTORY_SCHEDULING}"
require_bool RETRY_TRUNCATED_TRAJECTORIES "${RETRY_TRUNCATED_TRAJECTORIES}"
require_bool IGNORE_TRUNCATED_SAMPLES "${IGNORE_TRUNCATED_SAMPLES}"
require_bool EXCLUDE_TRUNCATED_FROM_ADVANTAGE_STATS "${EXCLUDE_TRUNCATED_FROM_ADVANTAGE_STATS}"
require_bool RECOVER_REPETITION_TRUNCATIONS "${RECOVER_REPETITION_TRUNCATIONS}"
require_positive_int REPETITION_MIN_BLOCK_TOKENS "${REPETITION_MIN_BLOCK_TOKENS}"
require_positive_int REPETITION_MAX_BLOCK_TOKENS "${REPETITION_MAX_BLOCK_TOKENS}"
require_positive_int REPETITION_MIN_REPEATS "${REPETITION_MIN_REPEATS}"
require_positive_int REPETITION_MIN_PREFIX_TOKENS "${REPETITION_MIN_PREFIX_TOKENS}"
require_non_negative_int REPETITION_TAIL_TOLERANCE_TOKENS "${REPETITION_TAIL_TOLERANCE_TOKENS}"
require_bool REPETITION_REQUIRE_COMPLETE_STEP "${REPETITION_REQUIRE_COMPLETE_STEP}"
(( REPETITION_MAX_BLOCK_TOKENS >= REPETITION_MIN_BLOCK_TOKENS )) || fail "REPETITION_MAX_BLOCK_TOKENS must be >= REPETITION_MIN_BLOCK_TOKENS"
(( REPETITION_MIN_REPEATS >= 2 )) || fail "REPETITION_MIN_REPEATS must be >= 2"
require_bool DYNAMIC_PROMPT_GROUP_REFILL "${DYNAMIC_PROMPT_GROUP_REFILL}"
[[ "${TRUNCATED_REWARD}" == "zero" || "${TRUNCATED_REWARD}" == "keep" || "${TRUNCATED_REWARD}" == "soft" ]] || \
    fail "TRUNCATED_REWARD must be zero, keep, or soft"
if [[ "${IGNORE_TRUNCATED_SAMPLES}" == "true" && "${TRUNCATED_REWARD}" != "zero" ]]; then
    fail "IGNORE_TRUNCATED_SAMPLES=true requires TRUNCATED_REWARD=zero"
fi
if [[ "${EXCLUDE_TRUNCATED_FROM_ADVANTAGE_STATS}" == "true" && "${IGNORE_TRUNCATED_SAMPLES}" != "true" ]]; then
    fail "EXCLUDE_TRUNCATED_FROM_ADVANTAGE_STATS=true requires IGNORE_TRUNCATED_SAMPLES=true"
fi
[[ "${OVERLONG_PENALTY_FACTOR}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
    fail "OVERLONG_PENALTY_FACTOR must be non-negative numeric, got ${OVERLONG_PENALTY_FACTOR}"
if [[ "${TRUNCATED_REWARD}" == "soft" ]]; then
    (( OVERLONG_BUFFER_LEN <= MAX_NEW_TOKENS )) || fail \
        "OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN} must not exceed MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
fi
require_positive_int MSE_STEPS "${MSE_STEPS}"
require_non_negative_int WARMUP_STEPS "${WARMUP_STEPS}"
[[ "${LR_SCHEDULER_TYPE}" == "constant" || "${LR_SCHEDULER_TYPE}" == "linear" || "${LR_SCHEDULER_TYPE}" == "cosine" ]] || \
    fail "LR_SCHEDULER_TYPE must be constant, linear, or cosine, got ${LR_SCHEDULER_TYPE}"
require_non_negative_int SAVE_INTERVAL "${SAVE_INTERVAL}"
[[ "${MAX_IMAGES}" =~ ^[0-9]+$ ]] || fail "MAX_IMAGES must be a non-negative integer, got ${MAX_IMAGES}"

TRAJECTORIES_PER_ROLLOUT=$((BATCH_SIZE * SAMPLES_PER_PROMPT))
if (( TRAJECTORIES_PER_ROLLOUT % NUM_DEVICES != 0 )); then
    fail "BATCH_SIZE*SAMPLES_PER_PROMPT=${TRAJECTORIES_PER_ROLLOUT} must be divisible by NUM_DEVICES=${NUM_DEVICES}"
fi
if (( GRADIENT_ACCUMULATION_STEPS > 1 && NUM_UPDATES_PER_BATCH > 1 )); then
    fail "GRADIENT_ACCUMULATION_STEPS>1 requires NUM_UPDATES_PER_BATCH=1"
fi
if (( GRADIENT_ACCUMULATION_STEPS > 1 )); then
    OPTIMIZER_TOTAL_STEPS=$(((NUM_ROLLOUTS + GRADIENT_ACCUMULATION_STEPS - 1) / GRADIENT_ACCUMULATION_STEPS))
else
    OPTIMIZER_TOTAL_STEPS=$((NUM_ROLLOUTS * NUM_UPDATES_PER_BATCH))
fi

if (( IMAGE_SIZE % 32 != 0 )); then
    fail "IMAGE_SIZE must be divisible by 32 for patch_size=16 and merge_size=2; got ${IMAGE_SIZE}"
fi
if (( DIFFUSION_STEPS * 2 / 10 < MSE_STEPS )); then
    fail "DIFFUSION_STEPS=${DIFFUSION_STEPS} leaves fewer than MSE_STEPS=${MSE_STEPS} steps in scheduler fraction [0, 0.2)"
fi

if [ -n "${LOAD_DIR}" ]; then
    LOAD_DIR="$(cd "${LOAD_DIR}" 2>/dev/null && pwd)" || fail "LOAD_DIR is not a directory: ${LOAD_DIR}"
    if [ ! -f "${LOAD_DIR}/checkpoint.pt" ] && [ ! -f "${LOAD_DIR}/.metadata" ]; then
        fail "LOAD_DIR is not a recognized torch/DCP checkpoint: ${LOAD_DIR}"
    fi
elif [ -d "${EXPERIMENT_DIR}" ] && [ -n "$(find "${EXPERIMENT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ] \
    && [ "${ALLOW_EXISTING_RUN}" != "true" ]; then
    fail "experiment already contains outputs: ${EXPERIMENT_DIR}; set LOAD_DIR to resume or ALLOW_EXISTING_RUN=true"
fi

if [ "${DRY_RUN:-0}" != "1" ]; then
    "${PYTHON_BIN}" - "${DATA_PATH}" "${BATCH_SIZE}" "${NUM_DEVICES}" <<'PYDATA'
import json
import sys

import torch

data_path, batch_size, num_devices = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
count = 0
with open(data_path, encoding="utf-8") as handle:
    for line_no, line in enumerate(handle, 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not str(item.get("prompt", "")).strip():
            raise SystemExit(f"{data_path}:{line_no}: missing prompt")
        metadata = item.get("metadata")
        if not isinstance(metadata, dict) or "answer" not in metadata:
            raise SystemExit(f"{data_path}:{line_no}: missing metadata.answer required by DashScope judge")
        count += 1
if count < batch_size:
    raise SystemExit(f"dataset has {count} samples, fewer than batch_size={batch_size}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the selected Python environment")
if torch.cuda.device_count() < num_devices:
    raise SystemExit(f"requested num_devices={num_devices}, but only {torch.cuda.device_count()} CUDA devices are visible")
try:
    with torch.inference_mode():
        probe = torch.randn(2, 3, 16, 16, device="cuda", dtype=torch.bfloat16)
        conv = torch.nn.Conv2d(3, 32, 3, device="cuda", dtype=torch.bfloat16)
        conv(probe)
        torch.cuda.synchronize()
except RuntimeError as exc:
    raise SystemExit(f"cuDNN Conv2d preflight failed: {exc}") from exc
print(
    f"[geoweave-train] preflight: dataset_samples={count}, "
    f"visible_gpus={torch.cuda.device_count()}, cudnn={torch.backends.cudnn.version()}"
)
PYDATA

    if [ "${REQUIRE_IDLE_GPUS}" = "true" ] && command -v nvidia-smi >/dev/null 2>&1; then
        GPU_QUERY=(nvidia-smi)
        if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
            GPU_QUERY+=(--id="${CUDA_VISIBLE_DEVICES}")
        else
            GPU_QUERY+=(--id="$(seq -s, 0 $((NUM_DEVICES - 1)))")
        fi
        BUSY_GPU_PROCS="$("${GPU_QUERY[@]}" --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"
        [ -z "${BUSY_GPU_PROCS}" ] || fail "GPU compute processes already exist; stop them or set REQUIRE_IDLE_GPUS=false:
${BUSY_GPU_PROCS}"
    fi
fi

CMD=(
    "${PYTHON_BIN}" -m unirl.train_unified_model
    "--config-name=${CONFIG_NAME}"
    "+devices_per_node=${NUM_DEVICES}"
    "num_devices=${NUM_DEVICES}"
    "batch_size=${BATCH_SIZE}"
    "num_rollouts=${NUM_ROLLOUTS}"
    "save_interval=${SAVE_INTERVAL}"
    "save_dir=${SAVE_DIR}"
    "logging.backend=${LOGGING_BACKEND}"
    "logging.report_to_tensorboard=${REPORT_TO_TENSORBOARD}"
    "logging.report_to_wandb=${REPORT_TO_WANDB}"
    "logging.logging_dir=${TENSORBOARD_LOG_DIR}"
    "logging.project_name=${WANDB_PROJECT}"
    "logging.run_name=${RUN_NAME}"
    "hydra.run.dir=${HYDRA_OUTPUT_DIR}"
    "bundle.config.base_shift=3.0"
    "bundle.config.model_precision=bf16"
    "bundle.config.autocast_precision=bf16"
    "bundle.config.trajectory_precision=bf16"
    "bundle.config.logprob_precision=fp32"
    "bundle.config.freeze_und=false"
    "bundle.config.freeze_gen=true"
    "bundle.config.freeze_fm_modules=true"
    "bundle.config.use_lora=false"
    "pipeline.autocast_precision=bf16"
    "pipeline.trajectory_precision=bf16"
    "pipeline.logprob_precision=fp32"
    "enable_fsdp_offload=false"
    "backend.fsdp_cfg.activation_checkpointing=true"
    "backend.fsdp_cfg.reshard_after_forward=false"
    "backend.fsdp_cfg.root_wrap=false"
    "backend.fsdp_cfg.cpu_offload=false"
    "backend.fsdp_cfg.use_torch_compile=false"
    "backend.fsdp_cfg.checkpoint_format=${CHECKPOINT_FORMAT}"
    "backend.optimizer_cfg.learning_rate=${LEARNING_RATE}"
    "backend.scheduler_cfg.type=${LR_SCHEDULER_TYPE}"
    "backend.scheduler_cfg.warmup_steps=${WARMUP_STEPS}"
    "reward.truncated_reward=${TRUNCATED_REWARD}"
    "reward.overlong_buffer_len=${OVERLONG_BUFFER_LEN}"
    "reward.overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR}"
    "algorithm.ar.clip_range=${CLIP_RANGE}"
    "algorithm.ar.loss_agg_mode=${LOSS_AGG_MODE}"
    "algorithm.ar.ar_kl_coef=${AR_KL_COEF}"
    "algorithm.ar.mse_weight=${MSE_WEIGHT}"
    "algorithm.ar.horizon=${MAX_NEW_TOKENS}"
    "algorithm.ar.train_reshard_after_forward=false"
    "stack.micro_batch_size=${MICRO_BATCH_SIZE}"
    "stack.num_updates_per_batch=${NUM_UPDATES_PER_BATCH}"
    "stack.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
    "stack.max_grad_norm=${MAX_GRAD_NORM}"
    "sampling.ar.samples_per_prompt=${SAMPLES_PER_PROMPT}"
    "sampling.ar.max_new_tokens=${MAX_NEW_TOKENS}"
    "stage_config.rollout_text_batch_size=${ROLLOUT_TEXT_BATCH_SIZE}"
    "stage_config.dynamic_trajectory_scheduling=${DYNAMIC_TRAJECTORY_SCHEDULING}"
    "stage_config.dynamic_rollout_chunk_size=${DYNAMIC_ROLLOUT_CHUNK_SIZE}"
    "stage_config.continuous_batching=${CONTINUOUS_BATCHING}"
    "stage_config.persistent_worker_session=${PERSISTENT_WORKER_SESSION}"
    "stage_config.continuous_request_admission=${CONTINUOUS_REQUEST_ADMISSION}"
    "stage_config.continuous_live_admission=${CONTINUOUS_LIVE_ADMISSION}"
    "stage_config.continuous_prompt_microbundle_size=${CONTINUOUS_PROMPT_MICROBUNDLE_SIZE}"
    "stage_config.continuous_session_window_size=${CONTINUOUS_SESSION_WINDOW_SIZE}"
    "stage_config.continuous_rollout_pool_size=${CONTINUOUS_ROLLOUT_POOL_SIZE}"
    "stage_config.continuous_text_batch_size=${CONTINUOUS_TEXT_BATCH_SIZE}"
    "stage_config.dynamic_reward_workers=${DYNAMIC_REWARD_WORKERS}"
    "stage_config.retry_truncated_trajectories=${RETRY_TRUNCATED_TRAJECTORIES}"
    "stage_config.ignore_truncated_samples=${IGNORE_TRUNCATED_SAMPLES}"
    "stage_config.exclude_truncated_from_advantage_stats=${EXCLUDE_TRUNCATED_FROM_ADVANTAGE_STATS}"
    "stage_config.recover_repetition_truncations=${RECOVER_REPETITION_TRUNCATIONS}"
    "stage_config.repetition_min_block_tokens=${REPETITION_MIN_BLOCK_TOKENS}"
    "stage_config.repetition_max_block_tokens=${REPETITION_MAX_BLOCK_TOKENS}"
    "stage_config.repetition_min_repeats=${REPETITION_MIN_REPEATS}"
    "stage_config.repetition_min_prefix_tokens=${REPETITION_MIN_PREFIX_TOKENS}"
    "stage_config.repetition_tail_tolerance_tokens=${REPETITION_TAIL_TOLERANCE_TOKENS}"
    "stage_config.repetition_require_complete_step=${REPETITION_REQUIRE_COMPLETE_STEP}"
    "stage_config.sca_outcome_correct_credit=${SCA_OUTCOME_CORRECT_CREDIT}"
    "stage_config.sca_process_error_penalty=${SCA_PROCESS_ERROR_PENALTY}"
    "stage_config.sca_advantage_eps=${SCA_ADVANTAGE_EPS}"
    "stage_config.max_attempts_per_candidate=${MAX_ATTEMPTS_PER_CANDIDATE}"
    "stage_config.max_truncated_refills_per_prompt=${MAX_TRUNCATED_REFILLS_PER_PROMPT}"
    "stage_config.dynamic_prompt_group_refill=${DYNAMIC_PROMPT_GROUP_REFILL}"
    "stage_config.reserve_prompt_count=${RESERVE_PROMPT_COUNT}"
    "stage_config.max_replacement_prompts=${MAX_REPLACEMENT_PROMPTS}"
    "stage_config.max_trajectory_multiplier=${MAX_TRAJECTORY_MULTIPLIER}"
    "dump_async=${DUMP_ASYNC}"
    "dump_image_workers=${DUMP_IMAGE_WORKERS}"
    "dump_jpeg_quality=${DUMP_JPEG_QUALITY}"
    "dump_max_pending=${DUMP_MAX_PENDING}"
    "sampling.diffusion.height=${IMAGE_SIZE}"
    "sampling.diffusion.width=${IMAGE_SIZE}"
    "sampling.diffusion.max_images=${MAX_IMAGES}"
    "sampling.diffusion.rollout_diffusion_batch_size=${ROLLOUT_DIFFUSION_BATCH_SIZE}"
    "sampling.diffusion.rollout_reencode_batch_size=${ROLLOUT_REENCODE_BATCH_SIZE}"
    "sampling.diffusion.num_inference_steps=${DIFFUSION_STEPS}"
    "sampling.diffusion.timestep_shift=1.0"
    "sampling.diffusion.eta=0.0"
    "sampling.diffusion.scheduler.num_sde_steps=${MSE_STEPS}"
)

if [ -n "${LOAD_DIR}" ]; then
    CMD+=("+load_dir=${LOAD_DIR}")
fi
if [ -n "${DUMP_DIR}" ]; then
    CMD+=("dump_dir=${DUMP_DIR}")
else
    CMD+=("dump_dir=null")
fi
CMD+=("$@")

echo "[geoweave-train] repository: ${REPO_ROOT}"
echo "[geoweave-train] model:      ${MODEL_PATH}"
echo "[geoweave-train] data:       ${DATA_PATH}"
echo "[geoweave-train] outcome judge: dashscope/${ANSWER_JUDGE_MODEL}"
echo "[geoweave-train] process judge: ${JUDGE_BASE_URL:-<not-set>} (${SCA_JUDGE_MODEL}), critiques=${SCA_NUM_CRITIQUES}, voting=${SCA_VOTING}, reasoning=${SCA_JUDGE_REASONING_EFFORT}, format=${SCA_JUDGE_RESPONSE_FORMAT}, workers=${SCA_JUDGE_MAX_WORKERS}, qps=${SCA_JUDGE_QPS}"
echo "[geoweave-train] text reward: policy=${TRUNCATED_REWARD}, Lmax=${MAX_NEW_TOKENS}, Lcache=${OVERLONG_BUFFER_LEN}, factor=${OVERLONG_PENALTY_FACTOR}"
echo "[geoweave-train] repetition recovery: enabled=${RECOVER_REPETITION_TRUNCATIONS}, block=${REPETITION_MIN_BLOCK_TOKENS}-${REPETITION_MAX_BLOCK_TOKENS}, repeats=${REPETITION_MIN_REPEATS}, prefix>=${REPETITION_MIN_PREFIX_TOKENS}, tail_tol=${REPETITION_TAIL_TOLERANCE_TOKENS}, step_backoff=${REPETITION_REQUIRE_COMPLETE_STEP}"
echo "[geoweave-train] experiment: ${EXPERIMENT_DIR}"
if [ -n "${LOAD_DIR}" ]; then
    echo "[geoweave-train] resume:     ${LOAD_DIR} (num_rollouts=${NUM_ROLLOUTS} is total budget)"
fi
echo "[geoweave-train] save_dir:   ${SAVE_DIR}"
echo "[geoweave-train] log_file:   ${LOG_FILE}"
echo "[geoweave-train] hydra_dir:  ${HYDRA_OUTPUT_DIR}"
echo "[geoweave-train] metrics:    ${LOGGING_BACKEND}"
if [ "${LOGGING_BACKEND}" = "tensorboard" ] && [ "${REPORT_TO_TENSORBOARD}" = "true" ]; then
    echo "[geoweave-train] tb_dir:     ${TENSORBOARD_LOG_DIR}"
fi
if [ -n "${DUMP_DIR}" ]; then
    echo "[geoweave-train] dump_dir:   ${DUMP_DIR}"
fi
echo "[geoweave-train] settings:   devices=${NUM_DEVICES}, batch=${BATCH_SIZE}, spp=${SAMPLES_PER_PROMPT}, trajectories=${TRAJECTORIES_PER_ROLLOUT}, grad_accum=${GRADIENT_ACCUMULATION_STEPS}, optimizer_steps=${OPTIMIZER_TOTAL_STEPS}, tokens=${MAX_NEW_TOKENS}, max_images=${MAX_IMAGES}, rollout_batches=${ROLLOUT_TEXT_BATCH_SIZE}/${ROLLOUT_DIFFUSION_BATCH_SIZE}/${ROLLOUT_REENCODE_BATCH_SIZE}, dump=${ENABLE_DUMP}/${DUMP_ASYNC}/${DUMP_IMAGE_WORKERS}x/jpeg-q${DUMP_JPEG_QUALITY}/pending-${DUMP_MAX_PENDING}, dynamic_queue=${DYNAMIC_TRAJECTORY_SCHEDULING}/chunk-${DYNAMIC_ROLLOUT_CHUNK_SIZE}, continuous=${CONTINUOUS_BATCHING}/persistent-${PERSISTENT_WORKER_SESSION}/admission-${CONTINUOUS_REQUEST_ADMISSION}/live-${CONTINUOUS_LIVE_ADMISSION}/microbundle-${CONTINUOUS_PROMPT_MICROBUNDLE_SIZE}/window-${CONTINUOUS_SESSION_WINDOW_SIZE}/pool-${CONTINUOUS_ROLLOUT_POOL_SIZE}/text-${CONTINUOUS_TEXT_BATCH_SIZE}, reward_workers=${DYNAMIC_REWARD_WORKERS}, trunc_retry=${RETRY_TRUNCATED_TRAJECTORIES}/${MAX_ATTEMPTS_PER_CANDIDATE}/prompt-${MAX_TRUNCATED_REFILLS_PER_PROMPT}, trunc_grad_mask=${IGNORE_TRUNCATED_SAMPLES}, trunc_adv_exclude=${EXCLUDE_TRUNCATED_FROM_ADVANTAGE_STATS}, group_refill=${DYNAMIC_PROMPT_GROUP_REFILL}/reserve-${RESERVE_PROMPT_COUNT}/replace-${MAX_REPLACEMENT_PROMPTS}/traj-x${MAX_TRAJECTORY_MULTIPLIER}, pixel_budget=${IMAGE_SIZE}^2, diffusion_steps=${DIFFUSION_STEPS}, clip_range=${CLIP_RANGE}, loss_agg=${LOSS_AGG_MODE}, ar_kl_coef=${AR_KL_COEF}, lr=${LEARNING_RATE}/${LR_SCHEDULER_TYPE}/warmup-${WARMUP_STEPS}, mse_weight=${MSE_WEIGHT}, checkpoint=${CHECKPOINT_FORMAT}, mem_profile=${SENSENOVA_MEM_PROFILE}"
echo "[geoweave-train] command:"
printf '  %q' "${CMD[@]}"
echo

if [ "${DRY_RUN:-0}" = "1" ]; then
    exit 0
fi

mkdir -p "${SAVE_DIR}" "${LOG_DIR}" "${HYDRA_OUTPUT_DIR}"
if [ "${LOGGING_BACKEND}" = "tensorboard" ] && [ "${REPORT_TO_TENSORBOARD}" = "true" ]; then
    mkdir -p "${TENSORBOARD_LOG_DIR}"
fi
if [ -n "${DUMP_DIR}" ]; then
    mkdir -p "${DUMP_DIR}"
fi

if command -v flock >/dev/null 2>&1; then
    exec 9>"${EXPERIMENT_DIR}/.train.lock"
    flock -n 9 || fail "another training process holds ${EXPERIMENT_DIR}/.train.lock"
fi

if [ -n "${LOAD_DIR}" ]; then
    "${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
    "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
fi
