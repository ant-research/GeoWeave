#!/usr/bin/env bash
#
# Multi-node GeoWeave Interleave-RL launcher.
#
# Run this same script once on every job node (SPMD). Rank 0 starts the Ray
# head and the single UniRL driver; every other node joins Ray and waits. The
# repository, model, dataset, experiment directory, and checkpoints must be on
# storage mounted at the same path on every node.
#
# Default topology is 4x8 (32 GPUs):
#   NUM_NODES=4 GPUS_PER_NODE=8 NODE_RANK=0 HEAD_IP=<head-ip> ...  # head
#   NUM_NODES=4 GPUS_PER_NODE=8 NODE_RANK=1 HEAD_IP=<head-ip> ...  # worker 1
#   NUM_NODES=4 GPUS_PER_NODE=8 NODE_RANK=2 HEAD_IP=<head-ip> ...  # worker 2
#   NUM_NODES=4 GPUS_PER_NODE=8 NODE_RANK=3 HEAD_IP=<head-ip> ...  # worker 3
#
# Taiji/PyTorchJob-style INDEX/CHIEF_IP/RANK/MASTER_ADDR variables are accepted
# as fallbacks. Extra positional arguments are forwarded as Hydra overrides.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-RL}"
CONFIG_NAME="${CONFIG_NAME:-unified_model/geoweave_interleave_rl}"

NUM_NODES="${NUM_NODES:-${HOST_NUM:-4}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${HOST_GPU_NUM:-8}}"
DEVICES_PER_NODE="${DEVICES_PER_NODE:-${GPUS_PER_NODE}}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_CLUSTER_WAIT_S="${RAY_CLUSTER_WAIT_S:-60}"

MODEL_PATH="${MODEL_PATH:-${SENSENOVA_U1_PATH:-/path/to/GeoWeave-HF}}"
DATA_PATH="${DATA_PATH:-/path/to/train.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-}"

TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
BATCH_SIZE="${BATCH_SIZE:-${TOTAL_GPUS}}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-8}"
# The formal dataset contains 9,203 prompts. With global batch=32, 720
# rollouts consume 23,040 prompt slots (approximately 2.50 dataset epochs) and,
# with two updates per rollout, produce 1,440 optimizer steps.
NUM_ROLLOUTS="${NUM_ROLLOUTS:-720}"
NUM_UPDATES_PER_BATCH="${NUM_UPDATES_PER_BATCH:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-512}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"
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
# IGNORE_TRUNCATED
RETRY_TRUNCATED_TRAJECTORIES="${RETRY_TRUNCATED_TRAJECTORIES:-false}"
TRUNCATED_REWARD="${TRUNCATED_REWARD:-keep}"
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
if [[ -z "${RESERVE_PROMPT_COUNT+x}" ]]; then
    if [[ "${DYNAMIC_PROMPT_GROUP_REFILL}" == "true" ]]; then
        RESERVE_PROMPT_COUNT="${BATCH_SIZE}"
    else
        RESERVE_PROMPT_COUNT=0
    fi
fi
MAX_REPLACEMENT_PROMPTS="${MAX_REPLACEMENT_PROMPTS:-12}"
MAX_TRAJECTORY_MULTIPLIER="${MAX_TRAJECTORY_MULTIPLIER:-1.5}"
# PPO clipping: lower ratio floor is 1-CLIP_RANGE; upper ratio ceiling is
# 1+CLIP_RANGE_HIGH (DAPO clip-higher). Default remains symmetric.
CLIP_RANGE="${CLIP_RANGE:-0.02}"
CLIP_RANGE_HIGH="${CLIP_RANGE_HIGH:-0.05}"
POLICY_ENTROPY_INTERVAL="${POLICY_ENTROPY_INTERVAL:-10}"
CLIP_SCHEDULE="${CLIP_SCHEDULE:-constant}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"
AR_KL_COEF="${AR_KL_COEF:-0.001}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
MSE_STEPS="${MSE_STEPS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
# Enable a 20-optimizer-step linear warmup by default. Constant scheduling ignores
# WARMUP_STEPS, so keep LR_SCHEDULER_TYPE=linear when warmup is required.
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
# Checkpoint intervals are measured in rollouts; with two updates per rollout,
# 100 rollouts correspond to approximately 200 optimizer steps.
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
CHECKPOINT_FORMAT="${CHECKPOINT_FORMAT:-dcp}"
LOAD_DIR="${LOAD_DIR:-}"
AUTO_RESUME="${AUTO_RESUME:-false}"
ENABLE_DUMP="${ENABLE_DUMP:-true}"
DUMP_ASYNC="${DUMP_ASYNC:-true}"
DUMP_IMAGE_WORKERS="${DUMP_IMAGE_WORKERS:-8}"
DUMP_JPEG_QUALITY="${DUMP_JPEG_QUALITY:-90}"
DUMP_MAX_PENDING="${DUMP_MAX_PENDING:-1}"

RUN_NAME="${RUN_NAME:-geoweave_multinode_${PAI_JOB_ID:-manual}}"
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-${REPO_ROOT}/experiments}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${EXPERIMENTS_DIR}/${RUN_NAME}}"
SAVE_DIR="${SAVE_DIR:-${EXPERIMENT_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${EXPERIMENT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-${EXPERIMENT_DIR}/runs}"
HYDRA_OUTPUT_DIR="${HYDRA_OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs}"
DUMP_DIR="${DUMP_DIR:-${EXPERIMENT_DIR}/dumps}"
LOGGING_BACKEND="${LOGGING_BACKEND:-tensorboard}"
REPORT_TO_TENSORBOARD="${REPORT_TO_TENSORBOARD:-true}"
REPORT_TO_WANDB="${REPORT_TO_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-geoweave_interleave_rl}"
ANSWER_JUDGE_MODEL="${ANSWER_JUDGE_MODEL:-qwen3.7-max}"
SCA_JUDGE_MODEL="${SCA_JUDGE_MODEL:-gemini-3.5-flash}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://your-judge-endpoint.example.com/v1/chat/completions}"
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

fail() {
    echo "[geoweave-multinode] ERROR: $*" >&2
    exit 2
}

require_positive_int() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer, got ${value}"
}

require_non_negative_int() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] || fail "${name} must be a non-negative integer, got ${value}"
}

require_bool() {
    local name="$1" value="$2"
    [[ "${value}" == "true" || "${value}" == "false" ]] || fail "${name} must be true or false, got ${value}"
}

for spec in \
    "NUM_NODES:${NUM_NODES}" \
    "GPUS_PER_NODE:${GPUS_PER_NODE}" \
    "DEVICES_PER_NODE:${DEVICES_PER_NODE}" \
    "BATCH_SIZE:${BATCH_SIZE}" \
    "SAMPLES_PER_PROMPT:${SAMPLES_PER_PROMPT}" \
    "NUM_ROLLOUTS:${NUM_ROLLOUTS}" \
    "NUM_UPDATES_PER_BATCH:${NUM_UPDATES_PER_BATCH}" \
    "GRADIENT_ACCUMULATION_STEPS:${GRADIENT_ACCUMULATION_STEPS}" \
    "MICRO_BATCH_SIZE:${MICRO_BATCH_SIZE}" \
    "MAX_NEW_TOKENS:${MAX_NEW_TOKENS}" \
    "OVERLONG_BUFFER_LEN:${OVERLONG_BUFFER_LEN}" \
    "IMAGE_SIZE:${IMAGE_SIZE}" \
    "DIFFUSION_STEPS:${DIFFUSION_STEPS}" \
    "ROLLOUT_TEXT_BATCH_SIZE:${ROLLOUT_TEXT_BATCH_SIZE}" \
    "ROLLOUT_DIFFUSION_BATCH_SIZE:${ROLLOUT_DIFFUSION_BATCH_SIZE}" \
    "ROLLOUT_REENCODE_BATCH_SIZE:${ROLLOUT_REENCODE_BATCH_SIZE}" \
    "DYNAMIC_REWARD_WORKERS:${DYNAMIC_REWARD_WORKERS}" \
    "DYNAMIC_ROLLOUT_CHUNK_SIZE:${DYNAMIC_ROLLOUT_CHUNK_SIZE}" \
    "CONTINUOUS_ROLLOUT_POOL_SIZE:${CONTINUOUS_ROLLOUT_POOL_SIZE}" \
    "CONTINUOUS_PROMPT_MICROBUNDLE_SIZE:${CONTINUOUS_PROMPT_MICROBUNDLE_SIZE}" \
    "CONTINUOUS_SESSION_WINDOW_SIZE:${CONTINUOUS_SESSION_WINDOW_SIZE}" \
    "CONTINUOUS_TEXT_BATCH_SIZE:${CONTINUOUS_TEXT_BATCH_SIZE}" \
    "MAX_ATTEMPTS_PER_CANDIDATE:${MAX_ATTEMPTS_PER_CANDIDATE}" \
    "MSE_STEPS:${MSE_STEPS}" \
    "DUMP_IMAGE_WORKERS:${DUMP_IMAGE_WORKERS}" \
    "DUMP_MAX_PENDING:${DUMP_MAX_PENDING}"; do
    require_positive_int "${spec%%:*}" "${spec#*:}"
done
require_non_negative_int SAVE_INTERVAL "${SAVE_INTERVAL}"
require_non_negative_int POLICY_ENTROPY_INTERVAL "${POLICY_ENTROPY_INTERVAL}"
require_non_negative_int MAX_TRUNCATED_REFILLS_PER_PROMPT "${MAX_TRUNCATED_REFILLS_PER_PROMPT}"
require_non_negative_int RESERVE_PROMPT_COUNT "${RESERVE_PROMPT_COUNT}"
require_non_negative_int MAX_REPLACEMENT_PROMPTS "${MAX_REPLACEMENT_PROMPTS}"
require_bool AUTO_RESUME "${AUTO_RESUME}"
require_bool ENABLE_DUMP "${ENABLE_DUMP}"
require_bool DUMP_ASYNC "${DUMP_ASYNC}"
require_bool DYNAMIC_TRAJECTORY_SCHEDULING "${DYNAMIC_TRAJECTORY_SCHEDULING}"
require_bool CONTINUOUS_BATCHING "${CONTINUOUS_BATCHING}"
require_bool PERSISTENT_WORKER_SESSION "${PERSISTENT_WORKER_SESSION}"
require_bool CONTINUOUS_REQUEST_ADMISSION "${CONTINUOUS_REQUEST_ADMISSION}"
require_bool CONTINUOUS_LIVE_ADMISSION "${CONTINUOUS_LIVE_ADMISSION}"
require_positive_int CONTINUOUS_SESSION_WINDOW_SIZE "${CONTINUOUS_SESSION_WINDOW_SIZE}"
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
require_bool REPORT_TO_TENSORBOARD "${REPORT_TO_TENSORBOARD}"
require_bool REPORT_TO_WANDB "${REPORT_TO_WANDB}"
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
[[ "${CLIP_RANGE}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
    fail "CLIP_RANGE must be non-negative numeric, got ${CLIP_RANGE}"
[[ "${CLIP_RANGE_HIGH}" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
    fail "CLIP_RANGE_HIGH must be non-negative numeric, got ${CLIP_RANGE_HIGH}"
[[ "${CLIP_SCHEDULE}" == "constant" || "${CLIP_SCHEDULE}" == "linear_decay" || "${CLIP_SCHEDULE}" == "cosine_decay" ]] || \
    fail "CLIP_SCHEDULE must be constant, linear_decay, or cosine_decay, got ${CLIP_SCHEDULE}"
if [[ "${TRUNCATED_REWARD}" == "soft" ]]; then
    (( OVERLONG_BUFFER_LEN <= MAX_NEW_TOKENS )) || fail \
        "OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN} must not exceed MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
fi
[[ "${DEVICES_PER_NODE}" -eq "${GPUS_PER_NODE}" ]] || fail "DEVICES_PER_NODE must equal GPUS_PER_NODE"
[[ "${IMAGE_SIZE}" -gt 0 && $((IMAGE_SIZE % 32)) -eq 0 ]] || fail "IMAGE_SIZE must be divisible by 32"
[[ "${MAX_IMAGES}" =~ ^[0-9]+$ ]] || fail "MAX_IMAGES must be non-negative"
[[ "${DUMP_JPEG_QUALITY}" =~ ^[1-9][0-9]?$|^100$ ]] || fail "DUMP_JPEG_QUALITY must be in [1, 100]"
[[ "${MAX_TRAJECTORY_MULTIPLIER}" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail \
    "MAX_TRAJECTORY_MULTIPLIER must be numeric, got ${MAX_TRAJECTORY_MULTIPLIER}"
[[ "${CHECKPOINT_FORMAT}" == "dcp" || "${CHECKPOINT_FORMAT}" == "torch" ]] || fail "CHECKPOINT_FORMAT must be dcp or torch"
[[ "${LOGGING_BACKEND}" == "tensorboard" || "${LOGGING_BACKEND}" == "wandb" ]] || fail "LOGGING_BACKEND must be tensorboard or wandb"
[[ "${LOSS_AGG_MODE}" == "token-mean" || "${LOSS_AGG_MODE}" == "seq-mean-token-mean" || "${LOSS_AGG_MODE}" == "seq-mean-token-sum-norm" ]] || \
    fail "LOSS_AGG_MODE must be token-mean, seq-mean-token-mean, or seq-mean-token-sum-norm; got ${LOSS_AGG_MODE}"
[[ -n "${DASHSCOPE_API_KEY:-}" ]] || fail "DASHSCOPE_API_KEY is required for the Qwen outcome judge"
[[ -n "${API_KEY_ENV:-}" ]] || fail "API_KEY_ENV is required for the SCA process judge"
require_positive_int SCA_JUDGE_MAX_TOKENS "${SCA_JUDGE_MAX_TOKENS}"
require_positive_int SCA_NUM_CRITIQUES "${SCA_NUM_CRITIQUES}"
require_positive_int SCA_JUDGE_MAX_WORKERS "${SCA_JUDGE_MAX_WORKERS}"
require_bool SCA_KEEP_RAW_OUTPUTS "${SCA_KEEP_RAW_OUTPUTS}"
[[ "${SCA_VOTING}" == "greedy" || "${SCA_VOTING}" == "majority" || "${SCA_VOTING}" == "intersection" || "${SCA_VOTING}" == "union" || "${SCA_VOTING}" == "average" ]] || \
    fail "SCA_VOTING must be greedy, majority, intersection, union, or average; got ${SCA_VOTING}"

TRAJECTORIES_PER_ROLLOUT=$((BATCH_SIZE * SAMPLES_PER_PROMPT))
(( TRAJECTORIES_PER_ROLLOUT % TOTAL_GPUS == 0 )) || fail \
    "BATCH_SIZE*SAMPLES_PER_PROMPT=${TRAJECTORIES_PER_ROLLOUT} must be divisible by total GPUs=${TOTAL_GPUS}"
if (( GRADIENT_ACCUMULATION_STEPS > 1 && NUM_UPDATES_PER_BATCH > 1 )); then
    fail "GRADIENT_ACCUMULATION_STEPS>1 requires NUM_UPDATES_PER_BATCH=1"
fi
if (( GRADIENT_ACCUMULATION_STEPS > 1 )); then
    OPTIMIZER_TOTAL_STEPS=$(((NUM_ROLLOUTS + GRADIENT_ACCUMULATION_STEPS - 1) / GRADIENT_ACCUMULATION_STEPS))
else
    OPTIMIZER_TOTAL_STEPS=$((NUM_ROLLOUTS * NUM_UPDATES_PER_BATCH))
fi
(( DIFFUSION_STEPS * 2 / 10 >= MSE_STEPS )) || fail \
    "DIFFUSION_STEPS=${DIFFUSION_STEPS} leaves fewer than MSE_STEPS=${MSE_STEPS} steps in [0, 0.2)"

# PyTorchJob commonly supplies RANK/WORLD_SIZE/MASTER_ADDR. Taiji-style
# INDEX/CHIEF_IP and explicit NODE_RANK/HEAD_IP are accepted as fallbacks.
NODE_RANK="${NODE_RANK:-${INDEX:-}}"
if [[ -z "${NODE_RANK}" ]]; then
    ROLE_NAME="${PAI_CURRENT_TASK_ROLE_NAME:-${TASK_ROLE:-}}"
    ROLE_INDEX="${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${TASK_INDEX:-0}}"
    case "${ROLE_NAME}" in
        master|chief) NODE_RANK=0 ;;
        worker) NODE_RANK=$((ROLE_INDEX + 1)) ;;
    esac
fi
NODE_RANK="${NODE_RANK:-${RANK:-}}"
[[ "${NODE_RANK}" =~ ^[0-9]+$ && "${NODE_RANK}" -lt "${NUM_NODES}" ]] || fail \
    "cannot determine a valid NODE_RANK; set NODE_RANK=0..$((NUM_NODES - 1))"

head_endpoint="${HEAD_IP:-${CHIEF_IP:-${MASTER_ADDR:-}}}"
[[ -n "${head_endpoint}" ]] || fail "cannot determine Ray head IP; set HEAD_IP or MASTER_ADDR"
resolved_head="$(getent ahostsv4 "${head_endpoint}" 2>/dev/null | awk 'NR == 1 {print $1}')"
HEAD_IP="${resolved_head:-${head_endpoint}}"

all_ips="$(hostname -I 2>/dev/null || true)"
NODE_IP="${NODE_IP:-${LOCAL_IP:-${POD_IP:-}}}"
# Multi-NIC PyPAI nodes expose several 33.*, 200.* and 10.* addresses. Match
# the successful debug launcher: rank 0 prefers the exact resolved head IP;
# other nodes select the route's source IP before falling back to subnet match.
if [[ -z "${NODE_IP}" && "${NODE_RANK}" -eq 0 ]] \
    && echo " ${all_ips} " | grep -Fq " ${HEAD_IP} "; then
    NODE_IP="${HEAD_IP}"
fi
if [[ -z "${NODE_IP}" ]] && command -v ip >/dev/null 2>&1; then
    NODE_IP="$(ip -4 route get "${HEAD_IP}" 2>/dev/null | awk '{for (i=1; i<=NF; i++) if ($i == "src") {print $(i+1); exit}}')"
fi
if [[ -z "${NODE_IP}" && "${HEAD_IP}" =~ ^[0-9]+\.[0-9]+\. ]]; then
    head_subnet="$(echo "${HEAD_IP}" | cut -d. -f1-2)"
    NODE_IP="$(echo "${all_ips}" | tr ' ' '\n' | grep "^${head_subnet}\." | head -1 || true)"
fi
NODE_IP="${NODE_IP:-$(echo "${all_ips}" | awk '{print $1}')}"
[[ -n "${NODE_IP}" ]] || fail "cannot determine this node's IP; set NODE_IP"

# PyTorchJob/Kubemaker injects its own torchrun rendezvous variables (commonly
# MASTER_ADDR=<master-pod-name>, MASTER_PORT=20173).  They are useful above as
# discovery fallbacks, but must not leak into Ray actors: UniRL creates separate
# TCPStores with dynamically selected addresses/ports for its internal process
# groups.  Keep the normalized NODE_RANK/HEAD_IP values and clear the originals
# before any Ray process is started.
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK GROUP_RANK \
    ROLE_RANK ROLE_WORLD_SIZE LOCAL_WORLD_SIZE

[[ -x "${VENV_DIR}/bin/python" ]] || fail "Python executable not found: ${VENV_DIR}/bin/python"
PYTHON_BIN="${VENV_DIR}/bin/python"

# Always launch Ray through the selected virtualenv interpreter.  Some shared
# copies of this venv contain Ray's Python package but no bin/ray console script;
# falling back to the image's /usr/local/bin/ray starts raylets and workers with
# the system Python and can load incompatible system packages.
RAY_CLI_DIR="${RAY_CLI_DIR:-/tmp/unirl-torch280-cli}"
mkdir -p "${RAY_CLI_DIR}"
RAY_BIN="${RAY_CLI_DIR}/ray"
cat > "${RAY_BIN}" <<'RAYCLI'
#!/usr/bin/env bash
exec "$VENV_DIR/bin/python" -m ray.scripts.scripts "$@"
RAYCLI
chmod +x "${RAY_BIN}"
export RAY_BIN

# Prefer the CUDA-12 cuDNN wheel in the shared virtualenv over a host-level
# cuDNN built for another CUDA major version. This export happens before each
# node starts its Ray process, so Ray workers inherit the corrected linker path.
PYTHON_SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
CUDNN_LIB_DIR="${CUDNN_LIB_DIR:-${PYTHON_SITE_PACKAGES}/nvidia/cudnn/lib}"
[[ -f "${CUDNN_LIB_DIR}/libcudnn.so.9" ]] || fail \
    "venv cuDNN library not found: ${CUDNN_LIB_DIR}/libcudnn.so.9"
export CUDNN_LIB_DIR
export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

[[ -d "${MODEL_PATH}" ]] || fail "MODEL_PATH is not visible on node ${NODE_RANK}: ${MODEL_PATH}"
[[ -f "${DATA_PATH}" ]] || fail "DATA_PATH is not visible on node ${NODE_RANK}: ${DATA_PATH}"
if [[ -n "${EVAL_DATA_PATH}" ]]; then
    [[ -f "${EVAL_DATA_PATH}" ]] || fail "EVAL_DATA_PATH is not visible on node ${NODE_RANK}: ${EVAL_DATA_PATH}"
fi
if [[ -n "${LOAD_DIR}" ]]; then
    [[ -d "${LOAD_DIR}" ]] || fail "LOAD_DIR is not visible on node ${NODE_RANK}: ${LOAD_DIR}"
fi

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="$(dirname "${RAY_BIN}"):${VENV_DIR}/bin:${PATH}"
unset PYTHONHOME
hash -r 2>/dev/null || true
export SENSENOVA_U1_PATH="${MODEL_PATH}"
export DATA_PATH RUN_NAME EXPERIMENT_DIR TENSORBOARD_LOG_DIR WANDB_PROJECT
export LOGGING_BACKEND REPORT_TO_TENSORBOARD REPORT_TO_WANDB
export ANSWER_JUDGE_MODEL
export SCA_JUDGE_MODEL JUDGE_BASE_URL SCA_JUDGE_TEMPERATURE SCA_JUDGE_TOP_P
export SCA_JUDGE_MAX_TOKENS SCA_JUDGE_REASONING_EFFORT SCA_JUDGE_RESPONSE_FORMAT
export SCA_NUM_CRITIQUES SCA_VOTING SCA_JUDGE_QPS
export SCA_JUDGE_TIMEOUT SCA_JUDGE_MAX_RETRIES SCA_JUDGE_MAX_WORKERS SCA_KEEP_RAW_OUTPUTS
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SENSENOVA_MEM_PROFILE="${SENSENOVA_MEM_PROFILE:-0}"
export SENSENOVA_ROLLOUT_RUN_ID="${SENSENOVA_ROLLOUT_RUN_ID:-${RUN_NAME}}"
if [[ -n "${EVAL_DATA_PATH}" ]]; then
    export EVAL_DATA_PATH
fi

ENV_CHECK_SCRIPT="${REPO_ROOT}/scripts/check_geoweave_rl_env.py"
[[ -f "${ENV_CHECK_SCRIPT}" ]] || fail "environment checker not found: ${ENV_CHECK_SCRIPT}"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${PYTHON_BIN}" "${ENV_CHECK_SCRIPT}" --expected-gpus "${GPUS_PER_NODE}"
fi

DATASET_SIZE="$(grep -cve '^[[:space:]]*$' "${DATA_PATH}")"
(( DATASET_SIZE >= BATCH_SIZE )) || fail "dataset has ${DATASET_SIZE} samples, fewer than BATCH_SIZE=${BATCH_SIZE}"

if [[ "${NODE_RANK}" -eq 0 ]]; then
    mkdir -p "${SAVE_DIR}" "${LOG_DIR}" "${TENSORBOARD_LOG_DIR}" "${HYDRA_OUTPUT_DIR}"
    if [[ "${ENABLE_DUMP}" == "true" ]]; then
        mkdir -p "${DUMP_DIR}"
    fi
fi

latest_complete_checkpoint() {
    local candidate
    while IFS= read -r candidate; do
        if [[ -f "${candidate}/.metadata" && -f "${candidate}/metadata.pt" && -f "${candidate}/trainer_state.json" ]]; then
            echo "${candidate}"
            return 0
        fi
    done < <(find "${SAVE_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V -r)
    return 1
}
if [[ -z "${LOAD_DIR}" && "${AUTO_RESUME}" == "true" ]]; then
    LOAD_DIR="$(latest_complete_checkpoint || true)"
fi

OVERRIDES=(
    "+devices_per_node=${DEVICES_PER_NODE}"
    "batch_size=${BATCH_SIZE}"
    "num_rollouts=${NUM_ROLLOUTS}"
    "save_interval=${SAVE_INTERVAL}"
    "save_dir=${SAVE_DIR}"
    "save_mode=full"
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
    "backend.fsdp_cfg.defer_grad_sync=false"
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
    "algorithm.ar.clip_range_high=${CLIP_RANGE_HIGH}"
    "algorithm.ar.policy_entropy_interval=${POLICY_ENTROPY_INTERVAL}"
    "algorithm.ar.clip_schedule=${CLIP_SCHEDULE}"
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
    "sampling.diffusion.height=${IMAGE_SIZE}"
    "sampling.diffusion.width=${IMAGE_SIZE}"
    "sampling.diffusion.max_images=${MAX_IMAGES}"
    "sampling.diffusion.rollout_diffusion_batch_size=${ROLLOUT_DIFFUSION_BATCH_SIZE}"
    "sampling.diffusion.rollout_reencode_batch_size=${ROLLOUT_REENCODE_BATCH_SIZE}"
    "sampling.diffusion.num_inference_steps=${DIFFUSION_STEPS}"
    "sampling.diffusion.timestep_shift=1.0"
    "sampling.diffusion.eta=0.0"
    "sampling.diffusion.scheduler.num_sde_steps=${MSE_STEPS}"
    "dump_async=${DUMP_ASYNC}"
    "dump_image_workers=${DUMP_IMAGE_WORKERS}"
    "dump_jpeg_quality=${DUMP_JPEG_QUALITY}"
    "dump_max_pending=${DUMP_MAX_PENDING}"
)
if [[ "${ENABLE_DUMP}" == "true" ]]; then
    OVERRIDES+=("dump_dir=${DUMP_DIR}")
else
    OVERRIDES+=("dump_dir=null")
fi
if [[ -n "${LOAD_DIR}" ]]; then
    OVERRIDES+=("+load_dir=${LOAD_DIR}")
fi
OVERRIDES+=("$@")

if [[ "${NODE_RANK}" -eq 0 ]]; then
    cat <<INFO
[geoweave-multinode] topology:          ${NUM_NODES} nodes x ${GPUS_PER_NODE} GPUs = ${TOTAL_GPUS}
[geoweave-multinode] prompts/rollout:   ${BATCH_SIZE}
[geoweave-multinode] trajectories:      ${TRAJECTORIES_PER_ROLLOUT} (${SAMPLES_PER_PROMPT}/prompt)
[geoweave-multinode] grad accumulation: ${GRADIENT_ACCUMULATION_STEPS}
[geoweave-multinode] optimizer steps:   ${OPTIMIZER_TOTAL_STEPS}
[geoweave-multinode] rollout budget:    ${NUM_ROLLOUTS}
[geoweave-multinode] outcome judge:     dashscope/${ANSWER_JUDGE_MODEL}
[geoweave-multinode] process judge:     ${JUDGE_BASE_URL} (${SCA_JUDGE_MODEL}), critiques=${SCA_NUM_CRITIQUES}, voting=${SCA_VOTING}, reasoning=${SCA_JUDGE_REASONING_EFFORT}, format=${SCA_JUDGE_RESPONSE_FORMAT}, workers=${SCA_JUDGE_MAX_WORKERS}, qps=${SCA_JUDGE_QPS}
[geoweave-multinode] SCA credit:       outcome=${SCA_OUTCOME_CORRECT_CREDIT}, process_penalty=${SCA_PROCESS_ERROR_PENALTY}, eps=${SCA_ADVANTAGE_EPS}
[geoweave-multinode] text truncation:   policy=${TRUNCATED_REWARD}, Lmax=${MAX_NEW_TOKENS}
[geoweave-multinode] repeat recovery:    enabled=${RECOVER_REPETITION_TRUNCATIONS}, block=${REPETITION_MIN_BLOCK_TOKENS}-${REPETITION_MAX_BLOCK_TOKENS}, repeats=${REPETITION_MIN_REPEATS}, prefix>=${REPETITION_MIN_PREFIX_TOKENS}, tail_tol=${REPETITION_TAIL_TOLERANCE_TOKENS}, step_backoff=${REPETITION_REQUIRE_COMPLETE_STEP}
[geoweave-multinode] PPO clipping:      low=${CLIP_RANGE}, high=${CLIP_RANGE_HIGH}, ratio=[$(awk "BEGIN {print 1-${CLIP_RANGE}}"), $(awk "BEGIN {print 1+${CLIP_RANGE_HIGH}}")], schedule=${CLIP_SCHEDULE}
[geoweave-multinode] Policy entropy:    every ${POLICY_ENTROPY_INTERVAL} optimizer steps (0=disabled)
[geoweave-multinode] regularization:    loss_agg=${LOSS_AGG_MODE}, ar_kl_coef=${AR_KL_COEF}, mse_weight=${MSE_WEIGHT}, mse_steps=${MSE_STEPS}
[geoweave-multinode] dynamic scheduler: retry=${RETRY_TRUNCATED_TRAJECTORIES}, ignore_truncated=${IGNORE_TRUNCATED_SAMPLES}, exclude_from_adv_stats=${EXCLUDE_TRUNCATED_FROM_ADVANTAGE_STATS}, group_refill=${DYNAMIC_PROMPT_GROUP_REFILL}, reserve=${RESERVE_PROMPT_COUNT}
[geoweave-multinode] continuous mode:   enabled=${CONTINUOUS_BATCHING}, persistent=${PERSISTENT_WORKER_SESSION}, admission=${CONTINUOUS_REQUEST_ADMISSION}, live=${CONTINUOUS_LIVE_ADMISSION}, microbundle=${CONTINUOUS_PROMPT_MICROBUNDLE_SIZE}, window=${CONTINUOUS_SESSION_WINDOW_SIZE}, pool=${CONTINUOUS_ROLLOUT_POOL_SIZE}, text=${CONTINUOUS_TEXT_BATCH_SIZE}
[geoweave-multinode] cuDNN library:      ${CUDNN_LIB_DIR}
[geoweave-multinode] Ray head:          ${HEAD_IP}:${RAY_PORT}
[geoweave-multinode] run dir:           ${EXPERIMENT_DIR}
[geoweave-multinode] resume:            ${LOAD_DIR:-fresh run}
INFO
fi

export ENTRY=train_unified_model
export INSTALL_EDITABLE=0
export LAUNCH=spmd
export NUM_NODES GPUS_PER_NODE NODE_RANK NODE_IP HEAD_IP RAY_PORT RAY_CLUSTER_WAIT_S VENV_DIR

LAUNCH_CMD=(
    bash examples/run_experiment_multinode_taiji.sh
    "${CONFIG_NAME}"
    "${OVERRIDES[@]}"
)

cd "${REPO_ROOT}"
if [[ "${NODE_RANK}" -eq 0 ]]; then
    set +e
    "${LAUNCH_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    status=${PIPESTATUS[0]}
    set -e
    "${RAY_BIN}" stop >/dev/null 2>&1 || true
    exit "${status}"
fi

exec "${LAUNCH_CMD[@]}"
