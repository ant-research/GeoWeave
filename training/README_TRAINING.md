# GeoWeave Reinforcement-Learning Training Guide

This guide describes how to set up and run GeoWeave Interleave-RL training.
Run the commands from the `training/` directory.

The available launchers are:

```text
scripts/setup_geoweave_env.sh       # Create the RL environment
scripts/train_geoweave.sh            # Single-node training
scripts/train_geoweave_multinode.sh  # Multi-node Ray training
```

## 1. Requirements

- Linux x86-64
- Python 3.12 and [`uv`](https://docs.astral.sh/uv/)
- NVIDIA GPUs with a compatible driver
- A Hugging Face-format GeoWeave checkpoint
- JSONL training data
- A DashScope API key and an OpenAI-compatible process-judge endpoint
- A compatible FlashAttention wheel for Python 3.12, PyTorch 2.8, CUDA 12.x,
  and the CXX11 ABI

The validated software stack uses PyTorch 2.8.0, CUDA 12.8, and
FlashAttention 2.8.3.post1.

## 2. Set up the RL environment

The RL environment is separate from the inference environment at the repository
root. Create it in `training/.venv-RL` using the setup script:

```bash
cd /path/to/GeoWeave/training
export FLASH_ATTN_WHEEL=/path/to/compatible/flash_attn.whl
bash scripts/setup_geoweave_env.sh
```

The script installs the pinned training dependencies and the local training
package. To use another environment location, set `ENV_DIR`:

```bash
ENV_DIR=/path/to/.venv-RL \
FLASH_ATTN_WHEEL=/path/to/compatible/flash_attn.whl \
  bash scripts/setup_geoweave_env.sh
```

If your environment requires a package mirror, set `PYPI_INDEX_URL` when
running the setup script.

## 3. Configure the reward judges

Training uses an outcome judge and a process judge. Set the required credentials
and endpoint before launching a job:

```bash
export DASHSCOPE_API_KEY="<your-dashscope-api-key>"
export JUDGE_BASE_URL="https://your-judge-endpoint.example.com/v1/chat/completions"
# Required by the current process-judge implementation.
export API_KEY_ENV="<your-process-judge-api-key>"
```

The default judge models are `qwen3.7-max` for answer judging and
`gemini-3.5-flash` for process judging. Override them if needed:

```bash
export ANSWER_JUDGE_MODEL="qwen3.7-max"
export SCA_JUDGE_MODEL="your-process-judge-model"
```

Do not commit API keys or private endpoints to the repository.

## 4. Prepare the training data

Pass a model checkpoint and a JSONL data file to the launcher:

```bash
MODEL_PATH=/path/to/GeoWeave-HF \
DATA_PATH=/path/to/train.jsonl \
  bash scripts/train_geoweave.sh
```

Each record must contain a non-empty `prompt` and `metadata.answer`:

```json
{
  "prompt": "problem text",
  "metadata": {
    "answer": "4"
  }
}
```

For image-conditioned problems, include the image in `media`:

```json
{
  "prompt_id": "sample-1",
  "prompt": "<image>\nProblem text",
  "media": [
    {
      "modality": "image",
      "role": "condition",
      "uri": "/path/to/image.png"
    }
  ],
  "metadata": {
    "answer": "4"
  }
}
```

The launcher checks that the data file exists, records contain the required
fields, and the dataset is large enough for the configured batch size.

## 5. Run single-node training

The default recipe is configured for 8 GPUs:

```bash
RUN_NAME=geoweave_interleave_rl_main \
MODEL_PATH=/path/to/GeoWeave-HF \
DATA_PATH=/path/to/train.jsonl \
NUM_ROLLOUTS=1000 \
SAVE_INTERVAL=100 \
  bash scripts/train_geoweave.sh
```

For a short smoke test, reduce the rollout budget and checkpoint interval:

```bash
RUN_NAME=geoweave_smoke \
MODEL_PATH=/path/to/GeoWeave-HF \
DATA_PATH=/path/to/train.jsonl \
NUM_ROLLOUTS=5 \
SAVE_INTERVAL=5 \
  bash scripts/train_geoweave.sh
```

Use `CUDA_VISIBLE_DEVICES` and `NUM_DEVICES` to select fewer GPUs. Ensure that
`BATCH_SIZE * SAMPLES_PER_PROMPT` is divisible by
`NUM_DEVICES * NUM_UPDATES_PER_BATCH`.

The launcher also accepts trailing Hydra overrides:

```bash
bash scripts/train_geoweave.sh \
  backend.optimizer_cfg.weight_decay=0.01 \
  sampling.ar.temperature=0.8
```

Run `bash scripts/train_geoweave.sh --help` to see all supported environment
variables.

## 6. Run multi-node training

Run the same multi-node launcher once on every node. The repository, checkpoint,
data, experiment directory, and checkpoint paths must be mounted at the same
paths on all nodes.

For two nodes with eight GPUs each:

```bash
# Head node
NUM_NODES=2 GPUS_PER_NODE=8 NODE_RANK=0 \
HEAD_IP=<head-ip> NODE_IP=<head-ip> \
MODEL_PATH=/path/to/GeoWeave-HF \
DATA_PATH=/path/to/train.jsonl \
RUN_NAME=geoweave_2x8 \
bash scripts/train_geoweave_multinode.sh

# Worker node
NUM_NODES=2 GPUS_PER_NODE=8 NODE_RANK=1 \
HEAD_IP=<head-ip> NODE_IP=<worker-ip> \
MODEL_PATH=/path/to/GeoWeave-HF \
DATA_PATH=/path/to/train.jsonl \
RUN_NAME=geoweave_2x8 \
bash scripts/train_geoweave_multinode.sh
```

The global batch size defaults to the total GPU count. Keep the following
constraints when changing the topology or batch size:

- `DEVICES_PER_NODE` must equal `GPUS_PER_NODE`.
- `BATCH_SIZE * SAMPLES_PER_PROMPT` must be divisible by the global GPU count
  and `NUM_UPDATES_PER_BATCH`.
- All nodes must be able to reach the Ray head at `HEAD_IP:RAY_PORT`.
- Use the same `RUN_NAME` and configuration on every node.

## 7. Resume a run

Set `LOAD_DIR` to a specific checkpoint directory:

```bash
RUN_NAME=geoweave_interleave_rl_main \
LOAD_DIR=/path/to/experiments/geoweave_interleave_rl_main/checkpoints/checkpoint-100 \
NUM_ROLLOUTS=1000 \
  bash scripts/train_geoweave.sh
```

`NUM_ROLLOUTS` is the total rollout budget, including rollouts completed before
resuming. Use the original run name and experiment directory when continuing a
run. Both DCP checkpoints (the default) and legacy Torch checkpoints are
supported.

## 8. Configuration and outputs

Important single-node defaults are:

| Variable | Default |
| --- | ---: |
| `NUM_DEVICES` | `8` |
| `BATCH_SIZE` | `8` |
| `NUM_ROLLOUTS` | `1000` |
| `SAMPLES_PER_PROMPT` | `8` |
| `MAX_NEW_TOKENS` | `4096` |
| `IMAGE_SIZE` | `512` |
| `DIFFUSION_STEPS` | `30` |
| `MAX_IMAGES` | `4` |
| `NUM_UPDATES_PER_BATCH` | `2` |
| `LEARNING_RATE` | `5e-7` |
| `MSE_WEIGHT` | `1.0` |
| `MSE_STEPS` | `3` |
| `SAVE_INTERVAL` | `100` |
| `CHECKPOINT_FORMAT` | `dcp` |
| `LOGGING_BACKEND` | `tensorboard` |

The default diffusion schedule uses `base_shift=3.0`,
`timestep_shift=1.0`, and `eta=0.0`.

Each run is stored under `experiments/<RUN_NAME>/`:

```text
experiments/<RUN_NAME>/
├── checkpoints/  # Resumable checkpoints
├── dumps/        # Rollout JSONL files and generated images
├── logs/         # Training logs
├── outputs/      # Hydra configuration and logs
└── runs/         # TensorBoard events
```

Start TensorBoard with:

```bash
tensorboard --logdir /path/to/experiments --port 6006 --bind_all
```

Rollout dumps can be disabled when they are not needed:

```bash
ENABLE_DUMP=false bash scripts/train_geoweave.sh
```

## 9. Common overrides

Adjust rollout sampling:

```bash
SAMPLES_PER_PROMPT=4 \
MAX_NEW_TOKENS=1024 \
MAX_IMAGES=2 \
  bash scripts/train_geoweave.sh
```

Adjust image generation:

```bash
IMAGE_SIZE=512 \
DIFFUSION_STEPS=20 \
MSE_STEPS=3 \
  bash scripts/train_geoweave.sh
```

Adjust optimization:

```bash
LEARNING_RATE=5e-7 \
GRADIENT_ACCUMULATION_STEPS=1 \
NUM_UPDATES_PER_BATCH=2 \
MICRO_BATCH_SIZE=1 \
  bash scripts/train_geoweave.sh
```

`IMAGE_SIZE` must be divisible by 32. Set `SAVE_INTERVAL=0` to disable
checkpoint creation for a temporary smoke test.

## 10. Troubleshooting

### Missing RL environment

Run the setup script from `training/` and provide a compatible FlashAttention
wheel:

```bash
FLASH_ATTN_WHEEL=/path/to/compatible/flash_attn.whl \
  bash scripts/setup_geoweave_env.sh
```

### Missing judge credentials

Make sure `DASHSCOPE_API_KEY`, `JUDGE_BASE_URL`, and `API_KEY_ENV` are exported
in the shell that launches training.

### Existing experiment directory

Use a new `RUN_NAME`, or set `LOAD_DIR` when resuming an existing run. Avoid
starting two processes with the same experiment directory.

### Checkpoint loading errors

For resume, `LOAD_DIR` must point to a complete `checkpoint-N` directory, not
the parent `checkpoints/` directory.
