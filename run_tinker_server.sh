#!/bin/bash
# Script to run the Tinker API server on TPU


set -e

cd ~/SkyRL/skyrl-tx
git pull

# HuggingFace cache configuration (use /dev/shm for fast access)
export HF_CACHE=/dev/shm/huggingface_cache
export HF_HOME=/dev/shm/huggingface_cache
export HF_HUB_CACHE=/dev/shm/huggingface_cache/hub
export TRANSFORMERS_CACHE=/dev/shm/huggingface_cache

ADDITIONAL_FLAGS=""
LOG_FILE="/tmp/tinker-api.log"
# MIN_SEQ_LEN=4096
# MIN_SEQ_LEN=8192
MIN_SEQ_LEN=65536
while [[ $# -gt 0 ]]; do
    case $1 in
        --scan-layers)
            ADDITIONAL_FLAGS="${ADDITIONAL_FLAGS} --scan-layers"
            shift
            ;;
        --log-file)
            LOG_FILE="$2"
            shift 2
            ;;
        --min-seq-len)
            MIN_SEQ_LEN="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# export JAX_LOG_COMPILES=1
sudo chown -R $(whoami) /dev/shm/huggingface_cache
# Date-based checkpoint path
DATE=$(date +%Y%m%d)
CHECKPOINTS_BASE="/dev/shm/huggingface_cache/lora-experiments/qwen3-4b/${DATE}"
EXTERNAL_LORA_BASE="gs://ohadrubin-docker-images/lora-experiments/qwen3-4b/${DATE}"
export USE_NNX_VALUE_AND_GRAD=1
# Model configuration
BASE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
TENSOR_PARALLEL_SIZE=4
MAX_LORA_ADAPTERS=4
MAX_LORA_RANK=8
TRAIN_MICRO_BATCH_SIZE=1

# External inference server
EXTERNAL_INFERENCE_URL="https://v6e-8-node-17.ohadrubin.com"

# Precompile common sequence lengths to avoid JIT during training
PRECOMPILE_SEQ_LENS=""



sudo rm /tmp/libtpu_lockfile || true
sleep 2

# Reinstall ringattention to get clean copy, then fix deprecated JAX API
uv pip install --reinstall ringattention --quiet
RING_INIT="/home/ohadr/SkyRL/skyrl-tx/.venv/lib/python3.12/site-packages/ringattention/__init__.py"
sed -i 's/jax.lib.xla_bridge.get_backend/jax.extend.backend.get_backend/' "$RING_INIT"
sed -i 's/^import jax$/import jax\nimport jax.extend/' "$RING_INIT"

# Run the server
# --gradient-checkpointing \
uv run --extra tinker --extra tpu  -m tx.tinker.api \
    --checkpoints-base "${CHECKPOINTS_BASE}" \
    ${ADDITIONAL_FLAGS} \
    --base-model "${BASE_MODEL}" \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --max-lora-adapters ${MAX_LORA_ADAPTERS} \
    --no-mlp-lora \
    --max-lora-rank ${MAX_LORA_RANK} \
    --train-micro-batch-size ${TRAIN_MICRO_BATCH_SIZE} \
    --external-inference-url "${EXTERNAL_INFERENCE_URL}" \
    --external-inference-lora-base "${EXTERNAL_LORA_BASE}" \
    --precompile-seq-lens "${PRECOMPILE_SEQ_LENS}" \
    --min-seq-len ${MIN_SEQ_LEN} \
    2>&1 | tee "${LOG_FILE}"
