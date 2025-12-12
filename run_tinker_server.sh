#!/bin/bash
# Script to run the Tinker API server on TPU

set -e

ts() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

ts "Starting Tinker server script"
cd ~/SkyRL/skyrl-tx
ts "Running git pull"
git pull

# HuggingFace cache configuration (use /dev/shm for fast access)
export HF_CACHE=/dev/shm/huggingface_cache
export HF_HOME=/dev/shm/huggingface_cache
export HF_HUB_CACHE=/dev/shm/huggingface_cache/hub
export TRANSFORMERS_CACHE=/dev/shm/huggingface_cache

ADDITIONAL_FLAGS=""
LOG_FILE=""
# MIN_SEQ_LEN=4096
# MIN_SEQ_LEN=8192
MIN_SEQ_LEN=59392
# 512*119=60928
# 2048*29=59392
while [[ $# -gt 0 ]]; do
    case $1 in
        --scan-layers)
            ADDITIONAL_FLAGS="${ADDITIONAL_FLAGS} --scan-layers --segment-length 8"
            shift
            ;;
        --dump-ts)
            JAX_DUMP_TS="$2"
            shift 2
            ;;
        --min-seq-len)
            MIN_SEQ_LEN="$2"
            shift 2
            ;;
        --no-load-safetensors)
            ADDITIONAL_FLAGS="${ADDITIONAL_FLAGS} --no-load-safetensors"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

export JAX_LOG_COMPILES=1
export JAX_TRACEBACK_FILTERING=off
export JAX_DUMP_IR_MODES='jaxpr'

# Use passed JAX_DUMP_TS or calculate it
if [ -z "$JAX_DUMP_TS" ]; then
    JAX_DUMP_DATE=$(date '+%Y-%m-%d')
    JAX_DUMP_TIME=$(date '+%H-%M')
    JAX_DUMP_TS="${JAX_DUMP_DATE}_${JAX_DUMP_TIME}"
else
    # Parse date and time from JAX_DUMP_TS (format: YYYY-MM-DD_HH-MM)
    JAX_DUMP_DATE="${JAX_DUMP_TS%_*}"
    JAX_DUMP_TIME="${JAX_DUMP_TS#*_}"
fi
export JAX_DUMP_IR_TO="/tmp/jax_ir_dump_${JAX_DUMP_TS}"
LOG_FILE="/tmp/logs/${JAX_DUMP_TS}/tinker-api.log"
mkdir -p "$(dirname "$LOG_FILE")"
ts "JAX IR dump dir: ${JAX_DUMP_IR_TO}"
ts "Log file: ${LOG_FILE}"
# export XLA_FLAGS='--xla_disable_hlo_passes=algsimp'

sudo chown -R $(whoami) /dev/shm/huggingface_cache
# Date-based checkpoint path
DATE=$(date +%Y%m%d)
CHECKPOINTS_BASE="/dev/shm/huggingface_cache/lora-experiments/qwen3-4b/${DATE}"
EXTERNAL_LORA_BASE="gs://ohadrubin-docker-images/lora-experiments/qwen3-4b/${DATE}"
export USE_NNX_VALUE_AND_GRAD=1
# Model configurationor
BASE_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
TENSOR_PARALLEL_SIZE=4
MAX_LORA_ADAPTERS=4
MAX_LORA_RANK=8
TRAIN_MICRO_BATCH_SIZE=1
# Disable algebraic simplification to prevent OOM during compilation


# External inference server
EXTERNAL_INFERENCE_URL="https://v6e-8-node-17.ohadrubin.com"

# Precompile common sequence lengths to avoid JIT during training
PRECOMPILE_SEQ_LENS="${MIN_SEQ_LEN}"



ts "Clearing TPU lockfile"
sudo rm /tmp/libtpu_lockfile || true
sleep 2

ts "Reinstalling ringattention"
# Reinstall ringattention to get clean copy, then fix deprecated JAX API
uv pip install --reinstall ringattention --quiet
rm -f uv.lock  # Remove lockfile to ensure local flax is used
uv sync --extra tpu --extra tinker
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
    --no-embed-lora \
    --max-lora-rank ${MAX_LORA_RANK} \
    --train-micro-batch-size ${TRAIN_MICRO_BATCH_SIZE} \
    --external-inference-url "${EXTERNAL_INFERENCE_URL}" \
    --external-inference-lora-base "${EXTERNAL_LORA_BASE}" \
    --precompile-seq-lens "${PRECOMPILE_SEQ_LENS}" \
    --min-seq-len ${MIN_SEQ_LEN} \
    2>&1 | tee "${LOG_FILE}"
