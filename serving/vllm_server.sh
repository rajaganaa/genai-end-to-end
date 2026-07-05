#!/usr/bin/env bash
# Launches vLLM's OpenAI-compatible server with the base model +
# hot-loaded LoRA adapter (medical fine-tune).
#
# --enable-lora + --lora-modules lets us serve the base model and the
# medical adapter from one process, and request either at inference time
# by setting `model` in the API call to "base" or "medical-lora-13b".
# This is what makes adapter iteration cheap: swap the adapter directory
# and restart, without re-downloading/re-sharding the (much larger) base
# model weights.

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3-13b}"
LORA_PATH="${LORA_PATH:-checkpoints/medical-lora-13b-vllm-ready}"
LORA_NAME="${VLLM_MODEL_NAME:-medical-lora-13b}"
TP_SIZE="${TENSOR_PARALLEL_SIZE:-2}"   # number of GPUs for tensor parallelism
PORT="${VLLM_PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "Starting vLLM server:"
echo "  base_model      = ${BASE_MODEL}"
echo "  lora_adapter    = ${LORA_NAME} (${LORA_PATH})"
echo "  tensor_parallel = ${TP_SIZE}"
echo "  port            = ${PORT}"

python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" \
    --served-model-name base "${LORA_NAME}" \
    --enable-lora \
    --lora-modules "${LORA_NAME}=${LORA_PATH}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.90 \
    --quantization awq \
    --dtype bfloat16 \
    --port "${PORT}" \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --disable-log-requests
    # --disable-log-requests: avoid logging raw prompts (may contain PHI-adjacent
    # clinical text) at the inference-server layer; structured, redacted
    # logging happens in serving/api.py instead.
