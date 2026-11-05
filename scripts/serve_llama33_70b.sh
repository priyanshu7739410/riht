#!/usr/bin/env bash
# Launch Llama 3.3 70B on a single NVIDIA DGX Spark (GB10, 128 GB unified memory)
# with vLLM, sized so it cannot run out of memory. See README.md for the reasoning.
set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.09-py3}"
MODEL="${MODEL:-nvidia/Llama-3.3-70B-Instruct-NVFP4}"
: "${HF_TOKEN:?export HF_TOKEN=<your Hugging Face token> (Llama 3.3 is a gated model)}"

# 1. Unified memory: the page cache holds RAM that the GPU needs. Flush it first.
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# 2. Remove any old container so two copies of the model never load at once.
docker rm -f llama33-70b >/dev/null 2>&1 || true

# 3. Start the server.
docker run -d --name llama33-70b \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --restart unless-stopped \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --served-model-name llama-3.3-70b \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8

echo "Loading the model takes a few minutes. Follow it with:  docker logs -f llama33-70b"
echo "Ready when this works:  curl http://localhost:8000/v1/models"
