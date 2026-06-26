#!/bin/bash
#SBATCH --job-name=glm-47-vllm-cuda-probe
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/logs/%x-%j.out

set -euo pipefail
DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda}"
export DEPLOY_DIR
export EDF_PATH="${DEPLOY_DIR}/deploy/glm-47-flash-bristen-vllm:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/home" "${DEPLOY_DIR}/cache/triton" \
  "${DEPLOY_DIR}/cache/vllm" "${DEPLOY_DIR}/cache/flashinfer" "${DEPLOY_DIR}/cache/pip"

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"
PORT="${PORT:-8000}"

srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -euo pipefail
  echo "=== transformers overlay (glm4_moe_lite) ==="
  python3 -c "import transformers; print(\"pre:\", transformers.__version__)"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'" || { echo "TRANSFORMERS_OVERLAY_FAILED"; exit 1; }
  python3 -c "import transformers; print(\"post:\", transformers.__version__)"
  echo "=== launch vllm serve TP=4 ==="
  t0=$(date +%s)
  vllm serve "'"${MODEL}"'" --host 127.0.0.1 --port '"${PORT}"' --served-model-name probe \
    --tensor-parallel-size 4 --trust-remote-code \
    --gpu-memory-utilization 0.90 --max-model-len 131072 --max-num-seqs 256 \
    > "${DEPLOY_DIR}/logs/probe-vllm.log" 2>&1 &
  PID=$!
  while :; do
    el=$(( $(date +%s) - t0 ))
    if kill -0 $PID 2>/dev/null && curl -fsS "http://127.0.0.1:'"${PORT}"'/health" >/dev/null 2>&1; then
      echo "PROBE READY at ${el}s"; break; fi
    if ! kill -0 $PID 2>/dev/null; then echo "PROBE SERVER_EXITED at ${el}s"; tail -40 "${DEPLOY_DIR}/logs/probe-vllm.log"; exit 1; fi
    if [ $el -gt 1200 ]; then echo "PROBE DEADLINE"; exit 1; fi
    sleep 5
  done
  RESP=$(curl -sS "http://127.0.0.1:'"${PORT}"'/v1/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"probe\",\"prompt\":\"The capital of France is\",\"max_tokens\":3,\"temperature\":0}")
  echo "PROBE completion: ${RESP}"
  echo "${RESP}" | grep -qi "paris" && echo "PROBE_CORRECT=1" || echo "PROBE_CORRECT=0"
  kill -9 $PID 2>/dev/null; pkill -9 -f "vllm serve" 2>/dev/null || true
'
