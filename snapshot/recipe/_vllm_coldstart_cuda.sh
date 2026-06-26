#!/bin/bash
# Measure vLLM-CUDA cold-start time to /health, with or without CUDA graph
# capture. Runs WITHOUT any LD_PRELOAD so the timing reflects real vLLM startup.
set -uo pipefail
DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda}"
cd "$DEPLOY_DIR"
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
TP="${TP:-4}"
GMU="${GMU:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
PORT="${PORT:-8821}"
MODE="${MODE:-graph}"   # graph | eager

LOG="${DEPLOY_DIR}/logs/vllm_coldstart_${MODE}_${SLURM_JOB_ID:-local}.log"
rm -f "$LOG"
DEADLINE="${DEADLINE:-1200}"

CAPTURE_ARGS=()
if [ -n "${CAPTURE_SIZES:-}" ]; then
  # shellcheck disable=SC2206
  CAPTURE_ARGS=(--cudagraph-capture-sizes $CAPTURE_SIZES)
fi
ARGS=(--host 127.0.0.1 --port "$PORT" --served-model-name cs \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --trust-remote-code \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" "${CAPTURE_ARGS[@]}")
if [ "$MODE" = "eager" ]; then ARGS+=(--enforce-eager); fi

echo "[coldstart] MODE=$MODE TP=$TP gmu=$GMU start=$(date +%T)"

# Vendor-neutral instrumentation (hooks torch.cuda.CUDAGraph): per-graph timing.
if [ -n "${VLLM_CG_INSTRUMENT:-}" ]; then
  export PYTHONPATH="${DEPLOY_DIR}/snapshot/recipe/cginst:${PYTHONPATH:-}"
  rm -f "$VLLM_CG_INSTRUMENT"
  echo "[coldstart] CG instrumentation active -> $VLLM_CG_INSTRUMENT"
fi
# Skip-capture probe (measure mode quantifies the capture-phase forward cost).
if [ -n "${VLLM_CG_SKIP_CAPTURE:-}" ]; then
  export PYTHONPATH="${DEPLOY_DIR}/snapshot/recipe/cginst_skip:${PYTHONPATH:-}"
  echo "[coldstart] CG skip-capture active mode=$VLLM_CG_SKIP_CAPTURE"
fi
# Pin the CUDA caches for a fair warm-cache A/B.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DEPLOY_DIR}/cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DEPLOY_DIR}/cache/vllm}"
export PYTHONHASHSEED=0

t0=$(date +%s)
vllm serve "$MODEL" "${ARGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  if kill -0 "$SERVER_PID" 2>/dev/null && \
     curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[coldstart] MODE=$MODE READY at ${elapsed}s"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[coldstart] MODE=$MODE SERVER_EXITED at ${elapsed}s"; tail -40 "$LOG"; break; fi
  if [ "$elapsed" -gt "$DEADLINE" ]; then echo "[coldstart] MODE=$MODE DEADLINE at ${elapsed}s"; break; fi
  sleep 2
done

RESP=$(curl -sS "http://127.0.0.1:${PORT}/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"cs","prompt":"The capital of France is","max_tokens":3,"temperature":0}' 2>/dev/null || true)
echo "[coldstart] MODE=$MODE completion: ${RESP}"
echo "${RESP}" | grep -qi "paris" && echo "[coldstart] MODE=$MODE inference=ok" || echo "[coldstart] MODE=$MODE inference=failed"

sleep 2
kill -9 "$SERVER_PID" 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f 'VLLM_RPC' 2>/dev/null
pkill -9 -f 'from multiprocessing' 2>/dev/null
# Brief pause for host-side CUDA context cleanup (separate srun steps ensure GPU is
# released at container exit; nvidia-smi is not in the enroot container).
sleep 5
echo "[coldstart] MODE=$MODE done=$(date +%T)"
