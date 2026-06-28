#!/bin/bash
# Measure vLLM cold-start time to /health, with and without CUDA graph capture.
# Runs WITHOUT our LD_PRELOAD libraries so the timing reflects real vLLM startup.
# Outputs T_GRAPH (with capture) and T_EAGER (--enforce-eager).
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
TP="${TP:-1}"
GMU="${GMU:-0.60}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PORT="${PORT:-8821}"
MODE="${MODE:-graph}"   # graph | eager

# Phase log on shared storage (node-local /tmp is wiped at job end; we need the
# weight-load / capture timestamps to decompose the cold start).
LOG=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/vllm_coldstart_${MODE}_${SLURM_JOB_ID:-local}.log
rm -f "$LOG"

DEADLINE="${DEADLINE:-800}"
# Build the optional --cudagraph-capture-sizes tail.
CAPTURE_ARGS=()
if [ -n "${CAPTURE_SIZES:-}" ]; then
  # shellcheck disable=SC2206
  CAPTURE_ARGS=(--cudagraph-capture-sizes $CAPTURE_SIZES)
fi

ARGS=(--host 127.0.0.1 --port "$PORT" --served-model-name cs \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --trust-remote-code \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" "${CAPTURE_ARGS[@]}")
if [ "$MODE" = "eager" ]; then
  ARGS+=(--enforce-eager)
fi

echo "[coldstart] MODE=$MODE TP=$TP gmu=$GMU start=$(date +%T)"

# Optional in-job instrumentation: when VLLM_CG_INSTRUMENT=<path> is set,
# activate cginst/sitecustomize.py which monkeypatches
# torch.cuda.CUDAGraph.capture_begin/end/replay — the ONE layer every capture
# path (vLLM FULL, breakable, torch.compile cudagraph_trees) funnels through.
# Emits per-call timing CSV. No in-container edits; just PYTHONPATH.
if [ -n "${VLLM_CG_INSTRUMENT:-}" ]; then
  export PYTHONPATH="/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe/cginst:${PYTHONPATH:-}"
  rm -f "$VLLM_CG_INSTRUMENT"
  echo "[coldstart] CG instrumentation active -> $VLLM_CG_INSTRUMENT"
fi
# Prototype: when VLLM_CG_SKIP_CAPTURE is set (measure|shim), load
# cginst_skip/sitecustomize.py which wraps create_forward_fn with a no-op,
# skipping all model forwards during cudagraph capture. 'measure' mode
# quantifies the forward-compute prize; 'shim' mode keeps capture_begin/end
# for the HIP interposer to rebuild graphs from snapshot.
if [ -n "${VLLM_CG_SKIP_CAPTURE:-}" ]; then
  export PYTHONPATH="/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe/cginst_skip:${PYTHONPATH:-}"
  echo "[coldstart] CG skip-capture prototype active mode=$VLLM_CG_SKIP_CAPTURE"
fi
# Pin Triton cache for a fair A/B vs the restore run (M3g).
export TRITON_CACHE_DIR="${SNAPSHOT_TRITON_CACHE_DIR:-${TRITON_CACHE_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm/cache/triton}}"
# Match the restore run's VLLM_CACHE_ROOT so baseline and restore share the
# SAME compiled artifacts (inductor + MoE autotune) for a fair A/B. Without
# this, srun --environment may leave vLLM's default (production) cache, which
# has different MoE-autotune warmth -> confounds the cold-start comparison.
export VLLM_CACHE_ROOT="${SNAPSHOT_VLLM_CACHE_ROOT:-${VLLM_CACHE_ROOT:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm/cache/vllm}}"
t0=$(date +%s)
vllm serve "$MODEL" "${ARGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!

# Poll /health.
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  if kill -0 "$SERVER_PID" 2>/dev/null && \
     curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[coldstart] MODE=$MODE READY at ${elapsed}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[coldstart] MODE=$MODE SERVER_EXITED at ${elapsed}s"
    break
  fi
  if [ "$elapsed" -gt "$DEADLINE" ]; then
    echo "[coldstart] MODE=$MODE DEADLINE at ${elapsed}s"
    break
  fi
  sleep 2
done

# Send one request to confirm inference works.
curl -sS "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"cs","prompt":"Hello","max_tokens":2,"temperature":0}' \
  >/dev/null 2>&1 && echo "[coldstart] MODE=$MODE inference=ok" || \
  echo "[coldstart] MODE=$MODE inference=failed"

sleep 2
kill -9 "$SERVER_PID" 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f 'VLLM_RPC' 2>/dev/null
sleep 3
echo "[coldstart] MODE=$MODE done=$(date +%T)"
