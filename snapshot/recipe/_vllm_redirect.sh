#!/bin/bash
# Runs a full vLLM cold start with EVERY device allocation redirected into the
# deterministic ARENA (libsnapshot_redirect.so, single hipMalloc + free-list) so
# the engine runs entirely on reproducible addresses. Measures the wall-clock
# cold start to server-ready (the end-to-end number) and logs the per-worker
# allocation sequence so two runs can be diffed for address determinism.
set -uo pipefail

PRELOAD=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/build/libsnapshot_redirect.so
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
DIR=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/redirect-vllm/${RUN_TAG:-run1}
rm -rf "$DIR"; mkdir -p "$DIR"

export LD_PRELOAD="$PRELOAD"
export SNAPSHOT_REDIRECT_ARENA="${ARENA:-1}"
export SNAPSHOT_REDIRECT_REGION_GIB="${REGION_GIB:-64}"
export SNAPSHOT_REDIRECT_ALLOC_DIR="$DIR"
export SNAPSHOT_REDIRECT_VERBOSE="${REDIR_VERBOSE:-1}"
export AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-0}"
export VLLM_LOGGING_LEVEL=INFO
TP="${TP:-1}"
GMU="${GMU:-0.60}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PORT="${PORT:-8799}"
DEADLINE="${RUN_SECS:-900}"
FULL="$DIR/vllm.log"

echo "[redir-vllm] tag=${RUN_TAG:-run1} TP=$TP arena=$SNAPSHOT_REDIRECT_ARENA region=${SNAPSHOT_REDIRECT_REGION_GIB}GiB gmu=$GMU start=$(date +%T)"

t0=$(date +%s)   # integer seconds (no bc in container)
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name redir \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --trust-remote-code \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  > "$FULL" 2>&1 &
SERVER_PID=$!

# Poll /health until the server reports ready or the deadline passes.
ready=""
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[redir-vllm] server process exited before ready (elapsed=${elapsed}s)"
    break
  fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    ready="$elapsed"
    echo "[redir-vllm] READY: COLD_START_SECONDS=${elapsed}"
    break
  fi
  if [ "$elapsed" -gt "$DEADLINE" ]; then
    echo "[redir-vllm] DEADLINE ${DEADLINE}s reached without ready (elapsed=${elapsed}s)"
    break
  fi
  sleep 2
done

# If ready, fire one real completion to prove the redirected engine computes.
if [ -n "$ready" ]; then
  echo "=== sample completion (proves redirected engine runs) ==="
  curl -s "http://127.0.0.1:${PORT}/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"redir","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
    | head -c 600; echo
fi

kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null
# Reap any orphaned engine-core / API-server workers so the next cold start
# starts on a clean, uncontended GPU.
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f 'from multiprocessing' 2>/dev/null
sleep 3
echo "[redir-vllm] exited $(date +%T)"

echo "=== [redirect] lines ==="; grep -E "\[redirect\]" "$FULL" | tail -20
echo "=== model load / KV / capture / ready markers ==="
grep -inE "Model loading took|Memory profiling|GPU KV blocks|Available KV|Maximum concurrency|Capturing|graph captur|Application startup complete|Uvicorn running|out of memory|HIP error|hipError|RuntimeError|Engine core init" "$FULL" | tail -50
echo "=== last 20 lines of vllm.log ==="; tail -20 "$FULL"
echo "=== alloc logs ==="; wc -l "$DIR"/redir-*.log 2>/dev/null || echo "(none)"
echo "[redir-vllm] COLD_START_RESULT tag=${RUN_TAG:-run1} ready=${ready:-NONE}"
