#!/bin/bash
# Per-phase decomposition of a warm-cache vLLM-CUDA start (no interposer).
# RUN_TAG + EXTRA_ARGS select the variant (EXTRA_ARGS="--enforce-eager" disables capture).
set -uo pipefail
DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda}"
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
DIR="${DEPLOY_DIR}/measure-init/${RUN_TAG:-graphs}"
rm -rf "$DIR"; mkdir -p "$DIR"
export VLLM_LOGGING_LEVEL=INFO
export PYTHONHASHSEED=0
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DEPLOY_DIR}/cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DEPLOY_DIR}/cache/vllm}"
TP="${TP:-4}"; GMU="${GMU:-0.90}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
PORT="${PORT:-8801}"; DEADLINE="${RUN_SECS:-1200}"; EXTRA_ARGS="${EXTRA_ARGS:-}"
FULL="$DIR/vllm.log"
echo "[measure] tag=${RUN_TAG:-graphs} extra='${EXTRA_ARGS}' tp=$TP gmu=$GMU start=$(date +%T)"
t0=$(date +%s)
# shellcheck disable=SC2086
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name m \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --trust-remote-code \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" $EXTRA_ARGS > "$FULL" 2>&1 &
SERVER_PID=$!
ready=""
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "[measure] server exited before ready (elapsed=${elapsed}s)"; break; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready="$elapsed"; echo "[measure] READY: COLD_START_SECONDS=${elapsed}"; break; fi
  if [ "$elapsed" -gt "$DEADLINE" ]; then echo "[measure] DEADLINE ${DEADLINE}s without ready"; break; fi
  sleep 2
done
kill -9 "$SERVER_PID" 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f 'from multiprocessing' 2>/dev/null; pkill -9 -f 'VLLM_RPC' 2>/dev/null
# Brief pause for host-side CUDA context cleanup (separate srun steps ensure GPU is
# released at container exit; nvidia-smi is not in the enroot container).
sleep 5
echo "=== sub-phase breakdown (${RUN_TAG:-graphs}) ==="
grep -aE "Model loading took|init engine .* took|Maximum concurrency|Application startup complete" "$FULL" | tail -6
echo "--- capture wall-clock (sum of PIECEWISE+FULL bars) ---"
grep -aoE "Capturing CUDA graphs \([^)]*\): 100%[^[]*\[[0-9]{2}:[0-9]{2}" "$FULL" | \
  awk '/PIECEWISE/{p=$0} /FULL/{f=$0} END{if(p)print p; if(f)print f}'
echo "[measure] RESULT tag=${RUN_TAG:-graphs} ready=${ready:-NONE}"
