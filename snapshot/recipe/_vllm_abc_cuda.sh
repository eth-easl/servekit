#!/bin/bash
# _vllm_abc_cuda.sh — N5b Task 8 (G5): ONE A/B/C cold-start-to-ready measurement.
# PROFILE selects the variant:
#   baseline = graph mode, NO interposer (A)
#   restore  = interposer, MODE=restore, replay recorded graphs (B)
#   eager    = --enforce-eager, NO interposer (C)
# A record run must have populated snap-n5b/ + meta-n5b/ before the restore
# profile. Warm weight cache is assumed (a warm-up serve precedes this driver).
# One serve per srun step. Writes N5B_ABC_READY=<profile> <seconds> to stdout
# and to ${MEASURE_DIR}/<profile>.ready; runs a focused serving probe for
# baseline/restore if SERVE_PROBE_OUT is set.
set -uo pipefail
DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda}"
SNAP_CUDA_DIR="${SNAP_CUDA_DIR:-/capstor/scratch/cscs/xyao/snapshot-cuda}"
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
cd "$DEPLOY_DIR"

RECORD_SO="${SNAP_CUDA_DIR}/snapshot/build-vllm/libsnapshot_record_cuda.so"
REDIRECT_SO="${SNAP_CUDA_DIR}/snapshot/build-vllm/libsnapshot_redirect_cuda.so"
PROFILE="${PROFILE:-baseline}"
TP="${TP:-4}"; GMU="${GMU:-0.85}"; REGION_GIB="${REGION_GIB:-72}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
PORT="${PORT:-8830}"; DEADLINE="${DEADLINE:-1500}"
MEASURE_DIR="${MEASURE_DIR:-${DEPLOY_DIR}/measure-n5b}"
LOG="${DEPLOY_DIR}/logs/vllm_abc_${PROFILE}_${SLURM_JOB_ID:-local}.log"
rm -f "$LOG"; mkdir -p "${MEASURE_DIR}" "${DEPLOY_DIR}/logs"

ARGS=(--host 127.0.0.1 --port "$PORT" --served-model-name cs \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --trust-remote-code \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" --disable-custom-all-reduce)
[ "$PROFILE" = "eager" ] && ARGS+=(--enforce-eager)

# Only the restore profile engages the interposer (Δ=0 + skip capture).
if [ "$PROFILE" = "restore" ]; then
  export LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}"
  export SNAPSHOT_RECORD_CUDA_MODE=restore
  export SNAPSHOT_RECORD_CUDA_DIR="${DEPLOY_DIR}/snap-n5b/rank%r"
  export SNAPSHOT_REDIRECT_REGION_GIB="$REGION_GIB"
  export PYTHONPATH="${SNAP_CUDA_DIR}/snapshot/recipe/cginst_cuda:${PYTHONPATH:-}"
  # The Python restore hook (skip-forward + entry.output reconstruction) is only
  # valid WITH the .snap rebuild (fake-begin reconstructs the graph). NCCL's
  # direct-handle kernel loading blocks the rebuild (proven), so restore runs
  # rebuild=OFF (default): the forward runs normally (real capture), and the
  # hook must be DISABLED or vLLM's engine core init fails (skipped forward +
  # no rebuilt graph). B therefore measures interposer+redirect overhead over a
  # real warm-cache cold start (B ~= A); the capture-skip win is unavailable.
  unset VLLM_CG_RESTORE_META
else
  unset LD_PRELOAD  # baseline / eager: unmodified vLLM
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DEPLOY_DIR}/cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DEPLOY_DIR}/cache/vllm}"
export PYTHONHASHSEED=0

echo "[n5b-abc] PROFILE=$PROFILE TP=$TP gmu=$GMU start=$(date +%T)"
t0=$(date +%s)
vllm serve "$MODEL" "${ARGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!
READY=""
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  if kill -0 "$SERVER_PID" 2>/dev/null && \
     curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY="$elapsed"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[n5b-abc] $PROFILE EXITED at ${elapsed}s"; tail -40 "$LOG"; break
  fi
  [ "$elapsed" -gt "$DEADLINE" ] && { echo "[n5b-abc] $PROFILE DEADLINE"; break; }
  sleep 2
done
echo "N5B_ABC_READY=${PROFILE} ${READY:-NONE}"
echo "${READY:-NONE}" > "${MEASURE_DIR}/${PROFILE}.ready"

# Focused serving probe (baseline + restore only; eager is the no-graph control).
if [ -n "${READY}" ] && [ "$PROFILE" != "eager" ] && [ -n "${SERVE_PROBE_OUT:-}" ]; then
  python3 "${SNAP_CUDA_DIR}/snapshot/recipe/_n5b_serve_probe.py" \
    "http://127.0.0.1:${PORT}/v1/completions" \
    "${SERVE_CONCURRENCY:-8}" "${SERVE_ROUNDS:-4}" > "${SERVE_PROBE_OUT}" 2>/dev/null || true
  echo "[n5b-abc] $PROFILE serve-probe -> ${SERVE_PROBE_OUT}"
fi

sleep 2
kill -9 "$SERVER_PID" 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f 'VLLM_RPC' 2>/dev/null
pkill -9 -f 'from multiprocessing' 2>/dev/null
sleep 5
echo "[n5b-abc] $PROFILE done=$(date +%T)"
