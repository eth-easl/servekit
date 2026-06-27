#!/bin/bash
# _vllm_record_cuda.sh — N5b Task 5/6: vLLM-CUDA TP=4 RECORD-mode cold start.
# LD_PRELOADs redirect+record; each worker records its CUDA graphs (.snap under
# SNAPSHOT_RECORD_CUDA_DIR=.../rank%r) and the Python cg_meta_cuda layer records
# entry.output (VLLM_CG_RECORD_META=.../rank%r.json). Mirrors
# _vllm_coldstart_cuda.sh but in record mode. One serve per srun step.
#
# Env (defaults):
#   MODE              record | restore   (record launcher defaults to record)
#   TP, GMU, REGION_GIB, MAX_NUM_SEQS, MAX_MODEL_LEN, PORT, DEADLINE, CAPTURE_SIZES
#   SNAP_ROOT, META_ROOT   per-rank .snap / meta roots (rank%r resolved per worker)
set -uo pipefail
DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda}"
SNAP_CUDA_DIR="${SNAP_CUDA_DIR:-/capstor/scratch/cscs/xyao/snapshot-cuda}"
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
cd "$DEPLOY_DIR"

# The record/redirect .so MUST be built inside the vllm serving image (glibc 2.35
# + CUDA 13) so it loads under LD_PRELOAD — the snapshot-cuda devel build
# (ubuntu24/glibc2.39 + CUDA 12.6) does not (the ldd gate catches this).
RECORD_SO="${SNAP_CUDA_DIR}/snapshot/build-vllm/libsnapshot_record_cuda.so"
REDIRECT_SO="${SNAP_CUDA_DIR}/snapshot/build-vllm/libsnapshot_redirect_cuda.so"
TP="${TP:-4}"; GMU="${GMU:-0.85}"; REGION_GIB="${REGION_GIB:-72}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
PORT="${PORT:-8821}"; DEADLINE="${DEADLINE:-1500}"
MODE="${MODE:-record}"
SNAP_ROOT="${SNAP_ROOT:-${DEPLOY_DIR}/snap-n5b}"
META_ROOT="${META_ROOT:-${DEPLOY_DIR}/meta-n5b}"
LOG="${DEPLOY_DIR}/logs/vllm_${MODE}_cuda_${SLURM_JOB_ID:-local}.log"
rm -f "$LOG"; mkdir -p "$SNAP_ROOT" "$META_ROOT" "$DEPLOY_DIR/logs"

CAPTURE_ARGS=()
if [ -n "${CAPTURE_SIZES:-}" ]; then
  # shellcheck disable=SC2206
  CAPTURE_ARGS=(--cudagraph-capture-sizes $CAPTURE_SIZES)
fi
# --disable-custom-all-reduce: vLLM's CustomAllreduce allocates an IPC-shared
# buffer (cudaMalloc + cudaIpcGetMemHandle). The redirect serves cudaMalloc from
# cuMemMap'd fixed-VMM memory, which does NOT support IPC handles → "invalid
# argument". Fall back to NCCL all-reduce (functional; the A/B/C baseline uses
# the same flag so the cold-start comparison is apples-to-apples).
ARGS=(--host 127.0.0.1 --port "$PORT" --served-model-name cs \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --trust-remote-code \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" --disable-custom-all-reduce "${CAPTURE_ARGS[@]}")

# --- interposer + Python-meta env (per-worker %r resolved inside each rank) ---
export LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}"
export SNAPSHOT_RECORD_CUDA_MODE="$MODE"
export SNAPSHOT_RECORD_CUDA_DIR="${SNAP_ROOT}/rank%r"
export SNAPSHOT_REDIRECT_REGION_GIB="$REGION_GIB"
export PYTHONPATH="${SNAP_CUDA_DIR}/snapshot/recipe/cginst_cuda:${PYTHONPATH:-}"
if [ "$MODE" = "record" ]; then
  export VLLM_CG_RECORD_META="${META_ROOT}/rank%r.json"
  unset VLLM_CG_RESTORE_META || true
else
  export VLLM_CG_RESTORE_META="${META_ROOT}/rank%r.json"
  unset VLLM_CG_RECORD_META || true
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DEPLOY_DIR}/cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DEPLOY_DIR}/cache/vllm}"
export PYTHONHASHSEED=0

echo "[n5b-${MODE}] MODE=$MODE TP=$TP gmu=$GMU region=${REGION_GIB}GiB start=$(date +%T)"
t0=$(date +%s)
vllm serve "$MODEL" "${ARGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!
READY_AT=""
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  if kill -0 "$SERVER_PID" 2>/dev/null && \
     curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY_AT="$elapsed"; echo "[n5b-${MODE}] READY at ${elapsed}s"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[n5b-${MODE}] SERVER_EXITED at ${elapsed}s"; tail -60 "$LOG"; break; fi
  if [ "$elapsed" -gt "$DEADLINE" ]; then echo "[n5b-${MODE}] DEADLINE at ${elapsed}s"; break; fi
  sleep 2
done
echo "N5B_${MODE^^}_READY_AT=${READY_AT:-NONE}"

# One greedy completion (correctness smoke; the same prompt is reused for the
# token-identical gate in the restore run).
RESP=$(curl -sS "http://127.0.0.1:${PORT}/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"cs","prompt":"The capital of France is","max_tokens":8,"temperature":0}' 2>/dev/null || true)
echo "[n5b-${MODE}] completion: ${RESP}"
echo "${RESP}" | grep -qi "paris" && echo "N5B_${MODE^^}_INFERENCE=ok" || echo "N5B_${MODE^^}_INFERENCE=failed"

# Fixed-prompt-set probe (Task 7 token-identical reference). The record run
# (transparent graph-mode serving) is the baseline; the restore run reproduces
# it. Save to PROBE_OUT if set.
if [ -n "${PROBE_OUT:-}" ]; then
  python3 "${SNAP_CUDA_DIR}/snapshot/recipe/_n5b_probe.py" \
    "http://127.0.0.1:${PORT}/v1/completions" > "${PROBE_OUT}" 2>/dev/null || true
  echo "[n5b-${MODE}] probe set -> ${PROBE_OUT} ($(wc -l < "${PROBE_OUT}" 2>/dev/null) prompts)"
fi

# Capture-phase presence in the log (honest accounting — Task 8 cross-check).
echo "--- capture bars (${MODE}) ---"
grep -aoE "Capturing CUDA graphs \([^)]*\): 100%[^[]*\[[0-9]{2}:[0-9]{2}" "$LOG" || true

sleep 2
kill -9 "$SERVER_PID" 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f 'VLLM_RPC' 2>/dev/null
pkill -9 -f 'from multiprocessing' 2>/dev/null
# One serve per srun step (enroot holds CUDA IPC shmem at container level — N3).
sleep 5
echo "[n5b-${MODE}] done=$(date +%T)"
