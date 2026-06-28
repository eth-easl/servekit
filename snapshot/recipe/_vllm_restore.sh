#!/bin/bash
# Measure vLLM cold-start time when CUDA graphs are RESTORED from snapshots
# instead of captured live. Same structure as _vllm_coldstart.sh, but sets
# LD_PRELOAD and SNAPSHOT_RESTORE_DIR so the recorder interposer returns
# pre-built graphs during vLLM's capture phase.
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
TP="${TP:-1}"
GMU="${GMU:-0.60}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PORT="${PORT:-8821}"
DEADLINE="${DEADLINE:-800}"
RESTORE_DIR="${RESTORE_DIR:-/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-vllm}"
LIB=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/build

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

# Optional --compilation-config override (must match the record run for
# deterministic HSACOs across cold starts).
if [ -n "${COMPILATION_CONFIG:-}" ]; then
  ARGS+=(--compilation-config "$COMPILATION_CONFIG")
fi

LOG=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/vllm_restore_${SLURM_JOB_ID:-local}.log
rm -f "$LOG"

export REGION_GIB="${REGION_GIB:-72}"
echo "[restore] start=$(date +%T) RESTORE_DIR=$RESTORE_DIR REGION_GIB=$REGION_GIB CAPTURE_SIZES=${CAPTURE_SIZES:-<none>}"
# Ensure ROCm libs are resolvable for the preloaded .so dependencies. Must be
# exported so that ALL processes (date/sleep/curl/vllm) can preload our .so.
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$LIB/libsnapshot_redirect.so $LIB/libsnapshot_record.so"
export SNAPSHOT_RESTORE_DIR="$RESTORE_DIR"
# Recreate the SAME deterministic arena the snapshots were recorded under, so
# the baked-in (arena-relative) device pointers in the .snap files resolve to
# valid memory. Default REGION_GIB=72 matches recipe/_vllm_record.sh; without
# this the redirect defaults to an 8 GiB arena and every weight alloc spills to
# passthrough -> addresses don't match -> restored graphs reference bad memory.
export SNAPSHOT_REDIRECT_ARENA=1
# FIXED_BASE=1 pins the arena at 0x600000000000 (matching record) so Δ=0 and
# every baked-in device pointer is valid without relocation. Must match the
# value used at record time.
export SNAPSHOT_REDIRECT_FIXED_BASE="${SNAPSHOT_REDIRECT_FIXED_BASE:-0}"
export SNAPSHOT_REDIRECT_REGION_GIB="$REGION_GIB"
export SNAPSHOT_REDIRECT_ALLOC_DIR="/tmp/restore-alloc-${SLURM_JOB_ID:-local}"
export SNAPSHOT_REDIRECT_VERBOSE=0
export SNAPSHOT_RECORD_NAMEREF_MODE=off
export SNAPSHOT_RECORD_VERBOSE=0
export SNAPSHOT_RECORD_OUT_DIR=/tmp/restore-dummy
# vLLM manages BOTH the Triton and inductor caches itself under
# VLLM_CACHE_ROOT (it overrides TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR to
# subdirs of $VLLM_CACHE_ROOT/torch_compile_cache/<hash>/). So the ONLY knob
# that matters for reproducible compiled artifacts is VLLM_CACHE_ROOT. Must
# point at the SAME dedicated frozen dir the graphs were recorded under so the
# record/live compiled HSACOs match byte-for-byte and the allocation patterns
# coincide. See _vllm_record.sh for the full rationale.
export VLLM_CACHE_ROOT="${SNAPSHOT_VLLM_CACHE_ROOT:-${VLLM_CACHE_ROOT:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm/cache/vllm}}"
# ALSO freeze the EAGER Triton kernels (flash-attn, MoE) which JIT-compile
# before vLLM's inductor override. Must match the record's dedicated dir.
export TRITON_CACHE_DIR="${SNAPSHOT_TRITON_CACHE_DIR:-${TRITON_CACHE_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm/cache/triton}}"
# Freeze Python's hash seed — MUST match the record run. See _vllm_record.sh
# for the full rationale (inductor fusion nondeterminism without this).
export PYTHONHASHSEED="${SNAPSHOT_PYTHONHASHSEED:-0}"

# cginst_skip/sitecustomize.py loads whichever hook its env var requests:
#   VLLM_CG_SKIP_CAPTURE=shim   — skip forward, graphs rebuilt from snapshot
#   VLLM_CG_RESTORE_META=<path> — lazy restore: skip forward + reconstruct
#                                 entry.output from recorded metadata JSON.
if [ -n "${VLLM_CG_SKIP_CAPTURE:-}" ] || [ -n "${VLLM_CG_RESTORE_META:-}" ]; then
  export PYTHONPATH="/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe/cginst_skip:${PYTHONPATH:-}"
  [ -n "${VLLM_CG_SKIP_CAPTURE:-}" ] && \
    echo "[restore] CG skip-capture (shim) active — forward suppressed, graphs rebuilt from snapshot"
  [ -n "${VLLM_CG_RESTORE_META:-}" ] && \
    echo "[restore] CG lazy-restore active — forward suppressed, entry.output reconstructed from ${VLLM_CG_RESTORE_META}"
fi

t0=$(date +%s)
vllm serve "$MODEL" "${ARGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!

# Poll /health.
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  if kill -0 "$SERVER_PID" 2>/dev/null && \
     curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[restore] READY at ${elapsed}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[restore] SERVER_EXITED at ${elapsed}s"
    break
  fi
  if [ "$elapsed" -gt "$DEADLINE" ]; then
    echo "[restore] DEADLINE at ${elapsed}s"
    break
  fi
  sleep 2
done

# Send one request to confirm inference works — and PRINT the output so a
# silently-broken restore (stale baked-in pointers) shows up as garbage rather
# than a misleading "inference=ok" from a 200 with empty/wrong content.
echo "[restore] inference probe (expect 'Paris'):"
curl -sS "http://127.0.0.1:${PORT}/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"cs","prompt":"The capital of France is","max_tokens":4,"temperature":0}' \
  2>&1 | head -c 600; echo

sleep 2
kill -9 "$SERVER_PID" 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f 'VLLM_RPC' 2>/dev/null
sleep 3
echo "[restore] done=$(date +%T)"
