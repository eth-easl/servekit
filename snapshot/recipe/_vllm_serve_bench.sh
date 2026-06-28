#!/bin/bash
# Serving-performance benchmark: drive `benchmaker llm` against a running vLLM
# server to verify the snapshot interposer (LD_PRELOAD) adds NO serving overhead.
#
# Two modes (select via MODE env):
#   MODE=baseline  — plain `vllm serve` (no LD_PRELOAD). Reference.
#   MODE=restore   — vllm serve under LD_PRELOAD snapshot restore (shim_pw:
#                    PIECEWISE graphs rebuilt from snapshot, FULL live-captured).
#
# Each run:
#   1. launches vllm serve (baseline or restore config), polls /health
#   2. after READY: a short warmup bench (discarded), then two measured phases:
#        PHASE 1 — closed:1   (pure single-stream latency; most sensitive to
#                              per-launch interposer overhead — TTFT/ITL)
#        PHASE 2 — closed:N   (saturating concurrency; throughput + tail latency)
#   3. writes benchmaker run-bundles (summary.json + samples.jsonl) under OUT
#   4. tears down the server
#
# Compare PHASE-1/PHASE-2 summary.json between MODE=baseline and MODE=restore:
#   ttft_s, itl_ms_mean (p50/p99), tokens_per_s, throughput_rps, latency_s p50/p99
#   must match within noise if the interposer is overhead-free.
set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
TP="${TP:-1}"
GMU="${GMU:-0.60}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PORT="${PORT:-8821}"
MODE="${MODE:-baseline}"          # baseline | restore
DEADLINE="${DEADLINE:-900}"
CAPTURE_SIZES="${CAPTURE_SIZES:-}"  # blank => default; "1" => cs=1
OUT="${OUT:-/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/bench-serving}"
PROMPTS="${PROMPTS:-/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/bench-serving/prompts.jsonl}"
TAG="${TAG:-${MODE}}"
CONC_LAT="${CONC_LAT:-1}"         # phase-1 concurrency (latency probe)
CONC_SAT="${CONC_SAT:-32}"        # phase-2 concurrency (saturation)
DUR_LAT="${DUR_LAT:-50s}"
DUR_SAT="${DUR_SAT:-50s}"
MAX_TOKENS="${MAX_TOKENS:-128}"
WARMUP_REQS="${WARMUP_REQS:-8}"
RUN_ID_PREFIX="${RUN_ID_PREFIX:-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "$OUT"

# ---- install benchmaker into a SHARED path (container instances are
#      ephemeral across srun jobs; a --target install on /capstor persists).
PYENV="$OUT/pyenv"
if [ ! -f "$PYENV/bin/benchmaker" ]; then
  echo "[bench] installing benchmaker -> $PYENV"
  # flock-serialize concurrent installs (A/B jobs may start together).
  (
    flock -x 200
    [ -f "$PYENV/bin/benchmaker" ] || \
      pip install --quiet --target="$PYENV" benchmaker aiohttp click pyyaml 2>&1 | tail -2
  ) 200>"$OUT/.pyenv.lock"
fi
export PYTHONPATH="$PYENV:${PYTHONPATH:-}"
export PATH="$PYENV/bin:${PATH}"
# Sanity-check the CLI is reachable before starting the server.
if ! command -v benchmaker >/dev/null 2>&1; then
  echo "[bench] FATAL: benchmaker CLI not found after install" >&2
  exit 1
fi

# Ensure prompts exist (deterministic; identical across runs).
if [ ! -f "$PROMPTS" ]; then
  python3 snapshot/recipe/_gen_bench_prompts.py 120 0 "$PROMPTS"
fi

# ---- build vllm serve args ------------------------------------------------
CAPTURE_ARGS=()
if [ -n "$CAPTURE_SIZES" ]; then
  # shellcheck disable=SC2206
  CAPTURE_ARGS=(--cudagraph-capture-sizes $CAPTURE_SIZES)
fi
ARGS=(--host 127.0.0.1 --port "$PORT" --served-model-name cs \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --trust-remote-code \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" "${CAPTURE_ARGS[@]}")

# ---- environment per mode -------------------------------------------------
VLLM_ENV=()
PRELOAD_ENV=()
if [ "$MODE" = "restore" ]; then
  LIB=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/build
  RESTORE_DIR="${RESTORE_DIR:-/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/record-default-pw}"
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$LIB/libsnapshot_redirect.so $LIB/libsnapshot_record.so"
  export SNAPSHOT_RESTORE_DIR="$RESTORE_DIR"
  export SNAPSHOT_REDIRECT_ARENA=1
  export SNAPSHOT_REDIRECT_FIXED_BASE="${SNAPSHOT_REDIRECT_FIXED_BASE:-1}"
  export SNAPSHOT_REDIRECT_REGION_GIB="${REGION_GIB:-72}"
  export SNAPSHOT_REDIRECT_ALLOC_DIR="/tmp/restore-alloc-${SLURM_JOB_ID:-local}"
  export SNAPSHOT_REDIRECT_VERBOSE=0
  export SNAPSHOT_RECORD_NAMEREF_MODE=off
  export SNAPSHOT_RECORD_VERBOSE=0
  export SNAPSHOT_RECORD_OUT_DIR=/tmp/restore-dummy
  export SNAPSHOT_RESTORE_SNAP_MODULES="${SNAPSHOT_RESTORE_SNAP_MODULES:-1}"
  export PYTHONPATH="/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe/cginst_skip:${PYTHONPATH:-}"
  export VLLM_CG_SKIP_CAPTURE="${VLLM_CG_SKIP_CAPTURE:-shim_pw}"
  echo "[bench] MODE=restore RESTORE_DIR=$RESTORE_DIR FIXED_BASE=$SNAPSHOT_REDIRECT_FIXED_BASE"
fi
# Shared frozen caches (identical compiled artifacts in both modes => fair A/B).
TRI="${SNAPSHOT_TRITON_CACHE_DIR:-/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-triton-v4}"
VCC="${SNAPSHOT_VLLM_CACHE_ROOT:-/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/e2e-vllm-cache4}"
export TRITON_CACHE_DIR="$TRI"
export VLLM_CACHE_ROOT="$VCC"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

LOG="$OUT/vllm_${MODE}_${SLURM_JOB_ID:-local}.log"
rm -f "$LOG"

echo "[bench] MODE=$MODE start=$(date +%T) TP=$TP gmu=$GMU capture='${CAPTURE_SIZES:-default}'"

t0=$(date +%s)
vllm serve "$MODEL" "${ARGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!

# ---- poll /health ---------------------------------------------------------
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  if kill -0 "$SERVER_PID" 2>/dev/null && \
     curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[bench] MODE=$MODE READY at ${elapsed}s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[bench] MODE=$MODE SERVER_EXITED at ${elapsed}s"
    tail -20 "$LOG" >&2
    exit 1
  fi
  if [ "$elapsed" -gt "$DEADLINE" ]; then
    echo "[bench] MODE=$MODE DEADLINE at ${elapsed}s"
    kill -9 "$SERVER_PID" 2>/dev/null
    exit 1
  fi
  sleep 3
done

URL="http://127.0.0.1:${PORT}/v1/chat/completions"

# ---- warmup (discard) — JIT the eager/first-request path -----------------
echo "[bench] warmup ($WARMUP_REQS reqs, discarded)..."
benchmaker llm --url "$URL" --model cs \
  --prompts-jsonl "$PROMPTS" --prompt-field prompt \
  --max-tokens 16 --temperature 0 \
  --rate "closed:2" --duration 20s --timeout 300 \
  --out-dir /tmp/bench-warmup-$$ >/dev/null 2>&1 || true
rm -rf /tmp/bench-warmup-$$

# ---- PHASE 1: latency probe (single stream) ------------------------------
echo "[bench] PHASE 1 latency: closed:$CONC_LAT $DUR_LAT"
benchmaker llm --url "$URL" --model cs \
  --prompts-jsonl "$PROMPTS" --prompt-field prompt \
  --max-tokens "$MAX_TOKENS" --temperature 0 \
  --rate "closed:$CONC_LAT" --duration "$DUR_LAT" --timeout 300 \
  --out-dir "$OUT/${TAG}-${RUN_ID_PREFIX}-lat" \
  --label "mode=$MODE" --label "phase=latency" \
  --label "capture=${CAPTURE_SIZES:-default}" 2>&1 | tee "$OUT/${TAG}-${RUN_ID_PREFIX}-lat.txt" | tail -40

# ---- PHASE 2: saturation throughput --------------------------------------
echo "[bench] PHASE 2 saturation: closed:$CONC_SAT $DUR_SAT"
benchmaker llm --url "$URL" --model cs \
  --prompts-jsonl "$PROMPTS" --prompt-field prompt \
  --max-tokens "$MAX_TOKENS" --temperature 0 \
  --rate "closed:$CONC_SAT" --duration "$DUR_SAT" --timeout 300 \
  --out-dir "$OUT/${TAG}-${RUN_ID_PREFIX}-sat" \
  --label "mode=$MODE" --label "phase=saturation" \
  --label "capture=${CAPTURE_SIZES:-default}" 2>&1 | tee "$OUT/${TAG}-${RUN_ID_PREFIX}-sat.txt" | tail -40

echo "[bench] MODE=$MODE done=$(date +%T)"

# ---- teardown -------------------------------------------------------------
sleep 2
kill -9 "$SERVER_PID" 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f 'VLLM_RPC' 2>/dev/null
sleep 3
echo "[bench] MODE=$MODE teardown complete"
