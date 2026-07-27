#!/usr/bin/env bash
# Baseline: serve Apertus-8B on ONE GPU (TP1) and profile + benchmark it with servekit.
#
# This is the reference cold-start we later compare checkpoint/restore against. It runs
# `servekit profile` (cold-start phase table) alongside `servekit bench` (throughput +
# correctness), which merge into one report JSON --
# wrapping the same launch command the cluster uses (profile/apertus-8b-bristen/
# serve_apertus_8b_sglang.sbatch) with knobs reduced to fit a 24 GB RTX 3090.
#
# TWO cold-start confounds, both controlled by --cold:
#   1. page cache (weights)   -- warm weights make weight_loading ~3x too fast. Evicted
#                                via cache_tools.py (posix_fadvise).
#   2. JIT/compile cache      -- the FIRST launch compiles flashinfer/triton/sglang
#      (~/.cache/*)              kernels (dominates cuda_graph_capture + warmup: ~38s the
#                                first time, ~3s once cached). The cluster neutralizes
#                                this with an ephemeral HOME=/root; we redirect the cache
#                                env vars to a WIPED throwaway dir so compilation is cold.
# A true cold node (fresh deploy) has BOTH cold -- that's what --cold reproduces, and
# what a checkpoint/restore must be compared against.
#
# Usage:
#   scripts/baseline.sh            # profile+bench in whatever cache state (warm/dirty)
#   scripts/baseline.sh --cold     # TRUE cold: evict weights AND wipe compile caches
#
# Env overrides: CKPT_VENV, MODEL_DIR, PORT, CONTEXT_LEN, MEM_FRACTION, MAX_RUNNING.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${CKPT_VENV:-$HOME/ckpt-venv}"
SERVEKIT="$VENV/bin/servekit"
PYBIN="$VENV/bin/python"
# Put the venv bin on PATH so child processes (e.g. SGLang's JIT calling `ninja`) find it.
export PATH="$VENV/bin:$PATH"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Apertus-8B-Instruct-2509}"
SERVED_NAME="swiss-ai/Apertus-8B-Instruct-2509"
PORT="${PORT:-8080}"

# 24 GB-tuned vs cluster (32768 / 0.85 / 256). 8B bf16 ~16 GB weights; the rest is KV +
# CUDA graphs. cuda graphs stay ENABLED -- graph capture is a cold-start phase we measure.
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_RUNNING="${MAX_RUNNING:-8}"

COLD=0
[ "${1:-}" = "--cold" ] && COLD=1

[ -x "$SERVEKIT" ] || { echo "servekit not found in $VENV -- run setup_env.sh install"; exit 2; }
ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || { echo "weights missing in $MODEL_DIR -- run setup_env.sh download"; exit 2; }

TS="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$HERE/results/baseline"
OUT="$HERE/results/baseline/baseline-${TS}-profile.json"

echo "==== baseline :: Apertus-8B TP1 profile + bench ===="
echo "model:   $MODEL_DIR"
echo "config:  ctx=$CONTEXT_LEN mem_frac=$MEM_FRACTION max_running=$MAX_RUNNING tp=1"
echo "out:     $OUT"

if [ "$COLD" = 1 ]; then
  echo; echo "-- [cold 1/2] evicting weights from page cache --"
  "$PYBIN" "$HERE/scripts/cache_tools.py" evict --verify "$MODEL_DIR"

  echo; echo "-- [cold 2/2] wiping compile caches (flashinfer/triton/inductor/sglang) --"
  # Redirect every JIT cache to a fresh throwaway dir so kernels compile from scratch,
  # mirroring the cluster's ephemeral HOME. Does NOT touch the user's real ~/.cache.
  COLD_CACHE="$HERE/results/baseline/.cold-compile-cache"
  rm -rf "$COLD_CACHE"; mkdir -p "$COLD_CACHE"
  export FLASHINFER_WORKSPACE_BASE="$COLD_CACHE"       # -> $COLD_CACHE/.cache/flashinfer
  export SGLANG_CACHE_DIR="$COLD_CACHE/sglang"
  export TRITON_CACHE_DIR="$COLD_CACHE/triton"
  export TORCHINDUCTOR_CACHE_DIR="$COLD_CACHE/inductor"
  export XDG_CACHE_HOME="$COLD_CACHE"                  # catch-all for anything else
  echo "   compile caches -> $COLD_CACHE (wiped)"
fi

# servekit profile and servekit bench are separate commands run side by side, joined
# by $OUT: profile writes the report the moment the server is ready, bench waits for
# that file to appear (so its load never lands inside the measured window) and then
# merges its results back into the same JSON. Same report shape as before.
echo; echo "-- launch (servekit profile wraps sglang.launch_server) --"
"$SERVEKIT" profile --out "$OUT" \
  -- \
  "$PYBIN" -m sglang.launch_server \
    --model-path "$MODEL_DIR" \
    --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --context-length "$CONTEXT_LEN" \
    --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$MAX_RUNNING" \
    --enable-metrics &
PROF_PID=$!
# Nothing else tears the server down now, so make sure an early exit still does.
trap 'kill $PROF_PID 2>/dev/null' EXIT

echo; echo "-- bench (waits for the profile report, then loads the server) --"
"$SERVEKIT" bench --url "http://127.0.0.1:$PORT" --into "$OUT" \
  --wait-ready 1800 --requests 64 --concurrency 16 --input-len 512 --output-len 128
RC=$?

kill $PROF_PID 2>/dev/null
wait $PROF_PID 2>/dev/null
trap - EXIT

echo; echo "==== baseline done (rc=$RC) -> $OUT ===="
exit $RC
