#!/bin/bash
# Spike: does a persistent TRITON_CACHE_DIR make Triton HSACOs byte-identical
# across cold starts? (If yes, the snapshot content-hash restore becomes viable.)
#
# Controlled A/B:
#   SPIKE_PHASE=A : clear SPIKE_CACHE_DIR, then run (autotune COLD, populates cache)
#   SPIKE_PHASE=B : keep SPIKE_CACHE_DIR, then run (cache WARM -> cache hits)
# Each phase writes HSACO dumps (SNAPSHOT_RECORD_DUMP_HSA=1) to SPIKE_OUT, which
# we later diff by .text content (see hsa_name_map.sh). Names are read from each
# dump's ELF symtab, so we do NOT depend on the recorder's namer draining.
set -uo pipefail

cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
REDIRECT="$PWD/snapshot/build/libsnapshot_redirect.so"
RECORDER="$PWD/snapshot/build/libsnapshot_record.so"
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash

SPIKE_PHASE="${SPIKE_PHASE:-A}"
SPIKE_CACHE_DIR="${SPIKE_CACHE_DIR:-/capstor/scratch/cscs/xyao/kimi-k25-vllm/spike-triton-cache}"
SPIKE_OUT="${SPIKE_OUT:-/capstor/scratch/cscs/xyao/kimi-k25-vllm/spike-out-${SPIKE_PHASE}}"
PORT="${PORT:-8831}"

# Phase A = cold cache (clear it so we control exactly what's warm for phase B).
if [ "$SPIKE_PHASE" = "A" ]; then
  rm -rf "$SPIKE_CACHE_DIR"
fi
mkdir -p "$SPIKE_CACHE_DIR" "$SPIKE_OUT"
rm -f "$SPIKE_OUT"/hsa-dump-*.co

# This is the lever under test: a persistent cache dir shared across phases.
export TRITON_CACHE_DIR="$SPIKE_CACHE_DIR"

export LD_PRELOAD="$REDIRECT $RECORDER"
export SNAPSHOT_RECORD_OUT_DIR="$SPIKE_OUT"
export SNAPSHOT_RECORD_MAX_GRAPHS=4
export SNAPSHOT_RECORD_VERBOSE=0
export SNAPSHOT_RECORD_DRAIN_MODE=auto      # naming not needed; names come from dumps
export SNAPSHOT_RECORD_NAMEREF_MODE=off
export SNAPSHOT_RECORD_DUMP_HSA=1           # write each HSACO to disk
export SNAPSHOT_REDIRECT_ARENA=1
export SNAPSHOT_REDIRECT_REGION_GIB=72
export SNAPSHOT_REDIRECT_ALLOC_DIR="$SPIKE_OUT"
export SNAPSHOT_REDIRECT_VERBOSE=0
export VLLM_LOGGING_LEVEL=WARNING

echo "[spike-$SPIKE_PHASE] TRITON_CACHE_DIR=$TRITON_CACHE_DIR (cache entries: $(find "$TRITON_CACHE_DIR" -type f 2>/dev/null | wc -l))"
echo "[spike-$SPIKE_PHASE] launching vLLM (TP=1 gmu=0.60 capture_sizes=1)..."

t0=$(date +%s)
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name sp \
  --tensor-parallel-size 1 --pipeline-parallel-size 1 --trust-remote-code \
  --gpu-memory-utilization 0.60 --max-model-len 8192 --max-num-seqs 64 \
  --cudagraph-capture-sizes 1 > "$SPIKE_OUT/vllm.log" 2>&1 &
SERVER_PID=$!

# Wait until ready, send a couple of completions to ensure capture finishes,
# then we have everything (HSACO dumps are written at module-load time, well
# before /health, so even an early exit captures them).
READY=0
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  if kill -0 "$SERVER_PID" 2>/dev/null && \
     curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    READY=1; echo "[spike-$SPIKE_PHASE] READY at ${elapsed}s"
    # drive a few requests so the capture phase completes and graphs FLUSH
    for i in 1 2 3; do
      curl -sS "http://127.0.0.1:${PORT}/v1/completions" -H 'Content-Type: application/json' \
        -d '{"model":"sp","prompt":"Hello","max_tokens":2,"temperature":0}' >/dev/null 2>&1 || true
    done
    sleep 5
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[spike-$SPIKE_PHASE] SERVER_EXITED at ${elapsed}s"; break
  fi
  [ "$elapsed" -gt 600 ] && { echo "[spike-$SPIKE_PHASE] DEADLINE"; break; }
  sleep 3
done

kill -9 "$SERVER_PID" 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f 'VLLM_RPC' 2>/dev/null
sleep 3

echo "[spike-$SPIKE_PHASE] HSACO dumps: $(ls "$SPIKE_OUT"/hsa-dump-*.co 2>/dev/null | wc -l)"
echo "[spike-$SPIKE_PHASE] cache entries now: $(find "$TRITON_CACHE_DIR" -type f 2>/dev/null | wc -l)"
echo "[spike-$SPIKE_PHASE] DONE"
