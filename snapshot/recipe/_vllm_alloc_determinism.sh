#!/bin/bash
# Runs vllm serve TWICE under the alloc-logging interposer and diffs the device
# allocation sequences across the two cold starts, to judge whether torch's HIP
# allocations (and thus the addresses embedded in captured graphs) are
# deterministic — the central feasibility question for snapshotting vLLM.
set -uo pipefail

PRELOAD=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/build/libsnapshot_preload.so
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
BASE=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/alloc-determinism
rm -rf "$BASE"; mkdir -p "$BASE/run1" "$BASE/run2"

export LD_PRELOAD="$PRELOAD"
export VLLM_LOGGING_LEVEL=INFO

run_once () {
  local dir="$1"
  export SNAPSHOT_PRELOAD_ALLOC_DIR="$dir"
  timeout --kill-after=30s --signal=TERM "${RUN_SECS:-360}" \
    vllm serve "$MODEL" --host 127.0.0.1 --port 8799 --served-model-name det \
      --tensor-parallel-size 4 --pipeline-parallel-size 1 --trust-remote-code \
      --gpu-memory-utilization 0.50 --max-model-len 8192 --max-num-seqs 64 \
      > "$dir/vllm.log" 2>&1 || true
}

echo "[det] run1 start $(date +%T)"; run_once "$BASE/run1"; echo "[det] run1 end $(date +%T)"
echo "[det] run2 start $(date +%T)"; run_once "$BASE/run2"; echo "[det] run2 end $(date +%T)"

echo "=== alloc log line counts ==="
wc -l "$BASE"/run1/alloc-*.log "$BASE"/run2/alloc-*.log 2>/dev/null

W1=$(ls -S "$BASE"/run1/alloc-*.log 2>/dev/null | head -1)
W2=$(ls -S "$BASE"/run2/alloc-*.log 2>/dev/null | head -1)
echo "[det] busiest worker: run1=$W1 run2=$W2"
[ -z "$W1" ] || [ -z "$W2" ] && { echo "[det] missing alloc logs (interposer saw no allocations?)"; exit 0; }

echo "=== first 15 allocations side by side (run1 | run2) ==="
paste <(awk '{print $2,$3,$4}' "$W1" | head -15) <(awk '{print $2,$3,$4}' "$W2" | head -15)

echo "=== SIZE/OP sequence identical across cold starts? ==="
if diff <(awk '{print $2,$3}' "$W1") <(awk '{print $2,$3}' "$W2") > "$BASE/size_seq.diff" 2>&1; then
  echo "RESULT: IDENTICAL size/op sequence ($(wc -l < "$W1") allocations)"
else
  echo "RESULT: size/op sequence DIFFERS (first diffs):"; head -20 "$BASE/size_seq.diff"
fi

echo "=== PTR sequence identical across cold starts (addresses reproducible)? ==="
if diff <(awk '{print $2,$3,$4}' "$W1") <(awk '{print $2,$3,$4}' "$W2") > "$BASE/ptr_seq.diff" 2>&1; then
  echo "RESULT: IDENTICAL ptr sequence -> addresses are byte-identical across cold starts (Delta=0)"
else
  echo "RESULT: ptr sequence differs (sizes may still match). first M/R ptr pairs (run1 vs run2):"
  paste <(grep -E ' [MR] ' "$W1" | awk '{print $4}') <(grep -E ' [MR] ' "$W2" | awk '{print $4}') | head -8
fi
