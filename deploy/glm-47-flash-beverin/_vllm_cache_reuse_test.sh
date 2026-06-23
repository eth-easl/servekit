#!/bin/bash
# Definitive vLLM compile-cache reuse test (runs INSIDE the ROCm container).
# Runs vllm serve TWICE with an isolated VLLM_CACHE_ROOT, snapshots the
# torch_compile_cache/<hash_key>/ dir + cache_key_factors.json after each,
# and diffs them. Reuse => identical hash_key. Miss => hash_key differs and
# the factors json shows which input changed.
set -uo pipefail   # no -e (vllm killed by timeout is expected)
DEPLOY_DIR=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
ISO_ROOT="$DEPLOY_DIR/cache/vllm_isolated_test"
LOG1="$DEPLOY_DIR/logs/vllm_reuse_test_run1.log"
LOG2="$DEPLOY_DIR/logs/vllm_reuse_test_run2.log"
rm -rf "$ISO_ROOT"; mkdir -p "$ISO_ROOT"

export VLLM_CACHE_ROOT="$ISO_ROOT"
export TRITON_CACHE_DIR="$DEPLOY_DIR/cache/triton"   # already warm; keep stable
export VLLM_LOGGING_LEVEL=INFO
export VLLM_NO_USAGE_STATS=1

run_once () {
  local label="$1" log="$2"
  echo "===== $label: start $(date +%T) ====="
  timeout "${RUN_SECS:-420}" vllm serve "$MODEL" \
    --host 127.0.0.1 --port 8799 --served-model-name reuse-test \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --enable-prefix-caching --trust-remote-code \
    --gpu-memory-utilization 0.50 --max-model-len 8192 --max-num-seqs 64 \
    > "$log" 2>&1 || true
  echo "===== $label: end $(date +%T) (exit) ====="
  echo "--- $label: hash_key dirs created ---"
  find "$ISO_ROOT/torch_compile_cache" -mindepth 1 -maxdepth 1 -type d 2>/dev/null
  echo "--- $label: first capture iter timing ---"
  tr '\r' '\n' < "$log" | grep -oE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, +[0-9.]+s/it\]" | head -2
  echo "--- $label: 'Using cache directory' line ---"
  grep -oE "Using cache directory: [^ ]+ for vLLM" "$log" | head -1
}

run_once "RUN1" "$LOG1"
HASHKEYS1=$(find "$ISO_ROOT/torch_compile_cache" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" 2>/dev/null | sort)
FACTORS1=$(find "$ISO_ROOT/torch_compile_cache" -name cache_key_factors.json 2>/dev/null | head -1)
cp "$FACTORS1" "$ISO_ROOT/factors_run1.json" 2>/dev/null

run_once "RUN2" "$LOG2"
HASHKEYS2=$(find "$ISO_ROOT/torch_compile_cache" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" 2>/dev/null | sort)
FACTORS2=$(find "$ISO_ROOT/torch_compile_cache" -name cache_key_factors.json 2>/dev/null | head -1)
cp "$FACTORS2" "$ISO_ROOT/factors_run2.json" 2>/dev/null

echo
echo "############ VERDICT ############"
echo "RUN1 hash_keys:"; echo "$HASHKEYS1"
echo "RUN2 hash_keys:"; echo "$HASHKEYS2"
if [ "$HASHKEYS1" = "$HASHKEYS2" ] && [ -n "$HASHKEYS1" ]; then
  echo ">>> SAME hash_key -> vLLM compile cache IS keyed consistently; miss is elsewhere"
else
  echo ">>> DIFFERENT hash_keys -> vLLM cache key is non-deterministic (root cause)"
  echo ">>> diff of cache_key_factors.json (run1 vs run2):"
  diff <(python3 -m json.tool "$ISO_ROOT/factors_run1.json" 2>/dev/null) \
       <(python3 -m json.tool "$ISO_ROOT/factors_run2.json" 2>/dev/null) | head -40
fi
