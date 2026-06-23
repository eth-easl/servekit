#!/bin/bash
# Runs INSIDE the ROCm container under the glm-47-flash EDF. Launches one real
# vllm serve cold start with the snapshot LD_PRELOAD interposer attached, so the
# interposer reports each HIP graph-capture window (duration + node count) that
# occurs during vLLM startup. Killed shortly after capture completes.
set -uo pipefail

PRELOAD=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/build/libsnapshot_preload.so
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash

export LD_PRELOAD="$PRELOAD"
export SNAPSHOT_PRELOAD_VERBOSE=1
export VLLM_LOGGING_LEVEL=INFO

echo "[probe] LD_PRELOAD=$PRELOAD"
echo "[probe] launching vllm serve at $(date +%T)"

# --kill-after forces SIGKILL 30s after SIGTERM so workers do not hang ~10min
# (timeout vllm serve does not shut down cleanly on ROCm).
timeout --kill-after=30s --signal=TERM "${RUN_SECS:-600}" \
  vllm serve "$MODEL" \
    --host 127.0.0.1 --port 8799 --served-model-name preload-probe \
    --tensor-parallel-size 4 --pipeline-parallel-size 1 \
    --trust-remote-code --gpu-memory-utilization 0.50 \
    --max-model-len 8192 --max-num-seqs 64 || true

echo "[probe] vllm exited at $(date +%T)"
