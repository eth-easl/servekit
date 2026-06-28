#!/bin/bash
#SBATCH --job-name=snapshot-record-vllm
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -uo pipefail

GLM_DEPLOY=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
RECIPE=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe
export EDF_PATH="${GLM_DEPLOY}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"
mkdir -p /capstor/scratch/cscs/xyao/kimi-k25-vllm/logs

echo "vllm-record start: $(date --iso-8601=seconds) node $(hostname)"

# Rebuild the recorder incrementally so the run always reflects current source
# (the .so is otherwise reused across jobs). cmake config already exists (HIP).
# Reconfigure first: refreshes CMAKE_COMMAND (portable across nodes) and runs
# the hipLaunchConfig_t try_compile so the launch-ex interposers build when the
# GPU node's ROCm provides them.
echo "=== incremental rebuild (recorder + CLI) ==="
srun --environment=glm-47-flash-rocm bash -c \
  'cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && cmake -S snapshot -B snapshot/build >/dev/null 2>&1; cmake --build snapshot/build -j"${SLURM_CPUS_PER_TASK:-8}" 2>&1 | tail -20' || true

# TP=1, gmu=0.60, 72 GiB arena = the M2.4 config proven to run end-to-end on the
# deterministic arena. M3b: CAPTURE_SIZES clamps vLLM's cuda-graph capture to a
# tiny explicit list (default "1") so the capture phase finishes in seconds and
# the model reaches the idle window the off-path namer needs to drain
# hipKernelNameRef (without this, ~528 piecewise sub-graphs keep the HIP capture
# lock busy for minutes and the namer never drains). MAX_GRAPHS=1 serializes
# exactly one real vLLM graph once its every node func is named + FLUSHed.
TP="${TP:-1}" GMU="${GMU:-0.60}" REGION_GIB="${REGION_GIB:-72}" \
  CAPTURE_SIZES="${CAPTURE_SIZES-1}" MAX_GRAPHS="${MAX_GRAPHS:-1}" \
  RUN_SECS="${RUN_SECS:-600}" NAMER_GRACE="${NAMER_GRACE:-90}" \
  srun --environment=glm-47-flash-rocm bash "${RECIPE}/_vllm_record.sh" || true

echo "vllm-record end: $(date --iso-8601=seconds)"
