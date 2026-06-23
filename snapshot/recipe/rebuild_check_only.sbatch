#!/bin/bash
#SBATCH --job-name=snapshot-rebuildchk
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -euo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
export EDF_PATH="/capstor/scratch/cscs/xyao/glm-47-flash-vllm/deploy/glm-47-flash-beverin:${HOME}/.edf"
SNAP=snapshot/build/snapshot
echo "rebuild-check start: $(date --iso-8601=seconds) node=$(hostname)"

srun --account=root --gres=gpu:1 --time=00:04:00 --environment=glm-47-flash-rocm \
  bash -c 'cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && \
    echo "=== inspect ===" && \
    SNAPSHOT_REBUILD_DEBUG=1 snapshot/build/snapshot inspect snapshot/record-vllm/graph-169374-0.snap && \
    echo "=== rebuild-check ===" && \
    SNAPSHOT_REBUILD_DEBUG=1 snapshot/build/snapshot rebuild-check snapshot/record-vllm/graph-169374-0.snap; \
    echo "REBUILD_CHECK_EXIT=$?"'

echo "rebuild-check finished: $(date --iso-8601=seconds)"
