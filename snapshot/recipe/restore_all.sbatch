#!/bin/bash
#SBATCH --job-name=snapshot-restore-all
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
export EDF_PATH="/capstor/scratch/cscs/xyao/glm-47-flash-vllm/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"

SNAP_DIR="${SNAP_DIR:-snapshot/record-vllm}"
echo "restore-all start: $(date --iso-8601=seconds) node=$(hostname) dir=$SNAP_DIR"

# Rebuild the CLI on this GPU node so it links against the matching ROCm.
srun --account=root --gres=gpu:1 --time=00:03:00 --environment=glm-47-flash-rocm \
  bash -c "cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && cmake -S snapshot -B snapshot/build >/dev/null 2>&1; cmake --build snapshot/build --target snapshot -j8 2>&1 | tail -3"

srun --account=root --gres=gpu:1 --time=00:05:00 --environment=glm-47-flash-rocm \
  bash -c "cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && \
    export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-} && \
    snapshot/build/snapshot restore-all $SNAP_DIR; \
    echo RESTORE_ALL_EXIT=\$?"

echo "restore-all finished: $(date --iso-8601=seconds)"
