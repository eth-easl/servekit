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
SNAP_FILE="${SNAPSHOT_PATH:-snapshot/record-vllm/graph-69920-0.snap}"
echo "rebuild-check start: $(date --iso-8601=seconds) node=$(hostname) snap=$SNAP_FILE"

# Rebuild the CLI on this GPU node so it links against the matching ROCm.
srun --account=root --gres=gpu:1 --time=00:03:00 --environment=glm-47-flash-rocm \
  bash -c "cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && cmake -S snapshot -B snapshot/build >/dev/null 2>&1; cmake --build snapshot/build --target snapshot -j8 2>&1 | tail -3"


srun --account=root --gres=gpu:1 --time=00:04:00 --environment=glm-47-flash-rocm \
  bash -c "cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && \
    export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-} && \
    echo '=== inspect ===' && \
    SNAPSHOT_REBUILD_DEBUG=1 snapshot/build/snapshot inspect $SNAP_FILE && \
    echo '=== analyze ===' && \
    snapshot/build/snapshot analyze $SNAP_FILE && \
    echo '=== rebuild-check ===' && \
    SNAPSHOT_REBUILD_DEBUG=1 SNAPSHOT_SKIP_LAUNCH=1 snapshot/build/snapshot rebuild-check $SNAP_FILE; \
    echo REBUILD_CHECK_EXIT=\$?"

echo "rebuild-check finished: $(date --iso-8601=seconds)"
