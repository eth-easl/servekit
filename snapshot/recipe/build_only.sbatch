#!/bin/bash
#SBATCH --job-name=snapshot-buildonly
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -euo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
export EDF_PATH="/capstor/scratch/cscs/xyao/glm-47-flash-vllm/deploy/glm-47-flash-beverin:${HOME}/.edf"
echo "build started: $(date --iso-8601=seconds) node=$(hostname)"
srun --account=root --gres=gpu:1 --time=00:12:00 --environment=glm-47-flash-rocm \
  bash -c 'cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && \
    cmake --build snapshot/build -j${SLURM_CPUS_PER_TASK:-8} 2>&1 | \
    grep -iE "error|warning|Built target snapshot" | head -20; echo BUILD_DONE'
echo "build finished: $(date --iso-8601=seconds)"
