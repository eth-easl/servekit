#!/bin/bash
#SBATCH --job-name=triton-spike
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:25:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -uo pipefail
GLM_DEPLOY=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
export EDF_PATH="${GLM_DEPLOY}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"
mkdir -p /capstor/scratch/cscs/xyao/kimi-k25-vllm/logs

echo "triton-cache-spike start: $(date --iso-8601=seconds) node=$(hostname) PHASE=${SPIKE_PHASE:-A}"

# Rebuild the recorder so the run reflects current source (the .so is reused
# across jobs otherwise).
srun --environment=glm-47-flash-rocm bash -c \
  'cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && cmake -S snapshot -B snapshot/build >/dev/null 2>&1; cmake --build snapshot/build -j"${SLURM_CPUS_PER_TASK:-8}" 2>&1 | tail -3' || true

srun --environment=glm-47-flash-rocm bash \
  /capstor/scratch/cscs/xyao/kimi-k25-vllm/deploy/bench-storage/_triton_cache_spike.sh

echo "triton-cache-spike end: $(date --iso-8601=seconds)"
