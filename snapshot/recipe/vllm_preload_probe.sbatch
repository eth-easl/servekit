#!/bin/bash
#SBATCH --job-name=snapshot-vllm-probe
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -uo pipefail

GLM_DEPLOY=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
RECIPE=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe
export EDF_PATH="${GLM_DEPLOY}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"
mkdir -p /capstor/scratch/cscs/xyao/kimi-k25-vllm/logs

echo "vllm preload probe start: $(date --iso-8601=seconds) node $(hostname)"
echo "EDF_PATH=${EDF_PATH}"

srun --environment=glm-47-flash-rocm "${RECIPE}/_vllm_preload_probe.sh" || true

echo "vllm preload probe end: $(date --iso-8601=seconds)"
