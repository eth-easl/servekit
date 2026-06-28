#!/bin/bash
#SBATCH --job-name=test-ldpreload
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:05:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
GLM_DEPLOY=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
export EDF_PATH="${GLM_DEPLOY}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"

srun --environment=glm-47-flash-rocm bash /capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe/_test_ldpreload.sh
