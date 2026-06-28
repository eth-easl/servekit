#!/bin/bash
#SBATCH --job-name=snap-serve-bench
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:45:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
GLM_DEPLOY=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
export EDF_PATH="${GLM_DEPLOY}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"

echo "serve-bench start: $(date --iso-8601=seconds) node=$(hostname) MODE=${MODE:-baseline}"

srun --environment=glm-47-flash-rocm bash /capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe/_vllm_serve_bench.sh

echo "serve-bench finished: $(date --iso-8601=seconds)"
