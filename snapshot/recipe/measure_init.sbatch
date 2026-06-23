#!/bin/bash
#SBATCH --job-name=snapshot-measure-init
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -uo pipefail

GLM_DEPLOY=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
RECIPE=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe
export EDF_PATH="${GLM_DEPLOY}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"
mkdir -p /capstor/scratch/cscs/xyao/kimi-k25-vllm/logs

echo "measure-init start: $(date --iso-8601=seconds) node $(hostname)"

# Warm-cache cold start WITH CUDA-graph capture (the engine-init we want to avoid)
echo "================= WITH GRAPHS (capture) ================="
RUN_TAG=graphs EXTRA_ARGS="" GMU="${GMU:-0.70}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" \
  srun --environment=glm-47-flash-rocm bash "${RECIPE}/_vllm_measure.sh" || true

# Same, but --enforce-eager: no CUDA-graph capture, no graph warmup. The delta in
# "init engine" is the capture+graph-warmup cost a persist/restore would avoid.
echo "================= ENFORCE EAGER (no capture) ================="
RUN_TAG=eager EXTRA_ARGS="--enforce-eager" GMU="${GMU:-0.70}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}" \
  srun --environment=glm-47-flash-rocm bash "${RECIPE}/_vllm_measure.sh" || true

echo "measure-init end: $(date --iso-8601=seconds)"
