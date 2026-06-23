#!/bin/bash
#SBATCH --job-name=snapshot-measure-repro
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:45:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -uo pipefail

GLM_DEPLOY=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
RECIPE=/capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/recipe
export EDF_PATH="${GLM_DEPLOY}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"
mkdir -p /capstor/scratch/cscs/xyao/kimi-k25-vllm/logs

echo "measure-repro start: $(date --iso-8601=seconds) node $(hostname)"

# Run the capture config (gmu=0.70, concurrency ~76x) TWICE. If capture stays
# ~292 s both runs, it is genuine graph-capture work (snapshot-eliminable). If
# run B is much faster, the cost was one-time aiter MoE autotuning (disk-cached,
# NOT eliminable by graph persistence).
for tag in graphsA graphsB; do
  echo "================= $tag (gmu=0.70, capture) ================="
  RUN_TAG="$tag" EXTRA_ARGS="" GMU=0.70 MAX_NUM_SEQS=64 MAX_MODEL_LEN=8192 \
    srun --environment=glm-47-flash-rocm bash "${RECIPE}/_vllm_measure.sh" || true
done

echo "measure-repro end: $(date --iso-8601=seconds)"
