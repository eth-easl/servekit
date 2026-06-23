#!/bin/bash
#SBATCH --job-name=snapshot-record-syn
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/kimi-k25-vllm}"
SCRIPT_DIR="${DEPLOY_DIR}/snapshot/recipe"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs"

echo "snapshot record-synthetic started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"

# M3a.2: the controlled bit-identical gate. The recorder (real interposer)
# captures our synthetic workload; a fresh process restores it bit-identically.
srun --environment=snapshot-rocm bash "${SCRIPT_DIR}/_record_synthetic.sh"

echo "snapshot record-synthetic finished: $(date --iso-8601=seconds)"
