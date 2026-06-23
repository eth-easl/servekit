#!/bin/bash
#SBATCH --job-name=snapshot-build
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
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/snapshot/build"

echo "snapshot build started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"
echo "EDF_PATH=${EDF_PATH}"

srun --environment=snapshot-rocm \
  bash -lc 'set -euo pipefail
    cmake -S snapshot -B snapshot/build -DSNAPSHOT_BACKEND=HIP
    cmake --build snapshot/build -j"${SLURM_CPUS_PER_TASK:-8}"
    ctest --test-dir snapshot/build --output-on-failure
    echo "--- probe-base (M1.0 fixed-base determinism) ---"
    snapshot/build/snapshot probe-base
    echo "--- verify (single-process bit-identical gate, relocation exercised) ---"
    snapshot/build/snapshot verify snapshot/build/verify.snap
  '

echo "snapshot build finished: $(date --iso-8601=seconds)"
