#!/bin/bash
#SBATCH --job-name=snapshot-cuda-record-smoke
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/xyao/snapshot-cuda/logs/%x-%j.out

# N5a Task 1 gate: build cuda_record_smoke, run twice, confirm CHECKSUM is identical.
# Task 5 will extend this script to add record/restore runs.

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/snapshot-cuda}"
export DEPLOY_DIR
SCRIPT_DIR="${DEPLOY_DIR}/snapshot/recipe"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/snapshot/build-cuda"

echo "cuda-record-smoke started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"

srun --environment=snapshot-cuda \
  bash -lc 'set -euo pipefail
    CMAKE_PREFIX="${DEPLOY_DIR}/cmake/cmake-3.30.8-linux-x86_64/bin"
    if [ -x "${CMAKE_PREFIX}/cmake" ]; then export PATH="${CMAKE_PREFIX}:${PATH}"; fi

    # A100 (sm_80): set CUDA_ARCHITECTURES so nvcc compiles the static k_add for the
    # right target.  Also passed to CMake so enable_language(CUDA) picks it up.
    cmake -S snapshot -B snapshot/build-cuda \
      -DSNAPSHOT_BACKEND=CUDA \
      -DCMAKE_CUDA_ARCHITECTURES=80
    cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" \
      --target cuda_record_smoke

    SMOKE="$PWD/snapshot/build-cuda/cuda_record_smoke"
    ls -l "$SMOKE"

    echo "--- run 1 ---"
    "$SMOKE" | tee /tmp/r1.txt
    echo "exit $?"

    echo "--- run 2 ---"
    "$SMOKE" | tee /tmp/r2.txt
    echo "exit $?"

    echo "--- determinism check ---"
    CS1=$(grep "^CHECKSUM" /tmp/r1.txt || echo "MISSING")
    CS2=$(grep "^CHECKSUM" /tmp/r2.txt || echo "MISSING")
    echo "run1: $CS1"
    echo "run2: $CS2"
    if [ "$CS1" = "$CS2" ] && [ "$CS1" != "MISSING" ]; then
      echo "CHECKSUM_DETERMINISTIC=1"
    else
      echo "CHECKSUM_DETERMINISTIC=0"
      exit 1
    fi
  '

echo "cuda-record-smoke finished: $(date --iso-8601=seconds)"
