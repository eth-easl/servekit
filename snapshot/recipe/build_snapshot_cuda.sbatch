#!/bin/bash
#SBATCH --job-name=snapshot-build-cuda
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/snapshot-cuda/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/snapshot-cuda}"
export DEPLOY_DIR
SCRIPT_DIR="${DEPLOY_DIR}/snapshot/recipe"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/snapshot/build-cuda"

echo "snapshot CUDA build started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"
echo "EDF_PATH=${EDF_PATH}"

srun --environment=snapshot-cuda \
  bash -lc 'set -euo pipefail
    # The nvidia/cuda devel image ships nvcc + g++ + nvrtc but NO cmake. Use a
    # standalone CMake binary tarball pre-fetched on the shared filesystem
    # (download once on the login node; see snapshot-cuda.toml comment). A system
    # cmake on PATH, if present, takes precedence.
    CMAKE_PREFIX="${DEPLOY_DIR}/cmake/cmake-3.30.8-linux-x86_64/bin"
    if [ -x "${CMAKE_PREFIX}/cmake" ]; then export PATH="${CMAKE_PREFIX}:${PATH}"; fi
    cmake --version | head -1
    cmake -S snapshot -B snapshot/build-cuda -DSNAPSHOT_BACKEND=CUDA
    cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}"
    echo "--- full ctest (host + GPU) on CUDA ---"
    ctest --test-dir snapshot/build-cuda --output-on-failure
    echo "--- probe-base (fixed-base determinism on A100) ---"
    snapshot/build-cuda/snapshot probe-base
    echo "--- capture (forward path: nvrtc -> module -> VMM -> capture -> serialize) ---"
    snapshot/build-cuda/snapshot capture snapshot/build-cuda/cap.snap
    ls -l snapshot/build-cuda/cap.snap
    echo "--- verify (single-process bit-identical, relocation exercised) ---"
    snapshot/build-cuda/snapshot verify snapshot/build-cuda/verify.snap
    echo "--- two-process: capture then restore in a fresh process ---"
    snapshot/build-cuda/snapshot capture snapshot/build-cuda/two.snap
    snapshot/build-cuda/snapshot restore snapshot/build-cuda/two.snap
    echo "--- bench (cold capture vs warm restore) ---"
    snapshot/build-cuda/snapshot bench snapshot/build-cuda/bench.snap --scaled
  '

echo "snapshot CUDA build finished: $(date --iso-8601=seconds)"
