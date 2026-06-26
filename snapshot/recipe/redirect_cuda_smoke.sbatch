#!/bin/bash
#SBATCH --job-name=snapshot-redirect-cuda-smoke
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/xyao/snapshot-cuda/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/snapshot-cuda}"
export DEPLOY_DIR
SCRIPT_DIR="${DEPLOY_DIR}/snapshot/recipe"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/snapshot/build-cuda"

echo "redirect-cuda smoke started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"

srun --environment=snapshot-cuda \
  bash -lc 'set -euo pipefail
    CMAKE_PREFIX="${DEPLOY_DIR}/cmake/cmake-3.30.8-linux-x86_64/bin"
    if [ -x "${CMAKE_PREFIX}/cmake" ]; then export PATH="${CMAKE_PREFIX}:${PATH}"; fi
    cmake -S snapshot -B snapshot/build-cuda -DSNAPSHOT_BACKEND=CUDA
    cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" \
      --target cuda_redirect_smoke snapshot_redirect_cuda
    LIB="$PWD/snapshot/build-cuda/libsnapshot_redirect_cuda.so"
    SMOKE="$PWD/snapshot/build-cuda/cuda_redirect_smoke"
    ls -l "$LIB" "$SMOKE"
    echo "--- linkage (expect libcuda only; no libstdc++/libnvrtc DT_NEEDED) ---"
    ldd "$LIB" || true

    echo "--- control: no preload (driver-chosen addresses) ---"
    "$SMOKE" | tee /tmp/r_ctrl.txt

    echo "--- run 1: fixed-base redirect ---"
    LD_PRELOAD="$LIB" "$SMOKE" | tee /tmp/r1.txt
    echo "--- run 2: fixed-base redirect ---"
    LD_PRELOAD="$LIB" "$SMOKE" | tee /tmp/r2.txt

    echo "--- determinism check ---"
    A1=$(grep "^addrs" /tmp/r1.txt); A2=$(grep "^addrs" /tmp/r2.txt)
    AC=$(grep "^addrs" /tmp/r_ctrl.txt)
    echo "run1: $A1"; echo "run2: $A2"; echo "ctrl: $AC"
    if [ "$A1" = "$A2" ]; then echo "REDIRECT_DETERMINISTIC=1"; else echo "REDIRECT_DETERMINISTIC=0"; exit 1; fi
    if grep -q "verify OK" /tmp/r1.txt && grep -q "verify OK" /tmp/r2.txt; then echo "COMPUTE_OK=1"; else echo "COMPUTE_OK=0"; exit 1; fi
    if [ "$A1" = "$AC" ]; then echo "CONTROL_DIFFERS=0 (addresses coincided)"; else echo "CONTROL_DIFFERS=1"; fi
    case "$A1" in
      *"A=0x600000000000"*) echo "FIXED_BASE_HONORED=1" ;;
      *) echo "FIXED_BASE_HONORED=0" ;;
    esac
  '

echo "redirect-cuda smoke finished: $(date --iso-8601=seconds)"
