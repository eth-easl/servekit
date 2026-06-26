#!/bin/bash
#SBATCH --job-name=snapshot-redirect-cuda-torch
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=/capstor/scratch/cscs/xyao/snapshot-cuda/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/snapshot-cuda}"
export DEPLOY_DIR
SCRIPT_DIR="${DEPLOY_DIR}/snapshot/recipe"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/snapshot/build-cuda" \
  "${DEPLOY_DIR}/home"

echo "redirect-cuda torch gate started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"

# Step 1 (devel container): build the redirect .so on shared /capstor.
srun --environment=snapshot-cuda \
  bash -lc 'set -euo pipefail
    CMAKE_PREFIX="${DEPLOY_DIR}/cmake/cmake-3.30.8-linux-x86_64/bin"
    if [ -x "${CMAKE_PREFIX}/cmake" ]; then export PATH="${CMAKE_PREFIX}:${PATH}"; fi
    cmake -S snapshot -B snapshot/build-cuda -DSNAPSHOT_BACKEND=CUDA
    cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" \
      --target snapshot_redirect_cuda
    ls -l snapshot/build-cuda/libsnapshot_redirect_cuda.so
  '

# Step 2 (torch container): load the .so under real torch, twice, + overhead.
srun --environment=snapshot-torch-cuda \
  bash -lc 'set -euo pipefail
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    LIB="$PWD/snapshot/build-cuda/libsnapshot_redirect_cuda.so"
    PY="$PWD/snapshot/recipe/_redirect_torch_smoke.py"
    echo "--- torch preflight ---"
    python -c "import torch; print(torch.__version__, torch.version.cuda)"
    echo "--- ldd .so under torch container (expect no \"not found\") ---"
    if ldd "$LIB" | grep -i "not found"; then echo "LDD_OK=0"; exit 1; else echo "LDD_OK=1"; fi

    echo "--- baseline (no preload) timing ---"
    t0=$(date +%s.%N); python "$PY"; t1=$(date +%s.%N)
    BASE_S=$(awk "BEGIN{print $t1-$t0}")

    echo "--- run 1: fixed-base redirect ---"
    t2=$(date +%s.%N)
    SNAPSHOT_REDIRECT_REGION_GIB=8 LD_PRELOAD="$LIB" python "$PY" | tee /tmp/t1.txt
    t3=$(date +%s.%N)
    PRELOAD_S=$(awk "BEGIN{print $t3-$t2}")
    echo "--- run 2: fixed-base redirect ---"
    SNAPSHOT_REDIRECT_REGION_GIB=8 LD_PRELOAD="$LIB" python "$PY" | tee /tmp/t2.txt

    echo "--- determinism + overhead ---"
    P1=$(grep "PTRS" /tmp/t1.txt); P2=$(grep "PTRS" /tmp/t2.txt)
    echo "run1: $P1"; echo "run2: $P2"
    if [ "$P1" = "$P2" ]; then echo "TORCH_DETERMINISTIC=1"; else echo "TORCH_DETERMINISTIC=0"; exit 1; fi
    if grep -q "COMPUTE=OK" /tmp/t1.txt && grep -q "COMPUTE=OK" /tmp/t2.txt; then echo "TORCH_COMPUTE_OK=1"; else echo "TORCH_COMPUTE_OK=0"; exit 1; fi
    OVH=$(awk "BEGIN{print $PRELOAD_S-$BASE_S}")
    echo "BASE_S=$BASE_S PRELOAD_S=$PRELOAD_S STARTUP_OVERHEAD_S=$OVH"
    awk "BEGIN{exit !($OVH < 10.0)}" && echo "STARTUP_OVERHEAD_OK=1" || { echo "STARTUP_OVERHEAD_OK=0"; exit 1; }
  '

echo "redirect-cuda torch gate finished: $(date --iso-8601=seconds)"
