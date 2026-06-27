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
# N5a Task 2 gate: build libsnapshot_record_cuda.so, run smoke under the
#   interposer in record mode, confirm:
#     (a) CHECKSUM is unchanged (interposer is transparent)
#     (b) SUMMARY reports "identity: 2 functions (1 fatbin, 1 module)"
#         i.e. k_add resolved via __cudaRegisterFunction (fatbin path) AND
#              k_mul resolved via cuModuleGetFunction (module/nvrtc path).

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

    # Task 1: build smoke binary.
    cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" \
      --target cuda_record_smoke

    # Task 2: build record interposer shared library.
    cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" \
      --target snapshot_record_cuda

    SMOKE="$PWD/snapshot/build-cuda/cuda_record_smoke"
    RECORD_SO="$PWD/snapshot/build-cuda/libsnapshot_record_cuda.so"
    ls -l "$SMOKE" "$RECORD_SO"

    echo "--- run 1 ---"
    "$SMOKE" | tee /tmp/r1.txt
    echo "exit $?"

    echo "--- run 2 ---"
    "$SMOKE" | tee /tmp/r2.txt
    echo "exit $?"

    echo "--- determinism check (Task 1 gate) ---"
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

    # -----------------------------------------------------------------------
    # Task 2 gate: run smoke under the record interposer.
    # Capture stdout+stderr together so we can grep both the CHECKSUM and the
    # [record-cuda] SUMMARY line from the same file.
    # -----------------------------------------------------------------------
    SNAP_DIR="${DEPLOY_DIR}/snap-record-smoke"
    mkdir -p "${SNAP_DIR}"

    echo "--- record interposer run (Task 2 gate) ---"
    LD_PRELOAD="${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=record \
      SNAPSHOT_RECORD_CUDA_DIR="${SNAP_DIR}" \
      "$SMOKE" 2>&1 | tee /tmp/r_record.txt
    RECORD_EXIT="${PIPESTATUS[0]}"

    echo "--- identity gate ---"
    SUMMARY=$(grep "\[record-cuda\].*SUMMARY" /tmp/r_record.txt || echo "SUMMARY_MISSING")
    echo "summary line: ${SUMMARY}"

    # Gate (a): smoke still exits clean and prints the same CHECKSUM.
    CS_RECORD=$(grep "^CHECKSUM" /tmp/r_record.txt || echo "MISSING")
    echo "checksum under interposer: ${CS_RECORD}"
    if [ "${CS_RECORD}" = "${CS1}" ] && [ "${CS_RECORD}" != "MISSING" ]; then
      echo "CHECKSUM_UNCHANGED=1"
    else
      echo "CHECKSUM_UNCHANGED=0 (expected ${CS1}, got ${CS_RECORD})"
      exit 1
    fi

    # Gate (b): SUMMARY reports both identities.
    if echo "${SUMMARY}" | grep -qF "identity: 2 functions (1 fatbin, 1 module)"; then
      echo "IDENTITY_GATE=1 (k_add via fatbin + k_mul via module: both resolved)"
    else
      echo "IDENTITY_GATE=0 (expected \"identity: 2 functions (1 fatbin, 1 module)\" not found)"
      exit 1
    fi

    if [ "${RECORD_EXIT}" -ne 0 ]; then
      echo "RECORD_RUN_EXIT=${RECORD_EXIT} (smoke exited non-zero under interposer)"
      exit 1
    fi

    echo "Task 2 gate PASSED"
  '

echo "cuda-record-smoke finished: $(date --iso-8601=seconds)"
