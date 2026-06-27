#!/bin/bash
#SBATCH --job-name=vllm-restore-cuda-n5b
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/logs/%x-%j.out

# N5b Task 5/7: vLLM-CUDA TP=4 RESTORE-mode cold start. Assumes a record run
# (vllm_record_cuda.sbatch) has populated SNAP_ROOT/rank{0..3}/*.snap and
# META_ROOT/rank{0..3}.json. Replays the recorded graphs (skip capture) and
# serves. One vllm serve per srun step.
set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda}"
SNAP_CUDA_DIR="${SNAP_CUDA_DIR:-/capstor/scratch/cscs/xyao/snapshot-cuda}"
export DEPLOY_DIR SNAP_CUDA_DIR
export EDF_PATH="${DEPLOY_DIR}/deploy/glm-47-flash-bristen-vllm:${SNAP_CUDA_DIR}/snapshot/recipe:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs"
export PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"
mkdir -p "${PIP_CACHE_DIR}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"

echo "=== Step 1: build redirect + record .so (snapshot-cuda devel env) ==="
srun --environment=snapshot-cuda bash -lc '
  set -euo pipefail
  cd "${SNAP_CUDA_DIR}"
  CMAKE_PREFIX="${SNAP_CUDA_DIR}/cmake/cmake-3.30.8-linux-x86_64/bin"
  if [ -x "${CMAKE_PREFIX}/cmake" ]; then export PATH="${CMAKE_PREFIX}:${PATH}"; fi
  cmake -S snapshot -B snapshot/build-cuda -DSNAPSHOT_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=80
  cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" \
    --target snapshot_record_cuda snapshot_redirect_cuda
  ls -l snapshot/build-cuda/libsnapshot_record_cuda.so \
        snapshot/build-cuda/libsnapshot_redirect_cuda.so
'

echo "=== Step 2: glibc check + RESTORE serve (glm-47-flash-vllm-cuda env) ==="
srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -uo pipefail
  cd "${DEPLOY_DIR}"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'"

  RECORD_SO="${SNAP_CUDA_DIR}/snapshot/build-cuda/libsnapshot_record_cuda.so"
  REDIRECT_SO="${SNAP_CUDA_DIR}/snapshot/build-cuda/libsnapshot_redirect_cuda.so"

  # Cross-container glibc gate (same as the record sbatch).
  echo "--- ldd check inside vllm image (must be empty) ---"
  MISS=$( { ldd "${RECORD_SO}"; ldd "${REDIRECT_SO}"; } | grep "not found" || true )
  if [ -n "${MISS}" ]; then
    echo "N5B_RESTORE_GATE: FAIL (missing libs in vllm image):"; echo "${MISS}"; exit 1
  fi
  echo "LDD_CLEAN=1"

  # Recorded snap + meta must exist (from a prior record run).
  SNAP_ROOT="${SNAP_ROOT:-${DEPLOY_DIR}/snap-n5b}"
  META_ROOT="${META_ROOT:-${DEPLOY_DIR}/meta-n5b}"
  TP_N="${TP:-4}"
  for r in $(seq 0 $((TP_N-1))); do
    [ -d "${SNAP_ROOT}/rank${r}" ] && [ -n "$(ls -A "${SNAP_ROOT}/rank${r}" 2>/dev/null)" ] || \
      { echo "N5B_RESTORE_GATE: FAIL (missing recorded snaps at ${SNAP_ROOT}/rank${r})"; exit 1; }
    [ -s "${META_ROOT}/rank${r}.json" ] || \
      { echo "N5B_RESTORE_GATE: FAIL (missing recorded meta ${META_ROOT}/rank${r}.json)"; exit 1; }
  done
  echo "N5B_RESTORE_INPUTS_PRESENT=1"

  # --- restore serve ---
  MODE=restore bash snapshot/recipe/_vllm_restore_cuda.sh
  LOG="${DEPLOY_DIR}/logs/vllm_restore_cuda_${SLURM_JOB_ID:-local}.log"

  # --- Task 5/7 engagement gates (per rank) ---
  REC_SUMMARY=$(grep -c "\[record-cuda\].*SUMMARY" "${LOG}" || true)
  echo "RECORD_summary_count=${REC_SUMMARY}"
  [ "${REC_SUMMARY}" -ge "${TP_N}" ] || { echo "N5B_RESTORE_GATE: FAIL (<${TP_N} record summaries; got ${REC_SUMMARY})"; exit 1; }
  # Each rank restore SUMMARY shows restored>=1, fallthrough=0 (capture skipped).
  REST_N=$(grep -oE "restored=[0-9]+" "${LOG}" | awk -F= "{s+=\$2} END{print s+0}")
  FALL_N=$(grep -oE "fallthrough=[0-9]+" "${LOG}" | awk -F= "{s+=\$2} END{print s+0}")
  echo "restored_total=${REST_N} fallthrough_total=${FALL_N}"
  [ "${REST_N}" -ge "${TP_N}" ] || { echo "N5B_RESTORE_GATE: FAIL (restored total ${REST_N} < ${TP_N})"; exit 1; }
  [ "${FALL_N}" -eq 0 ] || { echo "N5B_RESTORE_GATE: FAIL (fallthrough total ${FALL_N} > 0 — a rank fell through to real capture)"; exit 1; }
  # redirect still pinned Δ=0 in the restore run.
  REDIR_FIXED=$(grep -c "fixed_base_honored=1" "${LOG}" || true)
  [ "${REDIR_FIXED}" -ge "${TP_N}" ] || { echo "N5B_RESTORE_GATE: FAIL (<${TP_N} fixed_base_honored=1 in restore; got ${REDIR_FIXED})"; exit 1; }
  grep -q "N5B_RESTORE_INFERENCE=ok" "${LOG}" || { echo "N5B_RESTORE_GATE: FAIL (restore-mode inference did not return Paris)"; exit 1; }

  echo "N5B_RESTORE_GATE: PASS (restored>=1/fallthrough=0 per rank, Δ=0, inference ok)"
'

echo "vllm-restore-cuda-n5b finished: $(date --iso-8601=seconds)"
