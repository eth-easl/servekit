#!/bin/bash
#SBATCH --job-name=vllm-record-cuda-n5b
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/logs/%x-%j.out

# N5b Task 5/6: vLLM-CUDA TP=4 RECORD-mode cold start under redirect+record.
# Step 1 builds the interposer .so (snapshot-cuda devel env); step 2 runs the
# cross-container glibc check + the record serve (glm-47-flash-vllm-cuda env).
# One vllm serve per srun step (enroot holds CUDA IPC shmem at container level).
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

echo "=== Step 2: glibc check + RECORD serve (glm-47-flash-vllm-cuda env) ==="
srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -uo pipefail
  cd "${DEPLOY_DIR}"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'"
  python3 -c "import transformers; print(\"transformers:\", transformers.__version__)"

  RECORD_SO="${SNAP_CUDA_DIR}/snapshot/build-cuda/libsnapshot_record_cuda.so"
  REDIRECT_SO="${SNAP_CUDA_DIR}/snapshot/build-cuda/libsnapshot_redirect_cuda.so"

  # --- Task 5 Step 1: cross-container glibc gate (ldd must find everything) ---
  echo "--- ldd check inside vllm image (must be empty) ---"
  MISS=$( { ldd "${RECORD_SO}"; ldd "${REDIRECT_SO}"; } | grep "not found" || true )
  if [ -n "${MISS}" ]; then
    echo "N5B_RECORD_GATE: FAIL (missing libs in vllm image — rebuild .so against the image):"
    echo "${MISS}"
    exit 1
  fi
  echo "LDD_CLEAN=1"

  # --- record serve ---
  PROBE_OUT="${DEPLOY_DIR}/meta-n5b/record.probe" \
    MODE=record bash snapshot/recipe/_vllm_record_cuda.sh
  LOG="${DEPLOY_DIR}/logs/vllm_record_cuda_${SLURM_JOB_ID:-local}.log"

  # --- Task 5 engagement gates (per-rank redirect engaged) ---
  # 4 TP workers → 4 [redirect-cuda] SUMMARY lines, all with fixed_base_honored=1.
  REDIR_FIXED=$(grep -c "fixed_base_honored=1" "${LOG}" || true)
  REDIR_SUMMARY=$(grep -c "\[redirect-cuda\].*SUMMARY" "${LOG}" || true)
  REC_SUMMARY=$(grep -c "\[record-cuda\].*SUMMARY" "${LOG}" || true)
  echo "REDIRECT_fixed_base_honored_count=${REDIR_FIXED}"
  echo "REDIRECT_summary_count=${REDIR_SUMMARY}"
  echo "RECORD_summary_count=${REC_SUMMARY}"
  TP_N="${TP:-4}"
  [ "${REDIR_SUMMARY}" -ge "${TP_N}" ] || { echo "N5B_RECORD_GATE: FAIL (<${TP_N} redirect summaries; got ${REDIR_SUMMARY})"; exit 1; }
  [ "${REDIR_FIXED}"   -ge "${TP_N}" ] || { echo "N5B_RECORD_GATE: FAIL (<${TP_N} fixed_base_honored=1; got ${REDIR_FIXED})"; exit 1; }
  [ "${REC_SUMMARY}"   -ge "${TP_N}" ] || { echo "N5B_RECORD_GATE: FAIL (<${TP_N} record summaries; got ${REC_SUMMARY})"; exit 1; }
  grep -q "N5B_RECORD_INFERENCE=ok" "${LOG}" || { echo "N5B_RECORD_GATE: FAIL (record-mode inference did not return Paris)"; exit 1; }

  # --- Task 6 (G2): record-completeness gate ---
  # Every rank recorded >=1 graph, blind=0 across all ranks, and each rank wrote
  # an entry.output meta JSON with >=1 entries (covering CAPTURE_SIZES). Only
  # processes that actually capture write .snap / meta files, so non-capturing
  # helper processes (main/RPC) contribute graphs=0 and never create a rank dir.
  SNAP_ROOT="${SNAP_ROOT:-${DEPLOY_DIR}/snap-n5b}"
  META_ROOT="${META_ROOT:-${DEPLOY_DIR}/meta-n5b}"
  BLIND_TOTAL=$(grep -oE "blind=[0-9]+" "${LOG}" | awk -F= "{s+=\$2} END{print s+0}")
  GRAPH_TOTAL=$(grep -oE "recorded: graphs=[0-9]+" "${LOG}" | awk -F= "{s+=\$2} END{print s+0}")
  echo "recorded_graphs_total=${GRAPH_TOTAL} blind_total=${BLIND_TOTAL}"
  [ "${GRAPH_TOTAL}" -ge "${TP_N}" ] || { echo "N5B_RECORD_GATE: FAIL (graphs total ${GRAPH_TOTAL} < ${TP_N})"; exit 1; }
  [ "${BLIND_TOTAL}" -eq 0 ] || { echo "N5B_RECORD_GATE: FAIL (blind total ${BLIND_TOTAL} > 0)"; exit 1; }
  COMPLETE=1
  for r in $(seq 0 $((TP_N-1))); do
    NSNAP=$(ls "${SNAP_ROOT}/rank${r}"/graph-*.snap 2>/dev/null | wc -l)
    if [ -s "${META_ROOT}/rank${r}.json" ]; then
      NENT=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1])).get('entries',[])))" "${META_ROOT}/rank${r}.json" 2>/dev/null || echo 0)
    else
      NENT=0
    fi
    echo "rank${r}: snaps=${NSNAP} meta_entries=${NENT}"
    [ "${NSNAP}" -ge 1 ] || { echo "N5B_RECORD_GATE: FAIL (rank${r}: no graph-*.snap)"; COMPLETE=0; }
    [ "${NENT}"  -ge 1 ] || { echo "N5B_RECORD_GATE: FAIL (rank${r}: meta entries ${NENT} < 1)"; COMPLETE=0; }
  done
  [ "${COMPLETE}" -eq 1 ] || exit 1
  echo "N5B_RECORD: ranks=${TP_N} blind=0 complete=1"

  echo "N5B_RECORD_GATE: PASS (per-rank .snap + meta complete, blind=0, Δ=0, inference ok)"
'

echo "vllm-record-cuda-n5b finished: $(date --iso-8601=seconds)"
