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

echo "=== Step 1: build redirect + record .so INSIDE the vllm image (glibc 2.35 + CUDA 13 — required for cross-container LD_PRELOAD) ==="
srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -euo pipefail
  bash "${SNAP_CUDA_DIR}/snapshot/recipe/_build_cuda_so.sh" build-vllm
'

echo "=== Step 2: glibc check + RECORD serve (glm-47-flash-vllm-cuda env) ==="
srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -uo pipefail
  cd "${DEPLOY_DIR}"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'"
  python3 -c "import transformers; print(\"transformers:\", transformers.__version__)"

  RECORD_SO="${SNAP_CUDA_DIR}/snapshot/build-vllm/libsnapshot_record_cuda.so"
  REDIRECT_SO="${SNAP_CUDA_DIR}/snapshot/build-vllm/libsnapshot_redirect_cuda.so"

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
  STATUS_FILE="${DEPLOY_DIR}/meta-n5b/record.status" \
    MODE=record bash snapshot/recipe/_vllm_record_cuda.sh
  LOG="${DEPLOY_DIR}/logs/vllm_record_cuda_${SLURM_JOB_ID:-local}.log"

  # --- Task 5 engagement gates (per-rank redirect engaged) ---
  # The redirect/record atexit SUMMARY lines are LOST when the launcher SIGKILLs
  # vLLM after the probe, so gate on DURING-RUN signals instead: the FIXED_VMM
  # line (printed at region reserve) and the per-graph `graph=` walk lines
  # (printed at each cuStreamEndCapture, with their own blind= count).
  REDIR_FIXED=$(grep -c "fixed_base_honored=1" "${LOG}" || true)
  GRAPHS_WALKED=$(grep -cE "\[record-cuda\] pid=[0-9]+ graph=[0-9]+ nodes=" "${LOG}" || true)
  echo "REDIRECT_fixed_base_honored_count=${REDIR_FIXED}"
  echo "RECORD_graphs_walked=${GRAPHS_WALKED}"
  TP_N="${TP:-4}"
  [ "${REDIR_FIXED}"   -ge "${TP_N}" ] || { echo "N5B_RECORD_GATE: FAIL (<${TP_N} fixed_base_honored=1; got ${REDIR_FIXED})"; exit 1; }
  [ "${GRAPHS_WALKED}" -ge "${TP_N}" ] || { echo "N5B_RECORD_GATE: FAIL (<${TP_N} graphs walked; got ${GRAPHS_WALKED})"; exit 1; }
  grep -q "N5B_RECORD_INFERENCE=ok" "${DEPLOY_DIR}/meta-n5b/record.status" 2>/dev/null || { echo "N5B_RECORD_GATE: FAIL (record-mode inference did not return Paris)"; exit 1; }

  # --- Task 6 (G2): record-completeness gate ---
  # blind=0 across all walked graphs, and per-rank .snap + entry.output meta.
  # blind/graphs totals come from the during-run `graph=` walk lines (robust to
  # SIGKILL). Per-rank dirs are routed by device index (vLLM workers are not
  # given LOCAL_RANK in their env).
  SNAP_ROOT="${SNAP_ROOT:-${DEPLOY_DIR}/snap-n5b}"
  META_ROOT="${META_ROOT:-${DEPLOY_DIR}/meta-n5b}"
  BLIND_TOTAL=$(grep -oE "graph=[0-9]+ nodes=[0-9]+ edges=[0-9]+ blind=[0-9]+" "${LOG}" | grep -oE "blind=[0-9]+" | awk -F= "{s+=\$2} END{print s+0}")
  echo "recorded_graphs_walked=${GRAPHS_WALKED} blind_total=${BLIND_TOTAL}"
  [ "${BLIND_TOTAL}" -eq 0 ] || { echo "N5B_RECORD_GATE: FAIL (blind total ${BLIND_TOTAL} > 0)"; exit 1; }
  COMPLETE=1
  for r in $(seq 0 $((TP_N-1))); do
    NSNAP=$(ls "${SNAP_ROOT}/rank${r}"/graph-*.snap 2>/dev/null | wc -l)
    if [ -s "${META_ROOT}/rank${r}.json" ]; then
      # Count entry records (robust to python3/ACL issues inside the container;
      # each entry serializes with an "idx" field).
      NENT=$(grep -o '\"idx\"' "${META_ROOT}/rank${r}.json" 2>/dev/null | wc -l)
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
