#!/bin/bash
# End-to-end cold start at ONE {files_per_rank, max_threads} point.
#
# The question: Phase 3 fixed fastsafetensors' FILE-level concurrency
# (SGLANG_FST_FILES_PER_RANK, 4 -> 32 files in flight at fpr8, weight_loading
# 69-88 s -> 37-43 s). But NOTES.md's own follow-up found only ~13-18 s of
# fpr8's 38.2 s is actually reading (already near the raw dd ceiling of
# 7-10 GB/s); the rest is fixed, non-overlapped GPU-side work that more
# files-in-flight cannot touch. files_per_rank never touched fastsafetensors'
# OWN internal thread pool that pipelines each file's pread + H2D copy -
# that's SafeTensorsFileLoader(max_threads=...), hard-defaulted to 16 and
# never passed through by sglang. fst_max_threads.patch exposes it as
# SGLANG_FST_MAX_THREADS. Raising it to 64 also scales the pinned host bounce
# buffer 256 MB -> 1 GB (bbuf_size_kb * max_threads, see
# fastsafetensors/cpp/ext.cpp) - that growth is the accepted, intended side
# effect of the knob, not something this patch compensates for.
#
# FILES_PER_RANK=1, MAX_THREADS=16 is a TRUE NULL RUN: both knobs default to
# provably-upstream behavior (see fst_max_threads.patch's own comments). It
# must reproduce Phase 3's fpr1 baseline (weight_loading 69-88 s, total
# 242-258 s). If it does not, the harness is changing behavior and no other
# number here is worth reading.
#
# Model is the NATIVE production dir, same as phase3 - the question is
# whether raising max_threads (alone, or stacked on phase3's fpr8) closes
# more of the gap, with zero change to how the model is stored.
#
# Fresh exclusive node per submission (the page-cache rule).
#
# Submit (normally via fst_max_threads_submit_chain.sh):
#   sbatch --job-name=fstmt-mt64 --export=ALL,TAG=mt64,FILES_PER_RANK=1,MAX_THREADS=64 \
#          lustre-loading-exp/scripts/fastsafetensor_many_threads/fst_max_threads_e2e.sbatch
#
#SBATCH --job-name=fst-max-threads-e2e
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=lustre-loading-exp/results/fastsafetensor_many_threads/%x-%j.out

set -euo pipefail

TAG="${TAG:?set TAG, e.g. mt16_first, mt64, mt64_fpr8, mt16_last}"
FILES_PER_RANK="${FILES_PER_RANK:?set FILES_PER_RANK, e.g. 1 or 8}"
MAX_THREADS="${MAX_THREADS:?set MAX_THREADS, e.g. 16 or 64}"

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
RES="${EXP_DIR}/results/fastsafetensor_many_threads"
PROFILE_OUT="${RES}/fst-max-threads-${TAG}-${SLURM_JOB_ID}-${NODE}-profile.json"

echo "job=${SLURM_JOB_ID} node=${NODE} tag=${TAG} files_per_rank=${FILES_PER_RANK} max_threads=${MAX_THREADS} in_flight=$((FILES_PER_RANK * TP_SIZE)) model=${MODEL}"
echo "profile_out=${PROFILE_OUT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
pip install fastsafetensors -q

# Clone sglang @ the SHA pinned to this image, verify it matches the installed
# code, patch it, swap the one file in. Aborts loudly on any mismatch.
bash "${EXP_DIR}/scripts/lib/patch_sglang_in_container.sh" "${EXP_DIR}/scripts/fastsafetensor_many_threads/fst_max_threads.patch"
export SGLANG_FST_FILES_PER_RANK="${FILES_PER_RANK}"
export SGLANG_FST_MAX_THREADS="${MAX_THREADS}"

# servekit exits nonzero if the server never reached ready.
set +e
servekit profile --out "${PROFILE_OUT}" \
  --bench --bench-requests 64 --bench-concurrency 16 \
  --bench-input-len 512 --bench-output-len 128 \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${SGLANG_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --load-format fastsafetensors \
    --enable-metrics
SERVEKIT_RC=\$?
set -e
echo "servekit_rc=\${SERVEKIT_RC}"

exit \${SERVEKIT_RC}
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile result (${TAG}, files_per_rank=${FILES_PER_RANK}, max_threads=${MAX_THREADS}) ==="
cat "${PROFILE_OUT}" 2>/dev/null || echo "NO PROFILE JSON - run did not reach ready"
echo "finished=$(date --iso-8601=seconds)"
