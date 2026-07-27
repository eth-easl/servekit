#!/bin/bash
# Phase 3 - end-to-end cold start at ONE fastsafetensors reader fan-out.
#
# The question: is `weight_loading` bounded by capstor, or by the loader?
# Phase 2's contemporaneous native dd probe says capstor gives 3.6-5.5 GB/s on
# the UNMODIFIED layout at high reader concurrency, yet fastsafetensors gets
# 1.78 GB/s (132 GB / 73.7 s) - exactly where dd lands at N=4..8. The cause is
# in weight_utils.fastsafetensors_weights_iterator: it hands each rank ONE file
# per batch, so TP=4 keeps only 4 shards in flight and pays a collective
# barrier between batches (see the header of phase2_e2e_layout.sbatch, which
# first spotted this).
#
# fst_files_per_rank.patch turns that constant into SGLANG_FST_FILES_PER_RANK.
# FILES_PER_RANK 1/2/4/8 -> 4/8/16/30 shards in flight, straddling the dd
# sweep's N points so the two curves are directly comparable.
#
# FILES_PER_RANK=1 is a TRUE NULL RUN: the patched code provably reduces to
# upstream (single-file slices; trailing ranks dropped on a short final chunk,
# exactly as enumerate() does). It must reproduce the Phase-1.3 fastsafetensors
# baseline (weight_loading 73.7 s, total 237.2 s). If it does not, the harness
# is changing behavior and NO other number here is worth reading.
#
# Model is the NATIVE production dir on purpose - this asks whether a loader
# fix ALONE closes the gap, with zero change to how the model is stored. That
# is the plug-and-play deliverable.
#
# The dd probe runs AFTER the profiled launch, never before: reading all 141 GB
# up front would warm the Lustre OSS-side cache and flatter the very number we
# are measuring (O_DIRECT only bypasses the CLIENT page cache). It has to exist
# though - capstor drifts 2-6x over tens of minutes, so a 5-job sweep spread
# over hours is uninterpretable without a per-job bandwidth sample.
#
# Fresh exclusive node per submission (the page-cache rule: buffered reads are
# only cold on a node that has not seen this model).
#
# Submit (normally via phase3_submit_chain.sh):
#   sbatch --job-name=p3-fpr4 --export=ALL,TAG=fpr4,FILES_PER_RANK=4 \
#          lustre-loading-exp/scripts/phase3_loader_concurrency/phase3_loader_concurrency.sbatch
#
#SBATCH --job-name=phase3-loader-concurrency
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=lustre-loading-exp/results/phase3_loader_concurrency/%x-%j.out

set -euo pipefail

TAG="${TAG:?set TAG, e.g. fpr1_first, fpr2, fpr4, fpr8, fpr1_last}"
FILES_PER_RANK="${FILES_PER_RANK:?set FILES_PER_RANK, e.g. 1, 2, 4, 8}"

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
RES="${EXP_DIR}/results/phase3_loader_concurrency"
PROFILE_OUT="${RES}/phase3_loader_concurrency-${TAG}-${SLURM_JOB_ID}-${NODE}-profile.json"
DD_CSV="${RES}/phase3_ddprobe-${TAG}-${SLURM_JOB_ID}-${NODE}.csv"

echo "job=${SLURM_JOB_ID} node=${NODE} tag=${TAG} files_per_rank=${FILES_PER_RANK} in_flight=$((FILES_PER_RANK * TP_SIZE)) model=${MODEL}"
echo "profile_out=${PROFILE_OUT}"
echo "dd_csv=${DD_CSV}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
pip install fastsafetensors -q

# Clone sglang @ the SHA pinned to this image, verify it matches the installed
# code, patch it, swap the one file in. Aborts loudly on any mismatch.
bash "${EXP_DIR}/scripts/lib/patch_sglang_in_container.sh" "${EXP_DIR}/scripts/phase3_loader_concurrency/fst_files_per_rank.patch"
export SGLANG_FST_FILES_PER_RANK="${FILES_PER_RANK}"

# servekit exits nonzero if the server never reached ready. Keep that status,
# but still take the bandwidth sample - a failed run (e.g. an OOM at fpr8) is
# far easier to read next to the capstor bandwidth it was competing with.
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

# Contemporaneous capstor bandwidth, AFTER the measured read. O_DIRECT, N=30.
echo "--- dd probe (native layout, N=30, O_DIRECT) ---"
bash "${EXP_DIR}/scripts/lib/dd_read_sweep.sh" "${MODEL}" 16M "${DD_CSV}" "30"

exit \${SERVEKIT_RC}
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile result (${TAG}, files_per_rank=${FILES_PER_RANK}) ==="
cat "${PROFILE_OUT}" 2>/dev/null || echo "NO PROFILE JSON - run did not reach ready"
echo "=== dd probe (${TAG}) ==="
cat "${DD_CSV}" 2>/dev/null || echo "NO DD CSV"
echo "finished=$(date --iso-8601=seconds)"
