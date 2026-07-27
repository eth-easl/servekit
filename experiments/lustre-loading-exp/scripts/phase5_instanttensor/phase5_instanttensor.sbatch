#!/bin/bash
# Phase 5 - end-to-end cold start with ONE loader configuration.
#
# The question: Phase 3 fixed the fastsafetensors READ (fpr8: weight 38.2 s; of
# which only ~13-18 s is actually reading 132 GB, i.e. 7-10 GB/s, at or above
# raw dd). The residual ~20-25 s is NON-OVERLAPPED H2D + cross-rank NCCL
# broadcast + param materialization - priced independently by Phase 4, where
# /dev/shm removes storage cost entirely and the weight phase still takes 19.6 s.
# A wider read batch cannot touch that. A PIPELINED loader can: overlap the read
# of the next chunk with the broadcast of the current one.
#
# InstantTensor (PR sgl#28506) claims exactly that - "distributed loading,
# pipelined prefetching, and direct I/O". So: does it beat 38.2 s, and does it
# approach the ~19.6 s RAM-side floor?
#
# GDS IS NOT AVAILABLE HERE (phase5 preflight: libcufile loads, but
# /dev/nvidia-fs* and /proc/driver/nvidia-fs are absent -> the kernel driver is
# not exposed; capstor is Lustre). So InstantTensor's headline 35-45 GB/s
# numbers are unreachable on this system and it must win on pipelining + direct
# I/O alone. Setting expectations honestly BEFORE the run, not after.
#
# LOADER=fastsafetensors is the CONTROL: our Phase-3 best (fpr8, 30 files in
# flight). It runs first and last, bracketing the InstantTensor points, because
# the upstream-config baseline varies 69.5-86.1 s (21%) across nodes - node/time
# variance, not drift, is the dominant noise source, and a single-point
# comparison against a remembered number would be worthless.
#
# The dd probe runs AFTER the profiled launch, never before: reading all 141 GB
# up front would warm the Lustre OSS-side cache and flatter the very number we
# are measuring (O_DIRECT only bypasses the CLIENT page cache).
#
# Submit (normally via phase5_submit_chain.sh):
#   sbatch --job-name=p5-it_default --export=ALL,TAG=it_default,LOADER=instanttensor \
#          lustre-loading-exp/scripts/phase5_instanttensor/phase5_instanttensor.sbatch
#
#SBATCH --job-name=phase5-instanttensor
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=lustre-loading-exp/results/phase5_instanttensor/%x-%j.out

set -euo pipefail

TAG="${TAG:?set TAG, e.g. ctl_fst_first, it_default, it_tuned, ctl_fst_last}"
LOADER="${LOADER:?set LOADER=instanttensor or fastsafetensors}"

# Control knobs. FILES_PER_RANK applies only to fastsafetensors (phase3 patch);
# IT_CONCURRENCY / IT_IO_DEPTH only to instanttensor. Empty = library default,
# which for instanttensor reproduces PR #28506 exactly.
FILES_PER_RANK="${FILES_PER_RANK:-8}"
IT_CONCURRENCY="${IT_CONCURRENCY-}"
IT_IO_DEPTH="${IT_IO_DEPTH-}"

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
RES="${EXP_DIR}/results/phase5_instanttensor"
PROFILE_OUT="${RES}/phase5_instanttensor-${TAG}-${SLURM_JOB_ID}-${NODE}-profile.json"
DD_CSV="${RES}/phase5_ddprobe-${TAG}-${SLURM_JOB_ID}-${NODE}.csv"

echo "job=${SLURM_JOB_ID} node=${NODE} tag=${TAG} loader=${LOADER} fpr=${FILES_PER_RANK} it_conc=${IT_CONCURRENCY:-default} it_iodepth=${IT_IO_DEPTH:-default}"
echo "profile_out=${PROFILE_OUT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
pip install fastsafetensors -q
pip install 'instanttensor>=0.1.9' -q

# Both patches go in, in ONE call: phase3's files-per-rank knob (for the
# fastsafetensors control) and phase5's InstantTensor backport. Every point in
# this sweep therefore runs the SAME engine build - only --load-format and the
# env knobs differ. One variable at a time.
#
# They MUST be passed together. Two separate calls do not compose: the second
# would re-clone a pristine tree and diff it against an already-patched install,
# tripping the version check (both patches touch weight_utils.py). That is
# exactly how the first attempt at this sweep failed - correctly, loudly.
bash "${EXP_DIR}/scripts/lib/patch_sglang_in_container.sh" \
  "${EXP_DIR}/scripts/phase3_loader_concurrency/fst_files_per_rank.patch" \
  "${EXP_DIR}/scripts/phase5_instanttensor/instanttensor_backport.patch"

export SGLANG_FST_FILES_PER_RANK="${FILES_PER_RANK}"
[ -n "${IT_CONCURRENCY}" ] && export SGLANG_IT_CONCURRENCY="${IT_CONCURRENCY}"
[ -n "${IT_IO_DEPTH}" ] && export SGLANG_IT_IO_DEPTH="${IT_IO_DEPTH}"

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
    --load-format "${LOADER}" \
    --enable-metrics
SERVEKIT_RC=\$?
set -e
echo "servekit_rc=\${SERVEKIT_RC}"

echo "--- dd probe (native layout, N=30, O_DIRECT) ---"
bash "${EXP_DIR}/scripts/lib/dd_read_sweep.sh" "${MODEL}" 16M "${DD_CSV}" "30"

exit \${SERVEKIT_RC}
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile result (${TAG}, loader=${LOADER}) ==="
cat "${PROFILE_OUT}" 2>/dev/null || echo "NO PROFILE JSON - run did not reach ready"
echo "=== dd probe (${TAG}) ==="
cat "${DD_CSV}" 2>/dev/null || echo "NO DD CSV"
echo "finished=$(date --iso-8601=seconds)"
