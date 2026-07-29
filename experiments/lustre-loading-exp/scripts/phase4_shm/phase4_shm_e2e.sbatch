#!/bin/bash
# Phase 4 - stage the model into node-local /dev/shm (tmpfs), then run one
# end-to-end SGLang cold start against it. Source is the c8_s16M striped
# copy and staging uses N=60 O_DIRECT readers - the fastest raw config found
# in the Phase 2 stripe sweep (7.80 GB/s, see
# results/phase2_stripe_sweep/phase2_c8_s16M-73645-nid002317.csv), so the
# provisioning step itself is as fast as we've measured.
#
# Submit once per loader variant (mirrors phase1.3_e2e's separate mmap/nommap
# jobs - a single server process holds the port/GPUs for the whole job, so
# testing multiple variants means separate jobs, not separate launches in one
# job):
#   sbatch --job-name=p4shm-mmap        --export=ALL,TAG=shm_mmap,LOADER_FLAG="" \
#     lustre-loading-exp/scripts/phase4_shm/phase4_shm_e2e.sbatch
#   sbatch --job-name=p4shm-nommap      --export=ALL,TAG=shm_nommap,LOADER_FLAG="--weight-loader-disable-mmap" \
#     lustre-loading-exp/scripts/phase4_shm/phase4_shm_e2e.sbatch
#   sbatch --job-name=p4shm-fastsafetensors --export=ALL,TAG=shm_fastsafetensors,LOADER_FLAG="--load-format fastsafetensors" \
#     lustre-loading-exp/scripts/phase4_shm/phase4_shm_e2e.sbatch
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=lustre-loading-exp/results/phase4_shm/%x-%j.out

set -euo pipefail

TAG="${TAG:?set TAG, e.g. shm_mmap, shm_nommap, or shm_fastsafetensors}"
# "" for mmap (default), "--weight-loader-disable-mmap" to disable,
# "--load-format fastsafetensors" for the TP-rank-aware loader.
LOADER_FLAG="${LOADER_FLAG-}"

# Was llama70b_c8_s16M, one of the phase 2 stripe-layout copies, since deleted.
SRC="${SRC:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
WORKERS="${WORKERS:-60}"
SHM_DEST="${SHM_DEST:-/dev/shm/llama70b}"

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
PROFILE_OUT="${EXP_DIR}/results/phase4_shm/phase4_shm-${TAG}-${SLURM_JOB_ID}-${NODE}-profile.json"

echo "job=${SLURM_JOB_ID} node=${NODE} tag=${TAG} src=${SRC} workers=${WORKERS} shm_dest=${SHM_DEST} loader_flag='${LOADER_FLAG}'"
echo "profile_out=${PROFILE_OUT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
bash "${EXP_DIR}/scripts/phase4_shm/stage_to_shm.sh" "${SRC}" "${SHM_DEST}" "${WORKERS}"
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
pip install fastsafetensors -q
exec servekit profile --out "${PROFILE_OUT}" \
  --bench --bench-requests 64 --bench-concurrency 16 \
  --bench-input-len 512 --bench-output-len 128 \
  -- \
  python -m sglang.launch_server \
    --model-path "${SHM_DEST}" \
    --host 0.0.0.0 \
    --port "${SGLANG_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    ${LOADER_FLAG} \
    --enable-metrics
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile result (${TAG}) ==="
cat "${PROFILE_OUT}" 2>/dev/null || echo "NO PROFILE JSON - run did not reach ready"
echo "finished=$(date --iso-8601=seconds)"
