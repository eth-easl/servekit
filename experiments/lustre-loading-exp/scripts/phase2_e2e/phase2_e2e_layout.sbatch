#!/bin/bash
# Phase 2 (e2e) step 2 - end-to-end cold start against ONE model layout.
# This is the striping question asked the way that actually matters: does the
# Lustre layout change `weight_loading` for a real SGLang launch, not for `dd`.
#
# Loader: fastsafetensors. Chosen deliberately - it is TP-rank-aware, so it
# reads each shard from exactly ONE rank (pg.size()=4 -> only ~4 concurrent
# readers). That low reader count is precisely where per-file striping should
# help, since a single reader can then drive several OSTs at once. The
# multithread/mmap loaders already run ~32 readers and read the model 4x over,
# so they mask the layout effect.
#
# Fresh exclusive node per submission, and the copy was written by a DIFFERENT
# node (phase2_make_copies.sbatch) -> the weight read is genuinely cold.
#
# Submit (one per layout, serialized via --dependency so they never contend):
#   sbatch --job-name=p2e2e-c8_s4M \
#          --export=ALL,TAG=c8_s4M,MODEL_PATH=/capstor/.../llama70b_c8_s4M \
#          lustre-loading-exp/scripts/phase2_e2e/phase2_e2e_layout.sbatch
#
#SBATCH --job-name=phase2-e2e
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=lustre-loading-exp/results/phase2_e2e/%x-%j.out

set -euo pipefail

TAG="${TAG:?set TAG, e.g. native or c8_s4M}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the model dir for this layout}"

SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
RES="${SLURM_SUBMIT_DIR}/lustre-loading-exp/results/phase2_e2e"
PROFILE_OUT="${RES}/phase2_e2e-${TAG}-${SLURM_JOB_ID}-${NODE}-profile.json"

echo "job=${SLURM_JOB_ID} node=${NODE} tag=${TAG} model=${MODEL_PATH}"
echo "profile_out=${PROFILE_OUT}"
echo "started=$(date --iso-8601=seconds)"
echo "--- layout of one shard (stripe_count / stripe_size) ---"
lfs getstripe -c "${MODEL_PATH}"/model-00001-of-*.safetensors 2>/dev/null || true
lfs getstripe -S "${MODEL_PATH}"/model-00001-of-*.safetensors 2>/dev/null || true

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
pip install fastsafetensors -q
exec servekit profile --out "${PROFILE_OUT}" \
  --bench --bench-requests 64 --bench-concurrency 16 \
  --bench-input-len 512 --bench-output-len 128 \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
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
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile result (${TAG}) ==="
cat "${PROFILE_OUT}" 2>/dev/null || echo "NO PROFILE JSON - run did not reach ready"
echo "finished=$(date --iso-8601=seconds)"
