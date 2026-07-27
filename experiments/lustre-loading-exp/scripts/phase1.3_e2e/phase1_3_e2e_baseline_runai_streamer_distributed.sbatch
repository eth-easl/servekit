#!/bin/bash
# Phase 1.3 - end-to-end cold-start baseline for Llama-3.1-70B, NATIVE layout,
# with --load-format runai_streamer, distributed=true, CONCURRENCY swept
# (64/128/256/...) via --model-loader-extra-config.
#
# "distributed" defaults to False for local paths (it only auto-enables for
# s3://, gs://, az:// URIs) - see sglang/srt/model_loader/loader.py's
# RunaiModelStreamerLoader.__init__ / _prepare_weights. When False, the
# streamer stages every tensor on CPU before a separate H2D copy; when True
# it streams straight into the target CUDA device
# (weight_utils.py:runai_safetensors_weights_iterator sets device=cuda iff
# is_distributed and is_cuda_alike()).
#
# CONCURRENCY sizes the streamer's internal reader thread pool PER PROCESS
# (see runai_model_streamer/file_streamer/requests_iterator.py: files are
# batched up to RUNAI_STREAMER_MEMORY_LIMIT, default 40GB, and the thread
# pool pulls chunks across whatever files are in that batch - not scoped to
# a single file). Each of the TP_SIZE ranks runs its own streamer instance,
# so the actual node-wide reader fan-out is CONCURRENCY * TP_SIZE.
#
# Baselines so far (concurrency, distributed -> weight_loading):
#   64,  unset/false -> 89.81s  (job 75659, plain runai_streamer baseline)
#   64,  true         -> 84.91s (job 75660)
#   fastsafetensors reference: 73.7s
#
# Submit (normally via phase1_3_submit_runai_sweep.sh):
#   sbatch --job-name=p1.3-runai-c128 --export=ALL,TAG=c128,CONCURRENCY=128 \
#          lustre-loading-exp/scripts/phase1.3_e2e/phase1_3_e2e_baseline_runai_streamer_distributed.sbatch
#
# Do NOT run while a capstor-heavy job (dd sweep) is active - contention skews it.
#
#SBATCH --job-name=phase1.3-runai-streamer-distributed
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=lustre-loading-exp/results/phase1.3_e2e/runai_streamer/%x-%j.out

set -euo pipefail

TAG="${TAG:?set TAG, e.g. c64, c128, c256}"
CONCURRENCY="${CONCURRENCY:?set CONCURRENCY, e.g. 64, 128, 256}"

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
RES="${SLURM_SUBMIT_DIR}/lustre-loading-exp/results/phase1.3_e2e/runai_streamer"
PROFILE_OUT="${RES}/phase1_3_e2e-runai_streamer_distributed-${TAG}-${SLURM_JOB_ID}-${NODE}-profile.json"

echo "job=${SLURM_JOB_ID} node=${NODE} tag=${TAG} concurrency=${CONCURRENCY} model=${MODEL}"
echo "profile_out=${PROFILE_OUT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
exec servekit profile --out "${PROFILE_OUT}" \
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
    --load-format runai_streamer \
    --model-loader-extra-config '{"concurrency": ${CONCURRENCY}, "distributed": true}' \
    --enable-metrics
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile result (${TAG}, concurrency=${CONCURRENCY}) ==="
cat "${PROFILE_OUT}" 2>/dev/null || echo "NO PROFILE JSON - run did not reach ready"
echo "finished=$(date --iso-8601=seconds)"
