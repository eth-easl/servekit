#!/bin/bash
# Phase 1.3 - end-to-end cold-start baseline for Llama-3.1-70B, NATIVE layout,
# with --load-format runai_streamer (Run:ai Model Streamer). Reads the SAME
# .safetensors files with the streamer's own concurrent-chunk reader, decoupled
# from TP rank count (unlike fastsafetensors' per-rank fan-out). Concurrency
# set to 64 via --model-loader-extra-config '{"concurrency": 64}'.
# Directly submittable:  sbatch lustre-loading-exp/scripts/phase1.3_e2e/phase1_3_e2e_baseline_runai_streamer.sbatch
#
# runai-model-streamer ships in the lmsysorg/sglang:v0.5.10 image already, so
# unlike the fastsafetensors baseline there is no extra pip install here.
#
# Do NOT run while a capstor-heavy job (dd sweep) is active - contention skews it.
#
#SBATCH --job-name=e2e-baseline-runai-streamer
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

MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256
STREAMER_CONCURRENCY=64

NODE="$(hostname)"
PROFILE_OUT="${SLURM_SUBMIT_DIR}/lustre-loading-exp/results/phase1.3_e2e/runai_streamer/phase1_3_e2e-runai_streamer-${SLURM_JOB_ID}-${NODE}-profile.json"

echo "job=${SLURM_JOB_ID} node=${NODE} tag=runai_streamer concurrency=${STREAMER_CONCURRENCY} model=${MODEL}"
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
    --model-loader-extra-config '{"concurrency": ${STREAMER_CONCURRENCY}}' \
    --enable-metrics
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile result (runai_streamer) ==="
cat "${PROFILE_OUT}" 2>/dev/null || echo "NO PROFILE JSON - run did not reach ready"
echo "finished=$(date --iso-8601=seconds)"
