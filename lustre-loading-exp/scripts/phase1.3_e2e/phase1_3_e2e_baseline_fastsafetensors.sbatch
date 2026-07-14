#!/bin/bash
# Phase 1.3 - end-to-end cold-start baseline for Llama-3.1-70B, NATIVE layout,
# with --load-format fastsafetensors. Reads the SAME .safetensors files via the
# fastsafetensors loader: one shard per TP rank at a time (pg.size()=4 -> 4-way
# file parallelism), copied to GPU (GPU-Direct Storage if available on Lustre,
# else a pread/CPU fallback). The mmap flag does NOT apply to this loader.
# Directly submittable:  sbatch lustre-loading-exp/scripts/phase1.3_e2e/phase1_3_e2e_baseline_fastsafetensors.sbatch
#
# fastsafetensors is NOT in the image; installed into the ephemeral container
# overlay at job start (compute nodes have PyPI access), BEFORE the profiled
# launch so it's outside the measured cold-start window.
#
# Do NOT run while a capstor-heavy job (dd sweep) is active - contention skews it.
#
#SBATCH --job-name=e2e-baseline-fastsafetensors
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=lustre-loading-exp/results/phase1.3_e2e/fastsafetensors/%x-%j.out

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

NODE="$(hostname)"
PROFILE_OUT="${SLURM_SUBMIT_DIR}/lustre-loading-exp/results/phase1.3_e2e/fastsafetensors/phase1_3_e2e-fastsafetensors-${SLURM_JOB_ID}-${NODE}-profile.json"

echo "job=${SLURM_JOB_ID} node=${NODE} tag=fastsafetensors model=${MODEL}"
echo "profile_out=${PROFILE_OUT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
pip install fastsafetensors -q
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
    --load-format fastsafetensors \
    --enable-metrics
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile result (fastsafetensors) ==="
cat "${PROFILE_OUT}" 2>/dev/null || echo "NO PROFILE JSON - run did not reach ready"
echo "finished=$(date --iso-8601=seconds)"
