#!/bin/bash
#SBATCH --job-name=apertus-8b-sglang
#SBATCH --partition=debug
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/apertus-8b-bristen/apertus-8b-sglang.toml"
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509
SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256



HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"

# ServerKit directory
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
PROFILE_OUT="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-profile.json"

echo "job_id=${SLURM_JOB_ID}"
echo "head_node=${HEAD_NODE}"
echo "model=${MODEL}"
echo "served_model_name=${SERVED_MODEL_NAME}"
echo "tp=${TP_SIZE}"
echo "max_model_len=${MAX_MODEL_LEN} max_running_requests=${MAX_RUNNING_REQUESTS} mem_fraction_static=${MEM_FRACTION_STATIC}"
echo "port=${SGLANG_PORT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
exec servekit profile --out "${PROFILE_OUT}" -- \
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
    --enable-metrics
EOF
)

srun \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"
SRUN_EXIT=$?
echo "srun_exited code=${SRUN_EXIT} $(date --iso-8601=seconds)"
exit "${SRUN_EXIT}"
