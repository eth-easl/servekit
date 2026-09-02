#!/bin/bash
# `servekit profile`: per-phase timing off the engine's own log output, no
# servekit optimizations applied. Llama-3.1-70B, TP4, 1 node.
#SBATCH --job-name=profile-llama70b-sglang
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/sglang.toml"
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct

SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/.."
PROFILE_OUT="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.json"

INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
exec servekit profile --out "${PROFILE_OUT}" -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.85 \
    --max-running-requests 256 \
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
