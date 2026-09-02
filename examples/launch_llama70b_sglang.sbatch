#!/bin/bash
# `servekit launch`: fast weight loading + --overlap, Llama-3.1-70B, TP4, 1 node.
#SBATCH --job-name=launch-llama70b-sglang
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/sglang.toml"
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
ARTIFACT="${ARTIFACT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded}"
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct

SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/.."
OUT="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.json"

INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

servekit launch --servekit-artifact-path "${ARTIFACT}" --out "${OUT}" --overlap --timeout 3000 \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size 4 \
    --attention-backend flashinfer \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.85 \
    --max-running-requests 256 \
    --enable-metrics &
LAUNCH_PID=\$!

servekit bench --url "http://127.0.0.1:8080" --into "${OUT}" \
  --wait-ready 3000 --requests 64 --concurrency 16 --input-len 512 --output-len 128

kill \${LAUNCH_PID} 2>/dev/null || true
wait \${LAUNCH_PID} 2>/dev/null || true
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
