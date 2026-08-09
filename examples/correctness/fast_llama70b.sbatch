#!/bin/bash
# The arm under test: the presharded checkpoint staged into /dev/shm by servekit.
# Engine flags are identical to baseline_llama70b.sbatch except --model-path and
# --load-format, so tests/e2e/test_correctness_e2e.py compares two loads of the
# same weights and nothing else.
#SBATCH --job-name=correctness-fast-llama70b
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

EDF="${SLURM_SUBMIT_DIR}/../fast-weight-load/llama-3.1-70b-sglang.toml"
SHARDED_SRC="${SHARDED_SRC:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded}"
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct

SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../.."
PROBE="${SERVEKIT_DIR}/tests/e2e/probe_logprobs.py"
OUT="${SLURM_SUBMIT_DIR}/logs/fast-${SLURM_JOB_ID}.json"
REPORT="${SLURM_SUBMIT_DIR}/logs/fast-${SLURM_JOB_ID}.report.json"

INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

servekit launch --out "${REPORT}" --overlap --timeout 3000 \
  -- \
  python -m sglang.launch_server \
    --model-path "${SHARDED_SRC}" \
    --load-format sharded_state \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.85 \
    --max-running-requests 256 \
    --enable-metrics &
LAUNCH_PID=\$!

python "${PROBE}" --url "http://127.0.0.1:8080" --out "${OUT}" --wait-ready 3000

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
