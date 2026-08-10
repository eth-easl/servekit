#!/bin/bash
# Correctness test for weight loading: compares baseline (no servekit) vs fast (servekit staging).
#
# Run with MODE=baseline or MODE=fast environment variable:
#   MODE=baseline sbatch correctness-llama70b.sbatch
#   MODE=fast sbatch correctness-llama70b.sbatch
#
# Baseline: plain sglang reading original HF safetensors.
# Fast: presharded checkpoint staged into /dev/shm by servekit.


#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

export MODE="${MODE:-fast}"

case "$MODE" in
  baseline)
    export MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
    OUT_PREFIX="baseline"
    ;;
  fast)
    export SHARDED_SRC="${SHARDED_SRC:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded}"
    OUT_PREFIX="fast"
    ;;
  *)
    echo "MODE must be 'baseline' or 'fast', got: $MODE" >&2
    exit 1
    ;;
esac

EDF="${SLURM_SUBMIT_DIR}/../../../examples/fast-weight-load/llama-3.1-70b-sglang.toml"
export SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
export SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../.."
export OUT="${SLURM_SUBMIT_DIR}/logs/${OUT_PREFIX}-${SLURM_JOB_ID}.json"

if [ "$MODE" = "fast" ]; then
  export REPORT="${SLURM_SUBMIT_DIR}/logs/${OUT_PREFIX}-${SLURM_JOB_ID}.report.json"
  export REFERENCE="${SERVEKIT_DIR}/tests/e2e/fixtures/llama70b-tp4-bf16.json"
fi

INNER_CMD=$(cat <<'EOF'
set -euo pipefail
ulimit -c 0

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

if [ "$MODE" = "fast" ]; then
  servekit launch --out "${REPORT}" --timeout 3000 \
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
  LAUNCH_PID=$!
else
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
    --enable-metrics &
  SERVER_PID=$!
fi

VERIFY_RC=0
if [ "$MODE" = "fast" ]; then
  servekit verify --url "http://127.0.0.1:8080" --reference "${REFERENCE}" --out "${OUT}" --wait-ready 3000 || VERIFY_RC=$?
else
  servekit verify --url "http://127.0.0.1:8080" --record "${OUT}" --wait-ready 3000 || VERIFY_RC=$?
fi

if [ "$MODE" = "fast" ]; then
  kill ${LAUNCH_PID} 2>/dev/null || true
  wait ${LAUNCH_PID} 2>/dev/null || true
else
  kill ${SERVER_PID} 2>/dev/null || true
  wait ${SERVER_PID} 2>/dev/null || true
fi

exit ${VERIFY_RC}
EOF
)

srun \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "$INNER_CMD"
