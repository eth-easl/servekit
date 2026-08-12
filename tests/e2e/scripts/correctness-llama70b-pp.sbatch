#!/bin/bash
# Correctness test for weight loading under pipeline parallelism: compares
# baseline (plain sglang, TP=4/PP=2, reading the HF checkpoint directly) vs
# fast (servekit staging the presharded TP=4/PP=2 dump into /dev/shm). Both
# arms use the same parallel topology and the same nightly image, so the only
# variable is the loading path -- a TP=8 baseline would also differ in
# reduction order and make logprob deltas meaningless.
#
# Run with MODE=baseline or MODE=fast environment variable:
#   MODE=baseline sbatch correctness-llama70b-pp.sbatch
#   MODE=fast sbatch correctness-llama70b-pp.sbatch

#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

export MODE="${MODE:-fast}"
export MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
EDF="${SLURM_SUBMIT_DIR}/../../../examples/multinode-pp/sglang-nightly-clariden.toml"

case "$MODE" in
  baseline)
    OUT_PREFIX="baseline"
    ;;
  fast)
    export PRESHARDED_ROOT="${PRESHARDED_ROOT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-pp-presharded-tp4pp2}"
    OUT_PREFIX="fast"
    ;;
  *)
    echo "MODE must be 'baseline' or 'fast', got: $MODE" >&2
    exit 1
    ;;
esac

SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../.."
OUT="${SLURM_SUBMIT_DIR}/logs/${OUT_PREFIX}-${SLURM_JOB_ID}.json"

if [ "$MODE" = "fast" ]; then
  REPORT="${SLURM_SUBMIT_DIR}/logs/${OUT_PREFIX}-${SLURM_JOB_ID}.report.json"
  REFERENCE="${SERVEKIT_DIR}/tests/e2e/fixtures/llama70b-tp4pp2-bf16.json"
fi

NNODES=2
TP_SIZE=4
PP_SIZE=2
DIST_PORT=20000
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"

echo "mode: ${MODE}, head node: ${HEAD_NODE}, nodes: ${SLURM_JOB_NODELIST}"

INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
unset BASH_ENV

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

if [ "\${MODE}" = "fast" ]; then
  servekit launch --out "${REPORT:-}" --timeout 3000 \
    -- \
    python -m sglang.launch_server \
      --model-path "${MODEL:-}" \
      --load-format presharded \
      --model-loader-extra-config '{"presharded_path": "${PRESHARDED_ROOT:-}"}' \
      --tensor-parallel-size ${TP_SIZE} \
      --pipeline-parallel-size ${PP_SIZE} \
      --nnodes ${NNODES} \
      --node-rank \${SLURM_PROCID} \
      --dist-init-addr ${HEAD_NODE}:${DIST_PORT} \
      --attention-backend flashinfer \
      --host 0.0.0.0 \
      --port 8080 \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --trust-remote-code \
      --context-length 32768 \
      --mem-fraction-static 0.85 \
      --max-running-requests 256 \
      --enable-metrics &
  LAUNCH_PID=\$!
else
  python -m sglang.launch_server \
    --model-path "${MODEL:-}" \
    --tensor-parallel-size ${TP_SIZE} \
    --pipeline-parallel-size ${PP_SIZE} \
    --nnodes ${NNODES} \
    --node-rank \${SLURM_PROCID} \
    --dist-init-addr ${HEAD_NODE}:${DIST_PORT} \
    --attention-backend flashinfer \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.85 \
    --max-running-requests 256 \
    --enable-metrics &
  LAUNCH_PID=\$!
fi

if [ "\${SLURM_PROCID}" = "0" ]; then
  if [ "\${MODE}" = "fast" ]; then
    servekit verify --url "http://127.0.0.1:8080" --reference "${REFERENCE:-}" --out "${OUT}" --wait-ready 3000 || true
  else
    servekit verify --url "http://127.0.0.1:8080" --record "${OUT}" --wait-ready 3000 || true
  fi
  kill \${LAUNCH_PID} 2>/dev/null || true
fi
wait \${LAUNCH_PID} 2>/dev/null || true
EOF
)

srun \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --wait=60 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true
