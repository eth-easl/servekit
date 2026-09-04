#!/bin/bash
# GLM-5.1-FP8, TP4/PP4/EP4, 4 nodes, one pipeline stage a node: one job covers
# both cold-start speed and correctness against a presharded checkpoint.
#
# The first FP8 checkpoint and the first MLA model here, at 705 GB across 78
# layers. Block-wise FP8 keeps its scales in separate tensors that have to
# follow their weights into the right shard, which is what this arm proves.
#
# Build the checkpoint first with examples/run_glm51_fp8_sglang.sbatch. It must
# be dumped on an sglang carrying sgl-project/sglang#35715, or the canonical
# kv_b_proj weights are missing from the files and no amount of re-serving
# recovers them.
#SBATCH --partition=normal
#SBATCH --account=infra02
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

SLURM_SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"

export MODE="${MODE:-fast}"
export MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.1-FP8}"

case "$MODE" in
  baseline)
    OUT_PREFIX="baseline"
    ;;
  fast)
    export SHARDED_SRC="${SHARDED_SRC:-/capstor/store/cscs/swissai/infra01/servekit-artifacts/glm5.1-fp8-tp4-pp4}"
    OUT_PREFIX="fast"
    ;;
  *)
    echo "MODE must be 'baseline' or 'fast', got: $MODE" >&2
    exit 1
    ;;
esac

EDF="${SLURM_SUBMIT_DIR}/sglang5.16-glm.toml"
SERVED_MODEL_NAME=zai-org/GLM-5.1-FP8
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../.."
OUT="${SLURM_SUBMIT_DIR}/logs/${OUT_PREFIX}-${SLURM_JOB_ID}.json"

if [ "$MODE" = "fast" ]; then
  REPORT="${SLURM_SUBMIT_DIR}/logs/${OUT_PREFIX}-${SLURM_JOB_ID}.report.json"
  REFERENCE="${SERVEKIT_DIR}/tests/e2e/fixtures/glm5.1-tp4pp4-fp8.json"
fi

NNODES=4
TP_SIZE=4
PP_SIZE=4
EP_SIZE=4
DIST_PORT=20000
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"

echo "mode: ${MODE}, head node: ${HEAD_NODE}, nodes: ${SLURM_JOB_NODELIST}"

INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
unset BASH_ENV

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
# MLA registers kv_b_proj as a submodule of attn_mha on first forward, which
# makes save_model's storage dedup evict the canonical name. Both arms patch so
# they differ only in the loader.
python -m servekit._patches.sglang_35715

if [ "\${MODE}" = "fast" ]; then
  servekit launch --servekit-artifact-path "${SHARDED_SRC:-}" --out "${REPORT:-}" --timeout 5400 --overlap \
    -- \
    python -m sglang.launch_server \
      --model-path "${MODEL}" \
      --tensor-parallel-size ${TP_SIZE} \
      --pipeline-parallel-size ${PP_SIZE} \
      --expert-parallel-size ${EP_SIZE} \
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
      --random-seed 12345 \
      --enable-metrics &
  LAUNCH_PID=\$!
else
  python -m sglang.launch_server \
    --model-path "${MODEL:-}" \
    --tensor-parallel-size ${TP_SIZE} \
    --pipeline-parallel-size ${PP_SIZE} \
    --expert-parallel-size ${EP_SIZE} \
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
    --random-seed 12345 \
    --enable-metrics &
  LAUNCH_PID=\$!
fi

if [ "\${SLURM_PROCID}" = "0" ]; then
  if [ "\${MODE}" = "fast" ]; then
    # Verify first: the reference was recorded from a server that had served
    # nothing, and 64 batched bench requests ahead of the check would have the
    # two arms comparing server states rather than loaders.
    servekit verify --url "http://127.0.0.1:8080" --reference "${REFERENCE:-}" --out "${OUT}" --wait-ready 5400 || true
    servekit bench --url "http://127.0.0.1:8080" --into "${REPORT:-}" --wait-ready 5400 \
      --requests 64 --concurrency 16 --input-len 512 --output-len 128 --no-correctness || true
  else
    servekit verify --url "http://127.0.0.1:8080" --record "${OUT}" --wait-ready 5400 || true
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
