#!/bin/bash
# Build the TP=4/PP=2 presharded checkpoint, one pipeline stage a node.
#SBATCH --job-name=servekit-prepare-apertus8b-tp4pp2
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

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509}"
OUT="${OUT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/apertus8b-tp4pp2-sharded}"
TP="${TP:-4}"
PP="${PP:-2}"
NNODES=2
DIST_PORT=20000

EDF="${SLURM_SUBMIT_DIR}/sglang-clariden.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../.."
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"


INNER_CMD=$(cat <<EOF
set -euo pipefail
unset BASH_ENV
ulimit -c 0
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

servekit prepare --model "${MODEL}" --out "${OUT}" --tp "${TP}" --pp "${PP}" \
  --nnodes ${NNODES} \
  --node-rank \${SLURM_PROCID} \
  --dist-init-addr ${HEAD_NODE}:${DIST_PORT} \
  -- \
  --attention-backend flashinfer \
  --trust-remote-code \
  --context-length 32768 \
  --mem-fraction-static 0.85
EOF
)


srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --wait=60 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true


# Checks that all shards have been written. Shards are keyed on (pp_rank,
# tp_rank), so the two nodes write disjoint names into one directory.
echo
echo "=== prepared checkpoint ==="
for pp in $(seq 0 $((PP - 1))); do
  for tp in $(seq 0 $((TP - 1))); do
    [ -e "${OUT}/model-pp-${pp}-rank-${tp}-part-0.safetensors" ] || { echo "GATE FAIL: pp ${pp} tp ${tp} wrote no shards" >&2; exit 1; }
  done
done
[ -e "${OUT}/config.json" ] || { echo "GATE FAIL: no config.json" >&2; exit 1; }
[ -e "${OUT}/servekit.json" ] || { echo "GATE FAIL: no manifest" >&2; exit 1; }
du -sh "${OUT}"
ls "${OUT}"
cat "${OUT}/servekit.json"
