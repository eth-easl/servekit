#!/bin/bash
# Build the TP=8 presharded checkpoint that run_llama70b_sglang.sbatch consumes.
#SBATCH --job-name=servekit-prepare-llama70b-tp8
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

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
OUT="${OUT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp8-sharded}"
TP="${TP:-8}"
NNODES=2
DIST_PORT=20000

EDF="${SLURM_SUBMIT_DIR}/sglang-clariden.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../.."
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"


INNER_CMD=$(cat <<EOF
set -euo pipefail
unset BASH_ENV
ulimit -c 0
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

servekit prepare --model "${MODEL}" --out "${OUT}" --tp "${TP}" \
  --nnodes ${NNODES} \
  --node-rank \${SLURM_PROCID} \
  --dist-init-addr ${HEAD_NODE}:${DIST_PORT} \
  -- \
  --trust-remote-code \
  --context-length 32768 \
  --mem-fraction-static 0.85
EOF
)


srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --wait=60 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true



# Checks that all shards have been written. 
echo
echo "=== prepared checkpoint ==="
for rank in $(seq 0 $((TP - 1))); do
  [ -e "${OUT}/model-rank-${rank}-part-0.safetensors" ] || { echo "GATE FAIL: rank ${rank} wrote no shards" >&2; exit 1; }
done
[ -e "${OUT}/config.json" ] || { echo "GATE FAIL: no config.json" >&2; exit 1; }
[ -e "${OUT}/servekit.json" ] || { echo "GATE FAIL: no manifest" >&2; exit 1; }
du -sh "${OUT}"
ls "${OUT}"
cat "${OUT}/servekit.json"
