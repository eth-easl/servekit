#!/bin/bash
# Build the TP-presharded checkpoint that run_llama70b_sglang.sbatch consumes.
#SBATCH --job-name=servekit-prepare-llama70b
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
OUT="${OUT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-servekit}"
TP="${TP:-4}"

EDF="${SLURM_SUBMIT_DIR}/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../.."


INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

servekit prepare --model "${MODEL}" --out "${OUT}" --tp "${TP}" \
  -- \
  --trust-remote-code \
  --context-length 32768 \
  --mem-fraction-static 0.85
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"

echo
echo "=== prepared checkpoint ==="
du -sh "${OUT}"
ls "${OUT}"
cat "${OUT}/servekit.json"
