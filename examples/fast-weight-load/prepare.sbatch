#!/bin/bash
# Build the TP-presharded checkpoint that test_llama70b_sglang.sbatch consumes.
# Rebuilds from scratch, so run it after an engine upgrade or a TP change.
#
#   sbatch prepare.sbatch
#   MODEL=/store/.../Apertus-8B-Instruct-2509 OUT=/store/.../apertus8b-tp4 TP=4 sbatch prepare.sbatch
#
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

echo "job=${SLURM_JOB_ID} node=$(hostname)"
echo "model=${MODEL}"
echo "out=${OUT} tp=${TP}"

if [ -e "${OUT}" ]; then
  if [ -e "${OUT}/model-rank-0-part-0.safetensors" ] || [ -e "${OUT}/servekit.json" ]; then
    echo "removing the previous checkpoint at ${OUT}"
    rm -r "${OUT}"
  else
    echo "ERROR: ${OUT} exists but is not a servekit-prepared checkpoint." >&2
    echo "       Refusing to remove it. Point OUT elsewhere, or delete it by hand." >&2
    exit 1
  fi
fi

INNER_CMD=$(cat <<EOF
set -euo pipefail
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
