#!/bin/bash
#SBATCH --job-name=multinode-pp-glm51-fp8
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

EDF="${SLURM_SUBMIT_DIR}/sglang5.16-glm-clariden.toml"
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.1-FP8}"
ARTIFACT="${ARTIFACT:-/capstor/store/cscs/swissai/infra01/servekit-artifacts/glm5.1-fp8-tp4-pp4}"
SERVED_MODEL_NAME=zai-org/GLM-5.1-FP8

SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/.."
RUNDIR="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
OUT="${RUNDIR}/run.json"
mkdir -p "${RUNDIR}"

NNODES=4
TP=4
PP=4
EP=4
DIST_PORT=20000
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"

echo "head node: ${HEAD_NODE}, nodes: ${SLURM_JOB_NODELIST}"

INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
unset BASH_ENV

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
python -m servekit._patches.sglang_35715

servekit launch --servekit-artifact-path "${ARTIFACT}" --only-prepare --out "${OUT}" --timeout 5400 \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --tensor-parallel-size ${TP} \
    --pipeline-parallel-size ${PP} \
    --expert-parallel-size ${EP} \
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
    --enable-metrics
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

echo
echo "=== artifact ==="
for pp in $(seq 0 $((PP - 1))); do
  for tp in $(seq 0 $((TP - 1))); do
    [ -e "${ARTIFACT}/model-pp-${pp}-rank-${tp}-part-0.safetensors" ] || { echo "GATE FAIL: pp ${pp} tp ${tp} wrote no shards" >&2; exit 1; }
  done
done
[ -e "${ARTIFACT}/servekit.json" ] || { echo "GATE FAIL: no manifest" >&2; exit 1; }
du -sh "${ARTIFACT}"
cat "${ARTIFACT}/servekit.json"
