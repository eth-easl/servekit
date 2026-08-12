#!/bin/bash
# Build the TP=4 PP=2 presharded dump that run_llama70b_sglang.sbatch consumes.
#
# There is no save API for this format: the dump falls out of one ordinary load,
# so this job is the serving command itself, torn down as soon as it is up.
# --pipeline-parallel-size is what selects the format; servekit writes
# --load-format and --model-loader-extra-config itself.
#
# The dump is keyed by the FULL parallel and quantization config -- the cache
# subdirectory is TP-{tp}-sig-{hash over tp/dp/ep/pp/quant/dtype/...}. A dump
# made at TP=4 PP=2 will not be picked up by a TP=4 PP=4 run; the engine calls
# that a cache miss and silently re-dumps. Keep the flags here and in the run
# job identical.
#
# OUT is what the run job serves from: prepare copies the config and tokenizer in
# beside the dump, so it is a model path in its own right.
#
# OUT must be on a shared filesystem: every rank writes its own manifest into it
# and the head builds the plan from all of them. A node-local path fails with
# "the presharded dump directory must be on a filesystem shared by all
# ranks/nodes".
#SBATCH --job-name=servekit-prepare-llama70b-pp2
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

# /users is a symlink to /iopsstor and is not mounted in the container, so the
# submit dir must be resolved before it crosses into one.
SLURM_SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
OUT="${OUT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-pp-presharded-tp4pp2}"
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct

NNODES=2
TP_SIZE=4
PP_SIZE=2
DIST_PORT=20000

EDF="${SLURM_SUBMIT_DIR}/sglang-nightly-clariden.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../.."
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"

echo "head node: ${HEAD_NODE}, nodes: ${SLURM_JOB_NODELIST}"
echo "dump: ${OUT}"

# No backticks anywhere below, not even in comments: they would be command
# substituted on the submit host while this heredoc is built.
INNER_CMD=$(cat <<EOF
set -euo pipefail
unset BASH_ENV
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

servekit prepare --out "${OUT}" \
  --node-rank \${SLURM_PROCID} --timeout 3000 \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
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
    --max-running-requests 256
EOF
)

srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 --wait=120 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"

# Checks that the dump landed: one cache subdirectory, with the checksum the
# loader plans the stage from.
echo
echo "=== prepared dump ==="
DUMP="$(ls -d "${OUT}"/*/ 2>/dev/null | head -1)"
[ -n "${DUMP}" ] || { echo "GATE FAIL: no cache subdirectory under ${OUT}" >&2; exit 1; }
[ -e "${DUMP}/checksum.json" ] || { echo "GATE FAIL: no checksum.json in ${DUMP}" >&2; exit 1; }
du -sh "${DUMP}"
ls "${DUMP}" | head
