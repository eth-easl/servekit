#!/bin/bash
# ONE-OFF, OFFLINE: write the presharded dump this example's serve job loads.
#
# There is no save API for this format. The dump falls out of one ordinary load,
# so this job is the serving command itself, torn down as soon as it is up.
#
# The dump is keyed by the FULL parallel and quantization config -- the cache
# subdirectory is TP-{tp}-sig-{hash over tp/dp/ep/pp/quant/dtype/eplb/...}. A
# dump made at TP=4 PP=2 will not be picked up by a TP=4 PP=4 run: the engine
# calls that a cache miss and re-dumps, which at 70B or 1T is not something to
# discover from a log line afterwards. Both jobs source models.sh so their flags
# cannot drift.
#
# PRESHARDED_ROOT must be on a shared filesystem. Every rank writes its own
# manifest into it and the head builds the plan from all of them; a node-local
# path fails with "the presharded dump directory must be on a filesystem shared
# by all ranks/nodes".
#
#   ./submit.sh llama70b prepare
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00


set -euo pipefail

# Resolve before anything hands this path to the container: /users/$USER/scratch
# is a symlink to /iopsstor and /users is not among the EDF's mounts, so a
# submit from the symlinked path yields a directory the ranks cannot see.
SLURM_SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"

source "${SLURM_SUBMIT_DIR}/models.sh"

EDF="${SLURM_SUBMIT_DIR}/${EDF_NAME}"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../.."
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"

echo "preset=${MODEL_PRESET:-llama70b} tp=${TP_SIZE} pp=${PP_SIZE} ep=${EP_SIZE} nodes=${NNODES}"
echo "head node: ${HEAD_NODE}, nodes: ${SLURM_JOB_NODELIST}"
echo "model: ${MODEL}"
echo "dump:  ${PRESHARDED_ROOT}"
echo "started=$(date --iso-8601=seconds)"

# No backticks anywhere below, not even in comments: they would be command
# substituted on the submit host while this heredoc is built.
INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
unset BASH_ENV

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
export PYTHONPATH="${SLURM_SUBMIT_DIR}/patches"

servekit prepare --format presharded --out "${PRESHARDED_ROOT}" \
  --node-rank \${SLURM_PROCID} --timeout ${READY_TIMEOUT} \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --load-format presharded \
    --model-loader-extra-config '{"presharded_path": "${PRESHARDED_ROOT}"}' \
    --tensor-parallel-size ${TP_SIZE} \
    --pipeline-parallel-size ${PP_SIZE} \
    --expert-parallel-size ${EP_SIZE} \
    --nnodes ${NNODES} \
    --node-rank \${SLURM_PROCID} \
    --dist-init-addr ${HEAD_NODE}:${DIST_PORT} \
    --dist-timeout 3600 \
    ${ENGINE_EXTRA[@]} \
    --host 0.0.0.0 \
    --port ${SERVE_PORT} \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --context-length ${MAX_MODEL_LEN} \
    --mem-fraction-static ${MEM_FRACTION_STATIC} \
    --max-running-requests ${MAX_RUNNING_REQUESTS}
EOF
)

srun \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --wait=120 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"

echo
echo "=== dump ==="
du -sh "${PRESHARDED_ROOT}"/*/ 2>/dev/null || true
ls -la "${PRESHARDED_ROOT}"/*/ 2>/dev/null | head -20
echo "finished=$(date --iso-8601=seconds)"
