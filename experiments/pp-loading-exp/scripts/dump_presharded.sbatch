#!/bin/bash
# ONE-OFF, OFFLINE - produce the PP=4 presharded dump. Not part of any
# measurement.
#
# There is no separate save API: --load-format presharded takes a normal first
# load, runs post_load_weights and process_weights_after_loading, and dumps the
# resulting inference-ready state. So this job is just a server start that we
# tear down once it is ready.
#
# NOTE the dump is locked to the FULL parallelism + quantization config, not
# just TP. The cache subfolder is TP-{tp}-sig-{sha1 over tp/dp/ep/pp/quant/
# dtype/eplb/...}, so a PP=4 dump will not be picked up by a PP=2 run -- it
# silently misses and re-dumps rather than loading something wrong.
#
# This job also doubles as the per-model gate: a model that cannot load at PP=4
# under this image fails here for pennies instead of inside a measured run.
#
#   cd experiments/pp-loading-exp && ./submit.sh apertus8b dump
#
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/scripts/models.sh"

EDF="${SLURM_SUBMIT_DIR}/scripts/sglang-nightly.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../servekit"

if [ -d "${PRESHARDED_ROOT}" ] && [ -n "$(ls -A "${PRESHARDED_ROOT}" 2>/dev/null)" ]; then
  echo "ERROR: ${PRESHARDED_ROOT} is not empty." >&2
  echo "       Refusing to write over an existing dump. Remove it by hand:" >&2
  echo "         rm -r ${PRESHARDED_ROOT}" >&2
  exit 1
fi
mkdir -p "${PRESHARDED_ROOT}"

echo "job=${SLURM_JOB_ID} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
echo "preset=${MODEL_PRESET:-apertus8b} tp=${TP_SIZE} pp=${PP_SIZE}"
echo "src=${MODEL}"
echo "out=${PRESHARDED_ROOT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

# --timeout bounds the wait; the dump happens during startup, so reaching ready
# means it is written and the READY sentinel is in place.
servekit profile --out /dev/null --timeout "${READY_TIMEOUT}" \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${SERVER_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --pipeline-parallel-size "${PP_SIZE}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --trust-remote-code \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --load-format presharded \
    --model-loader-extra-config '{"presharded_path": "${PRESHARDED_ROOT}"}' &
PROF_PID=\$!

# Blocks until the server is ready, which is also when the dump is complete.
servekit bench --url "http://127.0.0.1:${SERVER_PORT}" --out /dev/null \
  --wait-ready "${READY_TIMEOUT}" \
  --requests 2 --concurrency 1 --input-len 64 --output-len 8 || true

kill \${PROF_PID} 2>/dev/null || true
wait \${PROF_PID} 2>/dev/null || true
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result (dump) ==="
python3 "${SLURM_SUBMIT_DIR}/scripts/inspect_dump.py" "${PRESHARDED_ROOT}" "${TP_SIZE}" "${PP_SIZE}"
echo "finished=$(date --iso-8601=seconds)"
