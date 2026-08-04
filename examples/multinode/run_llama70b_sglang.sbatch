#!/bin/bash
# Llama-3.1-70B at TP=8 across two 4-GPU Clariden nodes, served from /dev/shm.
#SBATCH --job-name=multinode-tp-llama70b
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

EDF="${SLURM_SUBMIT_DIR}/sglang-clariden.toml"
SHARDED_SRC="${SHARDED_SRC:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp8-sharded}"
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct

SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../.."
# One directory per run: `servekit report` merges every *.node*.json it finds,
# so a shared results dir would fold the last run's nodes into this one's.
RUNDIR="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
OUT="${RUNDIR}/run.json"
# What `servekit launch` writes on the head, once it knows --nnodes > 1.
OUT_HEAD="${RUNDIR}/run.node0.json"
MERGED="${RUNDIR}/merged.json"
mkdir -p "${RUNDIR}"

NNODES=2
TP_SIZE=8
DIST_PORT=20000
# Resolved here, not in the container: both tasks must be handed the same
# literal address, and only the submit host can expand the nodelist.
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"

echo "head node: ${HEAD_NODE}, nodes: ${SLURM_JOB_NODELIST}"

# No backticks anywhere below, not even in comments: they would be command
# substituted on the submit host while this heredoc is built.
INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
# The image sets BASH_ENV=/etc/bash.bashrc, which makes every one of the
# stager's ~900 subshells run nvidia-smi first: 0.41 GB/s instead of 16.89.
unset BASH_ENV

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

servekit launch --out "${OUT}" --timeout 3000 \
  -- \
  python -m sglang.launch_server \
    --model-path "${SHARDED_SRC}" \
    --load-format sharded_state \
    --tensor-parallel-size ${TP_SIZE} \
    --nnodes ${NNODES} \
    --node-rank \${SLURM_PROCID} \
    --dist-init-addr ${HEAD_NODE}:${DIST_PORT} \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.85 \
    --max-running-requests 256 \
    --enable-metrics &
LAUNCH_PID=\$!

# Only the head binds a port, so only the head benches. The worker just waits
# for its engine to go down with the head's.
if [ "\${SLURM_PROCID}" = "0" ]; then
  servekit bench --url "http://127.0.0.1:8080" --into "${OUT_HEAD}" \
    --wait-ready 3000 --requests 64 --concurrency 16 --input-len 512 --output-len 128 || true
  kill \${LAUNCH_PID} 2>/dev/null || true
fi
wait \${LAUNCH_PID} 2>/dev/null || true
EOF
)

# --wait=60: once the head's task ends, give the worker a minute to fall over on
# its own rather than holding the allocation to the time limit. The worker's
# exit code is meaningless here -- the gates below decide whether this worked.
srun \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --wait=60 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

# Each node maxed every phase over its own four ranks; the server was ready only
# when the slowest of all eight was done, so the phases have to be maxed again
# across nodes. Reading the head's report alone quotes ranks 0-3 as if they were
# the world.
srun --nodes=1 --ntasks=1 --environment="${EDF}" bash -c "
  set -euo pipefail
  pip install -e '${SERVEKIT_DIR}' --no-build-isolation --no-deps -q
  servekit report '${RUNDIR}' --out '${MERGED}'
"

# Cold-start thresholds are checked in tests/e2e/test_cold_start.py (pytest -m e2e)
