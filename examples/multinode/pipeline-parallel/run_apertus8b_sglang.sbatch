#!/bin/bash
# Apertus-8B at TP=4/PP=2 across two 4-GPU Clariden nodes, one pipeline stage a
# node, served from /dev/shm. The first run presharded the checkpoint into
# ARTIFACT; later runs find it there and only serve.
#SBATCH --job-name=multinode-pp-apertus8b
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
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509}"
ARTIFACT="${ARTIFACT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/apertus8b-tp4pp2-sharded}"
SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509

SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../.."
RUNDIR="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
OUT="${RUNDIR}/run.json"
# What `servekit launch` writes on the head, once it knows --nnodes > 1.
OUT_HEAD="${RUNDIR}/run.node0.json"
mkdir -p "${RUNDIR}"

NNODES=2
TP=4
PP=2
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

servekit launch --servekit-artifact-path "${ARTIFACT}" --out "${OUT}" --timeout 3000 \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --tensor-parallel-size ${TP} \
    --pipeline-parallel-size ${PP} \
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
# its own rather than holding the allocation to the time limit.
srun \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --wait=60 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

# Shards are keyed on (pp_rank, tp_rank), so the two nodes write disjoint names
# into one directory.
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
