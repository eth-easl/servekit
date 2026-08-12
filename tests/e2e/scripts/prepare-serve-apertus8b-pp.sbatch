#!/bin/bash
# The whole pipeline in one job, on Apertus-8B at TP4/PP2 across two nodes,
# with no external fixture: the job records its own baseline before checking
# servekit against it.
#   (0) plain sglang       -- serve the HF checkpoint directly, record logprobs
#   (1) servekit prepare   -- dump a fresh presharded checkpoint by starting the
#       engine once against an empty root
#   (2) servekit launch    -- stage that checkpoint and serve it
#   (3) servekit verify    -- check (2)'s logprobs against (0)'s
#
# Small model, real PP topology: cheap enough to run every time, while still
# exercising exactly the path that broke for llama70b/TP4PP2. It is also what
# proves prepare's output stands alone as a model path: (2) serves from it
# without the source checkpoint in the command at all.
#
#   sbatch prepare-serve-apertus8b-pp.sbatch
#   KEEP_PREPARED=1 sbatch prepare-serve-apertus8b-pp.sbatch   # keep (1) for inspection

#SBATCH --job-name=servekit-prepare-serve-apertus8b-pp
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

export MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509}"
export SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509
export TP="${TP:-4}"
export PP="${PP:-2}"
export NNODES=2
export DIST_PORT=20000
# Must be on a filesystem shared by both nodes: every rank writes its own
# manifest into it and the head builds the plan from all of them.
export PREPARED="${PREPARED:-${SCRATCH:-/tmp}/servekit-prepare-serve-pp-e2e-${SLURM_JOB_ID}}"

EDF="${SLURM_SUBMIT_DIR}/../../../examples/multinode-pp/sglang-nightly-clariden.toml"
export SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../.."
# Scoped to this job, not a committed fixture: the baseline is captured fresh
# every run, so there is nothing on disk for the fast arm to drift out of sync
# with.
export BASELINE="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.baseline.json"
export OUT="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.json"
export REPORT="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.report.json"
export MANIFEST_OUT="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.manifest.json"
export HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"

if [ -z "${KEEP_PREPARED:-}" ]; then
  trap 'rm -rf "${PREPARED}"' EXIT
fi

echo "head node: ${HEAD_NODE}, nodes: ${SLURM_JOB_NODELIST}, dump: ${PREPARED}"

# No backticks anywhere below, not even in comments: they would be command
# substituted on the submit host while this heredoc is built.
INNER_CMD=$(cat <<'EOF'
set -euo pipefail
ulimit -c 0
unset BASH_ENV

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

BASELINE_ARGS=(
  --model-path "${MODEL}"
  --tensor-parallel-size "${TP}"
  --pipeline-parallel-size "${PP}"
  --nnodes "${NNODES}"
  --node-rank "${SLURM_PROCID}"
  --dist-init-addr "${HEAD_NODE}:${DIST_PORT}"
  --attention-backend flashinfer
  --host 0.0.0.0
  --port 8080
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --context-length 32768
  --mem-fraction-static 0.85
  --max-running-requests 256
)

# (0) baseline: plain HF load, same TP/PP topology as the fast arm, so a
# structural mismatch (TP8 vs TP4/PP2) can't masquerade as a loading bug.
python -m sglang.launch_server "${BASELINE_ARGS[@]}" &
BASELINE_PID=$!
if [ "${SLURM_PROCID}" = "0" ]; then
  servekit verify --url "http://127.0.0.1:8080" --record "${BASELINE}" --wait-ready 3000
  kill ${BASELINE_PID} 2>/dev/null || true
fi
wait ${BASELINE_PID} 2>/dev/null || true

# No --model-path and no --load-format: prepare reads from the source checkpoint
# and launch reads from prepare's output, and servekit picks the format off
# --pipeline-parallel-size either way.
ENGINE_ARGS=(
  --tensor-parallel-size "${TP}"
  --pipeline-parallel-size "${PP}"
  --nnodes "${NNODES}"
  --node-rank "${SLURM_PROCID}"
  --dist-init-addr "${HEAD_NODE}:${DIST_PORT}"
  --attention-backend flashinfer
  --host 0.0.0.0
  --port 8080
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --context-length 32768
  --mem-fraction-static 0.85
  --max-running-requests 256
)

# (1) prepare: dump a fresh presharded checkpoint off the source checkpoint.
servekit prepare --out "${PREPARED}" \
  --node-rank "${SLURM_PROCID}" --timeout 3000 \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    "${ENGINE_ARGS[@]}"

if [ "${SLURM_PROCID}" = "0" ]; then
  python3 - "${PREPARED}" "${MODEL}" "${MANIFEST_OUT}" <<'PY'
import json, sys
from servekit import presharded

out, model, manifest_out = sys.argv[1:4]
dumps = presharded.find_dumps(out)
assert len(dumps) == 1, f"expected one dump under {out}, found {len(dumps)}"
plan = presharded.read_plan(dumps[0])
json.dump({
    "format": "presharded",
    "tp_size": plan.shard_config.get("tp"),
    "pp_size": plan.shard_config.get("pp"),
    "world_size": plan.world_size,
    "source": model,
}, open(manifest_out, "w"))
PY
fi

# (2) launch: stage and serve the fresh dump, which is now the model path.
servekit launch --out "${REPORT}" --timeout 3000 \
  -- \
  python -m sglang.launch_server \
    --model-path "${PREPARED}" \
    "${ENGINE_ARGS[@]}" \
    --enable-metrics &
LAUNCH_PID=$!

# (3) verify: check (2) against the baseline recorded in (0).
VERIFY_RC=0
if [ "${SLURM_PROCID}" = "0" ]; then
  servekit verify --url "http://127.0.0.1:8080" --reference "${BASELINE}" \
    --out "${OUT}" --wait-ready 3000 || VERIFY_RC=$?
  kill ${LAUNCH_PID} 2>/dev/null || true
fi
wait ${LAUNCH_PID} 2>/dev/null || true

exit ${VERIFY_RC}
EOF
)

srun \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"
