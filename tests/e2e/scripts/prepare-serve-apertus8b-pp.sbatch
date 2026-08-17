#!/bin/bash
# The whole pipeline in one job, on Apertus-8B at TP2/PP2 on a single node:
#   (0) plain sglang       -- serve the HF checkpoint, record logprobs
#   (1) servekit launch --only-prepare  -- dump a fresh sharded_state checkpoint
#   (2) servekit launch                 -- find that artifact prepared, stage it, serve it
#   (3) servekit verify    -- check (2)'s logprobs against (0)'s
#
# The baseline is recorded by this job at the same TP/PP, so a structural
# mismatch cannot masquerade as a loading bug and there is no fixture to go
# stale. TP2/PP2 fits four GPUs on one node, which keeps this off the sliced
# multi-node staging path.
#
#   sbatch prepare-serve-apertus8b-pp.sbatch
#   KEEP_PREPARED=1 sbatch prepare-serve-apertus8b-pp.sbatch   # keep (1) for inspection

#SBATCH --job-name=servekit-prepare-serve-apertus8b-pp
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

# The container mounts /capstor and /iopsstor only, so a submit dir reached
# through a /users symlink has to be resolved before it crosses into the job.
SLURM_SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"

export MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509}"
export SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509
export TP="${TP:-2}"
export PP="${PP:-2}"
export PREPARED="${PREPARED:-${SCRATCH:-/tmp}/servekit-prepare-serve-pp-e2e-${SLURM_JOB_ID}}"

# v0.5.11+, for the plugin framework the shim rides on.
EDF="${SLURM_SUBMIT_DIR}/../../../examples/multinode/pipeline-parallel/sglang-clariden.toml"
export SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../.."
export BASELINE="${SLURM_SUBMIT_DIR}/logs/prepared-pp-${SLURM_JOB_ID}.baseline.json"
export OUT="${SLURM_SUBMIT_DIR}/logs/prepared-pp-${SLURM_JOB_ID}.json"
export REPORT="${SLURM_SUBMIT_DIR}/logs/prepared-pp-${SLURM_JOB_ID}.report.json"
export MANIFEST_OUT="${SLURM_SUBMIT_DIR}/logs/prepared-pp-${SLURM_JOB_ID}.manifest.json"
export SHARDS_OUT="${SLURM_SUBMIT_DIR}/logs/prepared-pp-${SLURM_JOB_ID}.shards.json"

if [ -z "${KEEP_PREPARED:-}" ]; then
  trap 'rm -rf "${PREPARED}"' EXIT
fi

# No backticks below, not even in comments: they would be command substituted on
# the submit host while this heredoc is built.
INNER_CMD=$(cat <<'EOF'
set -euo pipefail
ulimit -c 0

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

COMMON_ARGS=(
  --tensor-parallel-size "${TP}"
  --pipeline-parallel-size "${PP}"
  --attention-backend flashinfer
  --host 0.0.0.0
  --port 8080
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --context-length 32768
  --mem-fraction-static 0.85
  --max-running-requests 256
)

# (0) baseline: plain HF load at the same TP/PP as the fast arm.
python -m sglang.launch_server --model-path "${MODEL}" "${COMMON_ARGS[@]}" &
BASELINE_PID=$!
servekit verify --url "http://127.0.0.1:8080" --record "${BASELINE}" --wait-ready 3000
kill ${BASELINE_PID} 2>/dev/null || true
wait ${BASELINE_PID} 2>/dev/null || true

# (1) prepare.
servekit launch --servekit-artifact-path "${PREPARED}" --only-prepare \
  -- \
  python -m sglang.launch_server --model-path "${MODEL}" "${COMMON_ARGS[@]}"

# The checkpoint is scratch, so keep what the test asserts on outside it.
cp "${PREPARED}/servekit.json" "${MANIFEST_OUT}"
python3 - "${PREPARED}" "${SHARDS_OUT}" <<'PY'
import json, os, sys
prepared, out = sys.argv[1:3]
names = sorted(os.listdir(prepared))
json.dump({
    "shards": [n for n in names if n.endswith(".safetensors")],
    "markers": [n for n in names if n.startswith(".sharded-state-done")],
}, open(out, "w"))
PY

# (2) launch: the same command as (0), pointed at the artifact from (1).
servekit launch --servekit-artifact-path "${PREPARED}" --out "${REPORT}" --timeout 3000 \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    "${COMMON_ARGS[@]}" \
    --enable-metrics &
LAUNCH_PID=$!

# (3) verify against the baseline from (0).
VERIFY_RC=0
servekit verify --url "http://127.0.0.1:8080" --reference "${BASELINE}" \
  --out "${OUT}" --wait-ready 3000 || VERIFY_RC=$?

kill ${LAUNCH_PID} 2>/dev/null || true
wait ${LAUNCH_PID} 2>/dev/null || true

exit ${VERIFY_RC}
EOF
)

srun \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "$INNER_CMD"
