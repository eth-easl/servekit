#!/bin/bash
# The whole pipeline in one job, on Apertus-8B at TP4 on a single node:
#   (0) plain sglang       -- serve the HF checkpoint, record logprobs
#   (1) servekit launch    -- stage the artifact and serve it under --overlap
#   (2) servekit verify    -- check (1)'s logprobs against (0)'s
#   (3) servekit bench     -- cold-start timing, after the check
#
# The baseline is recorded by this job at the same TP, on the same node, so a
# reference from other hardware cannot masquerade as a loading bug and there is
# no fixture to go stale.
#
#   sbatch prepare-serve-apertus8b.sbatch
#   FRESH=1 sbatch prepare-serve-apertus8b.sbatch      # shard it in this job
#   KEEP_ARTIFACT=1 FRESH=1 sbatch prepare-serve-apertus8b.sbatch
#
# FRESH points the artifact at an empty scratch dir, so `servekit launch` shards
# the checkpoint itself instead of reading one sharded long ago.


#SBATCH --job-name=servekit-prepare-serve-apertus8b
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
export TP="${TP:-4}"

if [ -n "${FRESH:-}" ]; then
  export ARTIFACT="${ARTIFACT:-${SCRATCH:-/tmp}/servekit-correctness-e2e-${SLURM_JOB_ID}}"
  [ -n "${KEEP_ARTIFACT:-}" ] || trap 'rm -rf "${ARTIFACT}"' EXIT
else
  export ARTIFACT="${ARTIFACT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/apertus8b-tp4-sharded}"
fi

EDF="${SLURM_SUBMIT_DIR}/sglang.toml"
export SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509
export SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../.."
export BASELINE="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.baseline.json"
export OUT="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.json"
export REPORT="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.report.json"
export MANIFEST_OUT="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.manifest.json"
export CACHES_OUT="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.caches.json"

INNER_CMD=$(cat <<'EOF'
set -euo pipefail
ulimit -c 0

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

# One list, both arms: an engine flag that reached only one of them would show up
# as a loading difference that isn't one.
COMMON_ARGS=(
  --host 0.0.0.0
  --port 8080
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP}"
  --trust-remote-code
  --context-length 32768
  --mem-fraction-static 0.85
  --max-running-requests 256
  --random-seed 12345
)

# (0) baseline: plain HF load at the same TP as the fast arm.
python -m sglang.launch_server --model-path "${MODEL}" "${COMMON_ARGS[@]}" &
BASELINE_PID=$!
servekit verify --url "http://127.0.0.1:8080" --record "${BASELINE}" --wait-ready 3000
kill ${BASELINE_PID} 2>/dev/null || true
wait ${BASELINE_PID} 2>/dev/null || true

# (1) launch: the same command as (0), pointed at the artifact, with the engine
# started concurrently with the stage.
servekit launch --servekit-artifact-path "${ARTIFACT}" --out "${REPORT}" --timeout 3000 --overlap \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    "${COMMON_ARGS[@]}" \
    --enable-metrics &
LAUNCH_PID=$!

# (2) verify against the baseline from (0), on a server as cold as the one that
# baseline was recorded from: 64 batched bench requests first would leave the two
# arms comparing different server states, not different loaders.
VERIFY_RC=0
servekit verify --url "http://127.0.0.1:8080" --reference "${BASELINE}" \
  --out "${OUT}" --wait-ready 3000 || VERIFY_RC=$?

# (3) bench for cold-start timing, once the comparison above is safely done.
servekit bench --url "http://127.0.0.1:8080" --into "${REPORT}" --wait-ready 3000 \
  --requests 64 --concurrency 16 --input-len 512 --output-len 128 --no-correctness

# The artifact may be scratch this job cleans up, so keep what the test
# asserts on outside it.
cp "${ARTIFACT}/servekit.json" "${MANIFEST_OUT}"
python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["ARTIFACT"]) / "servekit-caches"
dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
summary = {d.name: {"bytes": sum(p.stat().st_size for p in d.rglob("*") if p.is_file())} for d in dirs}
Path(os.environ["CACHES_OUT"]).write_text(json.dumps(summary))
print(f"[E2E] JIT caches under {root}: {summary}")
PY

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
