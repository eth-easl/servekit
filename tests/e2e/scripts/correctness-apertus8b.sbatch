#!/bin/bash
# Correctness test for weight loading: compares baseline (no servekit) vs fast (servekit staging).
#
# Run with MODE=baseline or MODE=fast environment variable:
#   MODE=baseline sbatch correctness-apertus8b.sbatch
#   MODE=fast sbatch correctness-apertus8b.sbatch
#   FRESH=1 sbatch correctness-apertus8b.sbatch     # shard the checkpoint in this job
#
# Baseline: plain sglang reading original HF safetensors.
# Fast: presharded checkpoint staged into /dev/shm by servekit.
#
# FRESH points the artifact at an empty scratch dir, so `servekit launch` shards
# the checkpoint itself instead of reading one sharded long ago.


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

export MODE="${MODE:-fast}"
export MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509}"
export TP="${TP:-4}"

case "$MODE" in
  baseline)
    OUT_PREFIX="baseline"
    ;;
  fast)
    if [ -n "${FRESH:-}" ]; then
      export ARTIFACT="${ARTIFACT:-${SCRATCH:-/tmp}/servekit-correctness-e2e-${SLURM_JOB_ID}}"
      [ -n "${KEEP_ARTIFACT:-}" ] || trap 'rm -rf "${ARTIFACT}"' EXIT
    else
      export ARTIFACT="${ARTIFACT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/apertus8b-tp4-sharded}"
    fi
    OUT_PREFIX="fast"
    ;;
  *)
    echo "MODE must be 'baseline' or 'fast', got: $MODE" >&2
    exit 1
    ;;
esac

EDF="${SLURM_SUBMIT_DIR}/../../../examples/fast-weight-load/sglang.toml"
export SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509
export SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../.."
export OUT="${SLURM_SUBMIT_DIR}/logs/${OUT_PREFIX}-${SLURM_JOB_ID}.json"

if [ "$MODE" = "fast" ]; then
  export REPORT="${SLURM_SUBMIT_DIR}/logs/${OUT_PREFIX}-${SLURM_JOB_ID}.report.json"
  export REFERENCE="${SERVEKIT_DIR}/tests/e2e/fixtures/apertus8b-tp4-bf16.json"
  export MANIFEST_OUT="${SLURM_SUBMIT_DIR}/logs/${OUT_PREFIX}-${SLURM_JOB_ID}.manifest.json"
  export CACHES_OUT="${SLURM_SUBMIT_DIR}/logs/${OUT_PREFIX}-${SLURM_JOB_ID}.caches.json"
fi

INNER_CMD=$(cat <<'EOF'
set -euo pipefail
ulimit -c 0

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

if [ "$MODE" = "fast" ]; then
  servekit launch --servekit-artifact-path "${ARTIFACT}" --out "${REPORT}" --timeout 3000 \
    -- \
    python -m sglang.launch_server \
      --model-path "${MODEL}" \
      --host 0.0.0.0 \
      --port 8080 \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size "${TP}" \
      --trust-remote-code \
      --context-length 32768 \
      --mem-fraction-static 0.85 \
      --max-running-requests 256 \
      --enable-metrics &
  LAUNCH_PID=$!
else
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP}" \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.85 \
    --max-running-requests 256 \
    --enable-metrics &
  SERVER_PID=$!
fi

VERIFY_RC=0
if [ "$MODE" = "fast" ]; then
  servekit verify --url "http://127.0.0.1:8080" --reference "${REFERENCE}" --out "${OUT}" --wait-ready 3000 || VERIFY_RC=$?

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
else
  servekit verify --url "http://127.0.0.1:8080" --record "${OUT}" --wait-ready 3000 || VERIFY_RC=$?
fi

if [ "$MODE" = "fast" ]; then
  kill ${LAUNCH_PID} 2>/dev/null || true
  wait ${LAUNCH_PID} 2>/dev/null || true
else
  kill ${SERVER_PID} 2>/dev/null || true
  wait ${SERVER_PID} 2>/dev/null || true
fi

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
