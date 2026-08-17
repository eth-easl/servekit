#!/bin/bash
# The whole pipeline in one job, on Apertus-8B at TP4:
#   (1) servekit launch --only-prepare  -- shard the HF checkpoint into a fresh scratch dir
#   (2) servekit launch                 -- find that artifact prepared, stage it, serve it
#   (3) servekit verify                 -- check the served logprobs against the recorded baseline
#
# correctness-apertus8b.sbatch checks (2) and (3) against a checkpoint sharded
# long ago; here the checkpoint is built by the servekit under test, so a
# regression in `prepare` fails instead of hiding behind an old artifact.
#
#   sbatch prepare-serve-apertus8b.sbatch
#   KEEP_PREPARED=1 sbatch prepare-serve-apertus8b.sbatch   # keep (1) for inspection

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
export SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509
export TP="${TP:-4}"
export PREPARED="${PREPARED:-${SCRATCH:-/tmp}/servekit-prepare-serve-e2e-${SLURM_JOB_ID}}"

EDF="${SLURM_SUBMIT_DIR}/../../../examples/fast-weight-load/sglang.toml"
export SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../.."
export REFERENCE="${SERVEKIT_DIR}/tests/e2e/fixtures/apertus8b-tp4-bf16.json"
export OUT="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.json"
export REPORT="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.report.json"
export MANIFEST_OUT="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.manifest.json"
export CACHES_OUT="${SLURM_SUBMIT_DIR}/logs/prepared-${SLURM_JOB_ID}.caches.json"

if [ -z "${KEEP_PREPARED:-}" ]; then
  trap 'rm -rf "${PREPARED}"' EXIT
fi

INNER_CMD=$(cat <<'EOF'
set -euo pipefail
ulimit -c 0

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

servekit launch --servekit-artifact-path "${PREPARED}" --only-prepare \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.85

# The checkpoint itself is scratch, so keep its manifest where the test can
# still read it once the job has cleaned up.
cp "${PREPARED}/servekit.json" "${MANIFEST_OUT}"

python - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["PREPARED"]) / "servekit-caches"
summary = {d.name: {"bytes": sum(p.stat().st_size for p in d.rglob("*") if p.is_file())} for d in root.iterdir() if d.is_dir()}
Path(os.environ["CACHES_OUT"]).write_text(json.dumps(summary))
print(f"[E2E] JIT caches under {root}: {summary}")
PY

servekit launch --servekit-artifact-path "${PREPARED}" --out "${REPORT}" --timeout 3000 \
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

VERIFY_RC=0
servekit verify --url "http://127.0.0.1:8080" --reference "${REFERENCE}" \
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
