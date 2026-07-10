#!/bin/bash
# Smoke test for the new `servekit profile --bench` feature on Apertus-8B.
# Profiles cold start, then (after ready) runs the post-ready benchmark against
# the live server: correctness probe (greedy hash) + throughput (tok/s). servekit
# tears the server down when the benchmark finishes, so the job exits on its own
# (no scancel watcher needed).
# Directly submittable:  sbatch lustre-loading-exp/scripts/reference/test_bench_apertus8b.sbatch
#
#SBATCH --job-name=apertus8b-bench
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=lustre-loading-exp/results/reference/%x-%j.out

set -euo pipefail

MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/swiss-ai/Apertus-8B-Instruct-2509
SERVED_MODEL_NAME=swiss-ai/Apertus-8B-Instruct-2509
EDF="${SLURM_SUBMIT_DIR}/profile/apertus-8b-bristen/apertus-8b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
RESULTS_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp/results"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
PROFILE_OUT="${RESULTS_DIR}/apertus8b-bench-${SLURM_JOB_ID}-${NODE}-profile.json"

echo "job=${SLURM_JOB_ID} node=${NODE} model=${MODEL}"
echo "profile_out=${PROFILE_OUT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
exec servekit profile --out "${PROFILE_OUT}" \
  --bench --bench-requests 64 --bench-concurrency 16 \
  --bench-input-len 512 --bench-output-len 128 \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${SGLANG_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --enable-metrics
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile + bench report ==="
cat "${PROFILE_OUT}" 2>/dev/null || echo "NO PROFILE JSON - run did not reach ready"
echo "finished=$(date --iso-8601=seconds)"
