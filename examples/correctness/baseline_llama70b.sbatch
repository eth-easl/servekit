#!/bin/bash
# The reference arm: plain sglang reading the original HF safetensors off Lustre
# with mmap. No servekit, nothing staged to /dev/shm, no sharded_state -- so a
# bug in `servekit prepare`'s resharding cannot cancel out against the fast arm.
#
# Run this twice on fresh nodes. The two captures differ only by kernel
# nondeterminism, which is the noise floor the test's tolerances are set from:
#   python probe_logprobs.py --compare logs/baseline-<A>.json logs/baseline-<B>.json
#SBATCH --job-name=correctness-baseline-llama70b
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/../fast-weight-load/llama-3.1-70b-sglang.toml"
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct

SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../.."
PROBE="${SERVEKIT_DIR}/tests/e2e/probe_logprobs.py"
OUT="${SLURM_SUBMIT_DIR}/logs/baseline-${SLURM_JOB_ID}.json"

# Every engine flag below matches fast_llama70b.sbatch. Only --model-path and
# the absent --load-format differ, so the capture can only reflect the load.
INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0

python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.85 \
    --max-running-requests 256 \
    --enable-metrics &
SERVER_PID=\$!

python "${PROBE}" --url "http://127.0.0.1:8080" --out "${OUT}" --wait-ready 3000

kill \${SERVER_PID} 2>/dev/null || true
wait \${SERVER_PID} 2>/dev/null || true
EOF
)

srun \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"
