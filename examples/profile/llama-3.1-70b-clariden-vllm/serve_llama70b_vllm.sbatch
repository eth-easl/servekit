#SBATCH --job-name=llama-3.1-70b-vllm
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

EDF="${VLLM_EDF:-${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-clariden-vllm/llama-3.1-70b-vllm.toml}"
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct

SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
PROFILE_OUT="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${NODE}-profile.json"


INNER_CMD=$(cat <<EOF
set -euo pipefail

python3 -m pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

# --disable-custom-all-reduce is required, not tuning: vLLM's custom all-reduce
# dies with an illegal memory access on GH200 during CUDA graph capture.
servekit profile --out "${PROFILE_OUT}" --timeout 3000 \
  -- \
  vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 \
    --disable-custom-all-reduce &
PROF_PID=\$!

servekit bench --url "http://127.0.0.1:8080" --into "${PROFILE_OUT}" \
  --wait-ready 3000 \
  --requests 64 --concurrency 16 --input-len 512 --output-len 128 &
BENCH_PID=\$!

wait \${BENCH_PID}
kill \${PROF_PID} 2>/dev/null || true
wait \${PROF_PID} 2>/dev/null || true
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
SRUN_EXIT=$?
echo "srun_exited code=${SRUN_EXIT} $(date --iso-8601=seconds)"
exit "${SRUN_EXIT}"
