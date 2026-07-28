#!/bin/bash
# vLLM cold-start profile of Llama-3.1-70B on clariden (GH200).
#
# Config mirrors experiments/clariden-loading-exp exactly (TP=4, 32768 context,
# 0.85 memory fraction, 256 concurrent requests) so the numbers sit next to the
# SGLang baseline for the same model on the same hardware.
#
# Run profile/llama-3.1-70b-clariden-vllm/preflight_vllm.sbatch FIRST.
#
#   cd <repo root> && sbatch profile/llama-3.1-70b-clariden-vllm/serve_llama70b_vllm.sbatch
#
# One FRESH node per run: the OS page cache survives across container runs, so a
# reused node would serve the weights from RAM and the load phase would be a lie.
#
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
VLLM_PORT=8080
TP_SIZE=4
GPU_MEM_UTIL=0.85
MAX_MODEL_LEN=32768
MAX_NUM_SEQS=256
READY_TIMEOUT=3000

SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
NODE="$(hostname)"
PROFILE_OUT="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-${NODE}-profile.json"

echo "job_id=${SLURM_JOB_ID} node=${NODE} cpus=${SLURM_CPUS_PER_TASK}"
echo "model=${MODEL} served_model_name=${SERVED_MODEL_NAME}"
echo "tp=${TP_SIZE} max_model_len=${MAX_MODEL_LEN} max_num_seqs=${MAX_NUM_SEQS} gpu_mem_util=${GPU_MEM_UTIL}"
echo "port=${VLLM_PORT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail

# Outside the measured window. python3, not python: the vllm image has no
# \`python\` alias, unlike sglang's.
python3 -m pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

# --disable-custom-all-reduce is required, not tuning: vLLM's custom all-reduce
# dies with an illegal memory access on GH200 during CUDA graph capture
# (custom_all_reduce.cuh:455, job 2917848; vllm-project/vllm#30044). NCCL takes
# over the TP all-reduce instead.
#
# The bench is not optional: vLLM reports ready without ever issuing a warmup
# request, so ready_wait_s is the only record of the first-request cost that
# SGLang's total already includes.
servekit profile --out "${PROFILE_OUT}" --timeout "${READY_TIMEOUT}" \
  -- \
  vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --disable-custom-all-reduce &
PROF_PID=\$!

servekit bench --url "http://127.0.0.1:${VLLM_PORT}" --into "${PROFILE_OUT}" \
  --wait-ready "${READY_TIMEOUT}" \
  --requests 64 --concurrency 16 --input-len 512 --output-len 128 &
BENCH_PID=\$!

# Stop benching the moment the server dies, or a crashed engine leaves the job
# polling a dead port for the rest of READY_TIMEOUT and looks like it is running.
while kill -0 \${PROF_PID} 2>/dev/null && kill -0 \${BENCH_PID} 2>/dev/null; do sleep 5; done
kill -0 \${PROF_PID} 2>/dev/null || { echo "server exited; abandoning bench"; kill \${BENCH_PID} 2>/dev/null || true; }
wait \${BENCH_PID} 2>/dev/null || true

# run_profile forwards SIGTERM to the server, so this tears down both.
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
