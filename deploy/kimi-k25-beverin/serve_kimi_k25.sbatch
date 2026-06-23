#!/bin/bash
#SBATCH --job-name=kimi-k27-vllm
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/kimi-k25-vllm}"
SCRIPT_DIR="${DEPLOY_DIR}/deploy/kimi-k25-beverin"
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K2.7-Code}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K2.7-Code}"
VLLM_PORT="${VLLM_PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-20G}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_PARALLEL_LOADING_WORKERS="${MAX_PARALLEL_LOADING_WORKERS:-2}"

OPENTELA_DIR="${OPENTELA_DIR:-/capstor/scratch/cscs/xyao/opentela}"
OPENTELA_BOOTSTRAP="${OPENTELA_BOOTSTRAP:-/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9}"
OPENTELA_TCP_PORT="${OPENTELA_TCP_PORT:-43905}"
OPENTELA_UDP_PORT="${OPENTELA_UDP_PORT:-59820}"

export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" \
  "${DEPLOY_DIR}/home" \
  "${DEPLOY_DIR}/cache/xdg" \
  "${DEPLOY_DIR}/cache/config" \
  "${DEPLOY_DIR}/cache/vllm" \
  "${DEPLOY_DIR}/cache/flashinfer" \
  "${DEPLOY_DIR}/cache/triton"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD_NODE="${NODES[0]}"

# Pick the head-node's high-speed IP for VLLM_HOST_IP discovery in the wrapper.
HEAD_IP="$(srun --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" hostname -i | awk '{print $1}')"

echo "job_id=${SLURM_JOB_ID}"
echo "nodes=${NODES[*]}"
echo "head_node=${HEAD_NODE}"
echo "head_ip=${HEAD_IP}"
echo "model=${MODEL}"
echo "served_model_name=${SERVED_MODEL_NAME}"
echo "tp=${TP_SIZE} pp=${PP_SIZE}"
echo "max_model_len=${MAX_MODEL_LEN} kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES} max_num_seqs=${MAX_NUM_SEQS}"
echo "port=${VLLM_PORT} master_port=${MASTER_PORT}"
echo "opentela_dir=${OPENTELA_DIR}"
echo "opentela_bootstrap=${OPENTELA_BOOTSTRAP}"
echo "started=$(date --iso-8601=seconds)"

# Generate the vLLM subprocess wrapper that OpenTela will manage.
# vLLM must run inside the ROCm container via srun --environment.
cat > "${DEPLOY_DIR}/vllm_subprocess.sh" <<VLLM_WRAPPER
#!/bin/bash
set -euo pipefail

export VLLM_HOST_IP="${HEAD_IP}"
export VLLM_LOGGING_LEVEL="\${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_NO_USAGE_STATS="1"
export RAY_DEDUP_LOGS="\${RAY_DEDUP_LOGS:-0}"

echo "vllm_subprocess_started head=${HEAD_NODE} ip=\${VLLM_HOST_IP} \$(date --iso-8601=seconds)"

exec srun \\
  --nodes=${SLURM_JOB_NUM_NODES} \\
  --ntasks=${SLURM_JOB_NUM_NODES} \\
  --ntasks-per-node=1 \\
  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\
  --gpus-per-task=4 \\
  --environment=kimi-k25-rocm \\
  bash -lc '
    set -euo pipefail

    NODE_RANK="\${SLURM_PROCID}"
    MASTER_ADDR="${HEAD_NODE}"
    VLLM_HOST_IP="\$(hostname -i | awk "{print \\\$1}")"
    export VLLM_HOST_IP
    export VLLM_LOGGING_LEVEL="\${VLLM_LOGGING_LEVEL:-INFO}"
    export RAY_DEDUP_LOGS="\${RAY_DEDUP_LOGS:-0}"

    echo "rank=\${NODE_RANK} host=\$(hostname) ip=\${VLLM_HOST_IP} rocr=\${ROCR_VISIBLE_DEVICES:-unset} hip=\${HIP_VISIBLE_DEVICES:-unset}"

    VLLM_ARGS=(
      "${MODEL}"
      --host 0.0.0.0
      --port "${VLLM_PORT}"
      --served-model-name "${SERVED_MODEL_NAME}"
      --tensor-parallel-size "${TP_SIZE}"
      --pipeline-parallel-size "${PP_SIZE}"
      --distributed-executor-backend mp
      --nnodes "${SLURM_JOB_NUM_NODES}"
      --node-rank "\${NODE_RANK}"
      --master-addr "\${MASTER_ADDR}"
      --master-port "${MASTER_PORT}"
      --mm-encoder-tp-mode data
      --block-size=1
      --tool-call-parser kimi_k2
      --reasoning-parser kimi_k2
      --enable-auto-tool-choice
      --enable-prefix-caching
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --max-model-len "${MAX_MODEL_LEN}"
      --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}"
      --max-num-seqs "${MAX_NUM_SEQS}"
      --max-parallel-loading-workers "${MAX_PARALLEL_LOADING_WORKERS}"
      --trust-remote-code
    )

    if [[ "\${NODE_RANK}" != "0" ]]; then
      VLLM_ARGS+=(--headless)
    fi

    exec vllm serve "\${VLLM_ARGS[@]}"
  '
VLLM_WRAPPER
chmod +x "${DEPLOY_DIR}/vllm_subprocess.sh"

if [[ -x "${OPENTELA_DIR}/otela" ]]; then
  OPENTELA_CONFIG="${OPENTELA_DIR}/cfg-kimi-k25-${SLURM_JOB_ID}.yaml"
  OPENTELA_LOG="${DEPLOY_DIR}/logs/opentela-${SLURM_JOB_ID}.log"

  "${OPENTELA_DIR}/otela" init --config "${OPENTELA_CONFIG}" >/dev/null 2>&1 || true

  # Start OpenTela in the background with stdin redirected from /dev/zero.
  # OpenTela exits when run as a foreground process in Slurm (likely because
  # stdin is EOF), but survives as a background process with live stdin.
  nohup "${OPENTELA_DIR}/otela" start \
    --config "${OPENTELA_CONFIG}" \
    --bootstrap.addr "${OPENTELA_BOOTSTRAP}" \
    --tcpport "${OPENTELA_TCP_PORT}" \
    --udpport "${OPENTELA_UDP_PORT}" \
    --mode node \
    --solana.skip_verification \
    --service.name llm \
    --service.port "${VLLM_PORT}" \
    --subprocess "${DEPLOY_DIR}/vllm_subprocess.sh" \
    --label "model=${SERVED_MODEL_NAME}" \
    --label "slurm_job_id=${SLURM_JOB_ID}" \
    --label "node=${HEAD_NODE}" \
    < /dev/zero \
    > "${OPENTELA_LOG}" 2>&1 &
  OPENTELA_PID=$!
  echo "opentela_started pid=${OPENTELA_PID} log=${OPENTELA_LOG}"
else
  echo "opentela_binary_not_found at ${OPENTELA_DIR}/otela"
  OPENTELA_PID=""
fi

cat > "${DEPLOY_DIR}/last_service.env" <<EOF
JOB_ID=${SLURM_JOB_ID}
HEAD_NODE=${HEAD_NODE}
PORT=${VLLM_PORT}
MODEL=${SERVED_MODEL_NAME}
SUBMITTED_AT=$(date --iso-8601=seconds)
LOG=${DEPLOY_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out
OPENTELA_PID=${OPENTELA_PID:-}
OPENTELA_LOG=${OPENTELA_LOG:-}
EOF

# Keep the Slurm job alive as long as OpenTela is running.
if [[ -n "${OPENTELA_PID}" ]]; then
  wait "${OPENTELA_PID}"
  echo "opentela_exited \$?=$(date --iso-8601=seconds)"
fi
