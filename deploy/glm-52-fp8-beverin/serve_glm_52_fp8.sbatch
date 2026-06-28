#!/bin/bash
#SBATCH --job-name=glm-52-fp8-vllm
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-52-fp8-vllm/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-52-fp8-vllm}"
SCRIPT_DIR="${DEPLOY_DIR}/deploy/glm-52-fp8-beverin"
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.2-FP8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-5.2-FP8-rocm}"
VLLM_PORT="${VLLM_PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29500}"
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-${SLURM_JOB_NUM_NODES:-2}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_PARALLEL_LOADING_WORKERS="${MAX_PARALLEL_LOADING_WORKERS:-1}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_MTP="${ENABLE_MTP:-0}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"

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
NUM_NODES="${#NODES[@]}"

# Pick the head-node's high-speed IP for VLLM_HOST_IP discovery in the wrapper.
HEAD_IP="$(srun --nodes=1 --ntasks=1 --nodelist="${HEAD_NODE}" hostname -i | awk '{print $1}')"

echo "job_id=${SLURM_JOB_ID}"
echo "num_nodes=${NUM_NODES}"
echo "nodes=${NODES[*]}"
echo "head_node=${HEAD_NODE}"
echo "head_ip=${HEAD_IP}"
echo "model=${MODEL}"
echo "served_model_name=${SERVED_MODEL_NAME}"
echo "tp=${TP_SIZE} pp=${PP_SIZE}"
echo "max_model_len=${MAX_MODEL_LEN} max_num_seqs=${MAX_NUM_SEQS}"
echo "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES:-unset} kv_cache_dtype=${KV_CACHE_DTYPE}"
echo "enforce_eager=${ENFORCE_EAGER}"
echo "enable_mtp=${ENABLE_MTP} mtp_num_speculative_tokens=${MTP_NUM_SPECULATIVE_TOKENS}"
echo "port=${VLLM_PORT} master_port=${MASTER_PORT}"
echo "opentela_dir=${OPENTELA_DIR}"
echo "opentela_bootstrap=${OPENTELA_BOOTSTRAP}"
echo "started=$(date --iso-8601=seconds)"

# Generate the vLLM subprocess wrapper that OpenTela will manage.
# vLLM must run inside the ROCm container via srun --environment.
VLLM_SUBPROCESS="${DEPLOY_DIR}/vllm_subprocess-${SLURM_JOB_ID}.sh"
cat > "${VLLM_SUBPROCESS}" <<VLLM_WRAPPER
#!/bin/bash
set -euo pipefail

export VLLM_HOST_IP="${HEAD_IP}"
export VLLM_LOGGING_LEVEL="\${VLLM_LOGGING_LEVEL:-INFO}"
export VLLM_NO_USAGE_STATS="1"
export RAY_DEDUP_LOGS="\${RAY_DEDUP_LOGS:-0}"

echo "vllm_subprocess_started head=${HEAD_NODE} ip=\${VLLM_HOST_IP} \$(date --iso-8601=seconds)"

exec srun \\
  --nodes=${NUM_NODES} \\
  --ntasks=${NUM_NODES} \\
  --ntasks-per-node=1 \\
  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\
  --gpus-per-task=4 \\
  --environment=glm-52-fp8-rocm \\
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
      --nnodes "${NUM_NODES}"
      --node-rank "\${NODE_RANK}"
      --master-addr "\${MASTER_ADDR}"
      --master-port "${MASTER_PORT}"
      --tool-call-parser glm47
      --reasoning-parser glm45
      --enable-auto-tool-choice
      --chat-template-content-format string
      --enable-prefix-caching
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --max-model-len "${MAX_MODEL_LEN}"
      --max-num-seqs "${MAX_NUM_SEQS}"
      --max-parallel-loading-workers "${MAX_PARALLEL_LOADING_WORKERS}"
      --trust-remote-code
    )

    if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]]; then
      VLLM_ARGS+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
    fi

    if [[ "${KV_CACHE_DTYPE}" != "auto" ]]; then
      VLLM_ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
    fi

    if [[ "${ENFORCE_EAGER}" == "1" ]]; then
      VLLM_ARGS+=(--enforce-eager)
    fi

    if [[ "${ENABLE_MTP}" == "1" ]]; then
      VLLM_ARGS+=(
        --speculative-config.method mtp
        --speculative-config.num_speculative_tokens "${MTP_NUM_SPECULATIVE_TOKENS}"
      )
    fi

    if [[ "\${NODE_RANK}" != "0" ]]; then
      VLLM_ARGS+=(--headless)
    fi

    exec vllm serve "\${VLLM_ARGS[@]}"
  '
VLLM_WRAPPER
chmod +x "${VLLM_SUBPROCESS}"

if [[ -x "${OPENTELA_DIR}/otela" ]]; then
  OPENTELA_CONFIG="${OPENTELA_DIR}/cfg-glm-52-fp8-${SLURM_JOB_ID}.yaml"
  OPENTELA_LOG="${DEPLOY_DIR}/logs/opentela-${SLURM_JOB_ID}.log"

  "${OPENTELA_DIR}/otela" init --config "${OPENTELA_CONFIG}" >/dev/null 2>&1 || true

  # Start OpenTela in the background with stdin redirected from /dev/zero.
  # OpenTela exits when run as a foreground process in Slurm with EOF on stdin.
  nohup "${OPENTELA_DIR}/otela" start \
    --config "${OPENTELA_CONFIG}" \
    --bootstrap.addr "${OPENTELA_BOOTSTRAP}" \
    --tcpport "${OPENTELA_TCP_PORT}" \
    --udpport "${OPENTELA_UDP_PORT}" \
    --mode node \
    --solana.skip_verification \
    --service.name llm \
    --service.port "${VLLM_PORT}" \
    --subprocess "${VLLM_SUBPROCESS}" \
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
JOBID=${SLURM_JOB_ID}
NUM_NODES=${NUM_NODES}
HEAD_NODE=${HEAD_NODE}
PORT=${VLLM_PORT}
MODEL=${SERVED_MODEL_NAME}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE}
ENFORCE_EAGER=${ENFORCE_EAGER}
ENABLE_MTP=${ENABLE_MTP}
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
