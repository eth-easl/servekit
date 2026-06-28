#!/bin/bash
#SBATCH --job-name=glm-47-flash-sglang
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-47-flash-sglang/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-sglang}"
SCRIPT_DIR="${DEPLOY_DIR}/deploy/glm-47-flash-bristen"
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-zai-org/GLM-4.7-Flash}"
SGLANG_PORT="${SGLANG_PORT:-8080}"
TP_SIZE="${TP_SIZE:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-256}"
SGLANG_REASONING_PARSER="${SGLANG_REASONING_PARSER:-glm45}"
SGLANG_TOOL_CALL_PARSER="${SGLANG_TOOL_CALL_PARSER:-glm47}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"

OPENTELA_DIR="${OPENTELA_DIR:-/capstor/scratch/cscs/xyao/opentela}"
OPENTELA_BOOTSTRAP="${OPENTELA_BOOTSTRAP:-/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9}"
OPENTELA_TCP_PORT="${OPENTELA_TCP_PORT:-43905}"
OPENTELA_UDP_PORT="${OPENTELA_UDP_PORT:-59820}"

METRICS_REMOTE_WRITE_URL="${METRICS_REMOTE_WRITE_URL:-https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write}"
METRICS_AGENT_BIN="${METRICS_AGENT_BIN:-/capstor/store/cscs/swissai/infra01/ocf-share/vmagent-amd64}"

export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" \
  "${DEPLOY_DIR}/home" \
  "${DEPLOY_DIR}/cache/hf" \
  "${DEPLOY_DIR}/cache/xdg" \
  "${DEPLOY_DIR}/cache/triton" \
  "${DEPLOY_DIR}/cache/pip"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
HEAD_NODE="${NODES[0]}"
NUM_NODES="${#NODES[@]}"

echo "job_id=${SLURM_JOB_ID}"
echo "num_nodes=${NUM_NODES}"
echo "nodes=${NODES[*]}"
echo "head_node=${HEAD_NODE}"
echo "model=${MODEL}"
echo "served_model_name=${SERVED_MODEL_NAME}"
echo "tp=${TP_SIZE}"
echo "max_model_len=${MAX_MODEL_LEN} max_running_requests=${MAX_RUNNING_REQUESTS} mem_fraction_static=${MEM_FRACTION_STATIC}"
echo "reasoning_parser=${SGLANG_REASONING_PARSER} tool_call_parser=${SGLANG_TOOL_CALL_PARSER:-<none>} disable_cuda_graph=${DISABLE_CUDA_GRAPH}"
echo "port=${SGLANG_PORT}"
echo "transformers_pin=${TRANSFORMERS_VERSION}"
echo "opentela_dir=${OPENTELA_DIR}"
echo "opentela_bootstrap=${OPENTELA_BOOTSTRAP}"
echo "metrics_remote_write_url=${METRICS_REMOTE_WRITE_URL}"
echo "metrics_agent_bin=${METRICS_AGENT_BIN}"
echo "started=$(date --iso-8601=seconds)"

# Shared SGLang subprocess wrapper used by every rank. Each rank runs on its own
# node, so one wrapper is safe to share. Single-node TP=4: no cross-node init,
# no torchrun, no --nnodes/--dist-init-addr.
SGLANG_SUBPROCESS="${DEPLOY_DIR}/sglang_subprocess-${SLURM_JOB_ID}.sh"
cat > "${SGLANG_SUBPROCESS}" <<SGLANG_WRAPPER
#!/bin/bash
set -euo pipefail

echo "sglang_subprocess_started rank=\${SLURM_PROCID} host=\$(hostname) \$(date --iso-8601=seconds)"

SGLANG_ARGS=(
  --model-path "${MODEL}"
  --host 0.0.0.0
  --port "${SGLANG_PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --trust-remote-code
  --context-length "${MAX_MODEL_LEN}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --enable-metrics
)

if [[ -n "${SGLANG_TOOL_CALL_PARSER}" ]]; then
  SGLANG_ARGS+=(--tool-call-parser "${SGLANG_TOOL_CALL_PARSER}")
fi
if [[ -n "${SGLANG_REASONING_PARSER}" ]]; then
  SGLANG_ARGS+=(--reasoning-parser "${SGLANG_REASONING_PARSER}")
fi
if [[ "${DISABLE_CUDA_GRAPH}" == "1" ]]; then
  SGLANG_ARGS+=(--disable-cuda-graph)
fi

# GLM-4.7-Flash (model_type glm4_moe_lite) is newer than the transformers
# pinned in the official SGLang 0.5.9 image (4.57.1), which rejects the config.
# The checkpoint ships no custom modeling code, so --trust-remote-code cannot
# help; native support lands later in the transformers 5.x line (verified on
# 5.12.1). Compute nodes have PyPI access, so upgrade at startup, pinned for
# reproducibility, with a persistent pip cache on /capstor for fast relaunch.
export PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"
python -m pip install --no-input --break-system-packages --upgrade \
  "transformers==${TRANSFORMERS_VERSION}" >&2 || true
echo "transformers_version=\$(python -c 'import transformers,sys;sys.stdout.write(transformers.__version__)' 2>/dev/null) rank=\${SLURM_PROCID} \$(date --iso-8601=seconds)"

exec python -m sglang.launch_server "\${SGLANG_ARGS[@]}"
SGLANG_WRAPPER
chmod +x "${SGLANG_SUBPROCESS}"

# Per-rank vmagent scrape config template. The actual container IP is filled in
# by the launcher because vmagent's Go resolver has trouble reaching SGLang via
# localhost in this container setup.
SCRAPE_CONFIG_TEMPLATE="${DEPLOY_DIR}/vmagent-scrape-${SLURM_JOB_ID}.yaml.tpl"
cat > "${SCRAPE_CONFIG_TEMPLATE}" <<SCRAPE
global:
  scrape_interval: 30s
scrape_configs:
  - job_name: slurm-llm
    metrics_path: /metrics
    static_configs:
      - targets: ['LOCALHOST_IP:${SGLANG_PORT}']
SCRAPE

# OpenTela launcher runs inside the SGLang container on every rank. It uses
# SLURM_PROCID to create a per-rank config/datastore directory on the shared
# filesystem so concurrent OpenTela instances do not collide.
OTELA_LAUNCHER="${DEPLOY_DIR}/opentela_launcher-${SLURM_JOB_ID}.sh"
cat > "${OTELA_LAUNCHER}" <<OTELA_LAUNCHER
#!/bin/bash
set -euo pipefail

RANK="\${SLURM_PROCID}"
NODE="\$(hostname)"
LOCALHOST_IP="\$(hostname -i | awk '{print \$1}')"
# Per-rank libp2p ports. Every peer used to bind the same fixed tcp/udp port,
# so multiple peers sharing one NAT egress collided on a single external port
# mapping to the bootstrap and only ~half registered on the gateway. Offsetting
# by SLURM_PROCID gives each peer a distinct port.
TCP_PORT=\$(( ${OPENTELA_TCP_PORT} + \${RANK} ))
UDP_PORT=\$(( ${OPENTELA_UDP_PORT} + \${RANK} ))
OPENTELA_HOME="${DEPLOY_DIR}/home/opentela-${SLURM_JOB_ID}-\${RANK}"
OPENTELA_CONFIG="\${OPENTELA_HOME}/.config/otela/cfg.yaml"
OPENTELA_LOG="${DEPLOY_DIR}/logs/opentela-${SLURM_JOB_ID}-\${RANK}.log"
VMAGENT_LOG="${DEPLOY_DIR}/logs/vmagent-${SLURM_JOB_ID}-\${RANK}.log"
SCRAPE_CONFIG="${DEPLOY_DIR}/vmagent-scrape-${SLURM_JOB_ID}-\${RANK}.yaml"
mkdir -p "\${OPENTELA_HOME}/.config/otela"

export HOME="\${OPENTELA_HOME}"

sed "s/LOCALHOST_IP/\${LOCALHOST_IP}/g" "${SCRAPE_CONFIG_TEMPLATE}" > "\${SCRAPE_CONFIG}"

"${OPENTELA_DIR}/otela" init --config "\${OPENTELA_CONFIG}" >/dev/null 2>&1 || true

echo "opentela_launcher_started rank=\${RANK} node=\${NODE} ip=\${LOCALHOST_IP} tcp=\${TCP_PORT} udp=\${UDP_PORT} home=\${OPENTELA_HOME} \$(date --iso-8601=seconds)"

# Stagger peer joins so all N ranks do not hit the single bootstrap in the same
# instant (the thundering herd was correlated with failed registrations).
sleep \$(( \${RANK} * 10 ))

# OpenTela is backgrounded with stdin from /dev/zero because it exits when run
# as a foreground process with EOF on stdin. It will start SGLang as its
# subprocess.
nohup "${OPENTELA_DIR}/otela" start \\
  --config "\${OPENTELA_CONFIG}" \\
  --bootstrap.addr "${OPENTELA_BOOTSTRAP}" \\
  --tcpport "\${TCP_PORT}" \\
  --udpport "\${UDP_PORT}" \\
  --mode node \\
  --solana.skip_verification \\
  --service.name llm \\
  --service.port "${SGLANG_PORT}" \\
  --subprocess "${SGLANG_SUBPROCESS}" \\
  --label "model=${SERVED_MODEL_NAME}" \\
  --label "framework=sglang" \\
  --label "slurm_job_id=${SLURM_JOB_ID}" \\
  --label "node=\${NODE}" \\
  < /dev/zero \\
  > "\${OPENTELA_LOG}" 2>&1 &
OPENTELA_PID=\$!
echo "opentela_started rank=\${RANK} pid=\${OPENTELA_PID} log=\${OPENTELA_LOG}"

# Wait for SGLang to be ready before starting vmagent. vmagent's Go HTTP client
# can get stuck if it starts scraping before the target is listening. SGLang's
# /get_model_info returns 200 only once the model is loaded (no inference).
echo "waiting_for_sglang rank=\${RANK} url=http://\${LOCALHOST_IP}:${SGLANG_PORT}/get_model_info \$(date --iso-8601=seconds)"
while true; do
  if curl -s "http://\${LOCALHOST_IP}:${SGLANG_PORT}/get_model_info" >/dev/null 2>&1; then
    echo "sglang_ready rank=\${RANK} \$(date --iso-8601=seconds)"
    break
  fi
  sleep 2
done

# Start vmagent to scrape SGLang /metrics and push to the remote write endpoint.
VMAGENT_PID=""
if [[ -x "${METRICS_AGENT_BIN}" ]]; then
  nohup "${METRICS_AGENT_BIN}" \\
    -promscrape.config="\${SCRAPE_CONFIG}" \\
    -remoteWrite.url="${METRICS_REMOTE_WRITE_URL}" \\
    -remoteWrite.label="slurm_job_id=${SLURM_JOB_ID}" \\
    -remoteWrite.label="model=${SERVED_MODEL_NAME}" \\
    -remoteWrite.label="framework=sglang" \\
    -remoteWrite.label="user=\${SLURM_JOB_USER}" \\
    -remoteWrite.label="node=\${NODE}" \\
    "-remoteWrite.tmpDataPath=${DEPLOY_DIR}/cache/vmagent-data-${SLURM_JOB_ID}-\${RANK}" \\
    > "\${VMAGENT_LOG}" 2>&1 &
  VMAGENT_PID=\$!
  echo "vmagent_started rank=\${RANK} pid=\${VMAGENT_PID} log=\${VMAGENT_LOG}"
else
  echo "vmagent_binary_not_found at ${METRICS_AGENT_BIN}"
fi

wait "\${OPENTELA_PID}"
OPENTELA_CODE=\$?
echo "opentela_exited rank=\${RANK} code=\${OPENTELA_CODE} \$(date --iso-8601=seconds)"

if [[ -n "\${VMAGENT_PID}" ]]; then
  kill "\${VMAGENT_PID}" 2>/dev/null || true
  echo "vmagent_stopped rank=\${RANK} \$(date --iso-8601=seconds)"
fi

exit "\${OPENTELA_CODE}"
OTELA_LAUNCHER
chmod +x "${OTELA_LAUNCHER}"

# Record endpoints for all nodes before srun starts so the metadata is
# available while the services are still running.
{
  echo "JOB_ID=${SLURM_JOB_ID}"
  echo "JOBID=${SLURM_JOB_ID}"
  echo "NUM_NODES=${NUM_NODES}"
  echo "PORT=${SGLANG_PORT}"
  echo "MODEL=${SERVED_MODEL_NAME}"
  echo "FRAMEWORK=sglang"
  echo "SUBMITTED_AT=$(date --iso-8601=seconds)"
  echo "LOG=${DEPLOY_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
  for i in "${!NODES[@]}"; do
    echo "ENDPOINT_${i}=${NODES[$i]}:${SGLANG_PORT}"
  done
} > "${DEPLOY_DIR}/last_service.env"

# Launch one OpenTela+SGLang service per node inside the SGLang container.
srun \
  --nodes="${NUM_NODES}" \
  --ntasks="${NUM_NODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment=glm-47-flash-sglang \
  bash -lc "${OTELA_LAUNCHER}"
SRUN_EXIT=$?
echo "srun_exited code=${SRUN_EXIT} $(date --iso-8601=seconds)"

exit "${SRUN_EXIT}"
