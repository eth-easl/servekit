#!/bin/bash
#SBATCH --job-name=glm-47-flash-sglang-router
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

# ============================================================================
# Cache-aware-router variant of serve_glm_47_flash_sglang.sbatch.
#
# Topology differs from the base serve script. The base script runs N
# *independent* OpenTela+SGLang replicas, each registered to the OpenTela
# gateway as its own `llm` peer -> the gateway load-balances cache-BLIND, so
# shared-prefix traffic scatters across replicas and each replica's RadixCache
# under-hits.
#
# Here we put the SGLang Router in front so prefix-affine traffic is steered to
# the SAME backend (cache_aware policy), maximising cross-replica RadixCache hit
# rate:
#
#   external client -> OpenTela gateway -> [rank 0] sglang_router (cache_aware)
#                                              |-> http://node0:8080  (SGLang TP=4)
#                                              |-> http://node1:8080  (SGLang TP=4)
#                                              |-> ...                 (one per node)
#
#   * Every node runs a plain SGLang backend (TP=4, port 8080). Backends are
#     NOT registered to OpenTela -- they are private, reachable only inside the
#     SLURM allocation by hostname.
#   * Rank 0 ADDITIONALLY runs the router (CPU-only, co-located with its own
#     backend) and ONE OpenTela peer that supervises the router as its
#     subprocess and registers it as the single public `llm` service.
#
# Trade-off vs base: the router is a single front-door (SPOF) and OpenTela no
# longer sees individual replicas. In exchange you get cache-aware routing,
# health-checked backends, retries, and circuit breaking from one Rust hop.
# ============================================================================

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

# --- Router knobs ----------------------------------------------------------
# sglang-router is absent from the lmsysorg/sglang:v0.5.9 image (confirmed by
# probe_router.sbatch) but installs from PyPI in ~1s on a compute node. Pinned
# for reproducibility, same as the transformers upgrade.
ROUTER_VERSION="${ROUTER_VERSION:-0.3.2}"
ROUTER_PORT="${ROUTER_PORT:-8090}"
ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"
ROUTER_PROM_PORT="${ROUTER_PROM_PORT:-29000}"
# Cache-aware tuning. Empty => use the router's own defaults
# (cache-threshold 0.3, balance-abs 64, balance-rel 1.5). Override at submit.
CACHE_THRESHOLD="${CACHE_THRESHOLD:-}"
BALANCE_ABS_THRESHOLD="${BALANCE_ABS_THRESHOLD:-}"
BALANCE_REL_THRESHOLD="${BALANCE_REL_THRESHOLD:-}"
# How long the router (and rank 0's readiness poll) waits for backends to load.
WORKER_STARTUP_TIMEOUT_SECS="${WORKER_STARTUP_TIMEOUT_SECS:-1800}"

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

# Space-separated worker URL list, resolved at submit time from the actual
# allocation. The fixed SGLANG_PORT is reachable node-to-node inside the
# allocation (intra-allocation connectivity is what NCCL/SSH already rely on;
# the OpenTela NAT caveat only applies to the EXTERNAL bootstrap).
WORKER_URLS=""
for n in "${NODES[@]}"; do
  WORKER_URLS+="http://${n}:${SGLANG_PORT} "
done
WORKER_URLS="${WORKER_URLS% }"

echo "job_id=${SLURM_JOB_ID}"
echo "num_nodes=${NUM_NODES}"
echo "nodes=${NODES[*]}"
echo "head_node=${HEAD_NODE}"
echo "model=${MODEL}"
echo "served_model_name=${SERVED_MODEL_NAME}"
echo "tp=${TP_SIZE}"
echo "backend_port=${SGLANG_PORT} max_model_len=${MAX_MODEL_LEN} max_running_requests=${MAX_RUNNING_REQUESTS} mem_fraction_static=${MEM_FRACTION_STATIC}"
echo "router_port=${ROUTER_PORT} router_policy=${ROUTER_POLICY} router_version=${ROUTER_VERSION} router_prom_port=${ROUTER_PROM_PORT}"
echo "cache_threshold=${CACHE_THRESHOLD:-<default>} balance_abs=${BALANCE_ABS_THRESHOLD:-<default>} balance_rel=${BALANCE_REL_THRESHOLD:-<default>}"
echo "worker_urls=${WORKER_URLS}"
echo "reasoning_parser=${SGLANG_REASONING_PARSER} tool_call_parser=${SGLANG_TOOL_CALL_PARSER:-<none>} disable_cuda_graph=${DISABLE_CUDA_GRAPH}"
echo "transformers_pin=${TRANSFORMERS_VERSION}"
echo "opentela_dir=${OPENTELA_DIR}"
echo "opentela_bootstrap=${OPENTELA_BOOTSTRAP}"
echo "started=$(date --iso-8601=seconds)"

# ---------------------------------------------------------------------------
# Backend wrapper: a plain SGLang server, run directly on every node (NOT as an
# OpenTela subprocess). Same flags/transformers-upgrade as the base serve
# script's SGLANG_SUBPROCESS.
# ---------------------------------------------------------------------------
BACKEND_WRAPPER="${DEPLOY_DIR}/sglang_backend-${SLURM_JOB_ID}.sh"
cat > "${BACKEND_WRAPPER}" <<BACKEND_WRAPPER_EOF
#!/bin/bash
set -euo pipefail

echo "sglang_backend_started rank=\${SLURM_PROCID} host=\$(hostname) \$(date --iso-8601=seconds)"

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
# pinned in the SGLang 0.5.9 image; upgrade at startup (compute nodes have
# PyPI access) with a persistent pip cache. See README "Model compatibility".
export PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"
python -m pip install --no-input --break-system-packages --upgrade \
  "transformers==${TRANSFORMERS_VERSION}" >&2 || true
echo "transformers_version=\$(python -c 'import transformers,sys;sys.stdout.write(transformers.__version__)' 2>/dev/null) rank=\${SLURM_PROCID} \$(date --iso-8601=seconds)"

exec python -m sglang.launch_server "\${SGLANG_ARGS[@]}"
BACKEND_WRAPPER_EOF
chmod +x "${BACKEND_WRAPPER}"

# ---------------------------------------------------------------------------
# Router wrapper: runs ONLY on rank 0, as OpenTela's --subprocess. Installs
# sglang-router (absent from the base image), then launches the cache-aware
# router across all backends. The worker list is read from a file that rank 0
# writes once every backend has reported ready.
# ---------------------------------------------------------------------------
WORKER_LIST_FILE="${DEPLOY_DIR}/router-workers-${SLURM_JOB_ID}.txt"
ROUTER_WRAPPER="${DEPLOY_DIR}/sglang_router-${SLURM_JOB_ID}.sh"
cat > "${ROUTER_WRAPPER}" <<ROUTER_WRAPPER_EOF
#!/bin/bash
set -euo pipefail

echo "sglang_router_starting host=\$(hostname) \$(date --iso-8601=seconds)"

export PIP_CACHE_DIR="${DEPLOY_DIR}/cache/pip"
python -m pip install --no-input --break-system-packages \
  "sglang-router==${ROUTER_VERSION}" >&2
echo "sglang_router_version=\$(python -c 'import importlib.metadata as m;print(m.version("sglang-router"))' 2>/dev/null) \$(date --iso-8601=seconds)"

# Space-separated http://node:port list (nargs for --worker-urls). Intentional
# word-splitting -- do not quote the expansion.
WORKER_URLS="\$(cat "${WORKER_LIST_FILE}")"
echo "router_worker_urls=\${WORKER_URLS}"

ROUTER_ARGS=(
  --host 0.0.0.0
  --port "${ROUTER_PORT}"
  --policy "${ROUTER_POLICY}"
  --worker-urls \${WORKER_URLS}
  --worker-startup-timeout-secs "${WORKER_STARTUP_TIMEOUT_SECS}"
  --prometheus-host 0.0.0.0
  --prometheus-port "${ROUTER_PROM_PORT}"
)
if [[ -n "${CACHE_THRESHOLD}" ]]; then
  ROUTER_ARGS+=(--cache-threshold "${CACHE_THRESHOLD}")
fi
if [[ -n "${BALANCE_ABS_THRESHOLD}" ]]; then
  ROUTER_ARGS+=(--balance-abs-threshold "${BALANCE_ABS_THRESHOLD}")
fi
if [[ -n "${BALANCE_REL_THRESHOLD}" ]]; then
  ROUTER_ARGS+=(--balance-rel-threshold "${BALANCE_REL_THRESHOLD}")
fi

exec python -m sglang_router.launch_router "\${ROUTER_ARGS[@]}"
ROUTER_WRAPPER_EOF
chmod +x "${ROUTER_WRAPPER}"

# ---------------------------------------------------------------------------
# Per-rank launcher. Every rank: start a backend, wait until it serves, start
# vmagent. Rank 0 additionally: wait for ALL backends, write the worker list,
# then start the OpenTela peer that supervises the router as its subprocess.
# ---------------------------------------------------------------------------
LAUNCHER="${DEPLOY_DIR}/router_launcher-${SLURM_JOB_ID}.sh"
cat > "${LAUNCHER}" <<LAUNCHER_EOF
#!/bin/bash
set -uo pipefail

RANK="\${SLURM_PROCID}"
NODE="\$(hostname)"
LOCALHOST_IP="\$(hostname -i | awk '{print \$1}')"
ALL_NODES=(${NODES[*]})

echo "router_launcher_started rank=\${RANK} node=\${NODE} ip=\${LOCALHOST_IP} \$(date --iso-8601=seconds)"

# --- 1. Start the local SGLang backend ------------------------------------
BACKEND_LOG="${DEPLOY_DIR}/logs/sglang-backend-${SLURM_JOB_ID}-\${RANK}.log"
nohup "${BACKEND_WRAPPER}" < /dev/zero > "\${BACKEND_LOG}" 2>&1 &
BACKEND_PID=\$!
echo "sglang_backend_launched rank=\${RANK} pid=\${BACKEND_PID} log=\${BACKEND_LOG}"

# --- 2. Wait for the local backend to load (no inference) ------------------
echo "waiting_for_backend rank=\${RANK} url=http://\${LOCALHOST_IP}:${SGLANG_PORT}/get_model_info \$(date --iso-8601=seconds)"
DEADLINE=\$(( \$(date +%s) + ${WORKER_STARTUP_TIMEOUT_SECS} ))
while true; do
  if curl -s "http://\${LOCALHOST_IP}:${SGLANG_PORT}/get_model_info" >/dev/null 2>&1; then
    echo "backend_ready rank=\${RANK} \$(date --iso-8601=seconds)"
    break
  fi
  if ! kill -0 "\${BACKEND_PID}" 2>/dev/null; then
    echo "backend_died rank=\${RANK} -- see \${BACKEND_LOG}"; exit 1
  fi
  if [[ \$(date +%s) -ge \${DEADLINE} ]]; then
    echo "backend_ready_timeout rank=\${RANK}"; exit 1
  fi
  sleep 2
done

# --- 3. Start vmagent scraping the local backend /metrics ------------------
SCRAPE_CONFIG="${DEPLOY_DIR}/vmagent-scrape-${SLURM_JOB_ID}-\${RANK}.yaml"
{
  echo "global:"
  echo "  scrape_interval: 30s"
  echo "scrape_configs:"
  echo "  - job_name: slurm-llm"
  echo "    metrics_path: /metrics"
  echo "    static_configs:"
  echo "      - targets: ['\${LOCALHOST_IP}:${SGLANG_PORT}']"
  # Rank 0 also scrapes the router's own Prometheus endpoint.
  if [[ "\${RANK}" == "0" ]]; then
    echo "  - job_name: sglang-router"
    echo "    metrics_path: /metrics"
    echo "    static_configs:"
    echo "      - targets: ['\${LOCALHOST_IP}:${ROUTER_PROM_PORT}']"
  fi
} > "\${SCRAPE_CONFIG}"

VMAGENT_LOG="${DEPLOY_DIR}/logs/vmagent-${SLURM_JOB_ID}-\${RANK}.log"
VMAGENT_PID=""
if [[ -x "${METRICS_AGENT_BIN}" ]]; then
  nohup "${METRICS_AGENT_BIN}" \\
    -promscrape.config="\${SCRAPE_CONFIG}" \\
    -remoteWrite.url="${METRICS_REMOTE_WRITE_URL}" \\
    -remoteWrite.label="slurm_job_id=${SLURM_JOB_ID}" \\
    -remoteWrite.label="model=${SERVED_MODEL_NAME}" \\
    -remoteWrite.label="framework=sglang" \\
    -remoteWrite.label="user=\${SLURM_JOB_USER:-\${USER}}" \\
    -remoteWrite.label="node=\${NODE}" \\
    "-remoteWrite.tmpDataPath=${DEPLOY_DIR}/cache/vmagent-data-${SLURM_JOB_ID}-\${RANK}" \\
    > "\${VMAGENT_LOG}" 2>&1 &
  VMAGENT_PID=\$!
  echo "vmagent_started rank=\${RANK} pid=\${VMAGENT_PID} log=\${VMAGENT_LOG}"
else
  echo "vmagent_binary_not_found at ${METRICS_AGENT_BIN}"
fi

# --- 4. Worker ranks: just hold the node and supervise the backend ---------
if [[ "\${RANK}" != "0" ]]; then
  echo "backend_only rank=\${RANK} -- holding node, backend pid=\${BACKEND_PID}"
  wait "\${BACKEND_PID}"
  BACKEND_CODE=\$?
  echo "backend_exited rank=\${RANK} code=\${BACKEND_CODE} \$(date --iso-8601=seconds)"
  [[ -n "\${VMAGENT_PID}" ]] && kill "\${VMAGENT_PID}" 2>/dev/null || true
  exit "\${BACKEND_CODE}"
fi

# --- 5. Rank 0: wait for EVERY backend, then publish the worker list -------
echo "rank0_waiting_for_all_backends count=\${#ALL_NODES[@]} \$(date --iso-8601=seconds)"
DEADLINE=\$(( \$(date +%s) + ${WORKER_STARTUP_TIMEOUT_SECS} ))
for n in "\${ALL_NODES[@]}"; do
  while true; do
    if curl -s "http://\${n}:${SGLANG_PORT}/get_model_info" >/dev/null 2>&1; then
      echo "remote_backend_ready node=\${n} \$(date --iso-8601=seconds)"
      break
    fi
    if [[ \$(date +%s) -ge \${DEADLINE} ]]; then
      echo "remote_backend_timeout node=\${n} -- continuing; router will keep health-checking it"
      break
    fi
    sleep 3
  done
done

# Hand the (now reachable) worker list to the router wrapper.
printf '%s' "${WORKER_URLS}" > "${WORKER_LIST_FILE}"
echo "wrote_worker_list file=${WORKER_LIST_FILE} urls='${WORKER_URLS}'"

# --- 6. Rank 0: OpenTela peer supervising the router as its subprocess ------
# Reuses the exact OpenTela invocation from the base serve script, but the
# subprocess is the ROUTER and the registered service.port is ROUTER_PORT.
RANK_TCP_PORT=\$(( ${OPENTELA_TCP_PORT} + \${RANK} ))
RANK_UDP_PORT=\$(( ${OPENTELA_UDP_PORT} + \${RANK} ))
OPENTELA_HOME="${DEPLOY_DIR}/home/opentela-${SLURM_JOB_ID}-\${RANK}"
OPENTELA_CONFIG="\${OPENTELA_HOME}/.config/otela/cfg.yaml"
OPENTELA_LOG="${DEPLOY_DIR}/logs/opentela-${SLURM_JOB_ID}-\${RANK}.log"
mkdir -p "\${OPENTELA_HOME}/.config/otela"
export HOME="\${OPENTELA_HOME}"

"${OPENTELA_DIR}/otela" init --config "\${OPENTELA_CONFIG}" >/dev/null 2>&1 || true

echo "opentela_router_launcher rank=\${RANK} node=\${NODE} tcp=\${RANK_TCP_PORT} udp=\${RANK_UDP_PORT} service_port=${ROUTER_PORT} \$(date --iso-8601=seconds)"

nohup "${OPENTELA_DIR}/otela" start \\
  --config "\${OPENTELA_CONFIG}" \\
  --bootstrap.addr "${OPENTELA_BOOTSTRAP}" \\
  --tcpport "\${RANK_TCP_PORT}" \\
  --udpport "\${RANK_UDP_PORT}" \\
  --mode node \\
  --solana.skip_verification \\
  --service.name llm \\
  --service.port "${ROUTER_PORT}" \\
  --subprocess "${ROUTER_WRAPPER}" \\
  --label "model=${SERVED_MODEL_NAME}" \\
  --label "framework=sglang-router" \\
  --label "role=router" \\
  --label "routing_policy=${ROUTER_POLICY}" \\
  --label "slurm_job_id=${SLURM_JOB_ID}" \\
  --label "node=\${NODE}" \\
  < /dev/zero \\
  > "\${OPENTELA_LOG}" 2>&1 &
OPENTELA_PID=\$!
echo "opentela_started rank=\${RANK} pid=\${OPENTELA_PID} log=\${OPENTELA_LOG}"

# Best-effort readiness log for the public router endpoint.
echo "waiting_for_router rank=\${RANK} url=http://\${LOCALHOST_IP}:${ROUTER_PORT}/health \$(date --iso-8601=seconds)"
for i in \$(seq 1 120); do
  if curl -s "http://\${LOCALHOST_IP}:${ROUTER_PORT}/health" >/dev/null 2>&1; then
    echo "router_ready rank=\${RANK} \$(date --iso-8601=seconds)"; break
  fi
  sleep 5
done

wait "\${OPENTELA_PID}"
OPENTELA_CODE=\$?
echo "opentela_exited rank=\${RANK} code=\${OPENTELA_CODE} \$(date --iso-8601=seconds)"
[[ -n "\${VMAGENT_PID}" ]] && kill "\${VMAGENT_PID}" 2>/dev/null || true
# If OpenTela/router exits, also stop the co-located backend.
kill "\${BACKEND_PID}" 2>/dev/null || true
exit "\${OPENTELA_CODE}"
LAUNCHER_EOF
chmod +x "${LAUNCHER}"

# Record endpoints before srun. The PUBLIC endpoint is the router on the head
# node; the per-node backends are listed for debugging only.
{
  echo "JOB_ID=${SLURM_JOB_ID}"
  echo "JOBID=${SLURM_JOB_ID}"
  echo "NUM_NODES=${NUM_NODES}"
  echo "PORT=${ROUTER_PORT}"
  echo "MODEL=${SERVED_MODEL_NAME}"
  echo "FRAMEWORK=sglang-router"
  echo "ROUTING_POLICY=${ROUTER_POLICY}"
  echo "SUBMITTED_AT=$(date --iso-8601=seconds)"
  echo "LOG=${DEPLOY_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
  echo "ENDPOINT=${HEAD_NODE}:${ROUTER_PORT}"
  echo "ROUTER_PROM=${HEAD_NODE}:${ROUTER_PROM_PORT}"
  for i in "${!NODES[@]}"; do
    echo "BACKEND_${i}=${NODES[$i]}:${SGLANG_PORT}"
  done
} > "${DEPLOY_DIR}/last_service.env"

# Launch one launcher per node. Each rank runs a TP=4 backend on its 4 GPUs;
# rank 0 also runs the (CPU-only) router + OpenTela peer.
srun \
  --nodes="${NUM_NODES}" \
  --ntasks="${NUM_NODES}" \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment=glm-47-flash-sglang \
  bash -lc "${LAUNCHER}"
SRUN_EXIT=$?
echo "srun_exited code=${SRUN_EXIT} $(date --iso-8601=seconds)"

exit "${SRUN_EXIT}"
