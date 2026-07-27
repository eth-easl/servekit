#!/bin/bash
# Trace the real SGLang cold start with nsys, to see the weight-loading H2D
# copies on the CUDA timeline -- the thing py-spy structurally could not show
# (it collapsed every copy into one `param_data.copy_` frame).
#
#   sbatch shm-weight-loading-exp/scripts/nsys_weight_load.sbatch
#
# The two flags that matter: --sample=none --cpuctxsw=none. Without BOTH,
# nsys hangs forever on Slurm while writing the report and produces no file
# at all (NVIDIA-confirmed bug, 2022.3+; nsys here is 2026.2.1). See
#   https://forums.developer.nvidia.com/t/nsys-hanging-on-slurm-cluster/239975
# Cost of the workaround: no CPU sampling/scheduling data, so this trace
# answers the CUDA side (bounce vs DMA, transfer sizes) but NOT page faults.
#
# Report is written node-local (/tmp) and copied to results/ immediately --
# the container tears down once the process tree exits.
#
# SCOPE THE CAPTURE. Job 76093 traced the whole startup and nsys spent 36 min
# failing to finalize, producing nothing: cuda_graph_capture (32.8 s) +
# piecewise_cuda_graph_capture (80.9 s, "Registering 4830 cuda graph
# addresses") across 4 TP ranks is millions of kernel events. From that run's
# phase table, weight_loading runs 43.5 s -> 63.8 s after launch, so
# --duration=70 covers it with margin and stops before the capture bloat.
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --output=shm-weight-loading-exp/results/%x-%j.out

set -euo pipefail

SRC="${SRC:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SLICES="${SLICES:-64}"
SHM_DEST="${SHM_DEST:-/dev/shm/llama70b}"

SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
LUSTRE_EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
EXP_DIR="${SLURM_SUBMIT_DIR}/shm-weight-loading-exp"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
TAG="nsys-${SLURM_JOB_ID}-${NODE}"
PROFILE_OUT="${EXP_DIR}/results/${TAG}-profile.json"
STAGE_LOG="${EXP_DIR}/results/${TAG}-stage.txt"
REP_OUT="${EXP_DIR}/results/${TAG}.nsys-rep"
STATS_OUT="${EXP_DIR}/results/${TAG}-stats.txt"

echo "job=${SLURM_JOB_ID} node=${NODE} src=${SRC} slices=${SLICES}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
bash "${LUSTRE_EXP_DIR}/scripts/phase4_shm/stage_to_shm_sliced.sh" "${SRC}" "${SHM_DEST}" "${SLICES}" | tee "${STAGE_LOG}"
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

# nsys WRAPS the launch (no ptrace attach, unlike the py-spy harness) and
# follows the TP worker subprocesses automatically.
nsys profile -t cuda --sample=none --cpuctxsw=none --duration=70 \
  -o /tmp/${TAG} -f true \
  servekit profile --out "${PROFILE_OUT}" \
  -- \
  python -m sglang.launch_server \
    --model-path "${SHM_DEST}" \
    --host 0.0.0.0 \
    --port "${SGLANG_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --enable-metrics || echo "nsys/servekit returned nonzero: \$?"

# Copy the report off node-local /tmp BEFORE anything else can tear us down.
# Tolerant: if nsys exits nonzero we still want whatever it managed to write.
ls -la /tmp/${TAG}* || echo "(no /tmp artifacts)"
if [ -f "/tmp/${TAG}.nsys-rep" ]; then
  cp "/tmp/${TAG}.nsys-rep" "${REP_OUT}"
  echo "report copied to ${REP_OUT}"
else
  echo "NO .nsys-rep produced"
  exit 0
fi

# Summaries: transfer count/size distribution is the smoking gun for whether
# the strided RowParallel narrow arrives as many small memcpys or one big one.
nsys stats --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  /tmp/${TAG}.nsys-rep 2>&1 | tee "${STATS_OUT}"
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result ==="
[ -f "${REP_OUT}" ] && echo "  nsys report: ${REP_OUT}" || echo "  NO nsys report produced"
[ -f "${PROFILE_OUT}" ] && python3 -c "
import json
d = json.load(open('${PROFILE_OUT}'))
for p in d['phases']:
    print('  %-22s %8.2f s' % (p['name'], p['duration_s']))
print('  %-22s %8.2f s' % ('TOTAL', d['total_duration_s']))
" || echo "  no servekit profile JSON"
echo "finished=$(date --iso-8601=seconds)"
