#!/bin/bash
# `servekit launch` end-to-end: the whole of Phase 1 in one command.
#
# The comparison is against experiments/clariden-loading-exp, whose numbers this
# reuses rather than re-measuring -- the model paths, TP size, context length and
# mem-fraction are SOURCED from that experiment's models.sh so the two cannot
# drift apart. Same image, same flags, same node type; the only difference is
# that `servekit launch --` is prepended and the model path is the one the user
# would have typed anyway.
#
#   clariden-loading-exp reference, SGLang / Llama-3.1-70B, TP=4:
#     default (mmap, capstor)     586.33 s total, 466.81 s weight_loading
#     preshard+shm+OVERLAPPED     127.06 s total,   6.19 s weight_loading  4.61x
#   Phase 1 is that run WITHOUT the overlap, so it should land at
#   ~127 + ~8.8 = ~136 s => ~4.3x, with the stage now a visible phase.
#
# LOADER selects what is being staged, because the two answer different
# questions:
#   sharded (default) - the presharded checkpoint + --load-format sharded_state.
#                       This is the configuration Phase 1 is claiming a number
#                       for.
#   mmap              - the stock checkpoint, no loader flag. Not a speed claim:
#                       it is the free-on-ready check. The mmap loader may still
#                       hold mappings at ready time, in which case the unlink
#                       succeeds and the RAM does NOT come back until the engine
#                       drops them. The memory samples below are what settles it.
#
#   cd experiments/servekit-fast-weight-load/phase-2-overlap
#   ./submit.sh <apertus8b|llama70b> <sharded|mmap> [sbatch args]
#
# One fresh node per run: the OS page cache survives across container runs, so a
# reused node serves the weights from RAM and manufactures a win.
#
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR}/../.."
source "${ROOT}/clariden-loading-exp/scripts/shared/models.sh"

ARM="${ARM:-seq}"
case "${ARM}" in
  seq) OVERLAP_FLAG="" ;;
  ovl) OVERLAP_FLAG="--overlap" ;;
  *) echo "ARM must be seq or ovl, got: ${ARM}" >&2; exit 1 ;;
esac

LOADER="${LOADER:-sharded}"
case "${LOADER}" in
  sharded) SRC="${SHARDED_SRC}"; LOAD_FORMAT="--load-format sharded_state" ;;
  mmap)    SRC="${MODEL}";       LOAD_FORMAT="" ;;
  *) echo "LOADER must be sharded or mmap, got: ${LOADER}" >&2; exit 1 ;;
esac

if [ ! -d "${SRC}" ]; then
  echo "ERROR: no checkpoint at ${SRC}" >&2
  exit 1
fi

EDF="${SLURM_SUBMIT_DIR}/scripts/sglang-clariden.toml"
SERVEKIT_DIR="${ROOT}/../servekit"
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/${RESULTS_SUBDIR}"

NODE="$(hostname)"
TAG="${ARM}-${LOADER}-${SLURM_JOB_ID}-${NODE}"
LAUNCH_OUT="${RESULTS_DIR}/${TAG}-launch.json"
MEM_LOG="${RESULTS_DIR}/${TAG}-mem.txt"

mkdir -p "${RESULTS_DIR}"
echo "job=${SLURM_JOB_ID} node=${NODE} cpus=${SLURM_CPUS_PER_TASK}"
echo "preset=${MODEL_PRESET:-llama70b} arm=${ARM} loader=${LOADER} src=${SRC}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail

# Outside the measured window.
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

echo "nproc=\$(nproc)"
df -h /dev/shm

# Does the unlink actually give the RAM back? Sampled for the whole run so the
# before/at-ready/after picture is read off one series rather than three probes.
( while true; do
    echo "\$(date +%s.%N) \$(awk '/MemAvailable/{print \$2}' /proc/meminfo) \$(df --output=used -k /dev/shm | tail -1)"
    sleep 1
  done ) > "${MEM_LOG}" &
MEM_PID=\$!

servekit launch --out "${LAUNCH_OUT}" --timeout "${READY_TIMEOUT}" ${OVERLAP_FLAG} \
  -- \
  python -m sglang.launch_server \
    --model-path "${SRC}" \
    --host 0.0.0.0 \
    --port "${SERVER_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    ${LOAD_FORMAT} \
    --enable-metrics &
LAUNCH_PID=\$!

# Blocks until launch writes the report, i.e. until the server reports ready --
# so the bench cannot load the server while the cold start is still being timed.
servekit bench --url "http://127.0.0.1:${SERVER_PORT}" --into "${LAUNCH_OUT}" \
  --wait-ready "${READY_TIMEOUT}" \
  --requests 64 --concurrency 16 --input-len 512 --output-len 128 || true

# A minute past ready, then bench again. Throughput must match the first repeat:
# the check that freeing the tmpfs copy did not leave the engine re-reading
# anything.
sleep 60
servekit bench --url "http://127.0.0.1:${SERVER_PORT}" --out "${LAUNCH_OUT%-launch.json}-bench2.json" \
  --requests 64 --concurrency 16 --input-len 512 --output-len 128 || true

echo "--- staged copies still present ---"
ls -la /dev/shm/servekit 2>/dev/null || echo "none"
df -h /dev/shm

kill \${MEM_PID} 2>/dev/null || true
# run_profile forwards SIGTERM to the server, so this tears down both.
kill \${LAUNCH_PID} 2>/dev/null || true
wait \${LAUNCH_PID} 2>/dev/null || true
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result (servekit launch, arm=${ARM}, ${LOADER}) ==="
if [ -f "${LAUNCH_OUT}" ]; then
  python3 "${SLURM_SUBMIT_DIR}/scripts/report.py" \
    "${LAUNCH_OUT}" "${MEM_LOG}" "${LAUNCH_OUT%-launch.json}-bench2.json" "${MODEL_PRESET:-llama70b}"
else
  echo "  MISSING launch json - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
