#!/bin/bash
# Drop the model from a live SGLang server, put it back, and time it.
#
# Same setup as lustre-loading-exp phase1.3 fastsafetensors baseline (70B, TP=4,
# --load-format fastsafetensors, profiled with servekit) plus ONE new variable:
# --enable-memory-saver, and the release -> resume -> update_weights_from_disk
# cycle. No engine patch: upstream loader, i.e. SGLANG_FST_FILES_PER_RANK unset.
# The phase3 fpr=8 tuning is a separate variable and would confound this.
#
# --enable-memory-saver IS REQUIRED. Without it TorchMemorySaverAdapter.create()
# returns a *Noop*: release_memory_occupation returns 200, frees nothing, and
# only logs "will not save memory because torch_memory_saver is not enabled".
# A green run that proves nothing is the main trap here -> the grep gate below.
#
# The dd probe runs LAST, never before the load: reading 141 GB up front warms
# the Lustre OSS-side cache and would flatter the very numbers we measure.
#
# Two points, submitted one at a time on DISTINCT nodes (--exclusive grants sole
# use of a node, NOT a different node - a dependency chain hands the same one
# straight back, which cost lustre-loading-exp an entire sweep):
#
#   sbatch --job-name=mrr-ctl_nosaver --export=ALL,TAG=ctl_nosaver,MEMORY_SAVER=0,DO_CYCLE=0 \
#          experiments/model-release-reload-exp/scripts/release_reload.sbatch
#   sbatch --exclude=<node from above> \
#          --job-name=mrr-saver_cycle --export=ALL,TAG=saver_cycle,MEMORY_SAVER=1,DO_CYCLE=1 \
#          experiments/model-release-reload-exp/scripts/release_reload.sbatch
#
# ctl_nosaver is the control: it prices what --enable-memory-saver itself costs
# at cold start (its VMM allocator hook is a real confound). It must be measured
# here, not remembered: the same config measured 69.5-86.1 s across nodes.
#
#SBATCH --job-name=model-release-reload
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:50:00
#SBATCH --output=experiments/model-release-reload-exp/results/%x-%j.out

set -euo pipefail

TAG="${TAG:?set TAG, e.g. ctl_nosaver or saver_cycle}"
MEMORY_SAVER="${MEMORY_SAVER:-1}"   # 1 -> --enable-memory-saver
DO_CYCLE="${DO_CYCLE:-1}"           # 1 -> run the post-ready cycle
DO_RELEASE="${DO_RELEASE:-1}"       # 0 -> reload only, no release/resume (the control)
RELEASE_TAGS="${RELEASE_TAGS:-weights}"   # subset of weights,kv_cache,cuda_graph
DD_PROBE="${DD_PROBE:-1}"           # 0 -> skip (irrelevant when the question is correctness)

# Model is a knob so the same script runs the 70B and the Apertus-8B repro. TP stays
# 4 for both: the distributed path is a suspect, so it must not vary with the model.
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-meta-llama/Llama-3.1-70B-Instruct}"
EDF="${EDF:-${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml}"
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
EXP_DIR="${SLURM_SUBMIT_DIR}/experiments/model-release-reload-exp"
DD_SWEEP="${SLURM_SUBMIT_DIR}/lustre-loading-exp/scripts/lib/dd_read_sweep.sh"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
RES="${EXP_DIR}/results"
# RUN_PREFIX keeps result sets from different models apart in one results/ dir
# (mrr- = the 70B runs, mrr8b- = the Apertus matrix) and is what the summarizers glob.
RUN_PREFIX="${RUN_PREFIX:-mrr}"
PROFILE_OUT="${RES}/${RUN_PREFIX}-${TAG}-${SLURM_JOB_ID}-${NODE}-profile.json"
CYCLE_OUT="${RES}/${RUN_PREFIX}-${TAG}-${SLURM_JOB_ID}-${NODE}-cycle.json"
DD_CSV="${RES}/${RUN_PREFIX}_ddprobe-${TAG}-${SLURM_JOB_ID}-${NODE}.csv"

SAVER_FLAG=""
[ "${MEMORY_SAVER}" = "1" ] && SAVER_FLAG="--enable-memory-saver"
NO_RELEASE_FLAG=""
[ "${DO_RELEASE}" = "0" ] && NO_RELEASE_FLAG="--no-release"

echo "job=${SLURM_JOB_ID} node=${NODE} tag=${TAG} model=${MODEL}"
echo "memory_saver=${MEMORY_SAVER} do_cycle=${DO_CYCLE} do_release=${DO_RELEASE} release_tags=${RELEASE_TAGS} load_format=${LOAD_FORMAT}"

# Guard: `--export=ALL,RELEASE_TAGS=weights,kv_cache` silently truncates to `weights`
# (SLURM splits --export on commas). That turned a whole tag sweep into duplicates of
# one point once already. If the job name advertises more tags than we actually got,
# stop rather than produce a convincing duplicate.
case "${SLURM_JOB_NAME}" in
  *_kv*|*all*)
    case "${RELEASE_TAGS}" in
      *,*) ;;
      *) echo "FATAL: job '${SLURM_JOB_NAME}' expects a multi-tag RELEASE_TAGS but got '${RELEASE_TAGS}' -- --export comma truncation. Aborting." >&2
         exit 2 ;;
    esac ;;
esac
echo "profile_out=${PROFILE_OUT}"
echo "cycle_out=${CYCLE_OUT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
pip install fastsafetensors -q
# torch_memory_saver is a hard dep of the image (sglang pyproject) - nothing to install.

# --keep-alive: servekit writes the report the moment the post-ready bench ends,
# then keeps draining the server's stdout instead of terminating it. It must stay
# alive rather than detach: it owns the stdout pipe, and exiting would SIGPIPE the
# server on its next log line.
set +e
servekit profile --out "${PROFILE_OUT}" --keep-alive \
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
    --load-format "${LOAD_FORMAT}" \
    ${SAVER_FLAG} \
    --enable-metrics &
SERVEKIT_PID=\$!

CYCLE_RC=0
if [ "${DO_CYCLE}" = "1" ]; then
  python "${EXP_DIR}/scripts/cycle_driver.py" \
    --profile-json "${PROFILE_OUT}" \
    --out "${CYCLE_OUT}" \
    --base-url "http://127.0.0.1:${SGLANG_PORT}" \
    --model "${MODEL}" \
    --load-format "${LOAD_FORMAT}" \
    --tags "${RELEASE_TAGS}" ${NO_RELEASE_FLAG}
  CYCLE_RC=\$?
else
  # Control: just wait for the profile+bench to land, then tear down.
  for i in \$(seq 1 360); do [ -f "${PROFILE_OUT}" ] && break; sleep 5; done
  sleep 5
fi

# Teardown: kill the SERVER (servekit's child), not servekit, and never via
# 'pkill -f sglang.launch_server' - that pattern also matches servekit's own
# cmdline, since the wrapped command is in its argv. Killing the child makes
# servekit see EOF and exit on its own.
kill \$(pgrep -P \${SERVEKIT_PID}) 2>/dev/null || true
wait \${SERVEKIT_PID} 2>/dev/null || true
set -e
echo "cycle_rc=\${CYCLE_RC}"

if [ "${DD_PROBE}" = "1" ]; then
  echo "--- dd probe (N=30, O_DIRECT) ---"
  bash "${DD_SWEEP}" "${MODEL}" 16M "${DD_CSV}" "30"
fi

exit \${CYCLE_RC}
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile result (${TAG}) ==="
cat "${PROFILE_OUT}" 2>/dev/null || echo "NO PROFILE JSON - run did not reach ready"

if [ "${DO_CYCLE}" = "1" ]; then
  echo "=== GATE: was the memory saver actually enabled? ==="
  # The Noop adapter is silent apart from this warning. If it fired, every
  # 'release freed N GB' number below is fiction. Check this FIRST.
  if grep -q "will not save memory because torch_memory_saver is not enabled" \
       "${RES}/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>/dev/null; then
    echo "!!! MEMORY SAVER WAS A NOOP - cycle results are meaningless"
  else
    echo "ok: no torch_memory_saver noop warning"
  fi
  echo "=== cycle result (${TAG}) ==="
  cat "${CYCLE_OUT}" 2>/dev/null || echo "NO CYCLE JSON"
fi

if [ "${DD_PROBE}" = "1" ]; then
  echo "=== dd probe (${TAG}) ==="
  cat "${DD_CSV}" 2>/dev/null || echo "NO DD CSV"
fi
echo "finished=$(date --iso-8601=seconds)"
