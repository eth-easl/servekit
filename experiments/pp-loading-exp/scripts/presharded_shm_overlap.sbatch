#!/bin/bash
# PRESHARDED + SHM + OVERLAP at PP=4 - the technique from clariden-loading-exp,
# carried over to pipeline parallelism with upstream's loader instead of a
# patched sharded_state.
#
# Same three ingredients as the TP round:
#   1. presharded  - each PP stage reads only its own layers, already in
#                    post-process layout. Built once, offline, by ./submit.sh
#                    <preset> dump.
#   2. /dev/shm    - the bytes come from tmpfs, not Lustre, at load time.
#   3. overlapped  - the engine spends tens of seconds in startup before it
#                    opens a weight file, and the stage is a plain host-side
#                    copy, so it runs concurrently inside that window.
#
# ORDER WITHIN THE STAGE IS LOAD-BEARING, and differs from the TP round.
# PreshardedModelLoader gates on a READY sentinel (_presharded_ready). So:
# checksum.json is copied synchronously FIRST, the .safetensor files stage in
# the background, and READY is created LAST.
#
# If the loader arrives before READY exists it does NOT read half-staged bytes
# -- it takes a cache miss. That is safer than sharded_state's silent
# corruption, but it is not harmless: a miss means a full HF load AND a re-dump,
# into /dev/shm, on top of the copy already staged there. For the 70B that is
# 141 GB of tmpfs writes with the staged copy still resident. Hence the gate
# below is mandatory, not decorative, and df /dev/shm is printed either side.
#
#   cd experiments/pp-loading-exp
#   ./submit.sh <preset> presharded --exclude=<node the default run used>
#
# The --exclude is not optional: the OS page cache survives container runs and
# --exclusive gives sole use of a node, NOT a different one.
#
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/scripts/models.sh"

SUBDIR="$(ls -d "${PRESHARDED_ROOT}"/*/ 2>/dev/null | head -1 || true)"
if [ -z "${SUBDIR}" ] || [ ! -f "${SUBDIR}/READY" ]; then
  echo "ERROR: no completed dump under ${PRESHARDED_ROOT}" >&2
  echo "       run  ./submit.sh ${MODEL_PRESET:-apertus8b} dump  first." >&2
  exit 1
fi
SUBDIR="$(basename "${SUBDIR}")"

SLICES="${SLICES:-64}"

EDF="${SLURM_SUBMIT_DIR}/scripts/sglang-nightly.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../servekit"
STAGER="${SLURM_SUBMIT_DIR}/scripts/stage_to_shm_sliced.sh"
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/${RESULTS_SUBDIR}"

SRC="${PRESHARDED_ROOT}/${SUBDIR}"
DEST="${SHM_PRESHARDED}/${SUBDIR}"

NODE="$(hostname)"
TAG="presharded-${PAR_TAG}-${SLURM_JOB_ID}-${NODE}"
PROFILE_OUT="${RESULTS_DIR}/${TAG}-profile.json"
PROBE_OUT="${RESULTS_DIR}/${TAG}-probe.jsonl"
STAGE_LOG="${RESULTS_DIR}/${TAG}-stage.txt"
TIMING="${RESULTS_DIR}/${TAG}-timing.txt"

mkdir -p "${RESULTS_DIR}"
echo "job=${SLURM_JOB_ID} node=${NODE} cpus=${SLURM_CPUS_PER_TASK}"
echo "preset=${MODEL_PRESET:-apertus8b} tp=${TP_SIZE} pp=${PP_SIZE}"
echo "dump=${SRC}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

echo "nproc=\$(nproc)"
df -h /dev/shm
mkdir -p "${DEST}"

# The plan must be readable before the loader looks; it is kilobytes.
cp -f "${SRC}/checksum.json" "${DEST}/"

# --- t0: the stage and the engine start together -------------------------
STAGE_START=\$(date +%s.%N)
echo "stage_start_epoch=\${STAGE_START}" > "${TIMING}"

(
  set +e
  # env -u BASH_ENV: in images that set it, each of the stager's ~1700 bash -c
  # slice workers sources it at startup. That cost the vLLM arm of the last
  # round a 348 s stage at 0.41 GB/s.
  FILE_PATTERN='*.safetensor' \
    env -u BASH_ENV bash "${STAGER}" "${SRC}" "${DEST}" "${SLICES}" > "${STAGE_LOG}" 2>&1
  rc=\$?
  if [ \${rc} -eq 0 ]; then
    # LAST, and only on success: this is what un-gates the loader.
    touch "${DEST}/READY"
    echo "ready_epoch=\$(date +%s.%N)" >> "${TIMING}"
  fi
  echo "stage_end_epoch=\$(date +%s.%N)" >> "${TIMING}"
  echo "stage_rc=\${rc}" >> "${TIMING}"
) &
STAGE_PID=\$!

servekit profile --out "${PROFILE_OUT}" --timeout "${READY_TIMEOUT}" \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${SERVER_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --pipeline-parallel-size "${PP_SIZE}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --trust-remote-code \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --load-format presharded \
    --model-loader-extra-config '{"presharded_path": "${SHM_PRESHARDED}"}' \
    --enable-metrics &
PROF_PID=\$!

# Do not sit in \`bench --wait-ready\` for the rest of the job if the server died.
while kill -0 \${PROF_PID} 2>/dev/null && [ ! -f "${PROFILE_OUT}" ]; do sleep 2; done
if ! python -c "import json,sys; sys.exit(0 if json.load(open('${PROFILE_OUT}'))['success'] else 1)" 2>/dev/null; then
  echo "SERVER NEVER REACHED READY -- skipping bench and probe"
  wait \${STAGE_PID} || true
  cat "${STAGE_LOG}"
  exit 1
fi

for i in \$(seq 1 ${BENCH_REPEATS}); do
  if [ "\$i" = "1" ]; then dest=(--into "${PROFILE_OUT}"); \
  else dest=(--out "${PROFILE_OUT%-profile.json}-bench\${i}.json"); fi
  echo "--- bench repeat \$i/${BENCH_REPEATS} ---"
  servekit bench --url "http://127.0.0.1:${SERVER_PORT}" "\${dest[@]}" \
    --wait-ready "${READY_TIMEOUT}" \
    --requests 64 --concurrency 16 --input-len 512 --output-len 128 || true
done

python "${SLURM_SUBMIT_DIR}/scripts/probe.py" \
  "http://127.0.0.1:${SERVER_PORT}" "${SERVED_MODEL_NAME}" "${PROBE_OUT}" || true

kill \${PROF_PID} 2>/dev/null || true
wait \${PROF_PID} 2>/dev/null || true

wait \${STAGE_PID} || true
cat "${STAGE_LOG}"
df -h /dev/shm
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result (presharded+shm+overlap) ==="
if [ -f "${PROFILE_OUT}" ] && [ -f "${TIMING}" ]; then
  python3 - "${PROFILE_OUT}" "${TIMING}" "${STAGE_LOG}" <<'PY'
import json, re, sys

prof, timing, stagelog = sys.argv[1:4]
d = json.load(open(prof))
phases = [(p['name'], p['duration_s']) for p in d['phases']]
by = dict(phases)
t = dict(re.findall(r'(\w+)=([\d.\-]+)', open(timing).read()))

stage_start = float(t['stage_start_epoch'])
stage_end = float(t.get('stage_end_epoch', 'nan'))
ready_epoch = float(t.get('ready_epoch', 'nan'))
stage_rc = t.get('stage_rc', '?')

m = re.search(r'stage_GBps=([\d.]+)', open(stagelog).read())
gbps = m.group(1) if m else '?'

# Absolute time the loader started = servekit's launch time plus every phase
# ahead of weight_loading. This is the deadline READY had to beat.
elapsed = 0.0
for name, dur in phases:
    if name == 'weight_loading':
        break
    elapsed += dur
wl_start = d['started_at'] + elapsed
slack = wl_start - ready_epoch
total = d['ready_at'] - stage_start
wl = by.get('weight_loading', float('nan'))

print('  stage (overlapped) = %7.2f s   (%s GB/s, rc=%s)' % (stage_end - stage_start, gbps, stage_rc))
print('  weight_loading     = %7.2f s' % wl)
print('  servekit total     = %7.2f s' % d['total_duration_s'])
print('  non-load           = %7.2f s' % (d['total_duration_s'] - wl))
print('  TOTAL COLD START   = %7.2f s   (ready_at - stage_start)' % total)
print()
print('  --- overlap validity gate ---')
print('  READY written %.2f s into the run; weight_loading began at %.2f s'
      % (ready_epoch - stage_start, wl_start - stage_start))
if stage_rc != '0':
    print('  INVALID: stage exited rc=%s' % stage_rc)
elif slack > 0:
    print('  VALID: READY beat the loader by %.2f s of slack' % slack)
else:
    print('  INVALID: the loader looked %.2f s BEFORE READY existed.' % -slack)
    print('           It took a cache MISS: full HF load plus a re-dump into')
    print('           /dev/shm. Discard this run; the number is not the technique.')
print()
b = (d.get('benchmark') or {}).get('throughput') or {}
print('  throughput = %s tok/s, %s/%s ok, errors=%s'
      % (b.get('output_tok_per_s'), b.get('completed'), b.get('requests'), b.get('errors')))
print()
print('  Correctness is NOT settled by these numbers. Diff the probe file against')
print('  the default run: identical bytes, or the stages loaded wrong.')
PY
else
  echo "  MISSING profile json or timing file - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
