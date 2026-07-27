#!/bin/bash
# Phase 7 - OVERLAP the /dev/shm stage with SGLang's startup, to hide the stage
# cost. Continues Phase 6 (job 76435), which established the two numbers this
# phase tries to fold into one:
#
#   phase 6 (serial): stage 11.89 s -> weight_loading 9.64 s -> total 180.70 s
#                     => TOTAL COLD START 192.59 s (stage + servekit total)
#
# Phase 6 ran the stage to completion BEFORE launching launch_server, so the two
# costs add. But nothing ties the shm stage to GPU/torch state -- it is a plain
# host-side file copy -- and SGLang spends ~42 s in process_startup (23.32) +
# tp_worker_spawn (15.87) + torch_distributed_init (3.09) before it opens a
# single weight file. So the stage can run CONCURRENTLY with that window and be
# finished by the time the loader wants the bytes. Ideal case: the 11.89 s
# disappears entirely and the win converges on the full weight_loading delta
# vs the phase1.3 mmap baseline (20.51 -> 9.64 s).
#
# SYNCHRONISATION: none, deliberately. This is the optimistic arm -- no engine
# patch, no barrier. Instead the run is VERIFIED AFTER THE FACT: the stage's end
# timestamp must land before the absolute time weight_loading began (derived
# from the servekit profile's started_at + the durations of the phases ahead of
# it). If it does not, the run is INVALID and is discarded, not interpreted --
# it means SGLang opened a half-staged file, and the throughput/error numbers
# below would be the only other clue. A production version of this needs the
# sentinel + atomic-rename + loader-wait design sketched in PLAN.md; that is not
# what is being measured here. What is being measured here is the ceiling.
#
# The small files (config.json, tokenizer*, generation_config.json) are copied
# SYNCHRONOUSLY first: they are tiny, and SGLang reads them within the first
# seconds of process_startup, far inside the overlap window. Only the 28
# *.safetensors shards (131 GB) go into the background stage.
#
# CPUs stay at 64, matching phase 6, so the overlap is the only variable. Note
# this means the stage now contends for CPU with process_startup and
# tp_worker_spawn -- Phase 4b showed the sliced stage is CPU-bound, so expect
# the stage itself to take LONGER than 11.89 s here. That interference is part
# of the result, not noise.
#
# Requires save_sharded_ckpt.sbatch (../phase6_preshard_shm/) to have run first.
#
#   sbatch lustre-loading-exp/scripts/phase7_overlap_stage/phase7_overlap_stage.sbatch
#
# One fresh node per run (page cache survives across container runs).
#
#SBATCH --job-name=phase7-overlap-stage
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=lustre-loading-exp/results/phase7_overlap_stage/%x-%j.out

set -euo pipefail

SHARDED_SRC="${SHARDED_SRC:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded}"

if [ ! -e "${SHARDED_SRC}/model-rank-0-part-0.safetensors" ]; then
  echo "ERROR: no sharded checkpoint at ${SHARDED_SRC}" >&2
  echo "       run ../phase6_preshard_shm/save_sharded_ckpt.sbatch first." >&2
  exit 1
fi

SLICES="${SLICES:-64}"
SHM_DEST="${SHM_DEST:-/dev/shm/llama70b}"

SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
LUSTRE_EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
RESULTS_DIR="${LUSTRE_EXP_DIR}/results/phase7_overlap_stage"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
PROFILE_OUT="${RESULTS_DIR}/phase7-overlap-${SLURM_JOB_ID}-${NODE}-profile.json"
STAGE_LOG="${RESULTS_DIR}/phase7-overlap-${SLURM_JOB_ID}-${NODE}-stage.txt"
TIMING="${RESULTS_DIR}/phase7-overlap-${SLURM_JOB_ID}-${NODE}-timing.txt"

mkdir -p "${RESULTS_DIR}"
echo "job=${SLURM_JOB_ID} node=${NODE}"
echo "sharded_src=${SHARDED_SRC}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail

# Outside the measured window, exactly as in phase 6.
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

mkdir -p "${SHM_DEST}"

# --- small files: synchronous, before anything starts --------------------
# SGLang reads config.json / tokenizer within the first seconds of startup.
find "${SHARDED_SRC}" -maxdepth 1 -type f ! -name '*.safetensors' \
  -exec cp -f {} "${SHM_DEST}/" \;
echo "sync-copied \$(find "${SHM_DEST}" -maxdepth 1 -type f ! -name '*.safetensors' | wc -l) non-safetensors files"

# --- t0: the stage and the engine start together -------------------------
STAGE_START=\$(date +%s.%N)
echo "stage_start_epoch=\${STAGE_START}" > "${TIMING}"

(
  set +e
  FILE_PATTERN='*.safetensors' \
    bash "${LUSTRE_EXP_DIR}/scripts/phase4_shm/stage_to_shm_sliced.sh" \
      "${SHARDED_SRC}" "${SHM_DEST}" "${SLICES}" > "${STAGE_LOG}" 2>&1
  rc=\$?
  echo "stage_end_epoch=\$(date +%s.%N)" >> "${TIMING}"
  echo "stage_rc=\${rc}" >> "${TIMING}"
) &
STAGE_PID=\$!

# No wait, no sentinel: the engine goes now, on purpose. Not exec'd, so the
# stage's exit status can still be collected afterwards.
servekit profile --out "${PROFILE_OUT}" \
  --bench --bench-requests 64 --bench-concurrency 16 \
  --bench-input-len 512 --bench-output-len 128 \
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
    --load-format sharded_state \
    --enable-metrics || true

wait \${STAGE_PID} || true
cat "${STAGE_LOG}"
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result (phase7_overlap_stage) ==="
if [ -f "${PROFILE_OUT}" ] && [ -f "${TIMING}" ]; then
  python3 - "${PROFILE_OUT}" "${TIMING}" "${STAGE_LOG}" <<'PY'
import json, sys, re

prof, timing, stagelog = sys.argv[1:4]
d = json.load(open(prof))
ph = [(p['name'], p['duration_s']) for p in d['phases']]
by = dict(ph)
t = dict(re.findall(r'(\w+)=([\d.\-]+)', open(timing).read()))
stage_start = float(t['stage_start_epoch'])
stage_end = float(t.get('stage_end_epoch', 'nan'))
stage_rc = t.get('stage_rc', '?')
stage_wall = stage_end - stage_start

m = re.search(r'stage_GBps=([\d.]+)', open(stagelog).read())
gbps = m.group(1) if m else '?'

# Absolute time the loader started, = servekit's launch time plus every phase
# ahead of weight_loading. This is the deadline the stage had to beat.
elapsed = 0.0
for name, dur in ph:
    if name == 'weight_loading':
        break
    elapsed += dur
wl_start = d['started_at'] + elapsed

slack = wl_start - stage_end
total = d['ready_at'] - stage_start

print('  stage (overlapped) = %7.2f s   (%s GB/s, rc=%s)' % (stage_wall, gbps, stage_rc))
print('  weight_loading     = %7.2f s' % by.get('weight_loading', float('nan')))
print('  servekit total     = %7.2f s' % d['total_duration_s'])
print('  TOTAL COLD START   = %7.2f s   (ready_at - stage_start)' % total)
print()
print('  --- overlap validity gate ---')
print('  stage finished %.2f s into the run; weight_loading began at %.2f s' %
      (stage_end - stage_start, wl_start - stage_start))
if stage_rc != '0':
    print('  INVALID: stage exited rc=%s' % stage_rc)
elif slack > 0:
    print('  VALID: stage beat the loader by %.2f s of slack' % slack)
else:
    print('  INVALID: loader opened weight files %.2f s BEFORE the stage finished' % -slack)
    print('           -> discard this run; it read partially-staged bytes.')
print()
b = d.get('benchmark', {}).get('throughput', {})
print('  throughput         = %s tok/s, %s/%s ok, errors=%s'
      % (b.get('output_tok_per_s'), b.get('completed'), b.get('requests'), b.get('errors')))
print()
print('  reference (phase 6, serial, job 76435, 64 CPUs):')
print('    stage 11.89 + weight_loading 9.64, servekit total 180.70')
print('    => TOTAL COLD START 192.59 s. The win here is that number minus this one.')
print('  NOTE: errors>0 or a throughput drop = shards loaded wrong. Valid only')
print('        if it SERVES correctly, not just if it is faster.')
PY
else
  echo "  MISSING profile json or timing file - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
