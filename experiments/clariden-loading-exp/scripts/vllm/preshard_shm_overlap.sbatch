#!/bin/bash
# PRESHARD+SHM+OVERLAP (vllm) - the best configuration found on bristen and
# reproduced on clariden with SGLang, now applied to vLLM unchanged:
# a TP=4-presharded checkpoint, staged into node-local /dev/shm by the sliced
# stager, with the stage OVERLAPPED with the engine's own startup so its cost is
# hidden rather than paid.
#
# Three ingredients, each doing a distinct job:
#   1. presharded  - the TP de-stride was done ONCE, offline; each rank now reads
#                    only its own ~35 GB in final layout. No strided narrow, no
#                    4x redundant read, no cross-rank collective.
#   2. /dev/shm    - the bytes come from tmpfs, not Lustre, at load time.
#   3. overlapped  - the engine spends ~60 s in process_startup + worker spawn +
#                    distributed init before it opens a single weight file (vLLM
#                    measured 14.80 + 45.42 s, wider than SGLang's ~35 s window).
#                    The stage is a plain host-side file copy with no GPU/torch
#                    dependency, so it runs concurrently inside that window.
#
# THE CHECKPOINT IS vLLM'S OWN, from scripts/vllm/save_sharded_ckpt.sbatch. The
# experiment first tried SGLang's -- `sharded_state` originated in vLLM and
# SGLang inherited it, so the format looked common -- and vLLM rejected it:
# `Missing keys ('model.layers.0.self_attn.attn._k_scale', ...)`, 320 attention
# scale params SGLang never writes (job 2918412). Presharded checkpoints are
# portable across architectures but NOT across engines.
#
# errors=0 with correct output below is still what proves the right bytes were
# loaded. A fast run that serves garbage is a failure of this config, not a
# result.
#
# SYNCHRONISATION: none, deliberately -- no engine patch, no barrier. The run is
# VERIFIED AFTER THE FACT: the stage's end timestamp must land before the
# absolute time weight_loading began (reconstructed from the servekit profile's
# started_at plus the durations of every phase ahead of it). If it does not, the
# run is INVALID and gets discarded, not interpreted: the loader would have
# opened a half-staged file. A production version needs a sentinel + atomic
# rename + loader-side wait. What is measured here is the ceiling.
#
#   cd experiments/clariden-loading-exp
#   ./submit.sh vllm <preset> preshard --exclude=<node the default run used>
#
# The --exclude is not optional: the OS page cache survives across container runs
# and --exclusive gives sole use of a node, NOT a different one. Landing on the
# default run's node would let this read the checkpoint from page cache and
# manufacture a fake win.
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=results/vllm/%x-%j.out

set -euo pipefail

ENGINE=vllm
source "${SLURM_SUBMIT_DIR}/scripts/shared/models.sh"

if [ ! -e "${SHARDED_SRC}/model-rank-0-part-0.safetensors" ]; then
  echo "ERROR: no sharded checkpoint at ${SHARDED_SRC}" >&2
  echo "       run  ./submit.sh vllm ${MODEL_PRESET:-llama70b} save  first." >&2
  echo "       (SGLang's shards do not load here -- see this file's header.)" >&2
  exit 1
fi

SLICES="${SLICES:-64}"

EDF="${SLURM_SUBMIT_DIR}/scripts/vllm/vllm-clariden.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../servekit"
STAGER="${SLURM_SUBMIT_DIR}/scripts/shared/stage_to_shm_sliced.sh"
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/vllm/${RESULTS_SUBDIR}"

NODE="$(hostname)"
TAG="preshard-shm-overlap-${SLURM_JOB_ID}-${NODE}"
PROFILE_OUT="${RESULTS_DIR}/${TAG}-profile.json"
STAGE_LOG="${RESULTS_DIR}/${TAG}-stage.txt"
TIMING="${RESULTS_DIR}/${TAG}-timing.txt"

mkdir -p "${RESULTS_DIR}"
echo "job=${SLURM_JOB_ID} node=${NODE} cpus=${SLURM_CPUS_PER_TASK}"
echo "preset=${MODEL_PRESET:-llama70b} sharded_src=${SHARDED_SRC}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail

# Outside the measured window. python3, not python: the vllm image has no
# \`python\` alias, unlike sglang's.
python3 -m pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

echo "nproc=\$(nproc)"
df -h /dev/shm
mkdir -p "${SHM_DEST}"

# --- small files: synchronous, before anything starts --------------------
# The engine reads config.json / tokenizer within the first seconds of startup.
find "${SHARDED_SRC}" -maxdepth 1 -type f ! -name '*.safetensors' \
  -exec cp -f {} "${SHM_DEST}/" \;
echo "sync-copied \$(find "${SHM_DEST}" -maxdepth 1 -type f ! -name '*.safetensors' | wc -l) non-safetensors files"

# --- t0: the stage and the engine start together -------------------------
STAGE_START=\$(date +%s.%N)
echo "stage_start_epoch=\${STAGE_START}" > "${TIMING}"

# BASH_ENV is unset for the stage subtree, and it is not a detail. The NGC image
# sets BASH_ENV=/etc/bash.bashrc, so EVERY non-interactive bash sources the
# NVIDIA banner, which shells out to nvidia-smi. The stager forks one `bash -c`
# per slice -- 1704 of them, concurrently -- giving 1704 concurrent nvidia-smi
# processes: load average 1658, not one dd actually running, the stage taking
# 350 s instead of 5 s AND starving vllm's own startup to 366 s against a normal
# 14.8 s (jobs 2918555, 2918623, 2918657). The bytes and the storage were never
# the problem. The stager is a verbatim copy of the bristen one and stays that
# way, so the fix goes here, at the call site.
echo "BASH_ENV=[\${BASH_ENV:-}] -- unset for the stager below"
(
  set +e
  FILE_PATTERN='*.safetensors' env -u BASH_ENV -u ENV \
    bash "${STAGER}" "${SHARDED_SRC}" "${SHM_DEST}" "${SLICES}" > "${STAGE_LOG}" 2>&1
  rc=\$?
  echo "stage_end_epoch=\$(date +%s.%N)" >> "${TIMING}"
  echo "stage_rc=\${rc}" >> "${TIMING}"
) &
STAGE_PID=\$!

# No wait, no sentinel: the engine goes now, on purpose. The stage's exit status
# is collected from STAGE_PID afterwards. Note the profile is ALREADY
# backgrounded here, alongside the stage -- hence two distinct PIDs, and \`wait\`
# is always given an explicit one so the two never get crossed.
#
# --disable-custom-all-reduce is required, not tuning: see baseline_mmap.sbatch.
servekit profile --out "${PROFILE_OUT}" --timeout "${READY_TIMEOUT}" \
  -- \
  vllm serve "${SHM_DEST}" \
    --host 0.0.0.0 \
    --port "${SERVER_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${MEM_FRACTION_STATIC}" \
    --max-num-seqs "${MAX_RUNNING_REQUESTS}" \
    --load-format sharded_state \
    --disable-custom-all-reduce &
PROF_PID=\$!

for i in \$(seq 1 ${BENCH_REPEATS}); do
  # Repeat 1 is the measurement and merges into the report. Later repeats are
  # diagnostic and go to their own files so they cannot overwrite it.
  if [ "\$i" = "1" ]; then dest=(--into "${PROFILE_OUT}"); \
  else dest=(--out "${PROFILE_OUT%-profile.json}-bench\${i}.json"); fi
  echo "--- bench repeat \$i/${BENCH_REPEATS} ---"
  servekit bench --url "http://127.0.0.1:${SERVER_PORT}" "\${dest[@]}" \
    --wait-ready "${READY_TIMEOUT}" \
    --requests 64 --concurrency 16 --input-len 512 --output-len 128 &
  BENCH_PID=\$!
  # Stop benching the moment the server dies -- a sharded_state mismatch is most
  # likely to show up as a hard load failure, and without this the bench would
  # poll a dead port for the rest of READY_TIMEOUT.
  while kill -0 \${PROF_PID} 2>/dev/null && kill -0 \${BENCH_PID} 2>/dev/null; do sleep 5; done
  kill -0 \${PROF_PID} 2>/dev/null || { echo "server exited; abandoning bench"; kill \${BENCH_PID} 2>/dev/null || true; }
  wait \${BENCH_PID} 2>/dev/null || true
  kill -0 \${PROF_PID} 2>/dev/null || break
done

# run_profile forwards SIGTERM to the server, so this tears down both.
kill \${PROF_PID} 2>/dev/null || true
wait \${PROF_PID} 2>/dev/null || true

wait \${STAGE_PID} || true
cat "${STAGE_LOG}"
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result (vllm preshard+shm+overlap) ==="
if [ -f "${PROFILE_OUT}" ] && [ -f "${TIMING}" ]; then
  python3 - "${PROFILE_OUT}" "${TIMING}" "${STAGE_LOG}" <<'PY'
import json, sys, re

prof, timing, stagelog = sys.argv[1:4]
d = json.load(open(prof))
ph = [(p['name'], p['duration_s']) for p in d['phases']]
by = {}
for name, dur in ph:                       # `unknown` appears more than once
    by[name] = by.get(name, 0.0) + dur
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
b = d.get('benchmark', {})
tp = b.get('throughput', {})
print('  ready_wait_s       = %s' % b.get('ready_wait_s'))
print('  throughput         = %s tok/s, %s/%s ok, errors=%s'
      % (tp.get('output_tok_per_s'), tp.get('completed'), tp.get('requests'), tp.get('errors')))
print()
print('  (vllm clariden reference: stage 16.89 s, weight_loading 7.50 s,')
print('   TOTAL 125.80 s vs a 322.01 s default => 2.56x)')
print('  (sglang clariden reference: TOTAL 127.06 s vs 586.33 s => 4.61x)')
print('  NOTE: errors>0 or a throughput drop = shards loaded wrong. This config is')
print('        valid only if it SERVES correctly, not just if it is faster.')
PY
else
  echo "  MISSING profile json or timing file - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
