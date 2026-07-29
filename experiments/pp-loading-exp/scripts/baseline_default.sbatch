#!/bin/bash
# BASELINE - PP=4, stock loader, straight off Lustre. What the presharded arm
# has to beat.
#
# The baseline is deliberately PP=4 on THIS image, not the TP=4 v0.5.10 numbers
# from clariden-loading-exp. Two things changed at once versus that round (the
# parallelism and the engine version), so the only honest comparison is against
# a run that differs from the presharded arm in the loader alone.
#
# Note the cold-start SHAPE differs from TP=4 by construction: pp_size>1 forces
# disable_overlap_schedule (server_args.py:2853) and disables piecewise CUDA
# graph (:1098), so piecewise_cuda_graph_capture will be absent. That is a
# property of PP, not of the loader.
#
#   cd experiments/pp-loading-exp && ./submit.sh <preset> default
#
# Record the node id it lands on: the presharded run must --exclude it, or it
# reads the same bytes out of page cache and manufactures a fake win.
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

EDF="${SLURM_SUBMIT_DIR}/scripts/sglang-nightly.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../servekit"
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/${RESULTS_SUBDIR}"

NODE="$(hostname)"
TAG="default-${PAR_TAG}-${SLURM_JOB_ID}-${NODE}"
PROFILE_OUT="${RESULTS_DIR}/${TAG}-profile.json"
PROBE_OUT="${RESULTS_DIR}/${TAG}-probe.jsonl"

mkdir -p "${RESULTS_DIR}"
echo "job=${SLURM_JOB_ID} node=${NODE} cpus=${SLURM_CPUS_PER_TASK}"
echo "preset=${MODEL_PRESET:-apertus8b} tp=${TP_SIZE} pp=${PP_SIZE} model=${MODEL}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

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
    --enable-metrics &
PROF_PID=\$!

# Do not sit in \`bench --wait-ready\` for the rest of the job if the server died.
while kill -0 \${PROF_PID} 2>/dev/null && [ ! -f "${PROFILE_OUT}" ]; do sleep 2; done
if ! python -c "import json,sys; sys.exit(0 if json.load(open('${PROFILE_OUT}'))['success'] else 1)" 2>/dev/null; then
  echo "SERVER NEVER REACHED READY -- skipping bench and probe"
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
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result (default) ==="
if [ -f "${PROFILE_OUT}" ]; then
  python3 - "${PROFILE_OUT}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
by = {p['name']: p['duration_s'] for p in d['phases']}
wl = by.get('weight_loading', float('nan'))
total = d['total_duration_s']
for name, dur in ((p['name'], p['duration_s']) for p in d['phases']):
    print('  %-32s %7.2f s' % (name, dur))
print('  %-32s %7.2f s' % ('TOTAL', total))
print('  %-32s %7.2f s' % ('non-load (total - weight_load)', total - wl))
b = (d.get('benchmark') or {}).get('throughput') or {}
print()
print('  throughput = %s tok/s, %s/%s ok, errors=%s'
      % (b.get('output_tok_per_s'), b.get('completed'), b.get('requests'), b.get('errors')))
PY
  echo
  echo "  NEXT: ./submit.sh ${MODEL_PRESET:-apertus8b} presharded --exclude=${NODE}"
else
  echo "  MISSING profile json - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
