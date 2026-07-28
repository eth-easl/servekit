#!/bin/bash
# DEFAULT - the untouched baseline: SGLang's default loader (mmap) reading the
# stock 30-shard checkpoint straight off capstor Lustre.
#
# Note there is NO loader flag here. mmap is the default; the point of this run
# is what a user gets with no intervention at all. This is the bristen
# phase1.3_e2e mmap run, ported to clariden -- only the SLURM header, the EDF and
# the output paths changed.
#
# bristen reference (n=4): weight_loading 634 s mean (430-939), TOTAL 812 s mean.
# That spread is why the time limit is 1 h rather than bristen's 30 min: the
# clariden capstor path is unmeasured and could be worse.
#
#   cd experiments/clariden-loading-exp && ./submit.sh <preset> default
#
# One fresh node per run: the OS page cache survives across container runs, so a
# reused node would serve the weights from RAM. Record the node id from the
# output and pass it to the preshard+shm+overlap --exclude.
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/models.sh"

EDF="${SLURM_SUBMIT_DIR}/sglang-clariden.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../servekit"
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/${RESULTS_SUBDIR}"

NODE="$(hostname)"
PROFILE_OUT="${RESULTS_DIR}/default-${SLURM_JOB_ID}-${NODE}-profile.json"

mkdir -p "${RESULTS_DIR}"
echo "job=${SLURM_JOB_ID} node=${NODE} cpus=${SLURM_CPUS_PER_TASK}"
echo "preset=${MODEL_PRESET:-llama70b} model=${MODEL}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail

# Outside the measured window.
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

echo "nproc=\$(nproc)"

# servekit's --bench flag was split into a second subcommand, so profile and
# bench now run side by side and join on the report file. The on-disk schema is
# unchanged, so these numbers stay comparable to bristen's. `bench --into` with
# --wait-ready blocks until profile writes the report, which happens the instant
# the server reports ready -- that sequencing is what keeps the benchmark from
# loading the server while the cold start is still being timed.
servekit profile --out "${PROFILE_OUT}" --timeout "${READY_TIMEOUT}" \
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
    --enable-metrics &
PROF_PID=\$!

for i in \$(seq 1 ${BENCH_REPEATS}); do
  # Repeat 1 is the measurement and merges into the report. Later repeats are
  # diagnostic and go to their own files so they cannot overwrite it.
  if [ "\$i" = "1" ]; then dest=(--into "${PROFILE_OUT}"); \
  else dest=(--out "${PROFILE_OUT%-profile.json}-bench\${i}.json"); fi
  echo "--- bench repeat \$i/${BENCH_REPEATS} ---"
  servekit bench --url "http://127.0.0.1:${SGLANG_PORT}" "\${dest[@]}" \
    --wait-ready "${READY_TIMEOUT}" \
    --requests 64 --concurrency 16 --input-len 512 --output-len 128 || true
done

# run_profile forwards SIGTERM to the server, so this tears down both.
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
print('  weight_loading     = %7.2f s' % by.get('weight_loading', float('nan')))
print('  TOTAL COLD START   = %7.2f s' % d['total_duration_s'])
b = d.get('benchmark', {}).get('throughput', {})
print('  throughput         = %s tok/s, %s/%s ok, errors=%s'
      % (b.get('output_tok_per_s'), b.get('completed'), b.get('requests'), b.get('errors')))
print()
print('  (llama70b bristen reference, n=4: weight_loading 634 s, TOTAL 812 s)')
PY
else
  echo "  MISSING profile json - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
echo
echo "NEXT: ./submit.sh ${MODEL_PRESET:-llama70b} preshard --exclude=${NODE}"
