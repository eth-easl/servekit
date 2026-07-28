#!/bin/bash
# DEFAULT (vllm) - the untouched baseline: vLLM's default loader reading the
# stock checkpoint straight off capstor Lustre.
#
# No loader flag. The point of this run is what a user gets with no intervention
# at all. It is the sglang baseline_mmap.sbatch with the engine swapped and
# nothing else: same model, same TP=4, same 32768 context, same 0.85 memory
# fraction, same bench, all read from scripts/shared/models.sh.
#
# Reference (profile/llama-3.1-70b-clariden-vllm, job 2918061, n=1):
# weight_loading 233.24 s, TOTAL 353.34 s, 820.8 tok/s. The SGLang default on
# the same hardware: weight_loading 510.27 s, TOTAL 624.58 s. Read neither gap
# as an engine result -- weight_loading on this storage ranged 430-939 s across
# the bristen reference runs, which is wider than the difference.
#
#   cd experiments/clariden-loading-exp && ./submit.sh vllm llama70b default
#
# One fresh node per run: the OS page cache survives across container runs, so a
# reused node would serve the weights from RAM. Record the node id from the
# output and pass it to the preshard --exclude.
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

EDF="${SLURM_SUBMIT_DIR}/scripts/vllm/vllm-clariden.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../servekit"
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/vllm/${RESULTS_SUBDIR}"

NODE="$(hostname)"
PROFILE_OUT="${RESULTS_DIR}/default-${SLURM_JOB_ID}-${NODE}-profile.json"

mkdir -p "${RESULTS_DIR}"
echo "job=${SLURM_JOB_ID} node=${NODE} cpus=${SLURM_CPUS_PER_TASK}"
echo "preset=${MODEL_PRESET:-llama70b} model=${MODEL}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail

# Outside the measured window. python3, not python: the vllm image has no
# \`python\` alias, unlike sglang's.
python3 -m pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

echo "nproc=\$(nproc)"

# --disable-custom-all-reduce is required, not tuning: vLLM's custom all-reduce
# dies with an illegal memory access on GH200 during CUDA graph capture
# (custom_all_reduce.cuh:455, job 2917848; vllm-project/vllm#30044). NCCL takes
# over the TP all-reduce instead.
#
# The bench is not optional here: vLLM reports ready WITHOUT issuing a warmup
# request, where SGLang's ready comes after its own. bench's ready_wait_s is the
# only record of that first-request cost, and totals are comparable across the
# two engines only once it is read alongside.
servekit profile --out "${PROFILE_OUT}" --timeout "${READY_TIMEOUT}" \
  -- \
  vllm serve "${MODEL}" \
    --host 0.0.0.0 \
    --port "${SERVER_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${MEM_FRACTION_STATIC}" \
    --max-num-seqs "${MAX_RUNNING_REQUESTS}" \
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
  # Stop benching the moment the server dies. Without this, a crashed engine
  # leaves the bench polling a dead port for the rest of READY_TIMEOUT and the
  # job looks like it is still running. vLLM crashed twice this way on GH200,
  # so it is a real failure mode here, not a hypothetical.
  while kill -0 \${PROF_PID} 2>/dev/null && kill -0 \${BENCH_PID} 2>/dev/null; do sleep 5; done
  kill -0 \${PROF_PID} 2>/dev/null || { echo "server exited; abandoning bench"; kill \${BENCH_PID} 2>/dev/null || true; }
  wait \${BENCH_PID} 2>/dev/null || true
  kill -0 \${PROF_PID} 2>/dev/null || break
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
echo "=== result (vllm default) ==="
if [ -f "${PROFILE_OUT}" ]; then
  python3 - "${PROFILE_OUT}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
by = {}
for p in d['phases']:                      # `unknown` appears more than once
    by[p['name']] = by.get(p['name'], 0.0) + p['duration_s']
print('  weight_loading     = %7.2f s' % by.get('weight_loading', float('nan')))
print('  TOTAL COLD START   = %7.2f s' % d['total_duration_s'])
b = d.get('benchmark', {})
t = b.get('throughput', {})
print('  ready_wait_s       = %s   (vLLM runs no warmup request; SGLang totals include one)'
      % b.get('ready_wait_s'))
print('  throughput         = %s tok/s, %s/%s ok, errors=%s'
      % (t.get('output_tok_per_s'), t.get('completed'), t.get('requests'), t.get('errors')))
print()
print('  (vllm clariden reference, n=1: weight_loading 233.24 s, TOTAL 353.34 s, 820.8 tok/s)')
PY
else
  echo "  MISSING profile json - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
echo
echo "NEXT: ./submit.sh vllm ${MODEL_PRESET:-llama70b} preshard --exclude=${NODE}"
