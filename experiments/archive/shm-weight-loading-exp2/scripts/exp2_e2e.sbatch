#!/bin/bash
# exp2 end-to-end: does NUMA locality close the gap to <5 s?
#
# Job 76361 (microbench, no engine) measured 4 concurrent ranks reading tmpfs:
#   remote pages, unbound ranks : 2.9-4.9 GB/s each  -> ~10-12 s for 35.3 GB
#   local pages, bound ranks    : 9.9-10.0 GB/s each -> ~3.6 s
# and every historical run logged, on all 4 ranks:
#   "NUMA affinity is already constrained for process, skipping NUMA node
#    configuration for GPU."
# i.e. SGLang's OWN automatic NUMA binding has been silently disabled the whole
# time, because --cpus-per-task=64 on a 128-logical-CPU node constrains the
# affinity mask and _is_numa_available() bails out when it sees that.
#
# Arms (each on a fresh node -- page cache survives container runs):
#   ctl       64 CPUs, plain stager. Reproduces today's configuration and
#             produces the golden answer file. Expect ~11.1 s.
#   numaproc  128 CPUs, plain stager. Removes the affinity constraint so
#             SGLang binds each rank to its GPU's node. Rank-local CPUs and
#             pinned buffers, but tmpfs pages still wherever the stager put
#             them -- isolates the process half of the effect.
#   numaboth  128 CPUs + NUMA-aware stager, so rank R's shard is allocated on
#             the node local to GPU R as well. Both halves.
#
#   for A in ctl numaproc numaboth; do
#     sbatch --job-name=e2-$A --export=ALL,ARM=$A \
#       shm-weight-loading-exp2/scripts/exp2_e2e.sbatch
#   done
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:50:00
#SBATCH --output=shm-weight-loading-exp2/results/%x-%j.out

set -uo pipefail

ARM="${ARM:?set ARM: ctl | numaproc | numaboth}"

SHARDED_SRC="${SHARDED_SRC:-/iopsstor/scratch/cscs/yboughizane/sharded-ckpt/llama70b-tp4}"
if [ ! -e "${SHARDED_SRC}/model-rank-0-part-0.safetensors" ]; then
  echo "ERROR: no sharded checkpoint at ${SHARDED_SRC}" >&2
  exit 1
fi

SLICES="${SLICES:-64}"
SHM_DEST="${SHM_DEST:-/dev/shm/llama70b-sharded}"

SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
LUSTRE_EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
EXP_DIR="${SLURM_SUBMIT_DIR}/shm-weight-loading-exp2"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
PROFILE_OUT="${EXP_DIR}/results/e2-${ARM}-${SLURM_JOB_ID}-${NODE}-profile.json"
STAGE_LOG="${EXP_DIR}/results/e2-${ARM}-${SLURM_JOB_ID}-${NODE}-stage.txt"

# The affinity mask the container sees is what decides whether SGLang's NUMA
# binding engages, and srun's --cpus-per-task is what sets it.
case "$ARM" in
  ctl)      SRUN_CPUS=64;  STAGER="${LUSTRE_EXP_DIR}/scripts/phase4_shm/stage_to_shm_sliced.sh"; EXP2=0 ;;
  numaproc) SRUN_CPUS=128; STAGER="${LUSTRE_EXP_DIR}/scripts/phase4_shm/stage_to_shm_sliced.sh"; EXP2=0 ;;
  numaboth) SRUN_CPUS=128; STAGER="${EXP_DIR}/scripts/stage_to_shm_numa.sh";               EXP2=0 ;;
  # numaboth + the exp2 loader: zero-copy views, pinned double-buffered
  # staging, multi-threaded gather. Bench says 4.46 s/rank (job 76372).
  zerocopy) SRUN_CPUS=128; STAGER="${EXP_DIR}/scripts/stage_to_shm_numa.sh";               EXP2=1; PINNED=0 ;;
  exp2)     SRUN_CPUS=128; STAGER="${EXP_DIR}/scripts/stage_to_shm_numa.sh";               EXP2=1; PINNED=1 ;;
  *) echo "bad ARM: $ARM" >&2; exit 1 ;;
esac
PINNED="${PINNED:-0}"

echo "job=${SLURM_JOB_ID} node=${NODE} arm=${ARM} srun_cpus=${SRUN_CPUS}"
echo "stager=$(basename "${STAGER}")"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -uo pipefail

echo "=== affinity as seen by the container ==="
python3 -c "import os; a=sorted(os.sched_getaffinity(0)); print('cpus_available=%d of %d' % (len(a), os.cpu_count()))"
python3 "${EXP_DIR}/scripts/numa_map.py" | sed 's/^/gpu_numa /'
echo

bash "${STAGER}" "${SHARDED_SRC}" "${SHM_DEST}" "${SLICES}" 2>&1 | tee "${STAGE_LOG}"

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

bash "${EXP_DIR}/scripts/patch_sglang_in_container.sh" \
  "${EXP_DIR}/scripts/exp2_verify.patch" "${EXP_DIR}/scripts/exp2_loader.patch"
export PYTHONPATH="${EXP_DIR}/scripts:\${PYTHONPATH:-}"
export SGLANG_EXP2_VERIFY=1
export SGLANG_EXP2_LOADER=${EXP2}
export SGLANG_EXP2_PINNED=${PINNED}
export SGLANG_EXP2_GATHER_THREADS=\${SGLANG_EXP2_GATHER_THREADS:-4}
echo "exp2 loader=\${SGLANG_EXP2_LOADER} pinned=\${SGLANG_EXP2_PINNED} gather_threads=\${SGLANG_EXP2_GATHER_THREADS}"

exec servekit profile --out "${PROFILE_OUT}" \
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
    --enable-metrics
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SRUN_CPUS}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"

LOG="${EXP_DIR}/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo
echo "=== result (${ARM}) ==="
STAGE_S=$(grep -o 'stage_wall_s=[0-9.]*' "${STAGE_LOG}" 2>/dev/null | cut -d= -f2 || true)
echo "  stage              = ${STAGE_S:-n/a} s"
echo "  NUMA binding engaged on all 4 ranks: $(grep -c 'NUMA affinity is already constrained' "${LOG}" 2>/dev/null | awk '{print ($1==0)?"yes":"NO ("$1" ranks skipped)"}')"
echo "  bit-exact gate     : $(grep -c 'EXP2_VERIFY PASS' "${LOG}" 2>/dev/null || echo 0)/4 ranks PASS, $(grep -c 'EXP2_VERIFY FAIL' "${LOG}" 2>/dev/null || echo 0) FAIL"
python3 "${EXP_DIR}/scripts/gated_weight_loading.py" "${LOG}" 2>/dev/null || true
if [ -f "${PROFILE_OUT}" ]; then
  python3 -c "
import json
d = json.load(open('${PROFILE_OUT}'))
b = d.get('benchmark', {}).get('throughput', {})
print('  servekit total     = %7.2f s' % d['total_duration_s'])
print('  throughput         = %s tok/s, %s/%s ok, errors=%s'
      % (b.get('output_tok_per_s'), b.get('completed'), b.get('requests'), b.get('errors')))
"
else
  echo "  NO PROFILE JSON - run did not reach ready"
fi
echo "  target: < 5.00 s gated. baseline 11.08 s (job 76115, corrected)."
echo "finished=$(date --iso-8601=seconds)"
