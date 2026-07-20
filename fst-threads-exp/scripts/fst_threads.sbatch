#!/bin/bash
# End-to-end cold start at ONE point of the (files_per_rank x max_threads) grid.
#
# The question: weight_loading is NOT storage-bound (capstor gives the whole
# 141 GB in ~7 s at 18.9 GB/s when asked with enough requests in flight; the
# loader takes 69-93 s at 1.5-1.9 GB/s). Phase 3 tuned files-in-flight and got
# 2.3x. This tunes the OTHER axis -- threads WITHIN each file, fastsafetensors'
# max_threads, default 16, never swept -- and asks whether they compose.
#
# MT unset + FPR=1 is a TRUE NULL RUN: the patched code provably reduces to
# upstream (empty fst_kwargs -> SafeTensorsFileLoader(pg, device); chunk =
# pg.size(); round-robin rank_file_map == enumerate()). It must reproduce the
# phase-1.3/phase-3 fastsafetensors baseline (weight_loading 69-93 s). If it
# does not, the harness is changing behavior and no other number here is worth
# reading.
#
# The dd probe is the SLICED one (T=8), not dd_read_sweep.sh. The old probe
# fans out across files only -> queue depth 1 per OST -> it reports the WORST
# OST's latency as if it were capstor's bandwidth (it read 0.72 GB/s while the
# node was doing 18.9). Normalizing these runs against that would be worse than
# having no probe. It runs AFTER the profiled launch so it cannot warm the
# OSS-side cache (O_DIRECT only bypasses the CLIENT page cache).
#
# Submit (normally via submit_sweep.sh):
#   sbatch --job-name=fst-mt64 --export=ALL,TAG=mt64,FILES_PER_RANK=1,MAX_THREADS=64 \
#          fst-threads-exp/scripts/fst_threads.sbatch
#
#SBATCH --job-name=fst-threads
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=fst-threads-exp/results/%x-%j.out

set -euo pipefail

TAG="${TAG:?set TAG, e.g. ctl_first, mt32, fpr8_mt64, ctl_last}"
FILES_PER_RANK="${FILES_PER_RANK:-1}"
MAX_THREADS="${MAX_THREADS:-}"          # empty -> leave fastsafetensors' default (16)
BBUF_SIZE_KB="${BBUF_SIZE_KB:-}"        # empty -> leave default (16384)

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
EXP_DIR="${SLURM_SUBMIT_DIR}/fst-threads-exp"
LOADING_EXP="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
CONTENTION_EXP="${SLURM_SUBMIT_DIR}/lustre-contention-exp"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
RES="${EXP_DIR}/results"
PROFILE_OUT="${RES}/fst-${TAG}-${SLURM_JOB_ID}-${NODE}-profile.json"
DD_CSV="${RES}/fst_ddprobe_sliced-${TAG}-${SLURM_JOB_ID}-${NODE}.csv"

echo "job=${SLURM_JOB_ID} node=${NODE} tag=${TAG}"
echo "files_per_rank=${FILES_PER_RANK} -> files_in_flight=$((FILES_PER_RANK * TP_SIZE))"
echo "max_threads=${MAX_THREADS:-<default 16>} bbuf_size_kb=${BBUF_SIZE_KB:-<default 16384>}"
echo "profile_out=${PROFILE_OUT}"
echo "started=$(date --iso-8601=seconds)"

# OST 8 is sick (~480 ms/RPC) and hosts model-00018. That is a LATENCY fault --
# exactly what deeper queues hide -- so max_threads could look artificially good
# while it is really just routing around a broken disk. Record the fault's size
# in every job so a winner can be re-checked once it is fixed.
echo "=== OST-8 straggler check (512 MiB, QD=1) ==="
for s in 00018 00017; do
  f="${MODEL}/model-${s}-of-00030.safetensors"
  t0=$(date +%s.%N); dd if="$f" of=/dev/null bs=16M count=32 iflag=direct status=none; t1=$(date +%s.%N)
  awk -v s=$t0 -v e=$t1 -v n="$s" -v o="$(lfs getstripe -i "$f" 2>/dev/null)" \
    'BEGIN{printf "  model-%s (ost %s): %6.1f MB/s\n", n, o, 512/(e-s)}'
done

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
pip install fastsafetensors -q
python -c 'import fastsafetensors as f; print("fastsafetensors", f.__version__)'

# Clone sglang @ the SHA pinned to this image, verify byte-for-byte, patch, swap.
# ONE patch only: fst_threads.patch subsumes phase3's fst_files_per_rank.patch
# (same lines, same function) and the harness cannot compose two patches over
# one region -- it would re-clone pristine and trip its own version check.
bash "${LOADING_EXP}/scripts/lib/patch_sglang_in_container.sh" "${EXP_DIR}/scripts/fst_threads.patch"

export SGLANG_FST_FILES_PER_RANK="${FILES_PER_RANK}"
[ -n "${MAX_THREADS}" ]  && export SGLANG_FST_MAX_THREADS="${MAX_THREADS}"
[ -n "${BBUF_SIZE_KB}" ] && export SGLANG_FST_BBUF_SIZE_KB="${BBUF_SIZE_KB}"
echo "env: SGLANG_FST_FILES_PER_RANK=\${SGLANG_FST_FILES_PER_RANK} SGLANG_FST_MAX_THREADS=\${SGLANG_FST_MAX_THREADS:-<unset>} SGLANG_FST_BBUF_SIZE_KB=\${SGLANG_FST_BBUF_SIZE_KB:-<unset>}"

set +e
servekit profile --out "${PROFILE_OUT}" \
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
    --load-format fastsafetensors \
    --enable-metrics
SERVEKIT_RC=\$?
set -e
echo "servekit_rc=\${SERVEKIT_RC}"

# Contemporaneous capstor sample, AFTER the measured read. SLICED (T=8): the
# across-files-only probe would report the worst OST, not the filesystem.
echo "--- sliced dd probe (30 files x 8 slices, O_DIRECT) ---"
bash "${CONTENTION_EXP}/scripts/lib/dd_read_sliced.sh" "${MODEL}" 16M "${DD_CSV}" "8" auto

exit \${SERVEKIT_RC}
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "=== profile result (${TAG}) ==="
if [ -f "${PROFILE_OUT}" ]; then
  python3 -c "
import json; d=json.load(open('${PROFILE_OUT}'))
w=[p for p in d['phases'] if p['name']=='weight_loading']
print('  weight_loading = %.2f s' % w[0]['duration_s'] if w else '  weight_loading = MISSING')
print('  total          = %.2f s' % d['total_duration_s'])
b=d.get('benchmark',{}).get('throughput',{})
print('  throughput     = %s tok/s, errors=%s' % (b.get('output_throughput'), b.get('errors')))
" 2>/dev/null || cat "${PROFILE_OUT}"
else
  echo "NO PROFILE JSON - run did not reach ready"
fi
echo "=== sliced dd probe (${TAG}) ==="
cat "${DD_CSV}" 2>/dev/null || echo "NO DD CSV"
echo "finished=$(date --iso-8601=seconds)"
