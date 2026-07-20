#!/bin/bash
# Phase 4 (sliced, BUFFERED read) - stage the NATIVE model into /dev/shm with
# 30 files x 64 within-file slices, then one end-to-end SGLang cold start.
#
# Identical to phase4_shm_sliced_e2e.sbatch except the stage reads WITHOUT
# O_DIRECT (READ_MODE=buffered): dd drops iflag=direct so the Lustre client
# caches and reads ahead. The question this run answers: does client-side
# readahead hide the OST-8 straggler that caps the O_DIRECT sliced stage, or does
# 30x64 in-flight slicing already saturate the link so buffering only adds a
# redundant RAM copy?
#
# TWO CONSEQUENCES OF DROPPING O_DIRECT, both deliberate here:
#  1. The source bytes land in the page cache too, so every byte is held twice in
#     RAM (page cache + tmpfs) until reclaim. 141 GB model + 141 GB tmpfs on a
#     node with plenty of RAM; the page cache is reclaimable so this should not
#     OOM, but it is why this variant exists as its own job, not a knob on a
#     shared node.
#  2. The read is NO LONGER COLD on a re-run: a second stage of the same source
#     on the same node hits warm page cache. This job MUST run on a fresh node to
#     price a genuine cold start (see the fresh-node-per-run rule).
#
#   sbatch --job-name=p4slb-mmap   --export=ALL,TAG=sliced_buf_mmap,LOADER_FLAG="" \
#     lustre-loading-exp/scripts/phase4_shm/phase4_shm_sliced_buffered_e2e.sbatch
#   sbatch --job-name=p4slb-nommap --export=ALL,TAG=sliced_buf_nommap,LOADER_FLAG="--weight-loader-disable-mmap" \
#     lustre-loading-exp/scripts/phase4_shm/phase4_shm_sliced_buffered_e2e.sbatch
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=lustre-loading-exp/results/phase4_shm/%x-%j.out

set -euo pipefail

TAG="${TAG:?set TAG, e.g. sliced_buf_mmap or sliced_buf_nommap}"
LOADER_FLAG="${LOADER_FLAG-}"   # "" = mmap (SGLang default)

SRC="${SRC:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SLICES="${SLICES:-64}"
READ_MODE=buffered              # the one thing this variant changes
SHM_DEST="${SHM_DEST:-/dev/shm/llama70b}"

SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
PROFILE_OUT="${EXP_DIR}/results/phase4_shm/phase4_sliced_buf-${TAG}-${SLURM_JOB_ID}-${NODE}-profile.json"
STAGE_LOG="${EXP_DIR}/results/phase4_shm/phase4_sliced_buf-${TAG}-${SLURM_JOB_ID}-${NODE}-stage.txt"

echo "job=${SLURM_JOB_ID} node=${NODE} tag=${TAG} src=${SRC} slices=${SLICES} read_mode=${READ_MODE} loader_flag='${LOADER_FLAG}'"
echo "started=$(date --iso-8601=seconds)"

# Record the straggler's size: the whole reason the control stager is slow today.
echo "=== OST-8 straggler check (512 MiB, QD=1) ==="
for s in 00018 00017; do
  f="${SRC}/model-${s}-of-00030.safetensors"
  t0=$(date +%s.%N); dd if="$f" of=/dev/null bs=16M count=32 iflag=direct status=none; t1=$(date +%s.%N)
  awk -v s=$t0 -v e=$t1 -v n="$s" -v o="$(lfs getstripe -i "$f" 2>/dev/null)" \
    'BEGIN{printf "  model-%s (ost %s): %6.1f MB/s\n", n, o, 512/(e-s)}'
done

INNER_CMD=$(cat <<EOF
set -euo pipefail
bash "${EXP_DIR}/scripts/phase4_shm/stage_to_shm_sliced.sh" "${SRC}" "${SHM_DEST}" "${SLICES}" "16M" "${READ_MODE}" | tee "${STAGE_LOG}"
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
pip install fastsafetensors -q
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
    ${LOADER_FLAG} \
    --enable-metrics
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result (${TAG}) ==="
STAGE_S=$(grep -o 'stage_wall_s=[0-9.]*' "${STAGE_LOG}" 2>/dev/null | cut -d= -f2)
STAGE_BW=$(grep -o 'stage_GBps=[0-9.]*' "${STAGE_LOG}" 2>/dev/null | cut -d= -f2)
if [ -f "${PROFILE_OUT}" ]; then
  python3 -c "
import json
d = json.load(open('${PROFILE_OUT}'))
stage = float('${STAGE_S:-0}' or 0)
w = [p for p in d['phases'] if p['name'] == 'weight_loading']
wl = w[0]['duration_s'] if w else float('nan')
tot = d['total_duration_s']
b = d.get('benchmark', {}).get('throughput', {})
print('  stage (sliced buffered 30x${SLICES}) = %7.2f s  @ ${STAGE_BW:-?} GB/s' % stage)
print('  weight_loading (from shm)          = %7.2f s' % wl)
print('  servekit total                     = %7.2f s' % tot)
print('  TOTAL COLD START                   = %7.2f s   (stage + total)' % (stage + tot))
print('  throughput                         = %s tok/s, %s/%s completed, errors=%s'
      % (b.get('output_tok_per_s'), b.get('completed'), b.get('requests'), b.get('errors')))
print()
print('  reference, same model, measured 2026-07-17:')
print('    sliced O_DIRECT 30x64        : see phase4_sliced-sliced_mmap stage log')
print('    fastsafetensors from capstor : weight_loading 76.75 s, total 245.17 s')
"
else
  echo "  NO PROFILE JSON - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
