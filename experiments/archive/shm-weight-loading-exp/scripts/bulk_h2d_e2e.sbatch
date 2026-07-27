#!/bin/bash
# End-to-end test of the whole-file bulk H2D loader on the sliced /dev/shm
# cold start. See ../NOTES.md and scripts/bulk_h2d_loader.patch.
#
# The loader reads each safetensors file's data region contiguously into a
# pinned buffer, does ONE contiguous H2D DMA to a GPU blob, and yields
# per-tensor GPU VIEWS. The engine's weight_loader then narrows on-device
# (D2D), so nothing about weight_loader changes -- the win, if any, is that a
# contiguous transfer replaces the strided per-shard copy the default loader
# does at ~2 GB/s. Selected by SGLANG_BULK_H2D_LOADER=1; off = upstream path.
#
# Two things this canNOT dodge, both under test here:
#   * it moves ~TP x the bytes (every rank DMAs the full unsharded tensor),
#   * the pinned memcpy of that 4x data is the ceiling -- double-buffered
#     against the DMA, but if the host copy is slower than the DMA it caps us.
# The host copy is torch-threaded, and 4 ranks x 64 threads on 64 cores thrash
# (prior finding, ../NOTES.md), so a thread-capped arm is required to tell
# "bulk loses" apart from "bulk is thread-starved".
#
# Three arms:
#   ctl       bulk off, threads default   <- baseline on THIS image (was 18.99 s)
#   bulk      bulk on,  threads default   <- honest default (may be thread-bound)
#   bulk_thr  bulk on,  threads 16        <- the actual proposal
#
#   for A in ctl bulk bulk_thr; do
#     sbatch --job-name=b-$A --export=ALL,ARM=$A \
#       shm-weight-loading-exp/scripts/bulk_h2d_e2e.sbatch
#   done
#
# One fresh node per arm (page cache survives container runs -- every
# cold-start number must come from its own allocation).
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=shm-weight-loading-exp/results/%x-%j.out

set -euo pipefail

ARM="${ARM:?set ARM: ctl | bulk | bulk_thr}"
case "$ARM" in
  ctl)      BULK=0; THREADS="" ;;
  bulk)     BULK=1; THREADS="" ;;
  bulk_thr) BULK=1; THREADS=16 ;;
  *) echo "bad ARM: $ARM" >&2; exit 1 ;;
esac

SRC="${SRC:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SLICES="${SLICES:-64}"
SHM_DEST="${SHM_DEST:-/dev/shm/llama70b}"

SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
LUSTRE_EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
EXP_DIR="${SLURM_SUBMIT_DIR}/shm-weight-loading-exp"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
PROFILE_OUT="${EXP_DIR}/results/b-${ARM}-${SLURM_JOB_ID}-${NODE}-profile.json"
STAGE_LOG="${EXP_DIR}/results/b-${ARM}-${SLURM_JOB_ID}-${NODE}-stage.txt"

echo "job=${SLURM_JOB_ID} node=${NODE} arm=${ARM} bulk=${BULK} threads=${THREADS:-default}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
bash "${LUSTRE_EXP_DIR}/scripts/phase4_shm/stage_to_shm_sliced.sh" "${SRC}" "${SHM_DEST}" "${SLICES}" | tee "${STAGE_LOG}"
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

# Clone sglang @ the SHA pinned to this image, verify every touched file is
# byte-identical to the installed copy, patch, swap in. Aborts loudly on
# mismatch. Applied in ALL arms so the code under test is identical -- the
# arms differ only by env var, never by which source is running.
bash "${LUSTRE_EXP_DIR}/scripts/lib/patch_sglang_in_container.sh" \
  "${EXP_DIR}/scripts/bulk_h2d_loader.patch"

export SGLANG_BULK_H2D_LOADER="${BULK}"
$( [ -n "${THREADS}" ] && echo "export OMP_NUM_THREADS=${THREADS}" )
echo "SGLANG_BULK_H2D_LOADER=\${SGLANG_BULK_H2D_LOADER} OMP_NUM_THREADS=\${OMP_NUM_THREADS:-unset}"
python3 -c "import torch; print('torch.get_num_threads() =', torch.get_num_threads())"

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
    --enable-metrics
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result (${ARM}) ==="
STAGE_S=$(grep -o 'stage_wall_s=[0-9.]*' "${STAGE_LOG}" 2>/dev/null | cut -d= -f2)
if [ -f "${PROFILE_OUT}" ]; then
  python3 -c "
import json
d = json.load(open('${PROFILE_OUT}'))
ph = {p['name']: p['duration_s'] for p in d['phases']}
b = d.get('benchmark', {}).get('throughput', {})
stage = float('${STAGE_S:-0}' or 0)
print('  arm                = ${ARM}')
print('  stage              = %7.2f s' % stage)
print('  weight_loading     = %7.2f s   <- the number under test' % ph.get('weight_loading', float('nan')))
print('  servekit total     = %7.2f s' % d['total_duration_s'])
print('  TOTAL COLD START   = %7.2f s' % (stage + d['total_duration_s']))
print('  throughput         = %s tok/s, %s/%s ok, errors=%s'
      % (b.get('output_tok_per_s'), b.get('completed'), b.get('requests'), b.get('errors')))
print()
print('  baseline (this image, p4-ctl job 75845): weight_loading 18.99 s')
print('  NOTE: errors>0 or a throughput drop = a GPU-view was mishandled; this')
print('        run is only valid if it serves correctly, not just faster.')
"
else
  echo "  NO PROFILE JSON - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
