#!/bin/bash
# Phase 4 - end-to-end test of pinned H2D staging (+ thread cap) on the
# sliced /dev/shm cold start. See ../NOTES.md.
#
# Microbenchmark says weight_loading should go 19.9 s -> ~9 s. Both halves
# are required and are varied together here:
#   * the PATCH   routes param_data.copy_ through a reusable pinned buffer
#   * OMP_NUM_THREADS caps each rank at cores/tp, without which staging is a
#     LOSS (4 ranks x 64 OMP threads on 64 cores: 2.47 vs 5.50 GB/s)
#
# Four arms, so the two factors are separable rather than confounded:
#   ctl        patch off, threads default   <- today's baseline (19.9 s)
#   thr        patch off, threads 16        <- does the env var alone do it?
#   stage      patch on,  threads default   <- staging without the cap (expect
#                                              flat or WORSE; guards the claim)
#   stage_thr  patch on,  threads 16        <- the actual proposal
#
#   for A in ctl thr stage stage_thr; do
#     sbatch --job-name=p4-$A --export=ALL,ARM=$A \
#       shm-weight-loading-exp/scripts/phase4_pinned_staging_e2e.sbatch
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

ARM="${ARM:?set ARM: ctl | thr | stage | stage_thr}"
case "$ARM" in
  ctl)       STAGING=0; THREADS="" ;;
  thr)       STAGING=0; THREADS=16 ;;
  stage)     STAGING=1; THREADS="" ;;
  stage_thr) STAGING=1; THREADS=16 ;;
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
PROFILE_OUT="${EXP_DIR}/results/p4-${ARM}-${SLURM_JOB_ID}-${NODE}-profile.json"
STAGE_LOG="${EXP_DIR}/results/p4-${ARM}-${SLURM_JOB_ID}-${NODE}-stage.txt"

echo "job=${SLURM_JOB_ID} node=${NODE} arm=${ARM} staging=${STAGING} threads=${THREADS:-default}"
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
  "${EXP_DIR}/scripts/h2d_pinned_staging.patch"

export SGLANG_H2D_PINNED_STAGING="${STAGING}"
$( [ -n "${THREADS}" ] && echo "export OMP_NUM_THREADS=${THREADS}" )
echo "SGLANG_H2D_PINNED_STAGING=\${SGLANG_H2D_PINNED_STAGING} OMP_NUM_THREADS=\${OMP_NUM_THREADS:-unset}"
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
print('  baseline (sliced mmap, jobs 75168/75196/75335): weight_loading 18.61-19.58 s')
print('  microbenchmark predicts stage_thr ~9 s')
"
else
  echo "  NO PROFILE JSON - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
