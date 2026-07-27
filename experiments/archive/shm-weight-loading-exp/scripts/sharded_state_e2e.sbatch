#!/bin/bash
# End-to-end test of a PRE-SHARDED checkpoint (--load-format sharded_state)
# on the /dev/shm cold start. Requires save_sharded_ckpt.sbatch to have run.
#
# The point (see ../NOTES.md): every runtime loader we tried had to pick two
# of three -- 1x bytes, contiguous layout, no cross-rank exchange:
#
#   default (narrow on CPU)  1x bytes, STRIDED,    no exchange   -> 20.5 s
#   bulk whole-file H2D      4x BYTES, contiguous, no exchange   -> 54 s
#   fastsafetensors (nogds)  1x bytes, contiguous, EXCHANGE tax  -> 25.1 s
#   sharded_state            1x bytes, contiguous, no exchange   -> under test
#
# Pre-sharding wins because the TP de-stride was done ONCE offline and frozen
# into each rank's file, so at serve time rank r reads only its own ~35 GB,
# already contiguous and in final layout. No engine patch is involved -- this
# is a stock SGLang load format, which also satisfies the plug-and-play goal.
#
# Arms:
#   ctl          original checkpoint, default loader    <- baseline    20.51 s
#   sharded      pre-sharded, sharded_state             <- measured    10.60 s
#   sharded_pin  sharded_state + pinned H2D staging     <- under test
#
# sharded_pin applies sharded_pinned_h2d.patch and sets
# SGLANG_SHARDED_PINNED_H2D=1. The stored shard is contiguous 1x-bytes data, so
# a bounce buffer finally helps here: the plain sharded arm still copies from a
# PAGEABLE mmap (~3.3 GB/s vs a ~20 GB/s DMA ceiling). The same staging trick
# failed on the default loader (strided) and on a whole-file loader (TP x bytes).
#
#   for A in ctl sharded sharded_pin; do
#     sbatch --job-name=s-$A --export=ALL,ARM=$A \
#       shm-weight-loading-exp/scripts/sharded_state_e2e.sbatch
#   done
#
# One fresh node per arm (page cache survives container runs).
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

ARM="${ARM:?set ARM: ctl | sharded}"

ORIG_SRC="${ORIG_SRC:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SHARDED_SRC="${SHARDED_SRC:-/iopsstor/scratch/cscs/yboughizane/sharded-ckpt/llama70b-tp4}"

case "$ARM" in
  ctl)         SRC="${ORIG_SRC}";    LOADER_FLAG="";                           PIN=0 ;;
  sharded)     SRC="${SHARDED_SRC}"; LOADER_FLAG="--load-format sharded_state"; PIN=0 ;;
  sharded_pin) SRC="${SHARDED_SRC}"; LOADER_FLAG="--load-format sharded_state"; PIN=1 ;;
  *) echo "bad ARM: $ARM" >&2; exit 1 ;;
esac

case "$ARM" in
  sharded|sharded_pin)
    if [ ! -e "${SHARDED_SRC}/model-rank-0-part-0.safetensors" ]; then
      echo "ERROR: no sharded checkpoint at ${SHARDED_SRC}" >&2
      echo "       run save_sharded_ckpt.sbatch first." >&2
      exit 1
    fi ;;
esac

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
PROFILE_OUT="${EXP_DIR}/results/s-${ARM}-${SLURM_JOB_ID}-${NODE}-profile.json"
STAGE_LOG="${EXP_DIR}/results/s-${ARM}-${SLURM_JOB_ID}-${NODE}-stage.txt"

echo "job=${SLURM_JOB_ID} node=${NODE} arm=${ARM}"
echo "src=${SRC}"
echo "loader_flag='${LOADER_FLAG}'"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
bash "${LUSTRE_EXP_DIR}/scripts/phase4_shm/stage_to_shm_sliced.sh" "${SRC}" "${SHM_DEST}" "${SLICES}" | tee "${STAGE_LOG}"
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

echo "staged contents:"
ls "${SHM_DEST}" | head -8
echo "  (\$(ls "${SHM_DEST}"/*.safetensors 2>/dev/null | wc -l) safetensors files)"

if [ "${PIN}" = "1" ]; then
  bash "${LUSTRE_EXP_DIR}/scripts/lib/patch_sglang_in_container.sh" \
    "${EXP_DIR}/scripts/sharded_pinned_h2d.patch"
  export SGLANG_SHARDED_PINNED_H2D=1
fi
echo "SGLANG_SHARDED_PINNED_H2D=\${SGLANG_SHARDED_PINNED_H2D:-unset}"

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
print('  reference: default 20.51 s (76051) | sharded 10.60 s (76115) | bulk 54 s (76052)')
print('  NOTE: errors>0 or a throughput drop = shards loaded wrong; this run is')
print('        only valid if it SERVES correctly, not just if it is faster.')
"
else
  echo "  NO PROFILE JSON - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
