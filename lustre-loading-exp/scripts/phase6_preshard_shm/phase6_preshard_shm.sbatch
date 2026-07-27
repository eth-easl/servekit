#!/bin/bash
# Phase 6 - end-to-end cold-start with a PRE-SHARDED checkpoint (per-TP-rank
# safetensors, --load-format sharded_state) staged into /dev/shm before serve.
# Compare weight_loading + TOTAL COLD START against the phase1.3_e2e baselines
# (mmap 20.51 s / nommap / fastsafetensors / runai_streamer), which all read
# the ORIGINAL checkpoint straight off capstor with no shm staging.
#
# Why pre-shard + shm together: every runtime loader tested in
# shm-weight-loading-exp had to pick two of three -- 1x bytes, contiguous
# layout, no cross-rank exchange. Pre-sharding does the TP de-stride ONCE
# offline and freezes it into each rank's file, so at serve time rank r reads
# only its own ~35 GB, already contiguous and in final layout. Staging that
# into /dev/shm first removes Lustre read latency from the timed window
# entirely, so weight_loading measures pure tmpfs->GPU H2D copy. No engine
# patch is involved -- this is a stock SGLang load format.
#
# Requires save_sharded_ckpt.sbatch (in this dir) to have run first.
#
#   sbatch lustre-loading-exp/scripts/phase6_preshard_shm/phase6_preshard_shm.sbatch
#
# One fresh node per run (page cache survives across container runs).
#
#SBATCH --job-name=phase6-preshard-shm
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=lustre-loading-exp/results/phase6_preshard_shm/%x-%j.out

set -euo pipefail

SHARDED_SRC="${SHARDED_SRC:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded}"

if [ ! -e "${SHARDED_SRC}/model-rank-0-part-0.safetensors" ]; then
  echo "ERROR: no sharded checkpoint at ${SHARDED_SRC}" >&2
  echo "       run save_sharded_ckpt.sbatch (in this dir) first." >&2
  exit 1
fi

SLICES="${SLICES:-64}"
SHM_DEST="${SHM_DEST:-/dev/shm/llama70b}"

SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
LUSTRE_EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
RESULTS_DIR="${LUSTRE_EXP_DIR}/results/phase6_preshard_shm"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
PROFILE_OUT="${RESULTS_DIR}/phase6-preshard-shm-${SLURM_JOB_ID}-${NODE}-profile.json"
STAGE_LOG="${RESULTS_DIR}/phase6-preshard-shm-${SLURM_JOB_ID}-${NODE}-stage.txt"

echo "job=${SLURM_JOB_ID} node=${NODE}"
echo "sharded_src=${SHARDED_SRC}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
bash "${LUSTRE_EXP_DIR}/scripts/phase4_shm/stage_to_shm_sliced.sh" "${SHARDED_SRC}" "${SHM_DEST}" "${SLICES}" | tee "${STAGE_LOG}"
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

echo "staged contents:"
ls "${SHM_DEST}" | head -8
echo "  (\$(ls "${SHM_DEST}"/*.safetensors 2>/dev/null | wc -l) safetensors files)"

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
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result (phase6_preshard_shm) ==="
STAGE_S=$(grep -o 'stage_wall_s=[0-9.]*' "${STAGE_LOG}" 2>/dev/null | cut -d= -f2)
if [ -f "${PROFILE_OUT}" ]; then
  python3 -c "
import json
d = json.load(open('${PROFILE_OUT}'))
ph = {p['name']: p['duration_s'] for p in d['phases']}
b = d.get('benchmark', {}).get('throughput', {})
stage = float('${STAGE_S:-0}' or 0)
print('  stage              = %7.2f s' % stage)
print('  weight_loading     = %7.2f s   <- the number under test' % ph.get('weight_loading', float('nan')))
print('  servekit total     = %7.2f s' % d['total_duration_s'])
print('  TOTAL COLD START   = %7.2f s' % (stage + d['total_duration_s']))
print('  throughput         = %s tok/s, %s/%s ok, errors=%s'
      % (b.get('output_tok_per_s'), b.get('completed'), b.get('requests'), b.get('errors')))
print()
print('  reference (phase1.3_e2e, no shm staging): mmap baseline 20.51 s')
print('  NOTE: errors>0 or a throughput drop = shards loaded wrong; this run is')
print('        only valid if it SERVES correctly, not just if it is faster.')
"
else
  echo "  NO PROFILE JSON - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
