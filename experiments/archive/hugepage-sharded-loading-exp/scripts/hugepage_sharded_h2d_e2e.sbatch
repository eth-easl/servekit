#!/bin/bash
# End-to-end test of hugepage-backed sharded_state weight loading. See
# hugepage-sharded-loading-exp/ plan (STEP4_PLAN.md Phase 1/1.5 follow-on).
#
# Arms:
#   ctl          sharded_state checkpoint, staged to /dev/shm, stock loader
#   sharded_pin  same staging, existing sharded_pinned_h2d.patch (pinned
#                bounce-buffer copy) applied         <- today's best number
#   hugepage     NO /dev/shm stage -- hugepage_stage_daemon.py stages the
#                checkpoint straight from Lustre into a memfd(MFD_HUGETLB)
#                buffer; hugepage_sharded_h2d.patch replaces safe_open +
#                the bounce with direct, pipelined cudaMemcpyAsync off the
#                registered buffer
#
#   for A in ctl sharded_pin hugepage; do
#     sbatch --job-name=hp-$A --export=ALL,ARM=$A \
#       hugepage-sharded-loading-exp/scripts/hugepage_sharded_h2d_e2e.sbatch
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
#SBATCH --output=hugepage-sharded-loading-exp/results/%x-%j.out

set -euo pipefail

ARM="${ARM:?set ARM: ctl | sharded_pin | hugepage}"

SHARDED_SRC="${SHARDED_SRC:-/iopsstor/scratch/cscs/yboughizane/sharded-ckpt/llama70b-tp4}"
if [ ! -e "${SHARDED_SRC}/model-rank-0-part-0.safetensors" ]; then
  echo "ERROR: no sharded checkpoint at ${SHARDED_SRC}" >&2
  exit 1
fi

SLICES="${SLICES:-64}"
SHM_DEST="${SHM_DEST:-/dev/shm/llama70b-sharded}"
HUGE_SOCK="/tmp/hugepage_sharded_${SLURM_JOB_ID}.sock"
HUGE_READY="/tmp/hugepage_sharded_${SLURM_JOB_ID}.ready"

SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
LUSTRE_EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
EXP_DIR="${SLURM_SUBMIT_DIR}/hugepage-sharded-loading-exp"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
PROFILE_OUT="${EXP_DIR}/results/hp-${ARM}-${SLURM_JOB_ID}-${NODE}-profile.json"
STAGE_LOG="${EXP_DIR}/results/hp-${ARM}-${SLURM_JOB_ID}-${NODE}-stage.txt"

echo "job=${SLURM_JOB_ID} node=${NODE} arm=${ARM}"
echo "started=$(date --iso-8601=seconds)"

case "$ARM" in
  ctl|sharded_pin)
    MODEL_PATH="${SHM_DEST}"
    INNER_PRELUDE=$(cat <<EOF
bash "${LUSTRE_EXP_DIR}/scripts/phase4_shm/stage_to_shm_sliced.sh" "${SHARDED_SRC}" "${SHM_DEST}" "${SLICES}" | tee "${STAGE_LOG}"
EOF
)
    if [ "$ARM" = "sharded_pin" ]; then
      INNER_PATCH=$(cat <<EOF
bash "${LUSTRE_EXP_DIR}/scripts/lib/patch_sglang_in_container.sh" \
  "${SLURM_SUBMIT_DIR}/shm-weight-loading-exp/scripts/sharded_pinned_h2d.patch"
export SGLANG_SHARDED_PINNED_H2D=1
EOF
)
    else
      INNER_PATCH="echo no-patch-ctl-arm"
    fi
    ;;
  hugepage)
    MODEL_PATH="${SHARDED_SRC}"
    INNER_PRELUDE=$(cat <<EOF
echo "no /dev/shm stage for hugepage arm -- daemon stages directly from Lustre" | tee "${STAGE_LOG}"
python3 "${EXP_DIR}/scripts/hugepage_stage_daemon.py" "${SHARDED_SRC}" \
  --sock "${HUGE_SOCK}" --ready-file "${HUGE_READY}" &
echo \$! > /tmp/hugepage_daemon_pid_${SLURM_JOB_ID}
for i in \$(seq 1 120); do
  [ -f "${HUGE_READY}" ] && break
  sleep 1
done
[ -f "${HUGE_READY}" ] || { echo "FATAL: hugepage daemon never became ready" >&2; exit 1; }
EOF
)
    INNER_PATCH=$(cat <<EOF
bash "${LUSTRE_EXP_DIR}/scripts/lib/patch_sglang_in_container.sh" \
  "${EXP_DIR}/scripts/hugepage_sharded_h2d.patch"
export SGLANG_HUGEPAGE_SHARDED_SOCK="${HUGE_SOCK}"
export PYTHONPATH="${EXP_DIR}/scripts:\${PYTHONPATH:-}"
export SGLANG_HUGEPAGE_CHUNK_BYTES="${CHUNK_BYTES:-1073741824}"
export SGLANG_HUGEPAGE_LOOKAHEAD_CHUNKS="${LOOKAHEAD_CHUNKS:-3}"
EOF
)
    ;;
  *) echo "bad ARM: $ARM" >&2; exit 1 ;;
esac

INNER_CMD=$(cat <<EOF
set -euo pipefail
${INNER_PRELUDE}
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

${INNER_PATCH}
echo "SGLANG_SHARDED_PINNED_H2D=\${SGLANG_SHARDED_PINNED_H2D:-unset} SGLANG_HUGEPAGE_SHARDED_SOCK=\${SGLANG_HUGEPAGE_SHARDED_SOCK:-unset} SGLANG_HUGEPAGE_CHUNK_BYTES=\${SGLANG_HUGEPAGE_CHUNK_BYTES:-unset} SGLANG_HUGEPAGE_LOOKAHEAD_CHUNKS=\${SGLANG_HUGEPAGE_LOOKAHEAD_CHUNKS:-unset}"

exec servekit profile --out "${PROFILE_OUT}" \
  --bench --bench-requests 64 --bench-concurrency 16 \
  --bench-input-len 512 --bench-output-len 128 \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
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

cleanup() {
  if [ -f "/tmp/hugepage_daemon_pid_${SLURM_JOB_ID}" ]; then
    kill "$(cat /tmp/hugepage_daemon_pid_${SLURM_JOB_ID})" 2>/dev/null || true
    rm -f "/tmp/hugepage_daemon_pid_${SLURM_JOB_ID}" "${HUGE_SOCK}" "${HUGE_READY}"
  fi
}
trap cleanup EXIT

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result (${ARM}) ==="
STAGE_S=$(grep -o 'stage_wall_s=[0-9.]*' "${STAGE_LOG}" 2>/dev/null | cut -d= -f2 || true)
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
print('  reference: sharded_state baseline ~10.60s weight_loading (job 76115, shm-weight-loading-exp)')
"
else
  echo "  NO PROFILE JSON - run did not reach ready"
fi
echo "finished=$(date --iso-8601=seconds)"
