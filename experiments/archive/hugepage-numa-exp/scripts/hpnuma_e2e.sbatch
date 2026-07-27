#!/bin/bash
# hugepage-numa-exp: does exp2's NUMA correction compose with the hugepage
# register-and-DMA loader?
#
# The two parents, and what each is missing:
#
#   hugepage-sharded-loading-exp -- best 7.03 s gated (job 76269). Stages the
#     checkpoint into one memfd(MFD_HUGETLB) and DMAs straight off the
#     registered buffer. Ran at --cpus-per-task=64, so SGLang's own NUMA
#     binding was silently disabled, exactly as in every other prior run.
#
#   shm-weight-loading-exp2 -- found that binding is worth 2.7 s on the tmpfs
#     bounce path (11.24 -> 8.47), split into two independent halves:
#       process side  --cpus-per-task=128 lets SGLang's numa_utils actually
#                     run numactl --cpunodebind/--membind per TP worker
#                     (_is_numa_available() bails on a constrained mask).
#       page side     the source pages themselves, placed by whoever first
#                     touches them -- i.e. the stager, not the rank.
#
# ANSWER, after 7 arms: no. See NOTES.md. `ctl64` -- the hugepage design with
# no NUMA work at all -- wins at 7.81 s, and every correction makes it worse.
# The corrections target host-memory locality; this path has no host gather
# for them to grip on, and binding costs a rank 2 of its 8 memory channels.
#
# Arms. ctl64 re-anchors the incumbent under this harness (bit-exactness gate
# + lookahead pinned to 1, so NOT byte-identical to 76269's 7.03 s -- it had
# to be re-measured, not assumed); everything else moves one variable off it.
#
#   for A in ctl64 numa128 page64 numaboth la3 ro ro_la3 ilv; do
#     sbatch --job-name=hn-$A --export=ALL,ARM=$A \
#       hugepage-numa-exp/scripts/hpnuma_e2e.sbatch
#   done
#
# One fresh node per arm (page cache survives container runs).
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:50:00
#SBATCH --output=hugepage-numa-exp/results/%x-%j.out

set -uo pipefail

ARM="${ARM:?set ARM: ctl64 | numa128}"

# PLACE=1 uses hpnuma_stage_daemon.py, which mbind()s each rank's region of
# the memfd to the node local to that rank's GPU before any page is faulted.
# PLACE=2 is MPOL_INTERLEAVE over the whole buffer instead.
#
# The NUMA arms are settled and negative (see NOTES.md): 7.81 unplaced /
# 8.52 placed at 64 CPUs, 9.51 / 8.59 at 128. Locality LOSES here because the
# DMA has no host gather to make local, and concentrating a rank's 35 GB on
# one node caps it at that node's 2 memory channels. The remaining arms chase
# the cost that actually dominates this path: cudaHostRegister.
case "$ARM" in
  ctl64)    SRUN_CPUS=64;  PLACE=0 ;;
  numa128)  SRUN_CPUS=128; PLACE=0 ;;
  page64)   SRUN_CPUS=64;  PLACE=1 ;;   # page half alone, no rank confinement
  numaboth) SRUN_CPUS=128; PLACE=1 ;;   # both halves
  # --- register-cost arms, all at the winning 64-CPU/unplaced baseline ---
  la3)      SRUN_CPUS=64;  PLACE=0; LOOKAHEAD_CHUNKS=3 ;;  # untested combo
  ro)       SRUN_CPUS=64;  PLACE=0; READONLY=1 ;;          # cheaper registration
  ro_la3)   SRUN_CPUS=64;  PLACE=0; READONLY=1; LOOKAHEAD_CHUNKS=3 ;;
  ilv)      SRUN_CPUS=64;  PLACE=2 ;;   # deliberate even spread vs accidental
  *) echo "bad ARM: $ARM" >&2; exit 1 ;;
esac
READONLY="${READONLY:-0}"

case "${PLACE}" in
  1) PLACE_FLAG="" ;;
  2) PLACE_FLAG="--interleave" ;;
  *) PLACE_FLAG="--no-place" ;;
esac

SHARDED_SRC="${SHARDED_SRC:-/iopsstor/scratch/cscs/yboughizane/sharded-ckpt/llama70b-tp4}"
if [ ! -e "${SHARDED_SRC}/model-rank-0-part-0.safetensors" ]; then
  echo "ERROR: no sharded checkpoint at ${SHARDED_SRC}" >&2
  exit 1
fi

SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
LUSTRE_EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
HP_DIR="${SLURM_SUBMIT_DIR}/hugepage-sharded-loading-exp"
E2_DIR="${SLURM_SUBMIT_DIR}/shm-weight-loading-exp2"
EXP_DIR="${SLURM_SUBMIT_DIR}/hugepage-numa-exp"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

# Pinned deliberately. 76269's best number used 1 GiB chunks with a shallow
# lookahead; the client's lookahead design was rewritten AFTER that sweep, so
# its current default of 3 is an untested combination. Holding it at 1 keeps
# this round's only moving variable the NUMA binding. Sweep it separately.
CHUNK_BYTES="${CHUNK_BYTES:-1073741824}"
LOOKAHEAD_CHUNKS="${LOOKAHEAD_CHUNKS:-1}"

NODE="$(hostname)"
PROFILE_OUT="${EXP_DIR}/results/hn-${ARM}-${SLURM_JOB_ID}-${NODE}-profile.json"
HUGE_SOCK="/tmp/hpnuma_${SLURM_JOB_ID}.sock"
HUGE_READY="/tmp/hpnuma_${SLURM_JOB_ID}.ready"

echo "job=${SLURM_JOB_ID} node=${NODE} arm=${ARM} srun_cpus=${SRUN_CPUS}"
echo "chunk_bytes=${CHUNK_BYTES} lookahead_chunks=${LOOKAHEAD_CHUNKS}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -uo pipefail

echo "=== affinity as seen by the container ==="
python3 -c "import os; a=sorted(os.sched_getaffinity(0)); print('cpus_available=%d of %d' % (len(a), os.cpu_count()))"
python3 "${E2_DIR}/scripts/numa_map.py" | sed 's/^/gpu_numa /'
echo

# No /dev/shm stage on this path at all -- the daemon reads Lustre straight
# into the memfd and serves that fd to every rank over SCM_RIGHTS.
python3 "${EXP_DIR}/scripts/hpnuma_stage_daemon.py" "${SHARDED_SRC}" \
  --sock "${HUGE_SOCK}" --ready-file "${HUGE_READY}" ${PLACE_FLAG} &
echo \$! > /tmp/hpnuma_daemon_pid_${SLURM_JOB_ID}
for i in \$(seq 1 180); do
  [ -f "${HUGE_READY}" ] && break
  sleep 1
done
[ -f "${HUGE_READY}" ] || { echo "FATAL: hugepage daemon never became ready" >&2; exit 1; }

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

# Both patches in ONE call -- the helper re-clones a pristine tree per
# invocation, so calling it twice trips its own version check (by design).
# They touch different files (loader.py, model_runner.py), so they compose.
bash "${LUSTRE_EXP_DIR}/scripts/lib/patch_sglang_in_container.sh" \
  "${HP_DIR}/scripts/hugepage_sharded_h2d.patch" \
  "${E2_DIR}/scripts/exp2_verify.patch"

export PYTHONPATH="${HP_DIR}/scripts:${E2_DIR}/scripts:\${PYTHONPATH:-}"
export SGLANG_HUGEPAGE_SHARDED_SOCK="${HUGE_SOCK}"
export SGLANG_HUGEPAGE_CHUNK_BYTES="${CHUNK_BYTES}"
export SGLANG_HUGEPAGE_LOOKAHEAD_CHUNKS="${LOOKAHEAD_CHUNKS}"
export SGLANG_HUGEPAGE_REGISTER_READONLY="${READONLY}"
export SGLANG_EXP2_VERIFY=1
echo "readonly=\${SGLANG_HUGEPAGE_REGISTER_READONLY} sock=\${SGLANG_HUGEPAGE_SHARDED_SOCK} chunk=\${SGLANG_HUGEPAGE_CHUNK_BYTES} lookahead=\${SGLANG_HUGEPAGE_LOOKAHEAD_CHUNKS} verify=\${SGLANG_EXP2_VERIFY}"

exec servekit profile --out "${PROFILE_OUT}" \
  --bench --bench-requests 64 --bench-concurrency 16 \
  --bench-input-len 512 --bench-output-len 128 \
  -- \
  python -m sglang.launch_server \
    --model-path "${SHARDED_SRC}" \
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
  if [ -f "/tmp/hpnuma_daemon_pid_${SLURM_JOB_ID}" ]; then
    kill "$(cat /tmp/hpnuma_daemon_pid_${SLURM_JOB_ID})" 2>/dev/null || true
    rm -f "/tmp/hpnuma_daemon_pid_${SLURM_JOB_ID}" "${HUGE_SOCK}" "${HUGE_READY}"
  fi
}
trap cleanup EXIT

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SRUN_CPUS}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"

LOG="${EXP_DIR}/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
echo
echo "=== result (${ARM}) ==="
echo "  stage              = $(grep -o 'staged [0-9.]* GB in [0-9.]*s' "${LOG}" 2>/dev/null | tail -1)"
# Count the numactl wrappers SGLang actually installed. The "already
# constrained" warning is NOT the signal: bound children still print it
# (numactl got there first, so the in-child check correctly declines to bind
# a second time), which made the previous version of this line report NO on a
# run where binding had in fact engaged on all four ranks.
echo "  numactl wrappers   : $(grep -c 'mp.set_executable.*numactl' "${LOG}" 2>/dev/null | head -1 || true)/4 ranks bound"
echo "  placement check    : $(grep -c '\[OK\]' "${LOG}" 2>/dev/null | head -1 || true) file(s) on their GPU-local node, $(grep -c 'MIXED/UNPLACED' "${LOG}" 2>/dev/null | head -1 || true) not"
echo "  bit-exact gate     : $(grep -c 'EXP2_VERIFY PASS' "${LOG}" 2>/dev/null | head -1 || true)/4 ranks PASS, $(grep -c 'EXP2_VERIFY FAIL' "${LOG}" 2>/dev/null | head -1 || true) FAIL"
echo "  registrar blocking : $(grep -c 'wait_for(' "${LOG}" 2>/dev/null | head -1 || true) blocking events across all ranks"
echo "  per-rank elapsed   : $(grep -o 'Load weight end. elapsed=[0-9.]* s' "${LOG}" 2>/dev/null | grep -o '[0-9.]*' | tr '\n' ' ')"
echo "  GATED weight_loading (max over ranks) = $(grep -o 'Load weight end. elapsed=[0-9.]* s' "${LOG}" 2>/dev/null | grep -o '[0-9.]*' | sort -g | tail -1) s"
if [ -f "${PROFILE_OUT}" ]; then
  # The batch node runs Python 3.6, so this stays walrus-free and tolerates a
  # null benchmark (a run that loaded but died before serving still has a
  # valid weight_loading number worth printing).
  python3 -c "
import json
d = json.load(open('${PROFILE_OUT}'))
print('  servekit total     = %7.2f s' % d['total_duration_s'])
b = (d.get('benchmark') or {}).get('throughput') or {}
if b:
    print('  throughput         = %s tok/s, %s/%s ok, errors=%s'
          % (b.get('output_tok_per_s'), b.get('completed'), b.get('requests'), b.get('errors')))
else:
    print('  throughput         = NONE - server never reached ready')
"
else
  echo "  NO PROFILE JSON - run did not reach ready"
fi
echo "  reference: 7.03 s gated (job 76269, hugepage, no NUMA); 8.47 s (job 76364, tmpfs bounce + both NUMA halves)"
echo "finished=$(date --iso-8601=seconds)"
