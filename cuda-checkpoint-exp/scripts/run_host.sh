#!/bin/bash
# STEP 2: the checkpoint/restore gates. Runs on the BARE compute node (no pyxis
# container -- see ckpt_restore.sbatch for why). Two independent halves, so a
# failure in one still yields a verdict on the other:
#
#   GATE cuda-ckpt : cuda-checkpoint ONLY (no criu), as the normal cluster user.
#                    Evicts GPU state to host RAM and restores it in a LIVE process.
#                    Needs only same-uid ptrace, which we have.
#   GATE map-files : root-cause probe for criu (see below).
#   GATE criu-cpu  : criu dump/restore of a plain process, inside `unshare -Urpf`
#                    (own user+pid ns -> full caps, no seccomp filter).
#   GATE criu-gpu  : full cuda-checkpoint + criu round trip (only if criu-cpu passed).
set -uo pipefail

EXP_DIR="${EXP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CRIU="${EXP_DIR}/bin/criu"
CCKPT="${EXP_DIR}/bin/cuda-checkpoint"
export LD_LIBRARY_PATH="${EXP_DIR}/hostlibs:${LD_LIBRARY_PATH:-}"

hr() { printf '\n========== %s ==========\n' "$*"; }
ok() { printf 'GATE %-10s : PASS  %s\n' "$1" "${2:-}"; }
no() { printf 'GATE %-10s : FAIL  %s\n' "$1" "${2:-}"; }
errs() { grep -iE '\)\s+Error' "$1" 2>/dev/null | tail -6; }

gpu_procs() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; }

################################################################################
# Phase B: criu gates, re-executed inside our own user+pid namespace.
if [ "${CUDACR_NS:-0}" = "1" ]; then
  WORK="$(mktemp -d /iopsstor/scratch/cscs/yboughizane/tmp.cudacr.XXXXXX)"
  CKPT="${WORK}/ckpt"; mkdir -p "${CKPT}"
  hr "criu environment (inside unshare -Urpf)"
  echo "uid=$(id -u) pid=$$ (own pidns)  Seccomp=$(grep -E '^Seccomp:' /proc/self/status | awk '{print $2}') (0=none)"
  echo "CapEff=$(grep -E '^CapEff:' /proc/self/status | awk '{print $2}')"
  echo "criu  : $("${CRIU}" --version 2>&1 | head -1)"

  hr "GATE map-files: can we OPEN /proc/PID/map_files/* ? (criu depends on it)"
  # NB: readlink() and open() on these links are different kernel paths. readlink
  # only needs ptrace access; open() goes through proc_map_files_get_link(), which
  # requires CAP_SYS_ADMIN/CAP_CHECKPOINT_RESTORE in the INITIAL userns. criu opens
  # them, so open() is the operation that actually matters -- test that one.
  sleep 300 & MP=$!
  sleep 0.3
  link=$(ls /proc/${MP}/map_files/ 2>/dev/null | head -1)
  if [ -z "${link}" ]; then
    no map-files "cannot even list /proc/${MP}/map_files"
  else
    rl=$(readlink "/proc/${MP}/map_files/${link}" 2>/dev/null && echo ok || echo EPERM)
    if head -c1 "/proc/${MP}/map_files/${link}" >/dev/null 2>&1; then
      ok map-files "open() works (readlink=${rl}) -> criu has a chance"
    else
      no map-files "open() EPERM (readlink=${rl})"
      echo "                   The kernel gates open() of /proc/PID/map_files behind"
      echo "                   checkpoint_restore_ns_capable(&init_user_ns): CAP_SYS_ADMIN or"
      echo "                   CAP_CHECKPOINT_RESTORE in the INITIAL userns. Namespace-root cannot"
      echo "                   satisfy it, so criu cannot dump ANY process without admin action."
    fi
  fi
  kill -9 "${MP}" 2>/dev/null 2>&1

  hr "GATE criu-cpu: criu dump/restore of a plain CPU process"
  ( i=0; while :; do echo $i > "${WORK}/plain.out"; i=$((i+1)); sleep 0.2; done ) &
  P=$!
  sleep 1; B=$(cat "${WORK}/plain.out" 2>/dev/null)
  "${CRIU}" dump -t "$P" -D "${CKPT}" --shell-job -o dump.log -v4 >/dev/null 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    no criu-cpu "criu dump rc=$rc"; errs "${CKPT}/dump.log"
    kill -9 "$P" 2>/dev/null
    echo; echo "criu-gpu gate skipped (criu cannot dump even a plain process)."
    exit 0
  fi
  "${CRIU}" restore -D "${CKPT}" --shell-job --restore-detached -o restore.log -v4 >/dev/null 2>&1
  sleep 2; A=$(cat "${WORK}/plain.out" 2>/dev/null)
  if [ -n "$A" ] && [ "$A" -gt "${B:-0}" ] 2>/dev/null; then
    ok criu-cpu "counter advanced ${B} -> ${A} across dump+restore"
  else
    no criu-cpu "did not resume (before=${B} after=${A})"; errs "${CKPT}/restore.log"
    exit 0
  fi
  pkill -9 -f plain.out 2>/dev/null; rm -rf "${CKPT:?}"/*

  hr "GATE criu-gpu: cuda-checkpoint + criu full round trip"
  LOG="${WORK}/counter.log"; : > "${LOG}"
  cd "${WORK}"; "${EXP_DIR}/src/counter" "${LOG}" </dev/null >/dev/null 2>&1 &
  CP=$!; sleep 3
  kill -0 "${CP}" 2>/dev/null || { no criu-gpu "counter died"; tail -3 "${LOG}"; exit 0; }
  "${CCKPT}" --action lock --pid "${CP}" --timeout 10000
  "${CCKPT}" --action checkpoint --pid "${CP}"
  FROZE="$(grep '^tick=' "${LOG}" | tail -1)"
  "${CRIU}" dump -t "${CP}" -D "${CKPT}" --shell-job -o dump.log -v4 >/dev/null 2>&1
  rc=$?; echo "   criu dump rc=${rc}"
  [ $rc -ne 0 ] && { no criu-gpu "criu dump failed"; errs "${CKPT}/dump.log"; exit 0; }
  "${CRIU}" restore -D "${CKPT}" --shell-job --restore-detached -o restore.log -v4 >/dev/null 2>&1
  sleep 2; NP="$(pgrep -xn counter)"
  [ -z "${NP}" ] && { no criu-gpu "restore failed"; errs "${CKPT}/restore.log"; exit 0; }
  "${CCKPT}" --action restore --pid "${NP}"; "${CCKPT}" --action unlock --pid "${NP}"
  sleep 3
  LAST="$(grep '^tick=' "${LOG}" | tail -1)"; NF="${FROZE#tick=}"; NL="${LAST#tick=}"
  if [ -n "${NL}" ] && [ "${NL}" -gt "${NF:-0}" ] 2>/dev/null; then
    ok criu-gpu "GPU counter resumed ${NF} -> ${NL} across cuda-checkpoint + CRIU"
  else
    no criu-gpu "counter did not resume (${NF} -> ${NL})"
  fi
  kill -9 "${NP}" 2>/dev/null
  exit 0
fi

################################################################################
# Phase A: cuda-checkpoint on its own, as the ordinary cluster user (no unshare,
# no container) -- i.e. exactly how a real serving job would run.
# This phase runs in BOTH environments (bare host in step 2, pyxis container in
# step 3), so say which one we're actually in rather than assuming.
if grep -qi sles /etc/os-release 2>/dev/null; then WHERE="bare host"; else WHERE="pyxis container"; fi
hr "environment (${WHERE}, normal user)"
echo "node=$(hostname) uid=$(id -u) Seccomp=$(grep -E '^Seccomp:' /proc/self/status | awk '{print $2}') (2 = seccomp filter active)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
echo "cuda-checkpoint: $([ -x "${CCKPT}" ] && echo present || echo MISSING)"

hr "GATE cuda-ckpt: evict GPU state to host RAM and restore it (live process, no criu)"
# TICK_MS paces the counter; FREEZE_S is how long we watch it while checkpointed.
# FREEZE_S must stay several ticks wide or "it stopped ticking" proves little --
# we report how many ticks *should* have happened in that window.
TICK_MS="${TICK_MS:-1000}"
FREEZE_S="${FREEZE_S:-8}"
WORK="$(mktemp -d /iopsstor/scratch/cscs/yboughizane/tmp.cudacr.XXXXXX)"
LOG="${WORK}/counter.log"; : > "${LOG}"
cd "${WORK}"
echo "tick_ms=${TICK_MS}  freeze_window=${FREEZE_S}s  (=> ~$((FREEZE_S * 1000 / TICK_MS)) ticks expected if NOT frozen)"
"${EXP_DIR}/src/counter" "${LOG}" "${TICK_MS}" </dev/null >/dev/null 2>&1 &
CP=$!; sleep $(( (TICK_MS * 5 / 1000) + 2 ))
if ! kill -0 "${CP}" 2>/dev/null; then
  no cuda-ckpt "counter failed to start"; tail -5 "${LOG}"; exit 1
fi
echo "counter pid=${CP}  $(grep '^tick=' "${LOG}" | tail -1)"
echo "GPU procs while running:"; gpu_procs | sed 's/^/  /'
echo "state: $("${CCKPT}" --get-state --pid "${CP}" 2>&1)"

echo "-- lock + checkpoint (device memory -> host RAM, GPU released) --"
"${CCKPT}" --action lock --pid "${CP}" --timeout 10000; echo "   lock rc=$?"
"${CCKPT}" --action checkpoint --pid "${CP}";            echo "   checkpoint rc=$?"
echo "   state: $("${CCKPT}" --get-state --pid "${CP}" 2>&1)"
FROZE="$(grep '^tick=' "${LOG}" | tail -1)"
echo "   last tick while checkpointed: ${FROZE}"
echo "GPU procs while checkpointed (expect NO entry for ${CP}):"; gpu_procs | sed 's/^/  /'
echo "   watching for ${FREEZE_S}s while checkpointed..."
sleep "${FREEZE_S}"
STILL="$(grep '^tick=' "${LOG}" | tail -1)"
EXPECTED=$(( FREEZE_S * 1000 / TICK_MS ))
if [ "${STILL}" = "${FROZE}" ]; then
  echo "   confirmed FROZEN: still ${STILL} after ${FREEZE_S}s (~${EXPECTED} ticks should have elapsed)"
else
  echo "   NOT frozen: advanced ${FROZE} -> ${STILL} while checkpointed (unexpected!)"
fi

echo "-- restore + unlock (reattach to GPU) --"
"${CCKPT}" --action restore --pid "${CP}"; echo "   restore rc=$?"
"${CCKPT}" --action unlock  --pid "${CP}"; echo "   unlock rc=$?"
echo "   state: $("${CCKPT}" --get-state --pid "${CP}" 2>&1)"
RESUME_S=$(( (TICK_MS * 5 / 1000) + 2 ))
echo "   letting it run ${RESUME_S}s to show it keeps ticking..."
sleep "${RESUME_S}"
NF="${FROZE#tick=}"
# The number that proves correctness is the FIRST tick after restore: it must be
# froze+1. The counter keeps ticking while we sleep, so whatever value we happen to
# read at the end is just "how far it got" -- it would be larger if we slept longer
# and carries no correctness information. Report both, clearly distinguished.
WANT=$(( NF + 1 ))
GOT_FIRST="$(grep '^tick=' "${LOG}" | sed -n "${WANT}p")"   # Nth tick line holds tick=N
LAST="$(grep '^tick=' "${LOG}" | tail -1)"; NL="${LAST#tick=}"
echo "   first tick after restore : ${GOT_FIRST:-NONE}  <- the proof; must be tick=${WANT} (= froze+1)"
echo "   counter now at           : ${LAST}  <- only how far it got during the ~${RESUME_S}s sleep; not a correctness signal"
if [ "${GOT_FIRST}" = "tick=${WANT}" ]; then
  ok cuda-ckpt "device memory intact: froze at ${NF}, resumed at exactly ${WANT}, still ticking (now ${NL})"
elif [ -n "${NL}" ] && [ "${NL}" -gt "${NF:-0}" ] 2>/dev/null; then
  no cuda-ckpt "resumed but NOT at froze+1: expected tick=${WANT}, got ${GOT_FIRST:-NONE}"
else
  no cuda-ckpt "counter did not resume (froze=${NF}, now=${NL})"
fi

# Don't just trust first/last. Verify the WHOLE sequence is 1,2,3,...,N with each
# value present exactly once and in order: that simultaneously rules out a reset to
# zero (would restart at 1), a gap across the checkpoint boundary, and a jump.
# The Nth tick line must hold the value N.
echo "-- sequence integrity check --"
grep '^tick=' "${LOG}" | sed 's/tick=//' \
  | awk -v froze="${NF}" '
      NR != $1 { printf("  BROKEN at line %d: expected tick=%d, got tick=%d\n", NR, NR, $1); bad=1 }
      END { if (bad) print "  RESULT: sequence BROKEN";
            else printf("  RESULT: contiguous tick=1..%d, no reset/gap/jump (froze at %s)\n", NR, froze) }'

# Show the actual boundary: the ticks straddling the freeze point, which is the
# claim that matters (did it really resume at froze+1?).
echo "-- ticks around the checkpoint boundary (froze at tick=${NF}) --"
lo=$(( NF > 2 ? NF - 2 : 1 )); hi=$(( NF + 3 ))
grep '^tick=' "${LOG}" | sed -n "${lo},${hi}p" | while read -r t; do
  v="${t#tick=}"
  if   [ "${v}" -le "${NF}" ]; then echo "  ${t}   <- before checkpoint"
  elif [ "${v}" -eq $((NF+1)) ]; then echo "  ${t}   <- FIRST tick after restore (must be $((NF+1)))"
  else echo "  ${t}"; fi
done
cp "${LOG}" "${EXP_DIR}/results/counter-boundary-${SLURM_JOB_ID:-local}.log" 2>/dev/null
kill -9 "${CP}" 2>/dev/null

################################################################################
# SKIP_CRIU=1 runs ONLY the cuda-ckpt gate above. Used to run this same gate INSIDE
# the pyxis container, which is where real SGLang serving actually runs -- the criu
# gates are pointless there (the container's seccomp filter makes criu impossible
# regardless), but cuda-checkpoint's behaviour in the container is what operationally
# matters for us.
if [ "${SKIP_CRIU:-0}" = "1" ]; then
  echo; echo "(SKIP_CRIU=1: criu gates skipped)"
  exit 0
fi

hr "now re-exec into unshare -Urpf for the criu gates"
CUDACR_NS=1 EXP_DIR="${EXP_DIR}" unshare -Urpf --mount-proc "$0"
