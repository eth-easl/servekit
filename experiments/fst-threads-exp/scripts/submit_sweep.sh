#!/bin/bash
# Driver: one e2e cold start per grid point, each on a node that has NEVER seen
# this model, strictly serialized so no two runs contend on capstor.
#
# Order: ctl_first -> mt32 -> mt64 -> mt128 -> fpr8 -> fpr8_mt64 -> ctl_last
#
# ctl_first/ctl_last are the NULL RUN (fpr=1, max_threads unset == upstream) and
# bracket the sweep. The same config has measured 69.5-86.1 s across nodes -- a
# 21% spread -- so one control cannot tell a real speedup from a quieter
# filesystem. Two bound the drift.
#
# FRESH NODE PER POINT - THE HARD PART. `--exclusive` grants sole use of a node,
# NOT a different node: with a --dependency chain SLURM hands the very same one
# straight back. So: submit ONE job at a time with --wait, learn its node, add it
# to a growing --exclude. Serialization falls out of --wait for free.
#
# Run from the repo root:
#   bash experiments/fst-threads-exp/scripts/submit_sweep.sh
set -uo pipefail

S="experiments/fst-threads-exp/scripts"
RES="experiments/fst-threads-exp/results"
LOG="${RES}/sweep.log"
mkdir -p "$RES"

[ -f "$S/fst_threads.sbatch" ] || { echo "run me from the repo root" >&2; exit 1; }

: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }

USED=""
# tag:files_per_rank:max_threads   ("-" = leave fastsafetensors' default of 16)
POINTS="
ctl_first:1:-
mt32:1:32
mt64:1:64
mt128:1:128
fpr8:8:-
fpr8_mt64:8:64
ctl_last:1:-
"

say "=== fst-threads sweep started $(date --iso-8601=seconds) ==="

for pt in $POINTS; do
  TAG="${pt%%:*}"; rest="${pt#*:}"; FPR="${rest%%:*}"; MT="${rest##*:}"
  [ "$MT" = "-" ] && MT=""

  EXCLUDE=""
  [ -n "$USED" ] && EXCLUDE="--exclude=$USED"

  say ""
  say "--- ${TAG}: files_per_rank=${FPR} max_threads=${MT:-<default 16>} ${EXCLUDE}"
  JID=$(sbatch --parsable --wait --job-name="fst-${TAG}" ${EXCLUDE} \
        --export=ALL,TAG="${TAG}",FILES_PER_RANK="${FPR}",MAX_THREADS="${MT}" \
        "$S/fst_threads.sbatch" 2>&1)
  RC=$?
  # --wait blocks; --parsable gives the id on stdout. On a submit failure JID is
  # the error text, so guard before using it as an id.
  case "$JID" in
    ''|*[!0-9]*) say "    SUBMIT FAILED: ${JID}"; continue ;;
  esac

  NODE=$(sacct -j "$JID" -o NodeList -n 2>/dev/null | head -1 | tr -d ' ')
  STATE=$(sacct -j "$JID" -o State -n 2>/dev/null | head -1 | tr -d ' ')
  say "    -> job ${JID} on ${NODE} (${STATE}, rc=${RC})"

  WL=$(grep -h "weight_loading = " "${RES}/fst-${TAG}-${JID}.out" 2>/dev/null | head -1)
  say "    -> ${WL:-no weight_loading parsed}"

  # Never hand a node back: fastsafetensors does not use the page cache, but the
  # OSS-side cache is not ours to reason about, and phase 3 was burned exactly here.
  [ -n "$NODE" ] && USED="${USED:+$USED,}${NODE}"
done

say ""
say "=== done $(date --iso-8601=seconds) ==="
say "nodes used (must all be distinct): ${USED}"
say ""
say "=== summary ==="
{
  printf "%-12s %14s %10s %10s\n" "tag" "weight_load_s" "total_s" "node"
  for f in "${RES}"/fst-*-profile.json; do
    [ -f "$f" ] || continue
    python3 -c "
import json,sys,os
f='$f'; d=json.load(open(f))
b=os.path.basename(f).split('-')
w=[p for p in d['phases'] if p['name']=='weight_loading']
print('%-12s %14.2f %10.2f %10s' % (b[1], w[0]['duration_s'] if w else -1, d['total_duration_s'], b[3]))
" 2>/dev/null
  done
} | tee -a "$LOG"
