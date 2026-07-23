#!/bin/bash
# Sweep the knob that actually creates read concurrency: max_copy_block_size.
#
# The max_threads sweep (submit_sweep.sh) came back NEGATIVE and the source says
# why: submit_io() chunks a file by max_copy_block_size (default 16 GiB) and
# submits ONE read per chunk, one thread each. A 5 GB shard therefore yields ONE
# submit -> ONE thread -> one sequential pread loop, whatever max_threads says.
# Lowering max_copy_block_size to 256 MB turns that shard into ~20 concurrent
# reads -- real queue depth on its OST, which is what the sliced dd probe showed
# is worth 26x.
#
# Order: ctl_first -> mcbs1024 -> mcbs256 -> mcbs64 -> fpr8 -> fpr8_mcbs256 -> ctl_last
# Same rules as the sibling sweep: null run brackets the whole thing; one job at
# a time with --wait; accumulate --exclude (--exclusive gives sole use of a node,
# NOT a different node).
#
#   bash experiments/fst-threads-exp/scripts/submit_mcbs_sweep.sh
set -uo pipefail

S="experiments/fst-threads-exp/scripts"
RES="experiments/fst-threads-exp/results"
LOG="${RES}/mcbs_sweep.log"
mkdir -p "$RES"
[ -f "$S/fst_knobs.sbatch" ] || { echo "run me from the repo root" >&2; exit 1; }

: > "$LOG"
say() { echo "$@" | tee -a "$LOG"; }

USED=""
# tag:files_per_rank:max_copy_block_size_mb   ("-" = leave the 16 GiB default)
POINTS="
ctl_first:1:-
mcbs1024:1:1024
mcbs256:1:256
mcbs64:1:64
fpr8:8:-
fpr8_mcbs256:8:256
ctl_last:1:-
"

say "=== mcbs sweep started $(date --iso-8601=seconds) ==="
for pt in $POINTS; do
  TAG="${pt%%:*}"; rest="${pt#*:}"; FPR="${rest%%:*}"; MC="${rest##*:}"
  [ "$MC" = "-" ] && MC=""
  EXCLUDE=""; [ -n "$USED" ] && EXCLUDE="--exclude=$USED"
  say ""
  say "--- ${TAG}: files_per_rank=${FPR} mcbs_mb=${MC:-<default 16384>} ${EXCLUDE}"
  JID=$(sbatch --parsable --wait --job-name="fstk-${TAG}" ${EXCLUDE} \
        --export=ALL,TAG="${TAG}",FILES_PER_RANK="${FPR}",MCBS_MB="${MC}" \
        "$S/fst_knobs.sbatch" 2>&1)
  case "$JID" in ''|*[!0-9]*) say "    SUBMIT FAILED: ${JID}"; continue ;; esac
  NODE=$(sacct -j "$JID" -o NodeList -n 2>/dev/null | head -1 | tr -d ' ')
  STATE=$(sacct -j "$JID" -o State -n 2>/dev/null | head -1 | tr -d ' ')
  say "    -> job ${JID} on ${NODE} (${STATE})"
  say "    -> $(grep -h 'weight_loading = ' "${RES}/fstk-${TAG}-${JID}.out" 2>/dev/null | head -1 || echo 'no weight_loading parsed')"
  say "    -> capstor: $(grep -hE '^8,[0-9]' "${RES}/fst_ddprobe_sliced-${TAG}-${JID}-${NODE}.csv" 2>/dev/null | head -1 | cut -d, -f5) GB/s (sliced T=8)"
  [ -n "$NODE" ] && USED="${USED:+$USED,}${NODE}"
done
say ""
say "=== done $(date --iso-8601=seconds) ==="
say "nodes used (must all be distinct): ${USED}"
