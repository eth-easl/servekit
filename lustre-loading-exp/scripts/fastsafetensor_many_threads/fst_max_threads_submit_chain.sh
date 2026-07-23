#!/bin/bash
# Driver: preflight, then one e2e cold start per {files_per_rank, max_threads}
# point, each on a node that has NEVER seen this model, strictly serialized so
# no two runs contend on capstor.
#
# Order: preflight -> mt16_first -> mt64 -> mt64_fpr8 -> mt128_fpr8 -> mt16_last
#
# mt16_first/mt16_last (fpr=1, max_threads=16 - the true null run, both knobs
# at their provably-upstream default) bracket the sweep, because capstor
# throughput drifts over tens of minutes (see phase3's own bracket, and
# NOTES Phase 1.2). Two null samples bound that drift.
#
# mt64 (fpr=1, max_threads=64) is the user's "run this alone" point: the
# max_threads effect in isolation, with upstream file concurrency.
# mt64_fpr8 (fpr=8, max_threads=64) stacks it on phase3's best file-fan-out
# point. Phase 3's own fpr8-alone result (37-43 s weight_loading) is reused as
# the comparison baseline for mt64_fpr8 - no need to rerun it here.
# mt128_fpr8 (fpr=8, max_threads=128) doubles max_threads again on top of
# fpr8, to see whether the mt64_fpr8 result keeps improving or has already
# plateaued. Doubles the pinned host bounce buffer again too (2 GB, still
# just bbuf_size_kb*1024*max_threads, unchanged bbuf_size_kb) - trivial next
# to a bristen A100 node's host RAM, same as the mt64 buffer growth.
#
# FRESH NODE PER POINT, same reasoning as phase3_submit_chain.sh: --exclusive
# grants sole use of a node, not a DIFFERENT node, so a --dependency chain
# would silently hand the same (now warm) node to the next job. Submit ONE job
# at a time with --wait, learn which node it landed on, exclude it from the
# next submission.
#
# Run from the repo root:
#   bash lustre-loading-exp/scripts/fastsafetensor_many_threads/fst_max_threads_submit_chain.sh
set -euo pipefail

S="lustre-loading-exp/scripts/fastsafetensor_many_threads"
RES="lustre-loading-exp/results/fastsafetensor_many_threads"
mkdir -p "$RES"

echo "=== preflight (harness must be sound before spending GPU nodes) ==="
sbatch --wait "$S/fst_max_threads_preflight.sbatch" || { echo "PREFLIGHT FAILED - aborting sweep" >&2; exit 1; }
echo "preflight OK"
echo

# Comma-separated nodes that have already read this model -> never reuse.
#   EXCLUDE_NODES=nid002324 bash .../fst_max_threads_submit_chain.sh
USED="${EXCLUDE_NODES:-}"

submit_and_wait() {  # submit_and_wait <tag> <files_per_rank> <max_threads>
  local tag="$1" fpr="$2" mt="$3" jid node
  local exclude_arg=()
  [ -n "$USED" ] && exclude_arg=(--exclude="$USED")

  jid="$(sbatch --parsable "${exclude_arg[@]}" \
        --job-name="fstmt-$tag" --export=ALL,TAG="$tag",FILES_PER_RANK="$fpr",MAX_THREADS="$mt" \
        "$S/fst_max_threads_e2e.sbatch")"
  printf 'e2e %-11s -> %-7s files_per_rank=%-2d max_threads=%-3d exclude=[%s]\n' \
        "$tag" "$jid" "$fpr" "$mt" "${USED:-none}"

  # block until it leaves the queue (COMPLETED or otherwise - an unexpected
  # failure is an expected possible outcome and must not stop the bracket)
  while squeue -j "$jid" -h -o %T 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do
    sleep 15
  done

  node="$(sacct -j "$jid" -X -n -o NodeList | head -1 | xargs)"
  local state
  state="$(sacct -j "$jid" -X -n -o State | head -1 | xargs)"
  echo "    -> $tag finished on $node ($state)"

  if [ -n "$node" ] && [ "$node" != "None" ]; then
    USED="${USED:+$USED,}$node"
  fi
}

submit_and_wait mt16_first 1 16
submit_and_wait mt64       1 64
submit_and_wait mt64_fpr8  8 64
submit_and_wait mt128_fpr8 8 128
submit_and_wait mt16_last  1 16

echo
echo "nodes used (all distinct, none reused): $USED"
echo "summarize: python lustre-loading-exp/scripts/analysis/summarize_e2e.py $RES"
