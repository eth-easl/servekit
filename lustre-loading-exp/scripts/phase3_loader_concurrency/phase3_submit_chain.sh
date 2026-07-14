#!/bin/bash
# Phase 3 driver: preflight, then one e2e cold start per reader fan-out, each on
# a node that has NEVER seen this model, strictly serialized so no two runs
# contend on capstor.
#
# Order: preflight -> fpr1 -> fpr2 -> fpr4 -> fpr8 -> fpr1
#
# fpr1 (the null run == upstream behavior) runs FIRST and LAST, bracketing the
# sweep, because capstor throughput drifts over tens of minutes (NOTES Phase
# 1.2, and the first attempt at this sweep saw the native dd probe climb
# 6.66 -> 8.55 GB/s across ~20 min). Two null samples bound that drift; one
# cannot tell a real speedup from a filesystem that merely got quieter.
#
# FRESH NODE PER POINT - THE HARD PART. `--exclusive` grants sole use of a node,
# it does NOT grant a DIFFERENT node: with a --dependency chain each job frees
# its node and SLURM hands the very same one straight back to the next job. The
# first run of this sweep put all 5 points on nid002324, where a 515 GB node
# with 402 GB free could hold the whole 141 GB model in page cache and quietly
# feed it to every subsequent run.
#
# So we do NOT pre-chain with --dependency. We submit ONE job at a time with
# --wait, learn which node it landed on, and add that node to a growing
# --exclude list. Serialization falls out of --wait for free. Jobs are only
# ~5 min (servekit --bench tears the server down), so the sequential cost is
# minutes, not hours.
#
# Run from the repo root:
#   bash lustre-loading-exp/scripts/phase3_loader_concurrency/phase3_submit_chain.sh
set -euo pipefail

S="lustre-loading-exp/scripts/phase3_loader_concurrency"
RES="lustre-loading-exp/results/phase3_loader_concurrency"
mkdir -p "$RES"

echo "=== preflight (harness must be sound before spending GPU nodes) ==="
sbatch --wait "$S/phase3_preflight.sbatch" || { echo "PREFLIGHT FAILED - aborting sweep" >&2; exit 1; }
echo "preflight OK"
echo

# Comma-separated nodes that have already read this model -> never reuse.
# Seed it with any node already contaminated by an earlier sweep, e.g.
#   EXCLUDE_NODES=nid002324 bash .../phase3_submit_chain.sh
USED="${EXCLUDE_NODES:-}"

submit_and_wait() {  # submit_and_wait <tag> <files_per_rank>
  local tag="$1" fpr="$2" jid node
  local exclude_arg=()
  [ -n "$USED" ] && exclude_arg=(--exclude="$USED")

  jid="$(sbatch --parsable "${exclude_arg[@]}" \
        --job-name="p3-$tag" --export=ALL,TAG="$tag",FILES_PER_RANK="$fpr" \
        "$S/phase3_loader_concurrency.sbatch")"
  printf 'e2e %-11s -> %-7s files_in_flight=%-2d exclude=[%s]\n' \
        "$tag" "$jid" "$((fpr * 4))" "${USED:-none}"

  # block until it leaves the queue (COMPLETED or otherwise - an OOM at fpr8 is
  # an expected outcome and must not stop the closing bracket)
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

submit_and_wait fpr1_first 1
submit_and_wait fpr2       2
submit_and_wait fpr4       4
submit_and_wait fpr8       8
submit_and_wait fpr1_last  1

echo
echo "nodes used (all distinct, none reused): $USED"
echo "summarize: python lustre-loading-exp/scripts/analysis/summarize_e2e.py $RES"
