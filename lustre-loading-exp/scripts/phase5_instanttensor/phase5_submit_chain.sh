#!/bin/bash
# Phase 5 driver: InstantTensor (PR sgl#28506) head-to-head against our Phase-3
# best (fastsafetensors + files_per_rank=8, weight 38.2 s).
#
# Order: preflight -> ctl_fst -> it_default -> it_default -> ctl_fst
#
# WHY TWO SAMPLES OF EACH, AND WHY BRACKET.
# Phase 3 measured the SAME upstream config at 69.5 s and 86.1 s on two
# different nodes - a 21% spread. Node/time variance, not capstor drift, is the
# dominant noise source here. So a single InstantTensor point compared against a
# remembered 38.2 s would be worthless. We therefore run the control (our best)
# FIRST and LAST, and InstantTensor TWICE in between, giving two samples of each
# and an honest bound on the noise. If InstantTensor's two points straddle the
# control's bracket, the answer is "no measurable difference" - and that is a
# real, reportable result, not a failure.
#
# `it_default` passes NO knobs -> library defaults -> reproduces PR #28506
# exactly. That is deliberately the first thing measured: "what does the PR
# actually give you, as written". The backport also wires SGLANG_IT_CONCURRENCY
# / SGLANG_IT_IO_DEPTH for a tuned follow-up round, which is worth doing only if
# the default is competitive (see NOTES).
#
# FRESH NODE PER POINT: `--exclusive` grants sole use, NOT a DIFFERENT node - a
# --dependency chain frees a node and SLURM hands the same one straight back.
# Phase 3 got burned by this. So we submit ONE job at a time, learn which node
# it landed on, and grow an --exclude list. Serialization is a free side effect.
#
# Run from the repo root:
#   bash lustre-loading-exp/scripts/phase5_instanttensor/phase5_submit_chain.sh
set -euo pipefail

S="lustre-loading-exp/scripts/phase5_instanttensor"
RES="lustre-loading-exp/results/phase5_instanttensor"
mkdir -p "$RES"

echo "=== preflight (backport + GDS check) ==="
sbatch --wait "$S/phase5_preflight.sbatch" || { echo "PREFLIGHT FAILED - aborting" >&2; exit 1; }
echo "preflight OK"
echo

USED="${EXCLUDE_NODES:-}"   # nodes that have already read the model -> never reuse

submit_and_wait() {  # submit_and_wait <tag> <loader> [extra --export vars]
  local tag="$1" loader="$2" extra="${3-}" jid node state
  local exclude_arg=()
  [ -n "$USED" ] && exclude_arg=(--exclude="$USED")

  jid="$(sbatch --parsable "${exclude_arg[@]}" \
        --job-name="p5-$tag" \
        --export=ALL,TAG="$tag",LOADER="$loader"${extra:+,$extra} \
        "$S/phase5_instanttensor.sbatch")"
  printf '%-14s -> %-7s loader=%-16s %s\n' "$tag" "$jid" "$loader" "${extra:-}"

  while squeue -j "$jid" -h -o %T 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do
    sleep 15
  done
  node="$(sacct -j "$jid" -X -n -o NodeList | head -1 | xargs)"
  state="$(sacct -j "$jid" -X -n -o State | head -1 | xargs)"
  echo "    -> finished on $node ($state)"
  [ -n "$node" ] && [ "$node" != "None" ] && USED="${USED:+$USED,}$node"
}

submit_and_wait ctl_fst_first fastsafetensors FILES_PER_RANK=8
submit_and_wait it_default    instanttensor
submit_and_wait it_default2   instanttensor
submit_and_wait ctl_fst_last  fastsafetensors FILES_PER_RANK=8

echo
echo "nodes used (all distinct): $USED"
echo "summarize: python lustre-loading-exp/scripts/analysis/summarize_e2e.py $RES"
