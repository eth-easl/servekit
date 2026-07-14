#!/bin/bash
# Phase 5b - is InstantTensor BAD, or just UNCONFIGURED?
#
# The A/B (phase5_submit_chain.sh) found PR #28506 as written is a ~9x cold-start
# REGRESSION here: weight_loading 352 s / 376 s vs 36.9 s / 42.7 s for our
# phase-3 fastsafetensors patch, reproduced on two nodes, with capstor healthy
# (6.2 / 7.3 GB/s) so it is not contention.
#
# The tell: InstantTensor's effective read bandwidth is 0.35-0.375 GB/s.
# Phase 1.2 measured SINGLE-STREAM dd at 0.35-0.41 GB/s. It is reading the model
# with effectively ONE stream - the same reader-starvation disease phase 3 found
# in fastsafetensors, only far worse. And instanttensor.safe_open() exposes
# `concurrency` and `io_depth`, both of which PR #28506 PASSES NEITHER OF.
#
# So: does turning the knobs up rescue it?
#   - it_c{16,32,64}   sweep SGLANG_IT_CONCURRENCY (the analogue of the
#                      files-per-rank lever that gave us 2x in phase 3)
#   - it_c64_d64       + SGLANG_IT_IO_DEPTH, the io_uring queue depth that
#                      actually governs in-flight reads on the URING backend
#                      (which the GDS probe showed it auto-selects, since
#                      nvidia_fs is not loaded and CUFILE is unavailable)
#
# If concurrency drags it from 352 s toward ~35 s, the library is fine and the
# PR is merely missing its own tuning - a concrete, upstreamable finding.
# If it stays slow, InstantTensor is a dead end on Lustre-without-GDS.
#
# Bracketed with our phase-3 best (fastsafetensors, files_per_rank=8) FIRST and
# LAST: the same config has measured 36.9-42.7 s here and 38.2-40.0 s in phase 3,
# so the bracket is what licenses any claim about a treatment point.
#
# NODE REUSE IS SAFE HERE, and seeding a warm-node exclude list is actively
# harmful. A first attempt seeded --exclude with all 7 nodes that had read the
# model today; bristen only has ~30 GPU nodes, the rest were allocated, and the
# only idle nodes left (nid001883/001887) are Gres=(null) - CPU-only. The sweep
# starved on (Resources) with "idle" nodes on screen.
#
# Both loaders under test are provably page-cache-immune, so the exclusion buys
# nothing:
#   - fastsafetensors: phase 3's bracket proved it. fpr1 on a COLD node = 65.0 s;
#     fpr1 on the SAME node after it had read the model 4x = 69.3 s. No speedup.
#   - InstantTensor: `it_default` (74768) ran on nid002324 - a node that had read
#     the model FIVE times during phase 3 - and still crawled at 0.375 GB/s. If
#     page cache were feeding it, it would have been fast. It also selects
#     Backend.URING, and the enum lists URING_BUFFERED separately, so URING is
#     the direct-I/O variant.
#
# Both read O_DIRECT. Neither fills nor uses the page cache. So we do NOT
# exclude - we just RECORD the node for every point, so any reuse stays
# auditable after the fact. (Set EXCLUDE_NODES=... to force exclusions back on.)
#
# Run from the repo root:
#   bash lustre-loading-exp/scripts/phase5_instanttensor/phase5_submit_knob_sweep.sh
set -euo pipefail

S="lustre-loading-exp/scripts/phase5_instanttensor"
RES="lustre-loading-exp/results/phase5_instanttensor"
mkdir -p "$RES"

USED="${EXCLUDE_NODES:-}"
SEEN_NODES=""
[ -n "$USED" ] && echo "excluding: $USED" || echo "no node exclusions (both loaders are O_DIRECT / cache-immune - see header)"
echo

submit_and_wait() {  # submit_and_wait <tag> <loader> [extra --export vars]
  local tag="$1" loader="$2" extra="${3-}" jid node state
  local exclude_arg=()
  [ -n "$USED" ] && exclude_arg=(--exclude="$USED")

  jid="$(sbatch --parsable "${exclude_arg[@]}" \
        --job-name="p5-$tag" \
        --export=ALL,TAG="$tag",LOADER="$loader"${extra:+,$extra} \
        "$S/phase5_instanttensor.sbatch")"
  printf '%-14s -> %-7s loader=%-16s %s\n' "$tag" "$jid" "$loader" "${extra:-defaults}"

  while squeue -j "$jid" -h -o %T 2>/dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; do
    sleep 15
  done
  node="$(sacct -j "$jid" -X -n -o NodeList | head -1 | xargs)"
  state="$(sacct -j "$jid" -X -n -o State | head -1 | xargs)"
  echo "    -> finished on $node ($state)"
  # Record only - do NOT grow the exclude list (see header: both loaders are
  # O_DIRECT, and excluding warm nodes starved the sweep out of the whole pool).
  SEEN_NODES="${SEEN_NODES:+$SEEN_NODES }$tag=$node"
}

submit_and_wait ctl_fst_first fastsafetensors FILES_PER_RANK=8
submit_and_wait it_c16        instanttensor   IT_CONCURRENCY=16
submit_and_wait it_c32        instanttensor   IT_CONCURRENCY=32
submit_and_wait it_c64        instanttensor   IT_CONCURRENCY=64
submit_and_wait it_c64_d64    instanttensor   IT_CONCURRENCY=64,IT_IO_DEPTH=64
submit_and_wait ctl_fst_last  fastsafetensors FILES_PER_RANK=8

echo
echo "nodes used: $SEEN_NODES"
echo "summarize: python lustre-loading-exp/scripts/analysis/summarize_e2e.py $RES"
