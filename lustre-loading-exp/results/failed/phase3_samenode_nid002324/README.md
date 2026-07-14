# Phase 3, first attempt — ALL 5 POINTS LANDED ON nid002324

Quarantined, not deleted. The `--dependency` chain freed the node and SLURM
handed the same one straight back each time, so every point ran on a node that
had already read the model. `--exclusive` grants sole use, not a *different*
node. 515 GB RAM / 402 GB free => the 141 GB model fits in page cache, so the
fresh-node rule (PLAN.md) was genuinely violated.

The data turned out NOT to be contaminated — the closing bracket proves it:

  fpr1_first, node cold           : weight 65.0 s
  fpr1_last,  node had read 4x    : weight 69.3 s   <- no speedup from cache

fastsafetensors reads O_DIRECT/GDS and neither fills nor uses the page cache.
But that was luck, not method. Superseded by the rerun on enforced-distinct
nodes (phase3_submit_chain.sh now accumulates --exclude and submits serially).

Kept because it is the cleanest page-cache probe we have for this loader.
