# Phase 3 — loader reader-concurrency (fastsafetensors `files_per_rank`)

**Status:** done — **this is the actual bottleneck, and the core result of the whole experiment**

## Goal

fastsafetensors (Phase 1.3's best one-flag loader) still loads well below
capstor's real raw-read capability. Find out why, and fix it if the fix is
plug-and-play.

## Method

Root cause: `weight_utils.fastsafetensors_weights_iterator` batches shards
`pg.size()` at a time and hands **one file per rank**
(`rank_file_map = {i: [f] ...}`). At TP=4 that's only **4 files in flight**
across 8 serial batches, each gated by a collective barrier on its slowest
file.

**Patch** (`scripts/phase3_loader_concurrency/fst_files_per_rank.patch`, 12
lines): makes files-per-rank an env knob, `SGLANG_FST_FILES_PER_RANK`,
defaulting to 1 = byte-for-byte upstream. `rank_file_map` already accepted a
list, so no fastsafetensors library change was needed, only the sglang call
site.

**Harness** (`scripts/lib/patch_sglang_in_container.sh`): each job clones
sglang at the SHA pinned to the container image, diffs the clone against the
sglang actually installed in the container (that diff *is* the version
check — a drifted pin aborts the job instead of silently serving different
engine code), then patches and swaps the single file into the live install.

**Node discipline**: one job per sweep point via `--wait`, growing an
`--exclude` list so no two points ever land on the same node — this phase
is also where the "fresh node per point" methodology was hardened, after a
first attempt broke it (see Caveats).

## Result

Native layout, TP=4, fastsafetensors, five **distinct** nodes, none of which
had ever read the model before:

| files/rank | in flight | node | weight_loading | eff BW | total | capstor same-job |
|---|---|---|---|---|---|---|
| 1 (upstream) | 4 | nid002292 | 86.1 s | 1.53 GB/s | 258.0 s | 7.43 |
| 2 | 8 | nid002293 | 82.2 s | 1.61 | 254.4 | 8.05 |
| **4** | **16** | nid002296 | **40.0 s** | **3.30** | **212.1** | 7.19 |
| **8** | **30** | nid002297 | **38.2 s** | **3.46** | **208.2** | 7.28 |
| 1 (bracket) | 4 | nid002312 | 69.5 s | 1.90 | 242.2 | 7.91 |

capstor was flat at 7.19–8.05 GB/s (in-job probe) across all five jobs, so
drift is not the explanation for the gains below.

## Verdict

**weight_loading ~78 s → 38.2 s (2.0×); total ~250 s → 208 s.** From a
12-line patch, on the untouched production model directory. Throughput
identical everywhere (401–402 tok/s), 0 errors.

Read against the upstream bracket (69.5–86.1 s, a 21% spread for an
*identical* config on different nodes — node/time variance is the dominant
noise source here), not against a single point:

- `fpr2` (8 in flight) lands **inside** the bracket → no real effect,
  doubling from 4→8 buys nothing.
- `fpr4`/`fpr8` land **far below** the bracket's best case → real.
- It's a **step function, not a smooth curve**: nothing until ~16 files in
  flight, then ~2×, then diminishing returns (40.0 → 38.2 s).

**Recommendation: `SGLANG_FST_FILES_PER_RANK=8`** — one env var + this
12-line patch, on top of `--load-format fastsafetensors`. This is the single
biggest lever found in this whole experiment (see top-level `SUMMARY.md`).

### Why dd is still ~2× faster than the loader (and why that's now fine)

dd reads to `/dev/null`. The loader must also copy H2D, broadcast every
tensor across the 4 ranks over NCCL, and materialize params — none of it
overlapped with the read. Of `fpr8`'s 38.2 s, only ~13–18 s is actually
reading 132 GB (→ 7–10 GB/s, matching raw dd in the same jobs); the rest is
fixed, non-overlapped GPU-side work that no amount of read concurrency can
touch (priced directly by
[`../phase4_shm/results.md`](../phase4_shm/results.md)'s tmpfs run: ~20–25 s).
This is exactly the residual that motivated Phase 5 (InstantTensor)'s
pipelining pitch — which turned out not to pay off; see
[`../phase5_instanttensor/results.md`](../phase5_instanttensor/results.md).

`scripts/fastsafetensor_many_threads/` later tested whether
fastsafetensors' *internal* thread pool (a second, independent concurrency
knob) could shave more off that residual — see
[`../fastsafetensor_many_threads/results.md`](../fastsafetensor_many_threads/results.md).

## Caveats

**A first sweep attempt was discarded and is not in the table above.** Jobs
74744–74748 all landed on nid002324: a `--dependency` chain frees the node
and SLURM hands the same one straight back, and `--exclusive` grants sole
use of a node, **not** a *different* node. That run produced a smooth, tidy,
and partly spurious curve (65 → 49 → 37 → 35.5 s) whose `fpr2` point did not
survive re-running on a clean node (49.0 s there vs 82.2 s in the table
above). The raw data was quarantined and has since been deleted; the lesson
is preserved here: `phase3_submit_chain.sh` now submits serially and
accumulates `--exclude` so no two points ever share a node.

That discarded sweep did leave one genuinely useful artifact — the best
page-cache probe available in this repo:

- `fpr1_first`, node cold: **65.0 s**
- `fpr1_last`, same node, model already read 4×: **69.3 s** — no speedup at
  all

→ **fastsafetensors reads O_DIRECT; it neither fills nor uses the page
cache.** This retroactively validates Phase 2b's node-reuse and the
Phase 1.3 warm/cold reverse-order pair.
