# Phase 2 (2b) — end-to-end per layout (fastsafetensors, TP=4)

**Status:** done — **striping is a dead end**

## Goal

Confirm the raw-bandwidth finding from
[`../phase2_stripe_sweep/results.md`](../phase2_stripe_sweep/results.md) (native
layout ≥ any striped layout) at the level that actually matters: end-to-end
SGLang cold start, not just dd.

## Method

`scripts/phase2_e2e/phase2_e2e_layout.sbatch` launches the full 70B server
with `--load-format fastsafetensors` against each striped copy from Phase 2a
plus native (bracketed first and last), capturing `weight_loading` + total
via the servekit profile JSON. Feasibility of the write location itself
(group-writable `infra01/cold-start-experiments/`, same Lustre FS as the
model so it's a clean striping comparison) was checked before any copies
were made.

## Result

| layout | weight_loading (s) | total (s) |
|---|---|---|
| native_first | 62.9 | 228.5 |
| c4_s4M | 49.5 | 210.6 |
| c4_s8M | 48.7 | 210.2 |
| c8_s4M | 63.2 | 225.7 |
| c8_s16M | 50.7 | 217.2 |
| c16_s4M | 72.8 | 236.8 |
| native_last | 59.4 | 220.2 |

Raw data: `p2e2e-*.out`, `phase2_e2e-*-profile.json`.

## Verdict

Striped layouts land both above and below the native bracket (59.4–62.9 s)
with no consistent ordering — **noise, not signal**. Expected in hindsight:
Phase 3 (run afterward) found the fastsafetensors loader only kept 4 files
in flight at this point, so layout cannot matter when the loader never asks
the filesystem for enough parallelism to notice it. Combined with Phase 2a's
raw-bandwidth result, this closes the striping question: **do not restripe.**
It requires copying the production model (141 GB per layout), it is not
plug-and-play, and it buys nothing at either the storage or the engine level.

## Caveats

- All seven points ran on a single reused node (nid002313), which violates
  the fresh-node-per-measurement rule elsewhere in this repo. Phase 3 later
  established that fastsafetensors reads O_DIRECT and neither fills nor uses
  the page cache, so these numbers stand — but that was confirmed after the
  fact, not guaranteed by this run's own design.
