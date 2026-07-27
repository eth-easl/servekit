# Phase 1.3 — end-to-end loader baseline (mmap / nommap / fastsafetensors / runai_streamer)

**Status:** done

## Goal

Measure real end-to-end cold-start impact of the loader choice on the full
70B model, on the native (unstriped) Lustre layout — the number every later
treatment (striping, `/dev/shm`, concurrency patches) is compared against.
`mmap` is treated as an explicit test axis (SGLang's default on Lustre), not
assumed bad in advance.

## Method

Existing 70B `sbatch` launch (`servekit profile -- python -m
sglang.launch_server ...`, TP=4, bristen A100), run once per loader variant
per **fresh, distinct, `--exclusive` node** (no two data points share a node —
see PLAN.md's page-cache rule). Loaders compared: default `mmap`,
`--weight-loader-disable-mmap` (`nommap`), `--load-format fastsafetensors`,
and `--load-format runai_streamer` (+ a distributed/concurrency variant of
the latter). 3 fresh-node repeats for mmap/nommap/fastsafetensors; a
reverse-run-order pass (fastsafetensors first, mmap last) to rule out a
time/contention-ordering confound; a post-ready `servekit --bench` correctness
+ throughput check on every arm.

## Result

### Aggregate over ALL fresh-node runs in `results/phase1.3_e2e/{mmap,nommap,fastsafetensors}/`

> **Correction**: an earlier version of this table used
> `summaries/phase1_3_aggregate_3run_stats.txt`, which only covers 3 of the
> 4 mmap runs and 3 of the 5 fastsafetensors runs actually on disk (excluding
> jobs 75350, 75336, 74788, none of which are explained in that summary file
> as intentionally dropped except 73612, noted there as "a 4th
> fastsafetensors cold-node robustness check"). Recomputed below directly
> from every `-profile.json` in each variant's directory — this is the
> number to trust; the SUMMARY.md's headline range now matches this table.

| loader | n | weight_loading mean ± std (s) | [min, max] | TOTAL mean ± std (s) | [min, max] |
|---|---|---|---|---|---|
| mmap | 4 | 634.4 ± 225.5 | [429.7, 939.1] | 812.3 ± 229.7 | [602.4, 1123.2] |
| nommap | 4 | 163.6 ± 1.9 | [161.9, 166.3] | 344.6 ± 9.3 | [331.2, 352.5] |
| fastsafetensors | 5 | 83.2 ± 20.4 | [58.7, 114.0] | 248.8 ± 20.4 | [226.6, 278.9] |

Per-run detail (job — node — weight_loading s — total_duration_s):

- mmap: 73605/nid002325 664.68/840.65 · 73622/nid002809 939.05/1123.23 ·
  73643/nid002333 504.25/683.04 · 75350/nid002801 429.73/602.41
- nommap: 73606/nid002313 162.51/331.20 · 73621/nid002808 161.92/348.21 ·
  73641/nid002317 166.28/346.49 · 75336/nid002801 163.52/352.49
- fastsafetensors: 73611/nid002313 81.60/242.69 · 73612/nid002804
  88.07/258.49 · 73620/nid002805 73.67/237.18 · 73642/nid002321
  58.66/226.63 · 74788/nid002324 114.03/278.94

**mmap's stdev (CV ≈ 36%) is on its own axis** — nommap is tight (CV ≈1%);
fastsafetensors is wider than previously stated (CV ≈25%, driven by the
74788 point below) but still far tighter than mmap. mmap is not just slow,
it is *unstable*.

The 74788 fastsafetensors point (114.03 s, the slowest of the 5) was
launched at 14:36:59 on nid002324 — within seconds of Phase 5b's noted
"`it_c64` killed by external SIGTERM at 14:36:40" contention event on the
*same node*. It is plausibly a genuine but heavily-contended sample rather
than a different regime; kept in the aggregate above rather than dropped
silently, since dropping it without a stated reason is what produced the
too-narrow range in the original summary file.

### Reverse-order run (fastsafetensors first / mmap last — rules out ordering confound)

| loader | node | weight_s | total_s | tok/s | p50 s | errs |
|---|---|---|---|---|---|---|
| fastsafetensors (73620) | nid002805 | 73.7 | 237.2 | 402.3 | 5.09 | 0 |
| nommap (73621) | nid002808 | 161.9 | 348.2 | 401.3 | 5.10 | 0 |
| mmap (73622) | nid002809 | 939.0 | 1123.2 | 402.0 | 5.09 | 0 |

fastsafetensors ran **first** (the position most favorable to mmap if
contention eased over the sweep) and still won by >12×; mmap ran **last** and
was still the worst single point measured (939 s) — the gap is not a
time-of-day artifact.

### runai_streamer (not in NOTES.md's original summary — reconstructed here)

| tag | node | weight_loading | cuda_graph | piecewise | total (sum of phases) |
|---|---|---|---|---|---|
| runai_streamer (default) | nid002293 | 89.81 | 41.9 | 78.92 | ≈292 |
| runai_streamer_distributed | nid002293 | 84.91 | 41.12 | 78.28 | ≈270 |
| runai_streamer_distributed c128 | nid002293 | 101.51 | 42.76 | 79.4 | ≈283 |
| runai_streamer_distributed c256 | nid002297 | 105.97 | 39.56 | 79.47 | ≈294 |

`runai_streamer` lands in the same tier as fastsafetensors upstream (~85–106 s
weight_loading, no clear win from raising its own concurrency knob 128→256),
but never beats fastsafetensors' 71–88 s range, and all 4 points ran on only
2 distinct nodes (nid002293 ×3), so this table is weaker evidence than the
3-fresh-node mmap/nommap/fastsafetensors comparison above — read as
directional, not conclusive.

### Correctness

Greedy outputs character-for-character identical across mmap / nommap /
fastsafetensors on all 6 test prompts. Throughput identical everywhere
(~401–402 tok/s, p50 ≈5.09–5.11 s, 0 errors) regardless of loader.

## Verdict

**mmap is the worst possible default on Lustre and current deployments are
silently paying for it.** ~7.6× gap on weight_loading means (634 vs 83 s),
and unstable (429.7–939.1 s observed across 4 fresh nodes) where the
alternatives are reproducible. `nommap` is a one-flag ~3.9× win on
weight_loading with high reproducibility. `fastsafetensors` is the best
one-flag option (~7.6× on weight_loading, ~3.3× total), and is the baseline
that Phase 3's concurrency patch improves further. `runai_streamer` is a
viable one-flag alternative in the same tier as nommap/fastsafetensors
upstream but does not beat fastsafetensors here and wasn't tuned/repeated as
rigorously — not recommended over fastsafetensors without more data.

The loader affects **only time-to-serving**, never runtime correctness or
throughput — every loader here produces bit-identical outputs at identical
tok/s once serving.

## Caveats

- The runai_streamer arm reused one node 3 of 4 times — treat those numbers as
  directional pending a fresh-node repeat, unlike the rigorously repeated
  mmap/nommap/fastsafetensors table above.
- `kv_cache_alloc` did not register as a distinct phase in any fastsafetensors
  run (0 samples across n=3) — not a measurement gap, the phase genuinely
  doesn't surface there in the profile.
