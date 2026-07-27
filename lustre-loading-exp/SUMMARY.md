# Lustre-aware model loading — summary

Model: `Llama-3.1-70B-Instruct` (~132 GB, 30 safetensors shards, capstor
Lustre). Node: bristen A100, TP=4. Full context/methodology: `PLAN.md`. Full
per-experiment detail: `results/<phase>/results.md` (linked below).

## TL;DR

**Cold start for Llama-3.1-70B is dominated by weight loading, and the
default loader is the worst possible choice on Lustre.** Best config found:

```
--load-format fastsafetensors   +   SGLANG_FST_FILES_PER_RANK=8   (12-line patch)
```

All figures below are recomputed directly from every profile JSON on disk
per variant (not a hand-picked bracket) — see each phase's `results.md` for
the full per-run breakdown.

| config | weight_loading (s) | staging (s) | total cold start (s) | n runs |
|---|---|---|---|---|
| mmap — **SGLang's default on Lustre** | 429.7–939.1 (mean 634) | — | 602.4–1123.2 (mean 812) | 4 |
| nommap | 161.9–166.3 | — | 331.2–352.5 | 4 |
| fastsafetensors (upstream) | 58.7–114.0 (mean 83) | — | 226.6–278.9 | 5 |
| **fastsafetensors + files_per_rank=8 (patch)** | **36.0–40.0** | — | **203.7–212.1** | 3 |
| `/dev/shm` staging + mmap (no code change) | 18.6–19.7 | 12–21 (stager-dependent) | 197–208 | 4 |
| `/dev/shm` + TP-presharded checkpoint (serial stage) | 9.6 | 11.9 | **192.6** | 1 |
| **`/dev/shm` + presharded + stage overlapped with startup** | **9.8** | *(hidden — overlapped, not serial)* | **183.4** | 1 |
| InstantTensor (PR sgl#28506, as written) ⚠️ dead end | 352.0–376.2 | — | 510.1–542.1 | 2 |
| InstantTensor, best tuned ⚠️ dead end | 244.0–245.0 | — | — | 2 |

**mmap → fastsafetensors+patch is ≈4× faster on total cold start** (812 → 208
s, mean-to-mean); **mmap → the fully overlapped shm+presharded pipeline is
≈4.4×** (812 → 183 s). Identical outputs and throughput (401–402 tok/s, 0
errors everywhere) across every arm — the loader only ever affects
time-to-serving.

With the shm+presharded+overlap pipeline applied, weight loading falls to
~10 s and **graph capture (`piecewise_cuda_graph_capture` +
`cuda_graph_capture`, ~58% of total) becomes the next bottleneck.**

## State-of-the-art progression (each step's best config → cumulative win)

| step | weight_loading (s) | total cold start (s) | vs. mmap baseline (812 s mean) | plug-and-play? |
|---|---|---|---|---|
| 1. mmap (SGLang default) | 429.7–939.1 | 602.4–1123.2 | 1.0× | — (never do this) |
| 2. nommap | 161.9–166.3 | 331.2–352.5 | ≈2.3–2.6× | one flag |
| 3. fastsafetensors (upstream) | 58.7–114.0 | 226.6–278.9 | ≈2.9–3.6× | one flag |
| 4. **fastsafetensors + `files_per_rank=8`** | 36.0–40.0 | 203.7–212.1 | ≈3.8–4.0× | one flag + 12-line patch |
| 5. `/dev/shm` staging (mmap, no presharding) | 18.6–19.7 (weight only) | ≈197–208 (incl. stage) | ≈3.9–4.1× | needs 141 GB RAM + a stager |
| 6. `/dev/shm` + TP-presharded checkpoint (serial stage) | 9.6 | 192.6 | ≈4.2× | needs offline preshard job + stager |
| 7. **`/dev/shm` + presharded + stage overlapped with engine init** | 9.8 | **183.4** | **≈4.4×** | same as (6), plus overlap logic (not yet hardened — see caveats below) |

Step 5 barely beats step 4 (≈197–208 s vs 203.7–212.1 s) and costs 141 GB of
RAM for it — its clearer win only shows up once combined with presharding
(steps 6–7). Steps 6–7 need an offline one-time preshard job in addition to
the stager, so they're a bigger lift than step 4's single patch; step 4
alone is the recommended plug-and-play default, steps 5–7 are the roadmap
for a deployment that can afford the extra machinery. Step 7's overlap has
no synchronization yet (see
[`results/phase7_overlap_stage/results.md`](results/phase7_overlap_stage/results.md)
Caveats) — it measures the ceiling, not a hardened design.

## Dead ends (measured, not adopted)

| experiment | what was tried | result | why it failed | detail |
|---|---|---|---|---|
| Lustre re-striping | copy the model onto `lfs setstripe -c{4,8,16} -S{4M,16M}` layouts | no benefit at raw-bandwidth or e2e level; native ≥ every striped layout | shards already scatter across 24 of ≥150 OSTs at default striping (Phase 1.1) — nothing left for per-file striping to add | [results/phase2_stripe_sweep/results.md](results/phase2_stripe_sweep/results.md), [results/phase2_e2e/results.md](results/phase2_e2e/results.md) |
| InstantTensor (PR sgl#28506) | backported pipelined/prefetching loader, tuned `concurrency`/`io_depth` up to 64 | 352–542 s total even tuned — 4–9× *worse* than the patched fastsafetensors baseline | structural: yields per-**tensor** (723 collectives) not per-**file** (30); fixed ~0.34 s/tensor sync that no read-concurrency knob touches. Its headline numbers also assume GPUDirect Storage, unavailable on bristen | [results/phase5_instanttensor/results.md](results/phase5_instanttensor/results.md), [results/probes/results.md](results/probes/results.md) |
| fastsafetensors internal `max_threads` (read/copy thread pool) | raised the library's own internal thread pool 16→64→128, alone and combined with `files_per_rank=8` | no measured win in isolation (64.0 s vs the 56.8–61.1 s upstream bracket) or combined (35.8–36.8 s vs `files_per_rank=8` alone's 37–43 s) | the residual time is fixed, non-overlapped GPU-side work (H2D copy, tensor materialization), not a read-throughput limit this knob can touch | [results/fastsafetensor_many_threads/results.md](results/fastsafetensor_many_threads/results.md) |
| Phase 1.2's first dd sweep ("1.75 GB/s ceiling") | raw dd O_DIRECT concurrency sweep on native layout, one job | not a failed *approach*, but a wrong measurement: contradicted by a 4–5× higher number measured properly in Phase 3 | single contended sample treated as a stable ceiling — capstor bandwidth swings 2–6× minute to minute; must probe in the same job as what it's normalizing | [results/phase1.2_dd_sweep/results.md](results/phase1.2_dd_sweep/results.md) |

## Recommendation

1. **Never use the mmap default on Lustre.** 429.7–939.1 s, wildly
   unpredictable (CV ≈36%). A ≈4× mean cold-start regression current
   deployments are silently paying, worse in the tail.
2. **`--load-format fastsafetensors` + `SGLANG_FST_FILES_PER_RANK=8`** — the
   plug-and-play recommendation. ~812 s mean → ~208 s (≈4× total), no
   change to model storage, identical outputs/throughput. Worth
   upstreaming.
3. **Ignore Lustre striping.** Measured at both raw-bandwidth and
   end-to-end level, no benefit, not plug-and-play (needs a 141 GB copy per
   layout).
4. **`/dev/shm` staging + presharding pushes further** (~192.6 s serial,
   ~183.4 s overlapped) but costs 141 GB RAM, a stager, and an offline
   preshard job; overlapping the stage with startup makes it effectively
   free. Real value is warm restarts / restart-heavy deployments, or
   squeezing the last ~25 s once step 4 alone isn't enough.
5. **Do not adopt InstantTensor (PR sgl#28506) on this platform.**
   Structural per-tensor sync cost (723 collectives vs fastsafetensors' 30),
   and its headline numbers depend on GPUDirect Storage, which bristen
   doesn't have.
6. **The bottleneck has moved.** With weight loading fixed, graph capture
   (~106 s of ~208 s, 51%) is the next lever.

## Phases

| phase | question | verdict | detail |
|---|---|---|---|
| 1.1 | Where do shards land on Lustre today? | Already scatter across 24 OSTs at default striping | [results/phase1.1_ost_map/results.md](results/phase1.1_ost_map/results.md) |
| 1.2 | Raw dd read ceiling? | Initial "1.75 GB/s ceiling" was a contended-sample artifact — corrected in Phase 3 to 6.7–8.6 GB/s | [results/phase1.2_dd_sweep/results.md](results/phase1.2_dd_sweep/results.md) |
| 1.3 | End-to-end loader baseline (mmap/nommap/fastsafetensors/runai_streamer) | mmap is worst by ~7.6× on weight_loading mean, unstable; fastsafetensors best one-flag option | [results/phase1.3_e2e/results.md](results/phase1.3_e2e/results.md) |
| 2 | Does Lustre striping help? | No — native ≥ every striped layout, at raw-bandwidth and e2e level | [results/phase2_stripe_sweep/results.md](results/phase2_stripe_sweep/results.md), [results/phase2_e2e/results.md](results/phase2_e2e/results.md) |
| 3 | Why is fastsafetensors still slow? | Concurrency-starved (4 files in flight); `files_per_rank` patch → 2× win | [results/phase3_loader_concurrency/results.md](results/phase3_loader_concurrency/results.md) |
| fastsafetensor_many_threads | Is the internal read/copy thread pool also starved? | No additional win beyond `files_per_rank=8` | [results/fastsafetensor_many_threads/results.md](results/fastsafetensor_many_threads/results.md) |
| 4 / 4b | Does `/dev/shm` staging beat Lustre reads? | Yes on tmpfs mmap flips to best-case; stager itself found to be CPU-bound | [results/phase4_shm/results.md](results/phase4_shm/results.md) |
| 5 | Does InstantTensor's pipelining beat the patch? | No — 4–9× regression, structural per-tensor sync cost | [results/phase5_instanttensor/results.md](results/phase5_instanttensor/results.md) |
| probe | Is GDS available on bristen? | No — `nvidia_fs` not loaded, falls back to io_uring | [results/probes/results.md](results/probes/results.md) |
| 6 | Does pre-sharding + shm staging help? | Cuts weight_loading ~2× but stage cost cancels it if serial | [results/phase6_preshard_shm/results.md](results/phase6_preshard_shm/results.md) |
| 7 | Can the stage be hidden entirely? | Yes — overlapped with startup, stage disappears from critical path (34.5 s slack) | [results/phase7_overlap_stage/results.md](results/phase7_overlap_stage/results.md) |
| reference | Apertus-8B cold-start scale point | weight_loading ≈64–74 s at 8B vs 429.7–939.1 s at 70B (mmap) — motivated this whole investigation | [results/reference/results.md](results/reference/results.md) |

## Methodology rules that turned out load-bearing

1. **A bandwidth number from a different job at a different time is
   worthless on shared storage.** capstor swings 2–6×. Every job that needs
   a bandwidth comparison carries its own dd probe, run *after* the measured
   load so it can't be warmed by it.
2. **`--exclusive` gives sole use of a node, NOT a different node.** A
   `--dependency` chain hands you the same node every time — submit serially
   and accumulate `--exclude` (cost one whole sweep in Phase 3 before this
   was caught).
3. **Bracket every sweep with a control, first and last.** The *same*
   config measures 21% apart across nodes/time — node/time variance is the
   dominant noise source, a single-point comparison is worthless.
