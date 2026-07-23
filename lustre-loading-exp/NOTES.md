# Lustre-aware model loading — experiment log

Model: `/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct`
(~132 GB, 30 safetensors shards, capstor Lustre). Node: bristen A100, TP=4.

---

## TL;DR

**Cold start for Llama-3.1-70B is dominated by weight loading, and the default
loader is the worst possible choice on Lustre.** Best config found:

```
--load-format fastsafetensors   +   SGLANG_FST_FILES_PER_RANK=8   (12-line patch)
```

| config | weight_loading | total cold start |
|---|---|---|
| mmap — **SGLang's default on Lustre** | 665–939 s | 841–1123 s |
| nommap | 162 s | 348 s |
| fastsafetensors (upstream) | 69–88 s | 242–258 s |
| **fastsafetensors + files_per_rank=8** | **37–43 s** | **208–212 s** |
| InstantTensor (PR sgl#28506, as written) | **352–376 s** ⚠️ | 510–542 s |
| InstantTensor, best tuned (`concurrency=16`) | **244 s** ⚠️ | — |

**≈5.4× faster total cold start** than the default, with identical outputs and
throughput (401–402 tok/s, 0 errors everywhere).

### The one idea behind all of it: reader concurrency

Every loader here is bottlenecked on how many shards it reads **at once**, not
on capstor. Contemporaneous in-job dd probes show capstor delivers **6.7–8.6
GB/s** on the *unmodified native layout*; the loaders were getting 0.35–1.8.

| what it does | files in flight | eff BW |
|---|---|---|
| InstantTensor (bottleneck is a per-tensor sync, not reads) | n/a | 0.36–0.54 GB/s |
| fastsafetensors upstream (TP=4) | 4 | 1.5–1.9 GB/s |
| **fastsafetensors + our patch** | **30** | **3.1–3.6 GB/s** |
| raw dd, O_DIRECT | 30 | 6.7–8.6 GB/s |

### What did NOT work (measured, not assumed)

- **Lustre striping** — native is as fast as or faster than every striped
  layout. Requires copying 141 GB/layout, not plug-and-play, buys nothing.
- **/dev/shm staging** — 198.8 s incl. staging vs 208.2 s. A wash, for 141 GB
  of RAM. Only wins for *warm restarts*.
- **InstantTensor (PR sgl#28506)** — a **9× regression** as written, and still
  **~4× slower than our patch even fully tuned**. It yields per-**tensor** (723
  of them, ~0.34 s each — a per-tensor cross-rank sync) where fastsafetensors
  yields per-**file** (30). Read-concurrency knobs cannot touch that, and its
  headline numbers come from GDS, which bristen does not have (`nvidia_fs` not
  loaded). **Structural, not configurational. Do not adopt.**

### Where the time goes now

With weights at ~38 s, **graph capture is the new bottleneck**: `cuda_graph`
~27 s + `piecewise` ~79 s = **~106 s of the ~208 s total (51%)**. That is where
the next win is.

### Methodology rules that turned out to be load-bearing

1. **A bandwidth number from a different job at a different time is worthless.**
   capstor swings 2–6×. Every job now carries its own dd probe (run *after* the
   measured load, so it cannot warm the OSS cache).
2. **`--exclusive` gives you sole use of a node, NOT a different node.** A
   `--dependency` chain hands you the same node every time. Submit serially and
   accumulate `--exclude`. (Cost us one whole sweep — Phase 3.)
3. **Bracket every sweep with a control, first and last.** The *same* config
   measures 69.5–86.1 s across nodes (21% spread). Node/time variance, not
   drift, is the dominant noise source; a single-point comparison is worthless.

---

## Phase 1.1 — shard→OST map (native layout)  ✅

Ran `scripts/shard_ost_map.sh` → `results/phase1_shard_ost_map.txt`.

- Every shard: `stripe_count=1, stripe_size=1M`.
- 30 shards land on **24 distinct OSTs** (capstor has ≥150 OSTs; indices up to 149).
- **Hot OSTs** (host >1 shard, become the concurrency tail):
  - OST 30 → **3** shards (~13 GB serialized on one OST)
  - OST 15, 46, 59, 93 → 2 shards each
- **Implication**: concurrent multi-shard reads already spread over 24 OSTs, so
  the win from restriping is NOT "use more OSTs" — it's about killing the
  straggler tail on hot OSTs (either per-file striping so each big shard spans
  multiple OSTs, or a balanced re-copy). Phase 2 must test this specifically.

## Phase 1.2 — raw read ceiling (dd O_DIRECT, full-model workload)  ✅ (see CORRECTION)

Tool: `scripts/dd_read_sweep.sh` (fio unavailable → `dd iflag=direct`). Constant
workload = all 30 shards (141 GB); N = worker-pool size draining the shard list.

**Full curve (job 73586, nid002313, COMPLETED, CONC=1 4 8 16 32 64):**
| N | full-model wall_s | agg GB/s |
|---|---|---|
| 1 | 405 | 0.35 |
| 4 | 175 | 0.80 |
| 8 | 81 | **1.75** (peak) |
| 16 | 128 | 1.10 |
| 32 | 209 | 0.68 |
| 64 | 202 | 0.70 |

Earlier partial run (job 73582) hit 0.91 @N=8 and 2.42 @N=16 — totally different.

> ## ⚠️ CORRECTION (Phase 3) — the "ceiling" below is WRONG. Read this first.
>
> The finding recorded here — "capstor peaks around 1.75 GB/s and collapses with
> concurrency" — is **an artifact of one badly-contended sample**, and it sent
> the investigation down the wrong path for a while. It is kept for the record,
> struck through, not deleted.
>
> Measured later, with a **native-layout dd probe run inside each Phase-3 job**
> (same node, minutes after the load, O_DIRECT, N=30), capstor delivers
> **6.7 – 8.6 GB/s on the unmodified native layout** — 4–5× the "ceiling" below.
> Five independent samples: 6.66, 7.44, 8.55, 6.91, 7.29 GB/s.
>
> **Methodological lesson**: a bandwidth number from a *different job at a
> different time* is worthless on shared storage. The probe must run **in the
> same job as the thing it is normalizing**. Every Phase-3 job now carries its
> own probe, and that is what made the loader result interpretable.

### ~~KEY FINDING~~ SUPERSEDED: capstor bandwidth is contention-dominated & non-reproducible
- Single-stream is stable (~0.35–0.41 GB/s) but **aggregate scaling is not**:
  this run *peaks at N=8 (1.75 GB/s) then declines* with more concurrency; the
  earlier run scaled to 5.26 @N=16. The difference is **whole-cluster load on
  the shared capstor OSTs**, which varies minute to minute.
- N=32/64 ≈ N=30 here: one dd worker per shard caps concurrency at the 30 shard
  count. Going beyond needs **striping** (one shard spanning multiple OSTs so a
  single reader drives several OSTs) → motivates Phase 2.
- **Consequences**: (1) don't trust a single dd curve as "the ceiling"; treat
  each as one contended sample. (2) The real lever may be **decoupling from
  capstor contention entirely** — i.e. stage once to node-local /dev/shm
  (Phase 4), after which reads are contention-immune. (3) Compare strategies by
  end-to-end time under realistic (contended) conditions, with repeats.

- Single-stream ≈ **0.38 GB/s** (matches the earlier per-shard run's 0.41).
- **Concurrency scaling collapsed to a ~0.9 GB/s ceiling** here, vs the earlier
  per-shard run that reached 5.26 GB/s at N=16. Single-stream agrees, so the
  difference is **capstor is shared production storage with time-varying
  contention** — the achievable aggregate depends on whole-cluster load.
  → **Methodology consequence**: repeat each layout across multiple submissions/
  times; treat a single run as one contended sample, not the ceiling.

## Phase 2 — feasibility of write location  ✅ (checked, nothing created)

- I'm in group **infra01**; `/capstor/store/cscs/swissai/infra01` is
  `drwxrws--x+` → group-writable + setgid. **Can create
  `infra01/cold-start-experiments/`** (inherits infra01 group). Doesn't exist yet.
- **Right place**: same Lustre FS as the model (clean striping comparison, not a
  capstor-vs-iopsstor difference). Capstor has ≥150 OSTs (shard indices up to
  149); iopsstor scratch only ~20.
- **TO VERIFY before creating copies**: infra01 space quota — `lfs quota`/`df`
  returned nothing usable. Each striped copy ≈ 141 GB × several layouts.
- Code ready (not run): `scripts/make_striped_copy.sh`,
  `scripts/phase2_stripe_sweep.sbatch` (make copy → sweep → delete per layout).

## Phase 1.3 — end-to-end baseline  (mmap on/off + fastsafetensors)

### mmap baseline (job 73605, nid002325) — the anchor
total cold start **840.6 s**; breakdown:
| phase | s | % |
|---|---|---|
| **weight_loading** | **664.7** | **79%** |
| piecewise_cuda_graph_capture | 79.0 | 9% |
| cuda_graph_capture | 29.4 | 4% |
| process_startup | 28.7 | 3% |
| tp_worker_spawn | 16.7 | 2% |
| warmup(JIT) | 15.0 | 2% |
| torch_distributed_init | 2.5 | - |

- **Weight loading = 79% of cold start** for 70B. Effective BW = 132 GB / 665 s
  = **0.20 GB/s** — *slower than one dd O_DIRECT stream* (0.35) and ~9× under the
  contended dd peak (1.75). The mmap demand-paging path over Lustre is the
  culprit. → nommap + multithread + fastsafetensors should win big.
- (Compare Apertus-8B earlier: weight_loading 74 s. 70B mmap is ~9× worse.)

### three-way weight_loading comparison (native layout, TP=4)
| variant | weight_loading | eff BW | total | notes |
|---|---|---|---|---|
| mmap (73605) | 664.7 s | 0.20 GB/s | 840.7 s | nid002325 |
| no-mmap (73606) | 162.5 s | 0.81 GB/s | 331.2 s | nid002313 |
| fastsafetensors (73611) | 81.6 s | 1.62 GB/s | 242.7 s | nid002313, warm, ran last |
| **fastsafetensors (73612)** | **88.1 s** | **1.50 GB/s** | 258.5 s | **nid002804, COLD, ran FIRST** |

**Ranking is robust.** Re-running fastsafetensors on a cold node, first in time
(73612: 88 s) matched the warm/last run (73611: 82 s) within noise → the win is
NOT a page-cache or contention-order artifact. Confirmed:
**fastsafetensors ≈85 s  <  no-mmap ≈163 s  <  mmap ≈665 s.**

- fastsafetensors = **~8× faster than mmap**, ~2× faster than no-mmap on weights.
- Total 70B cold start 840 → ~258 s just by changing the loader (no striping,
  no staging yet). mmap default is by far the worst thing here.
- Even fastsafetensors' ~1.5 GB/s is below the dd contended peak (1.75) and far
  below what /dev/shm could give → Phase 2/4 headroom remains.
- Note: with the loader fixed, the graph-capture phases (cuda_graph 28 +
  piecewise 79 ≈ 107 s) become the *next* biggest chunk of cold start.

### REVERSE-ORDER RERUN + post-ready benchmark (servekit `--bench`)
Run order reversed (fastsafetensors FIRST, mmap LAST), each pinned to a distinct
COLD node, serialized so nothing contends on capstor.

| loader | node | weight_s | total_s | tok/s | p50 s | errs |
|---|---|---|---|---|---|---|
| fastsafetensors (73620) | nid002805 | **73.7** | **237.2** | 402.3 | 5.09 | 0 |
| nommap (73621) | nid002808 | 161.9 | 348.2 | 401.3 | 5.10 | 0 |
| mmap (73622) | nid002809 | **939.0** | 1123.2 | 402.0 | 5.09 | 0 |

**Conclusions (all three confounders now ruled out):**
1. **Ordering/cache confound eliminated.** fastsafetensors ran first (worst
   position if contention eases over time) and still won; mmap ran last and got
   *worse* (939 s vs 664.7 s). Gap is now **12.7×** on the weight phase.
2. **Correctness verified — no weight corruption.** Greedy outputs are
   character-for-character identical across all three loaders (e.g. "The capital
   of France is" and "def fibonacci(n):" produce the same continuations).
   fastsafetensors loads bit-equivalent weights.
3. **Throughput identical (~402 tok/s, p50 5.09 s, 0 errors) across all three.**
   The loader affects only time-to-serving, not runtime performance.
4. **Reproducibility differs sharply**: nommap 162.5 → 161.9 s (rock steady);
   mmap 664.7 → 939.0 s (wild). Demand-paged mmap is the most
   contention-sensitive access pattern on Lustre — slow *and* unpredictable.

**Recommendation so far** *(superseded — see Phase 3 / Phase 5)*:
`--load-format fastsafetensors` is a one-flag, plug-and-play change taking 70B
cold start from ~1123 s (mmap worst case) to ~237 s (**4.7× total**), with
identical outputs and throughput. Never use the mmap default on Lustre.

> Phase 3 goes further: fastsafetensors is itself concurrency-starved (4 files
> in flight at TP=4). Adding `SGLANG_FST_FILES_PER_RANK=8` takes the weight
> phase ~78 → 38.2 s and the total to **208 s (5.4×)**.

## Phase 2 — layout sweep (striping)  ✅ — **striping is a dead end**

### 2a. Raw dd sweep on striped copies vs native

Aggregate GB/s, full 141 GB model, O_DIRECT. Native @N=30 comes from the
Phase-3 in-job probes (Phase 2 skipped N=30 on native — that omission is what
hid this conclusion for so long).

| layout | N=8 | N=16 | N=30 | N=60 |
|---|---|---|---|---|
| **native** | 1.6–2.3 | **3.6–5.5** | **6.7–8.6** | — |
| striped c8_s16M | 2.14 | 3.77 | 7.35 | 7.80 |
| striped c16_s4M | 1.64 | 3.53 | 6.43 | 6.08 |
| striped c8_s4M | 1.92 | 3.26 | 5.86 | 4.96 |
| striped c4_s4M | 2.27 | 4.43 | 4.30 | 5.17 |

**Native is as fast as, or faster than, every striped layout at N=30.** Phase
1.1 already explained why: the 30 shards independently scatter across 24 OSTs,
so concurrent multi-shard reads *already* exploit the whole filesystem. Per-file
striping adds nothing on top of shard-level concurrency.

### 2b. End-to-end per layout (fastsafetensors, TP=4)

| layout | weight_loading | total |
|---|---|---|
| native_first | 62.9 s | 228.5 s |
| c4_s4M | 49.5 | 210.6 |
| c4_s8M | 48.7 | 210.2 |
| c8_s4M | 63.2 | 225.7 |
| c8_s16M | 50.7 | 217.2 |
| c16_s4M | 72.8 | 236.8 |
| native_last | 59.4 | 220.2 |

Striped layouts land both above and below the native bracket (59–63 s) with no
consistent ordering — **noise, not signal**. Expected in hindsight: the loader
only kept 4 files in flight (see Phase 3), and layout cannot matter when you
never ask the filesystem for enough parallelism to notice it.

> **CONCLUSION: do not restripe.** It requires copying the production model
> (141 GB per layout), it is not plug-and-play, and it buys nothing. The
> bottleneck was never the layout.

(Caveat, recorded honestly: all 2b points ran on nid002313 — a reused node,
violating the fresh-node rule. Phase 3 later proved fastsafetensors reads
O_DIRECT and neither fills nor uses the page cache, so these numbers stand. It
was luck, not method.)

## Phase 4 — Strategy B (/dev/shm staging)  ✅

Staged from the c8_s16M striped copy with a **60-worker** parallel copy script,
then served with `--model-path /dev/shm/llama70b`.

| variant | stage | weight_loading | server total | **e2e incl. staging** |
|---|---|---|---|---|
| **shm + mmap** | 21.2 s @ 6.65 GB/s | **19.6 s** | 177.6 s | **198.8 s** |
| shm + fastsafetensors | 21.8 s @ 6.46 | 25.1 | 195.8 | 217.6 |
| shm + nommap | 23.8 s @ 5.93 | 103.6 | 270.4 | 294.2 |

Two inversions worth remembering:

1. **mmap is the BEST loader on tmpfs (19.6 s) and the WORST on Lustre (939 s).**
   The flag you must never use on Lustre is the one you want once the bytes are
   in RAM. Demand-paging is free when there is no disk behind the page.
2. **The staging script beat the engine's own loader at reading Lustre** — 6.65
   vs 1.78 GB/s. Not because tmpfs is magic, but because `cp` ran **60 workers**
   while the loader ran **4**. /dev/shm was never beating the storage; it was
   beating *the loader*. Phase 3 makes that explicit.

## Phase 3 — loader reader-concurrency  ✅ — **the actual bottleneck**

`weight_utils.fastsafetensors_weights_iterator` batches shards `pg.size()` at a
time and hands **one file per rank** (`rank_file_map = {i: [f] ...}`). At TP=4
that is **4 files in flight**, 8 serial batches, each a collective barrier gated
by its slowest file. dd at N=4–8 gives 1.6–2.3 GB/s; the loader gave 1.78. Exact
match — it was concurrency-starved.

**Patch** (`scripts/phase3_loader_concurrency/fst_files_per_rank.patch`, 12
lines): make files-per-rank an env knob, `SGLANG_FST_FILES_PER_RANK`, defaulting
to 1 = byte-for-byte upstream. Round-robins each chunk across ranks.
`rank_file_map` already accepted a list, so no fastsafetensors change is needed.

**Harness** (`patch_sglang_in_container.sh`): each job clones sglang at the SHA
pinned to the image, **diffs the clone against the sglang installed in the
container**, and only then patches and swaps the single file into the live
install. That diff *is* the version check — a drifted pin aborts the job instead
of silently serving different engine code. Nothing depends on the working tree.

### Results (native layout, TP=4, fastsafetensors)

**Jobs 74750–74754, five DISTINCT nodes, none of which had ever read the model.**

| files/rank | in flight | node | weight_loading | eff BW | total | capstor same-job |
|---|---|---|---|---|---|---|
| 1 (upstream) | 4 | nid002292 | 86.1 s | 1.53 GB/s | 258.0 s | 7.43 |
| 2 | 8 | nid002293 | 82.2 s | 1.61 | 254.4 | 8.05 |
| **4** | **16** | nid002296 | **40.0 s** | **3.30** | **212.1** | 7.19 |
| **8** | **30** | nid002297 | **38.2 s** | **3.46** | **208.2** | 7.28 |
| 1 (bracket) | 4 | nid002312 | 69.5 s | 1.90 | 242.2 | 7.91 |

**weight_loading ~78 s → 38.2 s (2.0×); total ~250 s → 208 s.** From a 12-line
patch, on the untouched production model dir. Throughput identical everywhere
(401–402 tok/s), 0 errors.

**Read it against the bracket, not against a single point.** The two upstream
(fpr1) runs give **69.5–86.1 s** — a 21% spread for an *identical config* on
different nodes. Node/time variance is the dominant noise source here, so:

- **fpr2 (8 in flight) lands INSIDE the bracket → no effect.** Doubling from 4 to
  8 buys nothing.
- **fpr4/fpr8 land far BELOW the bracket's best case → real.**
- It is a **step function, not a smooth curve**: nothing until ~16 files in
  flight, then ~2×, then diminishing returns (40.0 → 38.2).

**Drift is not the explanation**: capstor was flat at 7.19–8.05 GB/s across all
five jobs (the in-job probes exist precisely to license this claim).

> **Methodology note — earlier sweep discarded.** A first attempt (jobs
> 74744–74748) put all 5 points on nid002324: a `--dependency` chain frees the
> node and SLURM hands the same one straight back, and `--exclusive` grants sole
> use, NOT a *different* node. It produced a smooth, tidy, and partly spurious
> curve (65 → 49 → 37 → 35.5 s) whose fpr2 point did not survive re-running on a
> clean node (49.0 → 82.2 s). Quarantined in
> `results/failed/phase3_samenode_nid002324/`. `phase3_submit_chain.sh` now
> submits serially and accumulates `--exclude`.
>
> That botched sweep did leave one genuinely useful artifact — the best
> page-cache probe we have:

- `fpr1_first`, node cold: **65.0 s**
- `fpr1_last`, node had read the model **4×**: **69.3 s** — no speedup at all

→ **fastsafetensors reads O_DIRECT/GDS; it neither fills nor uses the page
cache.** (Retroactively validates Phase 2b and the Phase-1.3 warm/cold pair.)
First attempt quarantined in `results/failed/phase3_samenode_nid002324/`;
`phase3_submit_chain.sh` now submits serially and accumulates `--exclude`.

### Why dd is still ~2× faster than the loader — and why that's now fine

dd reads to `/dev/null`. The loader must also copy H2D, **broadcast every tensor
across the 4 ranks over NCCL**, and materialize params — none of it overlapped
with the read. Phase 4 prices that GPU-side work directly, because /dev/shm has
no storage cost: **~20–25 s**.

So of `fpr8`'s 38.2 s, only **~13–18 s is actually reading 132 GB → 7–10 GB/s,
in line with raw dd (7.2–8.1 GB/s measured in the same jobs).** The read is
*solved*. What remains is fixed, non-overlapped
GPU-side work that a wider read batch physically cannot touch — which is exactly
why the curve plateaus and why `fpr8` barely beat `fpr4`.

**This is the argument for InstantTensor (PR sgl#28506).** Its pitch is
*"pipelining and prefetching"* — overlap the read of batch N+1 with the
broadcast of batch N — which attacks precisely the ~20 s this patch cannot. The
next win is structural, not a tuning knob.

## RECOMMENDATION — the deliverable

| strategy | weight_loading | total cold start | plug-and-play? |
|---|---|---|---|
| mmap on Lustre (SGLang default) | 665–939 s | 841–1123 s | — (**never do this**) |
| nommap | 162 s | 348 s | one flag |
| fastsafetensors (upstream) | 69–88 s | 242–258 s | one flag |
| **fastsafetensors + files_per_rank=8** | **38.2 s** | **208.2 s** | one flag + 12-line patch |
| /dev/shm + mmap (60-worker stage) | 19.6 s | 198.8 s (incl. 21 s stage) | needs 141 GB RAM + a stager |
| striped Lustre copy | ~no change | ~no change | ✗ (**abandoned**) |

**Recommendation**

1. **Never use the mmap default on Lustre.** 665–939 s, wildly unpredictable.
   This alone is a 4–5× cold-start regression that current deployments are
   silently paying.
2. **`--load-format fastsafetensors` + `SGLANG_FST_FILES_PER_RANK=8`** — the
   plug-and-play recommendation. 1123 s → 208 s (**5.4× total**), no change to
   how the model is stored, identical outputs and throughput. Upstream the patch.
   (`=4` is worth ~the same: the effect is a step at ~16 files in flight, not a
   smooth curve.)
3. **Ignore striping.** Measured, no benefit, not plug-and-play.
4. **/dev/shm still edges it** (198.8 s incl. staging vs 208.2 s) but costs 141 GB of
   RAM. Its real value is elsewhere: it is contention-immune and it makes *warm
   restarts* nearly free. Revisit only for restart-heavy deployments, or if the
   21 s stage can be overlapped with import+CUDA init (~50 s of cover available).
5. **The bottleneck has moved.** With weights at ~38 s, graph capture
   (`cuda_graph` ~27 s + `piecewise` ~79 s = **~106 s of the ~208 s total, 51%**)
   is now the single largest phase. That is where the next real win is.

## Probe — GDS on bristen: NOT AVAILABLE  ✅ (measured, `scripts/probes/gds_probe.sbatch`)

Matters because InstantTensor's headline numbers (35–45 GB/s) are **GPUDirect
Storage**. If GDS were live here, that would be a different conversation.

**It is not, and this is measured, not inferred:**

1. **`nvidia_fs` is NOT in `/proc/modules`.** Decisive, and not a container
   artifact — containers share the host kernel, so `/proc/modules` is the
   *node's* module list. The nvidia-fs kernel driver is not loaded on bristen,
   so true GDS DMA is impossible regardless of what is bind-mounted. (Also: no
   `/dev/infiniband`.)
2. **InstantTensor selects `Backend.URING` for a real shard on capstor** — not
   `Backend.CUFILE`. The library's own auto-selection answers the question
   directly: it uses io_uring + direct I/O through the CPU.

→ On this system InstantTensor must win on **pipelining + direct I/O alone**,
against our 38.2 s. Expectations set before the run, not after.

- Backend menu: `AIO, AIO_BUFFERED, CUFILE, MMAP, URING, URING_BUFFERED`.
  Sweepable, alongside `concurrency` / `io_depth` (see Phase 5).
- **Trap**: `safe_open(..., load_now=False)` **segfaults** on teardown. The PR
  path uses the `load_now=True` default and is fine, but the library has sharp
  edges.
- **Do not trust a hand-rolled `CUfileDrvProps` ctypes struct** — `cufile.h`
  nests `size_t` fields, so an all-`c_uint` layout is misaligned and prints
  garbage (an early version of this probe confidently reported
  "COMPATIBILITY MODE: False" from junk bytes). Ask the *library* which backend
  it picked instead.

## Phase 5 — InstantTensor (PR sgl#28506)  ⏳

Backported to v0.5.10 (`scripts/phase5_instanttensor/instanttensor_backport.patch`,
4 files: `LoadFormat.INSTANTTENSOR`, `instanttensor_weights_iterator`, the
`loader.py` branch, and the `--load-format` allow-list). The PR targets `main`
and does not apply to v0.5.10, so this is a hand-backport, verified by the same
clone-and-diff harness as Phase 3 (now generalized to multi-file patches in
`scripts/lib/patch_sglang_in_container.sh`, which derives the file list from the
diff itself).

The PR passes **no** I/O knobs — it takes library defaults. Our backport adds
optional `SGLANG_IT_CONCURRENCY` / `SGLANG_IT_IO_DEPTH` pass-through, both
defaulting to `None`, so an unset run reproduces the PR exactly.

**Hypothesis**: Phase 3 fixed the *read*; the residual ~20–25 s is
non-overlapped H2D + NCCL broadcast + param copy (priced by Phase 4's /dev/shm
run: 19.6 s with storage cost removed). Pipelining is the only thing that can
hide it.

### 5a. A/B — PR #28506 *as written* is a **9× REGRESSION** here

Bracketed A/B, one variable (`--load-format`), 4 distinct nodes, n=2 per arm.

| point | node | weight | eff BW | total | capstor same-job | tok/s | errs |
|---|---|---|---|---|---|---|---|
| ctl (fst, fpr=8) | nid002292 | **36.9 s** | 3.58 GB/s | 208.4 s | 5.62 | 402 | 0 |
| **it_default** | nid002324 | **352.0 s** | **0.375 GB/s** | 510.1 s | 6.18 | 401 | 0 |
| **it_default2** | nid002289 | **376.2 s** | **0.351 GB/s** | 542.1 s | 7.28 | 401 | 0 |
| ctl (fst, fpr=8) | nid002293 | **42.7 s** | 3.09 GB/s | 211.6 s | 1.54 | 401 | 0 |

Reproduces on two nodes. **Not contention** — capstor was healthy (6.2 / 7.3
GB/s) during both InstantTensor runs. Outputs and throughput are fine, so the
weights load correctly; it is purely slow.

> ### THE TELL
> InstantTensor's effective read bandwidth is **0.35–0.375 GB/s**. Phase 1.2
> measured **single-stream** dd at **0.35–0.41 GB/s**.
>
> **It is reading the 132 GB model with effectively ONE stream.** Same
> reader-starvation disease Phase 3 found in fastsafetensors — far worse.
>
> `instanttensor.safe_open()` exposes `concurrency` and `io_depth`.
> **PR #28506 passes neither.** It inherits a library default that is
> catastrophic on Lustre.

**Why the headline numbers don't transfer**: InstantTensor's 32×/10× claims
(35–45 GB/s) are **GDS on local NVMe**. GDS is unavailable here (Phase 5b), and
it falls back to `Backend.URING`. With the read unaccelerated and untuned, there
is nothing left to carry it.

→ Merging sgl#28506 and setting `--load-format instanttensor` on this platform
would be a **~9× cold-start regression**, not an improvement.

### 5b. Knob sweep — it is **BAD**, not merely unconfigured  ✅

Ran under a heavily contended capstor (the control moved 37 → 60 s for identical
code), so **read these numbers only against their own bracket**, never against
the Phase-3 table.

| point | knobs | weight | eff BW |
|---|---|---|---|
| ctl (fst, fpr=8) | — | **62.6 s** | 2.11 GB/s |
| it_default | none (PR as written) | 352–376 s | 0.36 |
| it_c16 | `concurrency=16` | **244.0 s** | 0.54 |
| it_c32 | `concurrency=32` | **245.0 s** | 0.54 |
| it_c64_d64 | `concurrency=64, io_depth=64` | **276.7 s** | 0.48 |
| ctl (fst, fpr=8) | — | **58.6 s** | 2.25 GB/s |

Control bracket is tight (58.6 / 62.6 s) → the comparison is sound.

Concurrency buys a one-time ~30% (352 → 244 s) and then **flatlines**: 16, 32 and
64 are indistinguishable. **Still ~4× slower than our patched fastsafetensors,
fully tuned.**

> ### CORRECTION to 5a's diagnosis: it is NOT reading with one stream
> The 0.35 GB/s ≈ single-stream-dd coincidence was a red herring. The plateau
> gives the real cause away — look at its progress bar:
> ```
> Loading safetensors using InstantTensor loader: 0/723 [...] ~2-4 it/s
> ```
> **It iterates over 723 individual TENSORS at ~3/s. 723 ÷ 3 ≈ 240 s** — exactly
> what we measure, at *every* concurrency setting. The bottleneck is a fixed
> **~0.34 s per-tensor cost**, almost certainly a per-tensor cross-rank
> broadcast/sync on the TP=4 path.
>
> fastsafetensors yields per **file** (30). InstantTensor yields per **tensor**
> (723). At TP=4 that is 723 collectives instead of 8. Read concurrency cannot
> touch a per-tensor sync — which is precisely why turning it up does nothing.

**VERDICT: InstantTensor is a dead end on this platform.** The failure is
structural, not configurational, and its headline numbers come from GDS, which
bristen does not have. Do not adopt sgl#28506 here.

*(Caveat, recorded: `it_c64` was killed by an external SIGTERM at 14:36:40 —
not a timeout, not an OOM, unattributed. It does not change the verdict:
`it_c32` (245 s) and `it_c64_d64` (277 s) straddle it, and the plateau is
already established by c16 ≈ c32.)*

## Phase 4b — the sliced stage is CPU-bound, and `srun -c 64` was halving it  ✅

The sliced stager (`stage_to_shm_sliced.sh`, 64 slices, 1744 concurrent `dd`s)
measured **19.4–20.2 GB/s** in `stage_compare.sbatch` but only **11.5–12.2 GB/s**
in every e2e job that used it. Same script, same slices, same model — and on
**nid002297 five minutes apart** (75166 sliced leg 13:20 → 19.36; 75168 stage
13:25 → 11.66), so node variation and Lustre drift were already excluded.

Two things differed between those runs, not one:

- **CPU count.** `stage_compare.sbatch` calls the stager *directly in the batch
  step*. Under `--exclusive` that step gets the whole node — `sacct` confirms
  `75166.batch AllocCPUS=128` despite `ReqCPUS=32`. The e2e runs it under
  `srun --cpus-per-task=64`, which appears as step `.0` with **AllocCPUS=64**.
- **The container** (`--environment=$EDF`).

`scripts/phase4_shm/stage_isolate_container.sbatch` (job 75713, nid002280)
separates them — one job, one node, four legs, minutes apart:

| leg | CPUs | container | GB/s | stage |
|---|---|---|---|---|
| `batch_bare` | 128 | no | **20.54** | 6.87 s |
| `srun_bare_64` | 64 | no | 12.07 | 11.69 s |
| `srun_ctr_64` | 64 | yes | 11.59 | 12.18 s |
| `srun_ctr_128` | 128 | yes | **18.35** | 7.69 s |

> **KEY FINDING: it is CPU count, not the container.** At fixed 64 CPUs the
> container costs 12.07 → 11.59 = **4%**. At fixed container, 64 → 128 CPUs is
> 11.59 → 18.35 = **1.58×**. The stage is CPU-bound: O_DIRECT reads DMA into the
> user buffer, but all 141 GB are then **memcpy'd by the CPU** into tmpfs pages —
> ~40 GB/s of memory traffic at these rates.

`/dev/shm` came back byte-identical across all four legs (same tmpfs, same
`size=369160116k`; container adds only `nosuid,noexec`), so a differing tmpfs
mount is ruled out as well.

Mechanism, worth remembering: `TaskPlugin=task/affinity`, so srun restricts via
`sched_setaffinity`, **not** the cgroup cpuset. Every leg printed
`cpuset: 0-127` while `nproc` read 128/64/64/128. The affinity mask is inherited
by children, so all 1744 forked `dd`s are confined to it. Reading
`cpuset.cpus.effective` alone would have shown "no restriction" and missed this
entirely.

**RECOMMENDATION (not applied — recipe, not a code change).** The e2e scripts
(`phase4_shm_sliced_e2e.sbatch`, `shm-weight-loading-exp` phase1) declare
`#SBATCH --cpus-per-task=64` and the inner `srun` passes `${SLURM_CPUS_PER_TASK}`
through. Raising the **job-level** directive to 128 takes the stage
**~12.2 s → ~7.7 s, ≈4.5 s of cold start for free**. It must be the job-level
value: the `srun_ctr_128` leg only ran because `--exclusive` had 128 CPUs
available, and SLURM warned `Job step's --cpus-per-task value exceeds that of
job (128 > 64). Job step may never run.`

Two caveats: scaling is sublinear (2× CPUs → 1.58×), so 128 is about where this
stops paying and there is no further CPU headroom on the node; and giving SGLang
128 CPUs may shift other cold-start phases (TP worker spawn, torch thread
counts), so the e2e total is not guaranteed to drop by the full 4.5 s.

**Methodological lesson, same family as Phase 1.2's.** A bandwidth number from a
*differently launched step* is worth no more than one from a different job. The
first isolation script drafted for this held CPU count fixed at 64 in both legs
and varied only the container — it would have returned "bare ≈ container ≈ 12
GB/s", declared the container innocent, and read as a dead end while the real
variable was never moved. Vary the thing you have not yet excluded.

## Phase 6 (next) — graph capture  ⏳

