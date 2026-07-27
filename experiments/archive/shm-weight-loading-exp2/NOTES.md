# shm-weight-loading-exp2 — running log

Goal and rules: `SPEC.md`. Target is a hard gate: max-over-ranks
`weight_loading` **< 5.00 s**, loading strictly from `/dev/shm`.

## Baselines, corrected without spending a single job (2026-07-26)

servekit reports the first `Load weight end` line it sees, which is whichever
rank finished first. The gating number is the *last* rank. Both are already in
the logs — each rank prints a sub-second `elapsed=` — so
`scripts/gated_weight_loading.py` recovers the true number retroactively.
Full re-scoring of all 40 historical runs: `results/corrected_baselines.txt`.

| run | arm | servekit said | corrected |
|---|---|---|---|
| `b-ctl-76051` | shm, default loader | 20.51 | **20.51** |
| `s-sharded-76115` | shm + `sharded_state`, stock | 10.60 | **11.08** |
| `s-sharded_pin-76116` | + pinned-bounce patch | 11.93 | **14.37** |
| `hp-hugepage-76161` | hugepage, first e2e | 5.34 | **10.07** |
| `hp-hugepage-streamsync-76269` | hugepage, best | 6.84 | **7.03** |

Three findings:

1. **The number to beat is 11.08 s**, not 10.60 s. Under the strict-`/dev/shm`
   constraint that rules out the hugepage route, 5 s is a **2.2x** improvement.
2. **The existing pinned-bounce patch is worse than recorded** — 14.37 s vs
   11.08 s stock, a 3.3 s regression rather than 1.3 s. It is the nearest
   relative of what exp2 must build, so understanding *why* it lost is the
   first design input, not an afterthought.
3. **The error was not uniform, and it flattered exactly the wrong runs.**
   Most runs move < 0.1 s; the ones that move are those with rank imbalance —
   i.e. the designs that introduce per-rank contention. Rank balance is
   itself a signal that a design is not contending: the best hugepage run
   needed no correction at all, its ranks finishing within ~0.2 s.

`max_el` (largest per-rank `elapsed=`) is the figure used. It is exact to
sub-second; its only error term is true begin-skew across ranks, which the
logs bound below 1 s. The alternative last-end-minus-first-begin form also
inherits ~1 s of timestamp quantisation and reads high, so it is reported
alongside but not used.

## Budget arithmetic

Per rank: 35.3 GB in 5.00 s = **7.06 GB/s sustained**.

| | per-rank GB/s | implied phase |
|---|---|---|
| stock `sharded_state` from shm (measured) | 3.2 | 11.08 s |
| **target** | **7.06** | **5.00 s** |
| PCIe Gen4 per-GPU wire, 4 ranks active (exp1 `dma_only`) | 21–24 | ~1.6 s |

The wire has ~3x of headroom over the target. This is a host-feed problem.

## Job 76361 — first measurement: what rate is available at all

`scripts/shm_h2d_bench.py`, 4 concurrent ranks, 8 GiB each, no engine.
Modes `pageable` (models today's loader) / `pinned` (our own reusable buffer,
double-buffered async DMA) / `dma_only` (ceiling), sweeping thread cap, chunk
size, pipeline depth, and — never tested in exp1 — **NUMA locality**, binding
each rank to the node local to its GPU and first-touching its source pages
there. A ~32 GB/s aggregate host-copy ceiling on a node whose DRAM should do
several times that is the signature of cross-socket traffic.

Every mode verifies the GPU buffer against a rank-distinct pattern, per
SPEC.md §5 — a pipelined async copy that recycles a staging buffer early
corrupts weights in a way a throughput number hides.

**Result: NUMA locality is the entire gap, and it is worth 2.8x.**

Per-rank GB/s, 4 concurrent ranks, 8 GiB each (`results/e2-h2dbench-76361.out`):

| mode | threads | NUMA | per-rank GB/s | slowest | projected phase |
|---|---|---|---|---|---|
| `pageable` | 16 | no | 5.34 7.39 5.43 4.94 | 4.94 | 7.14 s |
| `pinned` | 8 | no | 3.94 3.67 3.68 4.86 | 3.67 | 9.61 s |
| `pinned` | 16 | no | 2.89 2.92 3.68 2.88 | 2.88 | 12.26 s |
| `pinned` | 32 | no | 3.55 3.76 3.91 3.54 | 3.54 | 9.96 s |
| `pinned` | 16 | no, 256M chunks | 3.71 4.04 4.40 3.63 | 3.63 | 9.72 s |
| `pinned` | 16 | no, 1G chunks | 2.87 2.95 3.21 2.91 | 2.87 | 12.29 s |
| `pinned` | 16 | no, depth 3 | 2.88 3.02 3.06 2.88 | 2.88 | 12.25 s |
| **`pinned`** | **16** | **yes** | **9.94 9.98 10.02 10.01** | **9.94** | **3.55 s** |
| `pageable` | 16 | yes | 8.12 8.08 8.12 8.08 | 8.08 | 4.37 s |
| `dma_only` | 16 | no | 24.22 24.45 23.78 24.33 | 23.78 | 1.48 s |
| `dma_only` | 16 | yes | 26.74 26.74 26.73 26.74 | 26.73 | 1.32 s |

All cells passed the bit-exactness check.

Four things fall out:

1. **Both NUMA-bound modes clear the 5 s bar** in projection — `pinned` at
   3.55 s and even plain `pageable`, the copy the stock loader already does,
   at 4.37 s. Unbound, nothing gets close.
2. **Without NUMA, explicit pinned staging is *worse* than pageable** (2.88
   vs 4.94 GB/s). That is the missing explanation for `sharded_pin`'s 14.37 s
   vs 11.08 s stock: the patch adds an explicit host gather, and a host
   gather across NUMA nodes is precisely the thing that costs. The patch
   wasn't wrong, it was unbound.
3. **Every unbound knob is noise.** Threads 8/16/32, chunk 256M/512M/1G,
   depth 2/3 all land between 2.9 and 4.9 GB/s with no clean ordering, and
   the exp1 thread-capping result does not reproduce. Tuning the wrong
   variable produced a scatter that looked like signal.
4. **Rank balance tracks binding.** Unbound: 5.34/7.39/5.43/4.94, a 1.5x
   spread. Bound: 9.94/9.98/10.02/10.01, uniform to 1%. Same signature as
   the corrected historical numbers, where only imbalanced runs moved.

Topology (this had never been looked at): single-socket **AMD EPYC 7713**,
64 physical cores / 128 threads, in **NPS4** — 4 NUMA nodes of ~128 GB each,
16 cores apiece. GPU→node is **reversed**: GPU0→node3, GPU1→node2,
GPU2→node1, GPU3→node0. Anything that assumes an identity mapping binds every
rank to the farthest node.

## Root cause: SGLang's own NUMA binding was silently disabled all along

SGLang v0.5.10 already does this automatically —
`srt/utils/numa_utils.py`, `numactl --cpunodebind=N --membind=N` on each
spawned TP worker, node queried from NVML. It never ran. Every historical
log, on all 4 ranks:

```
NUMA affinity is already constrained for process, skipping NUMA node
configuration for GPU. Remove your constraints to allow automatic configuration.
```

`_is_numa_available()` compares the process's CPU affinity mask against all
CPUs and gives up if it is constrained. The sbatch asks for
`--cpus-per-task=64` on a 128-logical-CPU node, so SLURM hands the container a
64-CPU mask, and the check trips — on every run of every prior experiment.

So the first thing to try needs **no patch and no engine flag at all**: give
the job all 128 CPUs and let SGLang bind itself.

Two halves to the effect, and they are separable:
- **process side** — rank CPUs and its pinned buffers on the local node. Fixed
  by removing the affinity constraint.
- **page side** — the tmpfs pages themselves. Placement is decided by whoever
  *writes* them, i.e. the stager, long before any rank binds. Fixed by
  `scripts/stage_to_shm_numa.sh`, which runs each `model-rank-R-part-*` writer
  under `numactl --membind=<node local to GPU R>`. The sharded_state layout
  makes this clean: rank R's bytes are already in their own files.

## Jobs 76362/76363/76364 — e2e, three arms

| arm | CPUs | stager | isolates |
|---|---|---|---|
| `ctl` | 64 | plain | today's configuration; produces the golden answers |
| `numaproc` | 128 | plain | process side only |
| `numaboth` | 128 | NUMA-aware | both halves |

`scripts/exp2_verify.patch` adds the SPEC §5 bit-exactness gate on every run,
hooked in immediately *after* model_runner's `Load weight end` log so the
`elapsed=` it validates is already computed and cannot be inflated by it.

| arm | gated `weight_loading` | throughput | bit-exact |
|---|---|---|---|
| `ctl` (76362) | 11.24 s | 401.5 tok/s, 0 err | 4/4 PASS |
| `numaproc` (76363) | 9.41 s | 402.3 tok/s, 0 err | 4/4 PASS |
| `numaboth` (76364) | **8.47 s** | 401.5 tok/s, 0 err | 4/4 PASS |

`ctl` reproduces the corrected 11.08 s baseline (11.24 s), so the harness is
sound. Removing the CPU-mask constraint is worth 1.8 s on its own and costs
nothing — the log confirms `numactl --cpunodebind=3 --membind=3` for GPU0
through `=0` for GPU3, matching the true reversed topology. NUMA-aware
staging is worth another 0.9 s.

The child ranks still print the "already constrained" warning, but that is
now expected and harmless: `numactl` has already bound them, so the in-child
check correctly declines to bind a second time.

## Jobs 76365-76372 — strategy bench on the real checkpoint

`scripts/shm_loader_bench.py`: one rank's real shards, 4 ranks concurrently,
no engine, every cell bit-exact-verified. Per-rank seconds for 35.3 GB.

| strategy | gated s | GB/s | note |
|---|---|---|---|
| `safe_open` (what stock does) | 7.08 | 5.00 | `get_tensor` materialises a host copy |
| `zerocopy` (mmap + `frombuffer`) | 5.51-5.63 | 6.3-6.5 | removes that copy |
| `pinned`, gather=torch | 4.80 | 7.36 | + reusable pinned buf, double-buffered DMA |
| **`pinned`, gather=mt/4** | **4.46** | **7.95** | multi-threaded `memmove` gather |
| `pinned`, gather=mt/8 | 4.48 | 7.88 | |
| `pinned`, gather=mt/16 | 5.00 | 7.06 | past the memory-channel wall |
| `pinned`, gather=memmove (1 thread) | 7.12 | 4.96 | |
| `register` + direct DMA | 13.2-15.5 | 2.5 | **reg 9.7-10.4 s, DMA 1.33 s** |

Two results decide the shape of everything:

**In-place registration is dead, now measured rather than assumed.** The DMA
straight out of a registered tmpfs mapping takes **1.33 s** — 26.5 GB/s, exact
line rate, 3.4x faster than the best bounce. But registering 35.3 GB of 4 KB
tmpfs pages costs 9.7-10.4 s even NUMA-local, and **it does not parallelise
at all**: 4 threads gave 9.5-10.8 s, 8 threads 9.3-10.5 s. The driver
serialises it. There is no arrangement in which a 10 s registration hides
behind a 1.33 s transfer.

**The bounce is memory-bandwidth-bound at ~8 GB/s per rank.** Each rank sits
on one NPS4 node with 2 DDR4-3200 channels (~51 GB/s peak). The bounce costs
~3x traffic per byte: read tmpfs, write pinned, DMA reads pinned. At 7.95
GB/s that is ~24 GB/s of traffic — roughly the practical ceiling for a copy
on 2 channels. Thread count past 4-8 makes it worse, not better, which is
the signature of a bandwidth wall rather than a core shortage. Realistic
headroom left: maybe 1.2x.

Also worth recording: the `register` strategy first failed with
`cudaErrorHostMemoryAlreadyRegistered` on file 3 of 7, every rank. Cause was
mine, not CUDA's — each `mmap` object fell out of scope, the kernel handed
the same address to the next one, and registration hit the stale range. Hold
every mapping alive for the duration.

## Jobs 76376-76378 — the exp2 loader, end to end

`exp2_loader.patch` + `exp2_shm_loader.py`: zero-copy views, pinned
double-buffered staging, mt/4 gather, thread cap from the affinity mask.

| arm | gated | throughput | bit-exact |
|---|---|---|---|
| `zerocopy` (76376) | 8.63 s | 401.8 tok/s, 0 err | 4/4 PASS |
| `exp2` (76377) | **7.96 s** | 402.1 tok/s, 0 err | 4/4 PASS |
| `exp2` + timing (76378) | 7.87 s | — | 4/4 PASS |

**11.24 s -> 7.87 s, all three SPEC §5 gates passing on every run.**

But job 76378's instrumentation shows why that is where it stops:

```
Load weight begin -> Load weight end     7.87 s
  exp2 copy_elapsed                      5.75 s
  everything else in the window          2.12 s
```

The 2.12 s is model construction (`_initialize_model` under
`with torch.device("cuda")`, allocating ~35 GB of parameter tensors) plus
`_filter_subtensors(model.state_dict())`. It is inside the measured window,
it is not a copy, and **no loading optimisation can touch it**.

So the budget for the copy is 5.00 - 2.12 = **2.88 s**, against a measured
bounce floor of 4.46 s (bench) / 5.75 s (in-engine). The bounce cannot get
there, and the one path that could — direct DMA at 1.33 s — is gated behind a
registration that costs 10 s on 4 KB pages and refuses to parallelise.

## Where this stands against the gate

Under SPEC §3.1 as written (strict `/dev/shm`, 4 KB tmpfs pages), **< 5.00 s
is not reachable**, and the blocking term is identified and measured rather
than inferred. Escalating per SPEC §1 rather than settling.

## Directions worth trying next

Budget to keep in mind: window = copy + 2.12 s of non-copy work. Copy must
reach **2.88 s** for a 5 s window, against a measured bounce floor of 4.46 s
(bench) / 5.75 s (in-engine). Nothing below is implemented — this is the
ranked list, cheapest and least invasive first.

**No constraint relaxed:**

- **Close the 1.3 s in-engine copy gap** (5.75 s in-engine vs 4.46 s bench,
  same code path, unexplained). Prime suspect: the bench streamed 512 MiB
  chunks, while the engine copies 483 separate tensors averaging 73 MB, and
  the double-buffer `evs[k].synchronize()` fires *per tensor* — so every
  tensor smaller than the chunk pays a full pipeline stall. Fix is to pack
  multiple tensors into one pinned chunk and sync per chunk, not per tensor.
  Free, no constraint change, and it is the single largest identified loss.
- **Decompose and attack the 2.12 s.** Never broken down — it is
  `_initialize_model` under `with torch.device("cuda")` plus
  `_filter_subtensors(model.state_dict())`, and nobody has measured the
  split. If it is dominated by ~35 GB of `cudaMalloc`, then
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a one-line test.
  Every second here is a second off the budget that no copy work can buy.
- **`cudaHostRegisterReadOnly` (flag 0x08).** Registration was measured only
  with flags=0. The read-only variant skips some write-tracking bookkeeping
  and may be materially cheaper on 4 KB pages. One-line test; if it turns
  9.7 s into 3-4 s the whole hybrid design below becomes comfortable.
- **Hybrid register + bounce, registrar started before model construction.**
  The two paths bottleneck on different resources (serialised kernel
  page-pinning vs DRAM bandwidth) and registration does not need the model to
  exist. Registrar claims files from the front from t=0, copier bounces from
  the back from t≈2.12 s: `3.6·T + 7.95·(T−2.12) = 35.3` → **T ≈ 4.5 s**.
  Half-built in `scripts/exp2_prefetch.py`. Main risks: the two may contend
  for CPU more than the arithmetic assumes, and 7 files × 5 GB is coarse
  granularity for a work-split that has to balance.
- **Spread each rank's source pages across 2 NUMA nodes.** Each rank is
  currently confined to one node's 2 memory channels, which is the bounce's
  wall. All 4 nodes are on **one socket** (distances 10/12 — remote is cheap
  here), so a rank reading from 2 nodes while its pinned buffer stays local
  could get 4 channels of read bandwidth. Note the earlier unbound result is
  *not* evidence against this: that was 4 ranks piling onto whatever nodes
  the stager happened to use, not a deliberate 2-node spread. Cheap
  microbench, no engine.
- **Overlap the copy with model construction.** Build layer N's parameters,
  copy layer N's weights while building N+1. Genuine in-window overlap, not
  relocation, worth up to the full 2.12 s. Invasive: touches
  `_initialize_model`, not just the loader.

**Relaxing "strict /dev/shm" (SPEC §3.1):**

- **Hugepage-backed staging behind a `/dev/shm`-shaped interface.**
  Registration on 2 MB pages measured 9.5-17.8 GB/s under 4-way contention vs
  3.6 GB/s here, i.e. registration in ~2-3.7 s with the 1.33 s line-rate DMA
  hidden behind it. `hugepage-sharded-loading-exp` already reached 7.03 s
  with *none* of the NUMA work in this experiment; the two are independent
  and should compose. Highest expected value of anything on this list.
- **Let registration start before the weight-loading window** (e.g. during
  imports or `torch.distributed` init, ~21 s of dead time). Forbidden by
  SPEC §3.2 as relocation, and rightly so for a fair phase number — but if
  the real goal is total cold start rather than the phase, it is close to
  free and would make the registered path (1.33 s DMA) simply win.
- **A staging daemon that outlives the server.** Register once, serve many
  restarts. Turns registration from a per-start cost into a per-node one.
  This is what the fergusfinn.com design does, and it reframes the problem
  from "load faster" to "don't reload".

**Relaxing "valid safetensors" (SPEC §3.5):**

- **One contiguous per-rank blob + offset manifest.** Collapses 483 per-tensor
  copies into a handful of multi-GB transfers, removes all per-tensor Python
  and pipeline-stall overhead, and makes the destination a single flat GPU
  buffer that params view into — which would also shrink the 2.12 s
  construction cost, since parameters become views rather than 483 separate
  allocations. Plausibly attacks the copy gap and the non-copy overhead at
  once, and pairs naturally with the packing fix above.
- **Layout the blob in load order, aligned to 2 MB.** Makes hugepage backing
  and single-shot registration trivially applicable, and lets a rank issue
  effectively one DMA.

**Relaxing "bf16 preserved" (SPEC §3.4):** fp8 halves the bytes and would put
the bounce at ~2.2 s, comfortably inside budget — but it changes model
numerics, so it is a product decision rather than a loading optimisation.
Recording it for completeness, not recommending it.
