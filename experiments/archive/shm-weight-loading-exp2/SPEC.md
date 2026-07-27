# shm-weight-loading-exp2 — SPEC

**Goal: bring SGLang's `weight_loading` phase under 5 s, loading strictly from
`/dev/shm`, for Llama-3.1-70B TP=4 on Bristen A100.**

Requirements only. No phase plan, no pre-committed technique — the route is
chosen from evidence as it comes in. This document is the contract that
decides whether a result counts.

---

## 1. Objective

| | |
|---|---|
| **Target** | max-over-ranks `weight_loading` **< 5.00 s** |
| **Today's best under this constraint** | **11.08 s** (`sharded` arm, job 76115, corrected — see §4) |
| **Status of the target** | **hard gate — iterate until achieved** |

Not an aspiration and not a best-effort. The experiment continues through
design iterations until the corrected max-over-ranks number is under 5.00 s.

Hitting a wall is not a stopping condition. If one is hit, the response is to
bring the measurement-backed case for relaxing a specific constraint in §3
and then keep going under the revised constraint — never to settle for the
best number achieved and call it done.

### Why 5 s is believed reachable

Per rank, 35.3 GB in 5 s = **7.06 GB/s sustained**. The per-GPU wire is not
the limit: `h2d_microbench.py`'s `dma_only` cell measured **21–24 GB/s per
rank with all 4 ranks copying concurrently** — PCIe Gen4 line rate on each
GPU, ~85 GB/s aggregate. The 23.8 GB/s aggregate that nsys reported for a
real run (job 76112) is what the current pipeline achieves, not a ceiling.

So this is a host-side problem with roughly 3× of headroom on the wire. The
binding constraint is whatever feeds the DMA engine, which is what this
experiment must find and fix.

---

## 2. Fixed configuration

Every number in this experiment comes from exactly this setup, so that all
results are comparable to each other and to the prior experiments' history.

| | |
|---|---|
| Model | `meta-llama/Llama-3.1-70B-Instruct`, bf16, 141.16 GB |
| Parallelism | TP = 4 |
| Node | Bristen, 4× NVIDIA A100-SXM4-80GB, 64 CPU, `--exclusive` |
| Engine | vendored SGLang v0.5.10 (`1519acf37c`), container EDF as in prior runs |
| Server flags | unchanged from the existing e2e sbatch — no new engine flags |
| Source | `/dev/shm`, pre-populated before the engine starts |

Generality to other models or TP degrees is **out of scope**.

---

## 3. Constraints

These are the rules a design must satisfy to count. Each is a candidate for
renegotiation if it turns out to be what blocks 5 s — but only explicitly,
by escalation, never silently.

1. **Strict `/dev/shm`.** The weights the loader reads must live in real
   tmpfs files under `/dev/shm`, openable by path independently by every TP
   rank. Anonymous `memfd(MFD_HUGETLB)` buffers, hugetlbfs mounts, and any
   other non-tmpfs backing are **out of scope for this experiment**, along
   with the fd-broker handoff they require. This deliberately gives up the
   2 MB-page advantage that `hugepage-sharded-loading-exp` exploits.

   Consequence to design around: tmpfs is 4 KB-paged (`shmem_enabled` is
   `[never]`, a root-only global knob), and `cudaHostRegister` on 4 KB tmpfs
   pages measures **2.59–4.56 GB/s** — 7.7–13.6 s to register one rank's
   35.3 GB, which alone exceeds the entire budget.

2. **Genuine speedup, not relocation.** The 5 s must come from making the
   RAM→GPU path faster. Starting transfers early to hide them behind imports
   or `torch.distributed` init, prefetching before the loader is entered, or
   any other move that shrinks the measured phase without shrinking the work
   does **not** count.

3. **Staging is out of the budget, in the report.** `/dev/shm` is assumed
   pre-populated; the ~12 s Lustre→tmpfs stage is not in the 5 s. Every e2e
   run still reports it, so the true end-to-end is never hidden.

4. **bf16 preserved.** Bit-identical weight values to the source checkpoint.
   No quantization, no precision change, as a route to moving fewer bytes.

5. **Valid safetensors on shm.** A one-time offline reshard is allowed and
   the converter may ship with the experiment, but what lands in `/dev/shm`
   must be real, rank-sharded safetensors files that `safe_open` can read
   (`sharded_state` style). Custom raw-blob layouts with sidecar manifests
   are out.

6. **Modest host RAM.** Beyond the 141 GB already in `/dev/shm`, a design may
   hold on the order of a few GB per rank of reusable buffers. A pinned
   mirror of the model, or anything else scaling with model size, is out —
   the result has to be deployable on nodes without large spare RAM.

7. **Engine invasiveness.** Patching the vendored SGLang via the existing
   clone-diff-verify harness is permitted and expected. Env-gate every arm so
   it is switchable without reverting.

---

## 4. The metric

**`weight_loading` = (last rank's "Load weight end") − (first rank's "Load
weight begin"), across all 4 TP ranks.**

This is the gating number: `Capture cuda graph begin` does not start on any
rank until every rank has finished loading. It is *not* what servekit
currently reports.

**servekit is wrong today and must be fixed.** It captures only the first
`Load weight end` line it sees, which is TP0's. Job 76161 shows the size of
the error: servekit reported 5.34 s while TP1–3 finished at ~10.07 s. The
fix — report max-over-ranks — lands in `servekit/` itself, not in this
experiment's directory. This is the one sanctioned exception to §6's
self-containment rule, since the alternative is leaving a known-misleading
number in place for every future experiment.

### The historical numbers, corrected

No reruns were needed. Every rank logs both a timestamp and a sub-second
`elapsed=`, so the gating time is recoverable from the existing logs:
`scripts/gated_weight_loading.py` does it for any run. `max_el` (the largest
per-rank `elapsed`) is the primary corrected figure — it is exact to
sub-second, and its only error term is true begin-skew between ranks, which
the logs bound at under 1 s. The alternative `gated` column (last end minus
earliest derived begin) additionally inherits ~1 s of timestamp quantisation,
so it reads high.

Corrected, for the runs that matter here:

| run | arm | servekit said | **corrected** |
|---|---|---|---|
| `b-ctl-76051` | shm, default loader | 20.51 | **20.51** |
| `s-sharded-76115` | shm + `sharded_state`, stock loader | 10.60 | **11.08** |
| `s-sharded_pin-76116` | + pinned-bounce patch | 11.93 | **14.37** |
| `p4shm-mmap-73665` | shm, mmap | 19.55 | **19.57** |
| `hp-hugepage-76161` | hugepage, first e2e | 5.34 | **10.07** |
| `hp-hugepage-streamsync-76269` | hugepage, best | 6.84 | **7.03** |

Three things follow:

- **The number to beat is 11.08 s, not 10.60 s.** Reaching 5 s is a 2.2×
  improvement, and it must be found without the hugepage technique (§3.1).
- **The pinned-bounce patch is worse than it looked.** 14.37 s vs 11.08 s
  stock — it lost 3.3 s, not 1.3 s. Whatever exp2 builds on the bounce idea
  starts from a worse position than the old numbers implied.
- **The error is not uniform.** Most runs shift by under 0.1 s; the ones that
  move are exactly the ones with rank imbalance, which are exactly the
  designs that introduce per-rank contention. The bug systematically
  flattered the interesting arms.

Note also that `hp-hugepage-streamsync-76269`'s 7.03 s needed no correction —
that design's ranks finish within ~0.2 s of each other. Balanced ranks are
themselves evidence a design is not contending.

---

## 5. Correctness bar

Three independent gates. **All three, on every run, including screening
runs.** A run that misses any of them produces no timing number at all.

**1. Bit-exact weights.** Every parameter resident on every GPU must be
byte-identical to the corresponding bytes in the source checkpoint. Not a
spot check, not a sample — all of them, all 4 ranks.

**2. Answers unchanged.** The greedy completions for the fixed prompt set
must be byte-identical to the recorded baseline set. A stored golden file,
compared automatically; "looks right" does not count.

**3. Performance unchanged.** 64/64 requests, 0 errors, and throughput within
**2 %** of the ~401 tok/s baseline. Prior runs land at 401.0–401.4 tok/s, so
that tolerance is wide relative to observed noise; anything outside it means
the load path changed something it had no business changing.

### How the bit-exact check is run without corrupting the measurement

It runs **post-ready, after the phase timings have already been captured**,
so it cannot perturb the number it exists to validate. Reading weights back
D2H and hashing them against the source costs wall time inside the job
(~141 GB of readback plus a source re-read) but zero time inside the measured
window. Runs stay one-per-node regardless, so the extra minutes are free.

This is deliberately stricter than screening-only checking, and the reason is
the failure mode itself: pipelined async H2D can reuse or unregister a host
buffer while a `cudaMemcpyAsync` is still draining. That corrupts weights
**nondeterministically** — a design can pass gates 2 and 3 on one run and
silently ship wrong weights on the next. Only gate 1, run every time, catches
it.

The standalone microbenchmark layer checksums against the source too, so the
race is caught before it ever reaches an e2e job.

---

## 6. Deliverables and layout

Everything lives in `shm-weight-loading-exp2/`:

```
SPEC.md          this file
NOTES.md         running log: every job, every number, including the dead ends
scripts/         stagers, patches, sbatch, microbenchmarks, analysis
  gated_weight_loading.py    max-over-ranks metric from SGLang logs (§4)
results/         raw job output, profile JSON, stage logs
  corrected_baselines.txt    all historical runs, re-scored with that metric
```

- **Code is copied in**, including anything borrowed from
  `shm-weight-loading-exp/`, `hugepage-sharded-loading-exp/`, or
  `lustre-loading-exp/` — the experiment must be readable in one place.
- **Large data is referenced by absolute path**: the 141 GB sharded
  checkpoint at `/iopsstor/scratch/cscs/yboughizane/sharded-ckpt/llama70b-tp4`
  and the container image. The script that regenerates the checkpoint is
  copied in, so the reference is reproducible rather than magic.
- The `servekit` metric fix (§4) is the sole change outside this directory.

## 7. Measurement protocol

- **One fresh node per end-to-end measurement.** Page cache survives
  container runs; reusing a node invalidates the number.
- **n=1 to screen, n=3 to confirm.** Single runs rank candidate designs;
  three replicates on fresh nodes are required before any number is reported
  as a result rather than a signal.
- **Microbench first.** Design selection happens in standalone
  single-node microbenchmarks (minutes, no engine) wherever it can. Full e2e
  jobs are spent at decision boundaries, not for exploration.
- Every e2e run reports, alongside the metric: stage time, total cold start,
  per-rank load begin/end timestamps, and all three §5 gates explicitly
  pass/fail — never "the benchmark looked fine".
- The **golden answer set and the source tensor hashes** are produced once, by
  the first stock-loader run of this experiment, and committed. Every later
  run compares against those, not against its own neighbours. (This run is
  needed for the golden files regardless; the *timing* baseline is already
  settled at 11.08 s from the corrected logs, so it is not a re-measurement.)

## 8. Definition of done

- max-over-ranks `weight_loading` **< 5.00 s**, n=3 on fresh nodes, all three
  clearing all three gates of §5 — bit-exact weights, unchanged answers,
  unchanged throughput;
- `NOTES.md` records the route taken and the branches killed, with the
  evidence for each;
- the winning design is a single env-gated patch that applies cleanly to
  vendored SGLang v0.5.10 via the clone-diff-verify harness;
- `servekit` reports the corrected max-over-ranks metric.

There is no "best effort" exit. If progress stalls, the deliverable is a
measurement-backed argument for which constraint in §3 is binding and what it
would cost to relax it — brought to you as a decision, after which iteration
resumes under the revised constraint.

---

## Appendix — inherited facts that constrain the design space

Established by prior experiments; re-verify before relying on any of them,
but do not re-discover them from scratch.

**Dead under §3's constraints:**

- *Whole-model `cudaHostRegister` on tmpfs* — 2.59–4.56 GB/s → 31–54 s for
  141 GB. Also dead per-rank (7.7–13.6 s for 35.3 GB), and it cannot be
  hidden behind DMA, since a rank's DMA at line rate is only ~1.7 s.
- *Hugepage-backed tmpfs* — `/dev/shm` is mounted without `huge=`;
  `shmem_enabled` is `[never]` and is root-only. No route from inside the
  container.
- *Read-optimizing load formats* — `fastsafetensors` (25.07 s),
  `runai_streamer`, `--weight-loader-disable-mmap` (103.57 s) all optimize
  reads, and reads are 0.8–9.8 % of this window once the model is in RAM.
  `fastsafetensors` is actively worse from tmpfs: GDS needs a real block
  device, so it falls back to a host bounce and still pays NCCL
  redistribution.
- *Cross-rank broadcast removal* — there is no broadcast. 0.0 % of py-spy
  samples touch `torch.distributed` during the window.

**Live, and known to matter:**

- *OMP oversubscription.* 4 ranks × 64 default threads on 64 cores erased a
  3.67× gather win. Any arm doing a host-side copy must cap threads
  (`OMP_NUM_THREADS` ≈ cores/TP), and the cap interacts with the loader's own
  `num_threads` prefetch pool.
- *The DMA engine is idle ~71 % of the window* (nsys, job 76112): 23.69 s of
  GPU-side H2D across 4 ranks against a 20.3 s phase. The pipeline is serial;
  the win is in keeping the wire busy.
- *Transfers are already large* — 3,956 H2D transfers, median 4.19 MB. Torch
  materializes a contiguous host buffer before the DMA, so a strided narrow
  costs a **host-side gather**, not fragmented DMA. (Moot for
  `sharded_state`, where tensors are contiguous by construction.)
- *`cudaMemcpyAsync` cannot span two separately-registered host regions* —
  `cudaErrorInvalidValue`, reproducibly. Any chunked registration must cut
  only at tensor boundaries.
- *Registration call count, not bytes, appears to be the cost driver* —
  1 GiB chunks (~35 registrations/rank) beat 100 MB chunks (~350/rank) even
  though the small chunks blocked the copy loop less often.
- *`cudaStreamSynchronize` on the rank's own stream* beat
  `torch.cuda.synchronize()` by ~0.5 s. Small, free, keep it.
- *Untested and cheap: NUMA locality.* Nothing has ever pinned tmpfs pages or
  copy threads to the socket local to each GPU. A ~32 GB/s aggregate
  host-copy ceiling on a node whose DRAM should do several times that is the
  signature of cross-socket traffic. This is the first thing I would measure.
- *The container tears down the instant the sglang process tree exits* —
  write artifacts before the server goes down or lose them.
