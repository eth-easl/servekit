# Case study — restriping GLM-4.7-Flash on bristen (`/capstor` Lustre)

Can we speed up model loads on bristen by restriping the safetensors off the
canonical `stripe_count=1` layout? We copied GLM-4.7-Flash (48 shards, 59 GiB)
into a sibling wide-striped dir and benchmarked. **Short answer: restriping is
necessary but not sufficient — it can win big or lose big depending on OST
contention, and on this shared production FS the contention dominates.**

## Setup

- **Source:** `/capstor/store/.../zai-org/GLM-4.7-Flash` — `stripe_count=1`,
  `stripe_size=1m` (every shard on a single OST). Canonical, mixed-ownership.
- **Restriped:** `/capstor/store/.../models/restriped/GLM-4.7-Flash` —
  `lfs setstripe -c 8 -S 4m` on the dir, then parallel-copied (58 GiB in 23–36 s,
  sizes verified, every shard confirmed `lmm_stripe_count=8`).
- 160 OSTs on this FS; `-c 8` → 48 shards × 8 ≈ 2.4 engagements/OST (balanced).
- Reader: parallel `dd iflag=direct` (512 MB/stream cap so sweeps finish under
  contention). `agg` = total bytes / wall; cached column = warm page cache.

## Results — the headline is the variance

Two windows, same files, same layout, same node family:

### Clean OST window (job 71668, nid002281) — restriped scales beautifully

| streams | source `c=1` | restriped `-c8 -S4m` | speedup |
|---|---|---|---|
| 1 | 359 MB/s | 188 MB/s | 0.5× (dd single-RPC artifact) |
| 8 | 0.76 GB/s | 0.96 GB/s | 1.3× |
| 16 | 1.34 GB/s | 1.63 GB/s | 1.2× |
| 32 | **0.72 GB/s (collapsed)** | **3.49 GB/s, still climbing** | **4.8×** |

### Contended OST window (job 71672, nid002293) — restriped *regresses*

Source sweep + 3 back-to-back restriped runs (restriped runs agree to ~5% — this
is real, not noise):

| streams | source `c=1` | restriped `-c8 -S4m` (run1 / 2 / 3) | restriped vs source |
|---|---|---|---|
| 1 | 220–244 MB/s | 301 / 238 / 166 MB/s | ~tie |
| 8 | 1.04 GB/s | 0.90 / 1.03 / 0.95 | ~tie |
| 16 | **1.97 GB/s** | **0.24 / 0.26 / 0.26** | **0.13× — 8× WORSE** |
| 32 | <0.09 GB/s (timeout) | 0.47 / 0.50 / 0.48 | — |

**Same restriped files: 3.49 GB/s (clean) vs 0.25 GB/s (contended) at 16 streams
— a 14× swing. That swing is the finding.**

## Why restriping *hurts* under contention

- **Source (`c=1`):** each shard lives on one OST. 16 parallel readers hit 16
  distinct OSTs, each streaming sequentially off its own disk → clean, ~130 MB/s
  per OST, scales to ~2 GB/s.
- **Restriped (`c=8`):** each shard is round-robined across 8 OSTs. 16 readers
  now fan onto 16×8 = 128 OST-engagements across the 160-OST pool. Under load
  you (a) lose per-OST sequential locality (each OST serves 8× more seeks), and
  (b) compete with other tenants on ~8× more OSTs — so a few busy OSTs stall the
  whole `wait`. Result: 0.25 GB/s, 8× worse than source.

The mechanism is the mirror image of the clean-window win: wide striping gives
you a **high ceiling but a low floor**; `c=1` gives a **low ceiling but a sturdy
floor**. On a contended shared FS, the floor is what you live on.

## What this means for the serving stack

1. **`stripe_count=1` is a real bottleneck only for single/few-stream readers.**
   A lone reader is capped at ~one OST (~250–360 MB/s here) and cannot exceed it.
   Restriping unambiguously helps *that* case (one rank pulling one shard fast).

2. **Restriping is NOT a robust win on shared `/capstor`.** The realized bandwidth
   is dominated by tenant load, not your `-c`. Do not expect a predictable
   speedup — you may regress. (The 14× run-to-run swing is the proof.)

3. **The robust lever is staging/broadcast (fast-loader §3), not striping (§1).**
   Get the weights *off* the contended OSTs:
   - **Page cache is the real fast tier here:** every cached read measured
     **12–14 GB/s** (DRAM). A 59 GiB model fits easily in the 503 GB node DRAM,
     so **the second read on a node is already at 13 GB/s** — the slow part is
     exclusively the *first* cold read per node. ⇒ read-once-per-node then
     broadcast over the interconnect, or keep replicas warm.
   - bristen compute nodes have **no node-local NVMe** (verified — only tiny Cray
     OS SSDs), so §3-to-local-disk isn't available; the page-cache / broadcast
     path is the available §3 flavor.

4. **If you do restripe, do it by copy-to-scratch (or a `restriped/` sibling),
   never `lfs migrate` in place** — the canonical store is shared (mixed
   ownership; 2 of 48 GLM-4.7-Flash shards aren't owned by xyao), migrate needs
   ownership, and it rewrites every byte anyway. The `-c 8 -S 4m` copy here is
   left in place at `models/restriped/GLM-4.7-Flash/` for anyone to A/B.

## Recommendation for bristen

Don't roll out a blanket restripe of GLM-5.2-FP8 expecting a deterministic win.
Instead:
- **Measure first at the deployment's actual concurrency and time-of-day.** The
  benefit is window-dependent; a single benchmark is uninformative.
- Prefer **read-once-per-node + broadcast / keep-warm** (page cache gives 13 GB/s
  on second read) as the primary cold-start mitigation.
- Keep the restriped copy as an option for the specific case of *few readers,
  clean window* — and re-benchmark in production before trusting it.

## Reproduce

```
# copy + side-by-side bench (restriped data persists at .../models/restriped/)
deploy/bench-storage/restripe_store_casestudy.sbatch
# variance: source sweep + 3x restriped sweep (what surfaced the contention)
deploy/bench-storage/restripe_variance.sbatch   # PER_STREAM_MB=512
```
Output logs: `/capstor/scratch/cscs/xyao/bench-loader/logs/{casestudy-restripe-71669,restripe-variance-71672}.out`
on bristen.
