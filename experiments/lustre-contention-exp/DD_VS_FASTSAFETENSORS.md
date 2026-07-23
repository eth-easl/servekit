# Why `dd` said 0.6 GB/s while fastsafetensors was fine

Investigation, 2026-07-17, bristen. Everything below is measured; every
comparison is same-node, minutes apart, in one job.

---

## TL;DR

**The `dd` probe was wrong by 26×, and the bug is in the probe, not capstor.**

It stacks two independent artifacts: `iflag=direct` throws away Lustre readahead
(**−6×**, §4b), and its across-files-only fan-out leaves **queue depth 1 per OST**
so one sick disk becomes the whole number (§3, §5).

| | GB/s | full-model wall |
|---|---|---|
| old probe (`dd_read_sweep.sh`, N=30) | **0.70–0.80** | 175–203 s |
| same shape, **just drop `iflag=direct`** | **4.9–5.6** | 25–29 s |
| **sliced probe, O_DIRECT, T=32** | **18.9** | **7.4 s** |
| *(buffered repeat — page-cache-served, NOT storage)* | *19–20* | *7.0 s* |
| fastsafetensors (whole model, TP=4) | 1.5–1.9 | 69–93 s |

`dd_read_sweep.sh` fans out **across files only**. Shards are `stripe_count=1`,
so 30 workers on 30 shards is **one reader per OST — queue depth 1 everywhere**.
One high-latency OST then takes ~200 s on its shard, and since `wall_s` is a
**max over the pool**, that single straggler *becomes* the reported aggregate.

fastsafetensors never suffered because it reads **buffered**, so the Lustre
client **reads ahead** up to `max_read_ahead_per_file_mb=160` — about **10
16 MiB RPCs in flight** instead of `dd`'s 1. `dd`'s `iflag=direct` disables
readahead outright, which is the very thing that was filling the queue.

**capstor was healthy the whole time.**

---

## 1. The trigger: OST 8 is sick

One OST hosts `model-00018-of-00030.safetensors`. Still faulty at 11:34 today:

```
model-00018 (ost 8):   24.1 MB/s
model-00017 (ost 16): 392.9 MB/s      # 16x
```

Confirmed independently by `lctl`: `avg_waittime` **15461 µs** on OST 8 vs
4547–6534 µs on OST 0010/0016/0069. Tunables identical
(`max_rpcs_in_flight=68`, `max_pages_per_rpc=4096`), `state: FULL`. Not a
config difference — the OST is just slow to answer.

## 2. It is a **latency** fault, not a capacity one

`bs` sweep at queue depth 1, sick shard vs healthy control:

| bs | ost8 MB/s | **ost8 ms/RPC** | ost16 MB/s | ost16 ms/RPC |
|---|---|---|---|---|
| 1M | 2.1 | **476** | 82.0 | 12 |
| 4M | 7.7 | **519** | 140.5 | 28 |
| 16M | 33.4 | **479** | 348.0 | 46 |
| 64M | 81.6 | **784** | 900.7 | 71 |

Throughput tracks `bs` almost exactly ⇒ **OST 8 charges ~480 ms per request
regardless of size**, ~40× OST 16's 12 ms. Nothing is saturated; it is a fixed
per-request toll.

Since `max_pages_per_rpc=4096` = 16 MiB, **`dd bs=16M` is exactly ONE RPC**, and
the client is allowed **68** in flight. So bandwidth on OST 8 is simply
*(RPCs in flight) × 16 MiB ÷ 0.48 s*. `dd` uses 1 of its 68 slots.

Parallel readers on the one sick file — same O_DIRECT, same bytes, **queue depth
the only variable**:

| readers | MB/s |
|---|---|
| 1 | 24.9 |
| 2 | 56.5 |
| 4 | 179.0 |
| 8 | 172.9 |
| 16 | 321.8 |
| **32** | **697.1** |

**28× from queue depth alone.** OST 8 has capacity; `dd` never asks for it.

## 3. The distinction that explains everything: two parallelisms

|  | across files | within a file (= per OST) | how |
|---|---|---|---|
| `dd_read_sweep.sh -P 30` | **30** | **1** | O_DIRECT — **readahead disabled** |
| fastsafetensors (upstream, TP=4) | 4 | **~10** | buffered — **Lustre readahead** (§4) |

`xargs -P 30` *looks* like heavy parallelism, but the 30 workers sit on 30
**different** OSTs. OST 8 gets one caller asking one question at a time. The 29
healthy readers finish in 11–20 s and **exit** — nobody helps the straggler.

**Fan-out across files cannot fix a sick disk. Only fan-out within a file can.**

Lustre's own in-flight histogram (`osc.*OST0008*.rpc_stats`) confirms each
reader's depth directly — this is observation, not inference:

| queue depth | dd | fastsafetensors |
|---|---|---|
| exactly 1 | **65%** | **36%** |
| cumulative ≤4 | 87% | 62% |
| cumulative ≤8 | 94% | 81% |

## 4. fastsafetensors on the *same* sick shard

```
dd, bs=16M, QD=1  ......  24.9 MB/s   -> 5.0 GB would take ~200 s
fastsafetensors   ...... 213.0 MB/s   -> 5.0 GB in 23.5 s        (8.6x)
```

**It reads the sick shard in 23.5 s, comfortably inside its 92.8 s whole-model
budget, overlapped with 3 other files.** Not luck, not page cache. The
mechanism is §4a — and it is **not** `max_threads`.

## 4a. The mechanism is Lustre READAHEAD, not thread fan-out

> ⚠️ **This section corrects an earlier claim in this document.** It previously
> said fastsafetensors "splits each file across 16 threads (`max_threads=16`),
> so every OST sees ~16 requests in flight", and that 213 MB/s landing between
> the 8- and 16-reader rungs of §2 was "exactly where `max_threads=16`
> predicts". **That was wrong** — a coincidence, and the reasoning behind it did
> not survive reading the source or testing it. The measurements were right; the
> explanation was not.

**`max_threads` cannot be the mechanism.** From the v0.3.3 source:

- `NoGdsFileCopier.submit_io()` chunks a file by **`max_copy_block_size`** and
  submits **one read per chunk**, one thread each. The default is **16 GiB**.
  Every shard is 5 GB — under 16 GiB — so each file yields **exactly ONE submit
  → ONE thread → one sequential `pread` loop.** `max_threads` only *caps*
  concurrency that `max_copy_block_size` never created; it otherwise just
  `cudaHostAlloc`s `bbuf_size_kb × max_threads` of pinned host memory per rank.
- `os.open(metadata.src, os.O_RDONLY)` — **no `O_DIRECT`**. The reads are
  **buffered**, so the Lustre client reads ahead.

**Tested on the real loader** (`fst-threads-exp`, jobs 75160/75161, fresh node
each, in-job storage probe flat at 5.20 vs 5.41 GB/s):

| max_threads | weight_loading |
|---|---|
| default (16) | **76.75 s** |
| 32 | **104.01 s** — 35% **worse** |

Raising it *hurts*, exactly as "more pinned memory, zero extra reads" predicts.

**The real mechanism, measured** (`scripts/readahead_probe.py` — one fd, long
sequential stream, per-128MiB timing; a fresh `dd` per chunk resets readahead
per-fd, which is precisely how the earlier spot check hid this):

```
SICK shard (OST 8), disjoint 512MiB regions, one fd each:
  O_DIRECT  ramp (MB/s per 128MiB):  32   19   31   22     flat      -> 24.6 MB/s
  buffered  ramp (MB/s per 128MiB):  30   89  356  263     8.8x ramp -> 78.1 MB/s
```

**The buffered stream starts at 30 MB/s — identical to O_DIRECT, window closed —
then climbs to 356 as readahead opens.** O_DIRECT never ramps, because O_DIRECT
has no window to open. The tunable predicts the ceiling:

```
llite.*.max_read_ahead_per_file_mb = 160        # NOT the 64 default
max_pages_per_rpc = 4096 = 16 MiB               # so 160 MiB = ~10 RPCs in flight
160 MiB / 0.48 s  =  333 MB/s                   # vs 356 MB/s measured peak
16 MiB  / 0.48 s  =   33 MB/s                   # vs 24.9 MB/s dd at QD=1
```

A longer stream (1536 MiB, fresh region) holds **146.6 MB/s average, 130–250
steady state** — and fastsafetensors measured **213 MB/s** over the whole 5 GB
file. Same regime. **Everything is consistent with ~10 RPCs in flight from
readahead, and nothing requires threads.**

### Why the earlier "buffered doesn't help" check was wrong

It reported O_DIRECT 25.2 vs buffered 31.0 MB/s — 1.2×, "so buffering is not the
explanation". **It read only 128 MiB.** The ramp above shows 128 MiB never
leaves the *first rung* (~30 MB/s): the readahead window had not opened yet. The
test measured the pre-ramp regime and concluded the effect did not exist.

**The lesson generalises past this repo:** readahead is per-fd and ramps, so any
short read, or any probe that opens a fresh fd per chunk, measures the
un-ramped floor. Both of this document's `dd` probes do exactly one of those.

The top-level conclusion is unchanged and now better supported: **queue depth is
what matters.** What changed is where fastsafetensors' queue depth comes from —
**the kernel's readahead, not the library's threads.**

## 4b. Whole model, one flag: `iflag=direct` alone costs 6–7× (jobs 75163/75164)

§4a proved the mechanism on one shard. This proves it on the **whole 141 GB
model, in the old probe's exact shape** (30 files × 1 reader). The only variable
is `iflag=direct`. Two fresh nodes, both orderings:

| pass | direct_first (75163) | buffered_first (75164) |
|---|---|---|
| **O_DIRECT (cold)** | 175.4 s → **0.80 GB/s** | 180.4 s → **0.78 GB/s** |
| **buffered (cold)** | 29.0 s → **4.86 GB/s** | 25.3 s → **5.58 GB/s** |
| buffered (WARM/RAM) | 7.5 s → 18.89 GB/s | 7.0 s → 20.13 GB/s |

**Removing one flag turns 0.8 GB/s into ~5 GB/s.** Same files, same pool, same
node, minutes apart.

Every pass is validated by `/proc/meminfo`, which is why the numbers can be
trusted rather than merely reported:

- `Cached 5.7 → 5.7 GB` across O_DIRECT — it populates **nothing**, so the
  buffered pass that follows is genuinely cold.
- `Cached 5.7 → 74.0 GB` across buffered — it **does** cache, so a repeat is
  RAM-served.
- The WARM repeat at 19–20 GB/s is the **positive control**: the cache really
  filled, and that is what a page-cache-served read looks like. Any probe
  reporting ~20 GB/s is measuring RAM.
- **Order control kills the OSS-cache objection.** O_DIRECT does not bypass the
  *server*-side cache, so pass 1 could warm pass 2. But buffered ran **faster
  cold-first (5.58) than second (4.86)**, and O_DIRECT is identical either way
  (0.80 / 0.78). The bias, if any, runs against the conclusion.
- **The sharpest single fact:** in 75164 the O_DIRECT pass ran with **71.2 GB of
  the model already in page cache** and still took 180 s at 0.78 GB/s
  (`Cached 71.2 → 71.2`). It refused to read from RAM. That is O_DIRECT doing
  exactly what it says — and why it also refuses readahead.

### The straggler shrinks 6× but survives

```
              slowest shard          median shard    max/min
O_DIRECT       28.5 MB/s (ost 8)       338 MB/s       15.8x
buffered      172.7 MB/s (ost 8)       811 MB/s        5.2x
```

OST 8 is still the wall (28.95 s of a 29.01 s pass): **readahead hides the
latency, it does not fix the disk.** But healthy OSTs gain 2.4× too
(338 → 811 MB/s), so this is not only about the fault.

### The probe stacked TWO artifacts

The old probe gave up readahead (`iflag=direct`, −6×) **and** supplied no queue
depth of its own (one reader per OST → straggler-dominated). Worst of both.

**The fix is not to drop O_DIRECT.** O_DIRECT is *correct* for a cold probe —
buffered reads pollute the page cache and make repeats meaningless (that 20 GB/s
warm pass is what a "fast" contaminated result looks like). The rule is:

> **If you take O_DIRECT, you must supply the queue depth yourself.**

Which is what §5's sliced probe does: 18.9 GB/s, cold, repeatable, no cache.

## 5. The fix: slice each shard (job 75151, one node, one job)

`scripts/lib/dd_read_sliced.sh` keeps the across-files fan-out and adds
**T contiguous slices per shard** (`dd skip=/count=`), so every OST sees T
requests in flight. Slices tile each file exactly (verified: no gaps, no
overlap, dd returns byte-for-byte the full size at every T), so every row reads
the identical 141 GB and `wall_s` stays comparable.

**Control (old probe) and sweep in the same job, same node, minutes apart:**

| probe | slices/file | pool | wall_s | **GB/s** |
|---|---|---|---|---|
| **old probe** (`dd_read_sweep.sh`, N=30) | 1 | 30 | 202.8 | **0.70** |
| sliced, T=1 *(reduces to the old probe)* | 1 | 30 | 192.6 | 0.73 |
| sliced, T=2 | 2 | 60 | 77.2 | 1.83 |
| sliced, T=4 | 4 | 120 | 87.8 | 1.61 |
| sliced, T=8 | 8 | 240 | 38.7 | 3.64 |
| **sliced, T=16** | 16 | 480 | **14.7** | **9.59** |

**13.7× faster, same node, same minute, identical bytes.** T=1 reproducing the
old probe (0.73 vs 0.70) is the control that makes the rest trustworthy.

### It is queue depth, not "more processes"

Pool pinned to **30 readers total** — the same process count as the old probe,
so T>1 means *fewer files in flight but T readers each*:

| slices/file | pool | wall_s | GB/s |
|---|---|---|---|
| 1 | 30 | 198.3 | 0.71 |
| 2 | 30 | 105.4 | **1.34** (1.9×) |
| 4 | 30 | 57.3 | **2.46** (3.5×) |

Same 30 processes. **3.5× purely from where those processes point.** This is the
control that rules out "you just threw more CPUs at it".

## 5b. It does not stop at T=16 (job 75154)

T=16 was the last rung of §5 and was still climbing steeply, so it was pushed
further. Every T also ran the **identical task list with `true` in place of the
read** — the fork-storm floor — because at these speeds the probe starts
measuring itself. 2 repeats each.

| slices/file | pool | tasks | wall_s (r1/r2) | **GB/s** | spawn floor | spawn share |
|---|---|---|---|---|---|---|
| 1 *(old probe)* | 30 | 30 | 196.6 / 189.9 | **0.72 / 0.74** | 0.11 s | 0.1% |
| 16 | 480 | 480 | 14.67 / 11.69 | 9.62 / 12.07 | 0.63 s | 5% |
| **32** | 960 | 919 | 7.45 / 7.42 | **18.94 / 19.02** | 1.25 s | **17%** |
| 48 | 1440 | 1356 | 6.52 / 6.38 | 21.64 / 22.12 | 1.88 s | 29% |
| 64 | 1920 | 1734 | 6.16 / 5.26 | 22.91 / 26.83 | 2.40 s | **45%** |

**The whole 141 GB model in ~7 s, vs 197 s from the old probe minutes earlier on
the same node — a 26× understatement.**

**Read the top rows with care.** T=32 is trustworthy: reproducible to 0.4%,
spawn only 17%. T=48/64 are **lower bounds contaminated by fork overhead**, not
a plateau — at T=64, 45% of the wall is `bash -c`. Naive spawn subtraction
implies 30–50 GB/s, i.e. the curve likely has not flattened at all; we ran out
of `dd`. **Past ~T=32, `dd` measures itself.**

Two caveats on this table specifically:

- **T ran in ascending order, so high-T ran last.** O_DIRECT bypasses the
  *client* page cache but **not** the OSS-side cache, and this job read the same
  141 GB ten times. Later rungs may be partly OST-RAM-served — a systematic bias
  *in favour of exactly the rows claimed as the win*. The T=1↔T=16 step is safe
  (reproduced across two independent jobs, 75151 and 75154); **19 GB/s should not
  be quoted as capstor's number** until T order is randomised or each T gets a
  fresh job.
- **This may be the NIC, not capstor.** The link is Slingshot (`26859@kfi`); a
  single 200 Gb/s port is 25 GB/s, and 22.9–26.8 would be line rate. Link speed
  unverified — "capstor can do 27 GB/s" is not yet a supportable claim; it may be
  "this node's NIC can do 25".

### The consequence for the project

fastsafetensors loads the whole model in 69–93 s at 1.5–1.9 GB/s. capstor hands
over the same bytes in **~7 s**. **Weight loading is not storage-bound and never
was** — storage has ~10–25× more headroom than the loader uses. The cold-start
lever is **loader request concurrency**, not staging, striping, or `/dev/shm`
(all three of which `lustre-loading-exp` measured as no-ops or near-no-ops —
consistent with this, in hindsight).

Followed up in **`experiments/fst-threads-exp/`**. First attempt (`max_threads`) was a
**negative result** — it is not a read-concurrency knob at all (§4a), and
raising it cost 35%. The live candidate is **`max_copy_block_size`** (default
16 GiB ⇒ one read, one thread, per shard): lowering it to e.g. 256 MB turns a
5 GB shard into ~20 concurrent reads. That matters even on healthy OSTs, because
each thread's loop is `pread → cudaMemcpy → pread` **serially** — with one
thread per file, the H2D copy never overlaps the read.

## 6. What this invalidates

1. **The 24h contention campaign's premise.** Its "degraded window" (Jul 16
   14:01–18:15, 0.48–0.70 GiB/s, 45 samples, five nodes) is **not a diurnal
   contention curve** — it is OST 8's latency, sampled 45 times. The tell was
   already in the data: the *slow* mode had CoV **7.2%** vs the healthy mode's
   25.5%. Real contention is bursty; a straggler is metronomic.
   `plot_contention.py` plots this aggregate and should not be trusted.

2. **`model-release-reload-exp`'s "storage was degraded" correction.** It argued
   capstor really was at ~0.65 GB/s because "two independent experiments,
   different nodes, agreeing". **Both instruments were `dd`.** Two `dd` probes
   agreeing shows `dd` is reproducible, not correct. The independent instrument
   was its own loader, which read 1.52/1.59 GB/s on the same nodes minutes
   apart — and *that* number is dead-center of the 1.5–1.9 GB/s band measured on
   days when `dd` said 6.7–8.6. **The loader was flat across an 11× swing in
   `dd`'s number, because it never depended on it.**

3. **`lustre-loading-exp`'s "capstor peaks at 1.75 GB/s" (Phase 1.2).** Already
   struck through there as "one badly-contended sample". This is the mechanism:
   almost certainly the same straggler artifact.

### The methodology rule this earns

> **An aggregate whose `wall_s` is a `max` over workers is a measurement of the
> worst worker.** Never gate on it.

`lustre-loading-exp`'s rule #1 ("a bandwidth number from a different job at a
different time is worthless") is necessary but **not sufficient** — these probes
were contemporaneous *and* same-node, and still misled by 26×.

## 7. Reproduce in 20 seconds

```bash
M=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct
dd if=$M/model-00018-of-00030.safetensors of=/dev/null bs=16M count=32 iflag=direct  # ~24 MB/s (ost 8)
dd if=$M/model-00017-of-00030.safetensors of=/dev/null bs=16M count=32 iflag=direct  # ~343 MB/s (ost 16)
```

Full sweeps:

```bash
sbatch experiments/lustre-contention-exp/scripts/ost8_queue_depth.sbatch      # latency, QD sweep, fastsafetensors, rpc_stats
sbatch experiments/lustre-contention-exp/scripts/slice_sweep.sbatch           # old probe vs sliced, e2e full model
sbatch experiments/lustre-contention-exp/scripts/slice_sweep_deep.sbatch      # T=32/48/64 + fork-storm floor
sbatch experiments/lustre-contention-exp/scripts/buffered_vs_direct.sbatch    # one flag, whole model, cache-accounted
sbatch --export=ALL,ORDER=buffered_first \
       experiments/lustre-contention-exp/scripts/buffered_vs_direct.sbatch    #   ... and the order control
# readahead ramp on one fd (login node, ~30 s):
python3 experiments/lustre-contention-exp/scripts/readahead_probe.py $M/model-00018-of-00030.safetensors 1024 512 16 buffered
python3 experiments/lustre-contention-exp/scripts/readahead_probe.py $M/model-00018-of-00030.safetensors 0    512 16 direct
```

Artifacts: `results/ost8-qd-75150.out`, `results/slice-sweep-75151.out`,
`results/slice-deep-75154.out`, `results/buf-vs-direct-7516{3,4}.out`,
`results/contention_pershard-7514*.csv`, `results/bufdirect_pershard-7516*.csv`.

## 8. Next

1. **Report OST 8 to CSCS.** Live production fault; evidence in §1–2. It is
   silently taxing every job that touches `model-00018`.
2. **Retire or fix the old probe.** It stacks two artifacts (§4b): `iflag=direct`
   throws away readahead (−6×) and its across-files-only fan-out leaves queue
   depth 1 per OST (straggler-dominated). Keep O_DIRECT — it is the only way to
   measure *cold* — but **supply the queue depth explicitly**: `dd_read_sliced.sh`
   at T≥8. `dd_read_sweep.sh` now also takes `PERSHARD_CSV=<path>`, which records
   per-shard/per-OST times and makes a straggler visible instantly.
   **Never "fix" it by dropping `iflag=direct`**: buffered fills the page cache
   and the second repeat of any slot would report ~20 GB/s of RAM as storage.
3. **A possible real cold-start lever.** `max_threads=16` is fastsafetensors'
   *per-file* fan-out; `SGLANG_FST_FILES_PER_RANK` (phase 3) tunes the *other*
   axis. **The two compose: files-in-flight × threads-per-file.** Phase 3 only
   ever tuned the first, and the sliced sweep shows the second is worth 26× to
   raw dd. **`max_threads` was the wrong guess at it and is now a measured
   negative** (`experiments/fst-threads-exp/NOTES.md`); the live candidate is
   `max_copy_block_size`.
4. Re-run the contention campaign once OST 8 is fixed, with the sliced probe, to
   see whether *any* diurnal signal exists underneath. Current best guess: very
   little — capstor delivered 9.59 GB/s at 11:47 on a Friday.

## Caveats

- T=4 (1.61 GB/s) came in below T=2 (1.83). Non-monotonic; single samples on
  shared storage. The trend across T=1→16 is 13×, far outside that noise, but
  no individual rung should be quoted as precise.
- Jobs 75141/75142 (the per-shard runs) were submitted **concurrently** and
  interfered with each other; their absolute aggregates (0.93–1.06 GB/s) are not
  comparable to the campaign's one-slot-at-a-time numbers. The straggler finding
  is unaffected — both nodes independently named OST 8.
- OST 8 is confirmed sick on **Jul 17**. The Jul 16 campaign had no per-shard
  probe, so "the degraded window was OST 8" is *consistent* (its implied
  straggler rate, ~21 MB/s, matches Jul 17's 24–38 MB/s) but strictly unproven.
- fastsafetensors was measured on **one shard** via `SingleGroup()`, not in a
  live TP=4 server. The whole-model loader numbers (1.52/1.59 GB/s) come from
  `model-release-reload-exp` on Jul 16.
