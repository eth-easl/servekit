# Notes — capstor read bandwidth over 24h

Probe: full 132 GB Llama-3.1-70B (30 shards), O_DIRECT `dd`, 32-worker pool,
5 repeats/slot, one slot per 30 min, fresh node per slot. See `plan.md`.

> **→ `DD_VS_FASTSAFETENSORS.md` is the full write-up of why this probe was
> wrong by 26× and why the loader never noticed.** It supersedes the
> mechanism sections below and adds the fix (`scripts/lib/dd_read_sliced.sh`:
> 0.70 → 9.59 GB/s, same node, same minute). Read it first.

---

## TL;DR — the question in the plan was the wrong question

The plan asked "how does capstor bandwidth vary over a day/night cycle?" and
assumed the answer would be a diurnal contention curve. It isn't.

**capstor's aggregate did not swing. One OST went sick, and `dd`'s aggregate is
a measurement of the sickest OST — not of capstor.**

| | |
|---|---|
| What the campaign showed | bimodal, not diurnal: healthy 6.76 GiB/s median → a **step** to 0.61 at 14:01 that held 4.5 h |
| What the per-shard probe showed | the pool's `wall_s` **is** OST 8's `wall_s`, to within 0.07 s, every repeat, both nodes |
| Culprit | **OST 8**, hosting `model-00018-of-00030.safetensors` |
| capstor excluding OST 8 | **3.0–8.6 GB/s** — the normal healthy band |
| The loader (fastsafetensors) | **never noticed** — 1.52/1.59 GB/s, dead-center of its healthy-day 1.5–1.9 band |

---

## The 24h campaign: bimodal, and the slow mode is the *steady* one

240 samples, Jul 15 18:31 → Jul 16 18:15:

| regime | n | min | median | max | CoV | within-slot max/min |
|---|---|---|---|---|---|---|
| healthy (Jul 15 18:31 → Jul 16 13:33) | 195 | 1.22 | **6.76** | 9.53 | 25.5% | 1.23× |
| degraded (Jul 16 14:01 → 18:15) | 45 | 0.48 | **0.61** | 0.70 | **7.2%** | **1.10×** |

13:33 read 6.66 GiB/s; 14:01 read 0.577. It then held 0.48–0.70 for 4.5 h across
**five different nodes** (nid002324, 002288, 002300, 002293, and both
`model-release-reload-exp` nodes), at ±10% within each slot.

**That shape is the whole clue.** Real contention is bursty — see the 20:01 slot
(1.33 → 5.72 GiB/s within five minutes) or 01:31 (1.22 → 3.57). A step function
that then holds flat for 4.5 h across five nodes is not a crowd. It is a fixed
cost added to every read.

## The per-shard probe: the aggregate *is* one OST

Shards are `stripe_count=1` (phase 1.1), so **one shard == one OST**, and timing
each shard is a per-OST bandwidth map. Jobs **75141** (nid002320) / **75142**
(nid002321), Jul 17 10:45:

```
   job run | pool wall_s  ost8 wall_s | agg reported  agg w/o ost8
 75141   1 |     151.02       150.95  |   0.93 GB/s     2.97 GB/s
 75141   2 |     132.92       132.86  |   1.06 GB/s     5.30 GB/s
 75141   3 |     147.23       147.17  |   0.96 GB/s     8.63 GB/s
 75141   4 |     144.97       144.91  |   0.97 GB/s     6.65 GB/s
 75142   3 |     147.83       147.77  |   0.95 GB/s     8.63 GB/s
```

The pool's wall clock is the **max** over 30 workers, and OST 8 *is* that max —
to within 0.07 s, in all 10 repeats across both nodes. OST 8 read its shard at
**0.029–0.038 GB/s (130–151 s)**; the other 29 shards ran at **0.28–0.53 GB/s
(11–20 s)**. Drop OST 8 and capstor delivers 3.0–8.6 GB/s.

So the reported aggregate carried **no information about capstor**. It was a
stopwatch on one sick disk, divided by 141 GB.

## Why the loader doesn't care — MEASURED (job 75150, `scripts/ost8_queue_depth.sbatch`)

**OST 8's fault is per-request *latency*, not lost capacity.** A `bs` sweep at
QD=1 on the sick shard vs a healthy control (OST 16):

| bs | ost8 MB/s | **ost8 ms/RPC** | ost16 MB/s | ost16 ms/RPC |
|---|---|---|---|---|
| 1M | 2.1 | **476** | 82.0 | 12 |
| 4M | 7.7 | **519** | 140.5 | 28 |
| 16M | 33.4 | **479** | 348.0 | 46 |
| 64M | 81.6 | **784** | 900.7 | 71 |

Throughput tracks `bs` *exactly*: OST 8 charges **~480 ms per request regardless
of size**, ~40× OST 16's 12 ms. Nothing is saturated.

**The client is allowed 68 RPCs in flight (`max_rpcs_in_flight=68`) and
`max_pages_per_rpc=4096` makes a 16 MiB read exactly ONE RPC.** So bandwidth on
OST 8 is just *(RPCs in flight) × 16 MiB / 0.48 s*. Parallel readers on the one
sick file — same O_DIRECT, same bytes, queue depth the only variable:

| readers | MB/s |
|---|---|
| 1 | 24.9 |
| 4 | 179.0 |
| 16 | 321.8 |
| **32** | **697.1** |

**28×, from queue depth alone.** OST 8 has capacity; `dd` never asks for it.

### The two parallelisms — the distinction the probe misses

`dd_read_sweep.sh` runs `xargs -P 30` over **30 different files**. Shards are
`stripe_count=1`, so that is *one reader on each of 30 OSTs* — **queue depth 1
everywhere**, including OST 8. The 29 healthy readers finish in 11–20 s and
exit; nobody helps the straggler. Fan-out **across files** cannot fix a sick
disk. Only fan-out **within a file** can, and `dd` has none.

### fastsafetensors, on the same shard, same node

```
  dd, bs=16M, QD=1  ......  24.9 MB/s   -> 5.0 GB in ~200 s
  fastsafetensors   ...... 213.0 MB/s   -> 5.0 GB in   23.5 s   (8.6x)
```

The mechanism is **Lustre readahead**, not threads — see
`DD_VS_FASTSAFETENSORS.md` §4a, which corrects an earlier claim here that
`max_threads=16` "splits each file across 16 reader threads". It does not:
fastsafetensors submits **one read per `max_copy_block_size` chunk** (default
16 GiB ⇒ one read, one thread, for a 5 GB shard) and opens the file **buffered**
(no `O_DIRECT`), so the client reads ahead to
`max_read_ahead_per_file_mb=160` ≈ **10 × 16 MiB RPCs in flight**. Raising
`max_threads` to 32 on the real loader made weight_loading **worse** (76.75 →
104.01 s). Lustre's in-flight histogram (`rpc_stats`, OST 8) shows the depth
difference regardless of its source:

| queue depth | dd | fastsafetensors |
|---|---|---|
| exactly 1 | **65%** | **36%** |
| cumulative ≤4 | 87% | 62% |
| cumulative ≤8 | 94% | 81% |

**fastsafetensors reads the sick shard in 23.5 s, comfortably inside its 92.8 s
whole-model budget and overlapped with 3 other files. That is why it never
noticed.** Not luck, not page cache — readahead keeps ~10 RPCs in flight where
`dd`'s `iflag=direct` keeps 1. The arithmetic matches the tunable:
160 MiB / 0.48 s = 333 MB/s predicted, 356 MB/s measured at the top of the ramp.

Corollary: on Jul 16 evening `dd` read 0.62–0.70 GB/s while upstream
fastsafetensors read the whole model at **1.52 and 1.59 GB/s on the same nodes
minutes apart** — dead-center of the 1.5–1.9 GB/s band `lustre-loading-exp`
measured on days when `dd` said 6.7–8.6. The loader was flat across an 11× swing
in `dd`'s number, because it never depended on it.

### ⚠️ Buffered reads DO rescue dd — an earlier check here was wrong

This section previously read: *"O_DIRECT 25.2 vs buffered 31.0 MB/s on the sick
shard. Readahead buys 1.2×, not 8×. Buffering is not the explanation."*
**That check read only 128 MiB.** Readahead is **per-fd and ramps**, and 128 MiB
never leaves the first rung of the ramp — it measured the un-ramped floor and
concluded the effect did not exist. Over a long sequential stream on one fd:

```
O_DIRECT  ramp (MB/s per 128MiB):  32   19   31   22    flat
buffered  ramp (MB/s per 128MiB):  30   89  356  263    8.8x
```

Buffered *starts* at O_DIRECT's rate and climbs 8.8× as the window opens.
Buffering **is** the explanation — it is how the queue gets deep.
`DD_VS_FASTSAFETENSORS.md` §4a has the full correction.

## ⚠️ Consequences — two published conclusions are wrong

1. **This experiment's own framing.** There is no diurnal curve here to report.
   Do not plot the campaign's aggregate as "capstor bandwidth vs time of day" —
   most of the y-axis is OST 8's health. `plot_contention.py` plots exactly that
   and should be rebuilt on the per-shard CSVs (median-shard, not aggregate).

2. **`experiments/model-release-reload-exp/NOTES.md`'s correction.** It concludes "capstor
   really was running at ~0.65 GB/s that evening" from "two independent
   experiments, different nodes, agreeing." They are not independent — **both
   instruments are `dd`**. Two `dd` probes agreeing shows `dd` is reproducible,
   not that it is right. The independent instrument is the loader, and it
   disagreed by 2.3×. See "Why the loader doesn't care" above.

**Methodology rule to add to `lustre-loading-exp`'s list:** *an aggregate whose
wall clock is a max over workers is a measurement of the worst worker.* Never
gate on it. `lustre-loading-exp`'s rule #1 ("a bandwidth number from a different
job at a different time is worthless") is necessary but not sufficient — these
probes were contemporaneous and same-node, and still misled.

## Caveats

- **75141/75142 ran concurrently** (2 nodes × 30 streams), unlike the campaign's
  one-slot-at-a-time. Their absolute aggregates are therefore *not* comparable
  to the campaign's, and the two jobs' near-identical per-repeat numbers show
  they were interfering. The straggler finding is unaffected — both nodes
  independently name OST 8 — but do not quote 0.93–1.06 GB/s as a slot value.
- **OST 8 is confirmed for Jul 17, not Jul 16.** The per-shard probe did not
  exist during the campaign. The Jul 16 straggler rate implied by a 220 s wall
  (~21 MB/s) is the same ballpark as Jul 17's 29–38 MB/s, so it is *consistent*
  with the same fault — but unproven.
- OST 8 hosts 1 of 30 shards here. A model whose shards miss OST 8 entirely
  would show no degradation at all, which is worth remembering before treating
  any single model's read time as a storage metric.

## Next

1. **Report OST 8 to CSCS.** Live production fault, still present Jul 17 11:28.
   Evidence: ~480 ms/RPC vs ~12 ms on OST 16 (bs sweep above), `avg_waittime`
   15461 µs vs 4547–6534 µs on OST 0010/0016/0069, identical tunables
   (`max_rpcs_in_flight=68`, `max_pages_per_rpc=4096`), `state: FULL`.
   Reproduce in 20 s from a login node:
   `dd if=$MODEL/model-00018-of-00030.safetensors of=/dev/null bs=16M count=32 iflag=direct`
   (~24 MB/s) vs `model-00017` (~343 MB/s).
2. **Fix the probe's blind spot.** `dd_read_sweep.sh` has fan-out across files
   but queue depth 1 per OST, so it reports the worst OST's latency as if it
   were the filesystem's bandwidth. Either read each shard with N parallel
   offset ranges, or stop treating its aggregate as a storage metric.
3. **Rebuild `plot_contention.py`** on median-shard GB/s + a per-OST heatmap;
   the aggregate line is not worth plotting.
4. Re-run a slot once OST 8 is fixed to confirm the "degraded mode" disappears.

## Bonus: a real cold-start lever — but NOT the one guessed first

The first guess was `max_threads`. **It was wrong and is now measured wrong**
(`fst-threads-exp`): raising it 16 → 32 made weight_loading *worse*, 76.75 →
104.01 s, with the in-job storage probe flat. It is not a read-concurrency knob
— fastsafetensors submits **one read per `max_copy_block_size` chunk** (default
**16 GiB**, so one read and one thread for a whole 5 GB shard); `max_threads`
only caps concurrency that was never created, while eagerly allocating
`bbuf_size_kb × max_threads` of pinned host memory per rank.

The live candidate is **`max_copy_block_size`**. At 256 MB a 5 GB shard becomes
~20 concurrent reads. This should help even on healthy OSTs, because each
thread's loop is `pread → cudaMemcpy → pread` **serially** — with one thread per
file the H2D copy never overlaps the read, which is exactly the ~20 s of
non-overlapped GPU-side work `lustre-loading-exp` phase 3 could not explain away.
Wired as `SGLANG_FST_MAX_COPY_BLOCK_SIZE_MB` in
`experiments/fst-threads-exp/scripts/fst_knobs.patch`; sweep not yet run.
