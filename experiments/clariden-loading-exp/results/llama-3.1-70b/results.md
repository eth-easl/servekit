# Clariden (GH200) reproduction — default vs preshard+shm+overlap

**Status:** done — the speedup reproduces, and is **larger** on Clariden than on
bristen: **4.61x** total cold start (bristen: 4.4x), with `weight_loading` cut
**75x** (466.81 s → 6.19 s).

> **Correction (weight_loading).** An earlier version of this file quoted
> `weight_loading` as 466.20 → 3.72 s (125x), taken straight from servekit.
> **servekit reports the FIRST `Load weight end` line, i.e. the FASTEST TP rank.**
> Loading is not finished until the slowest rank is, so the correct figure is the
> max over ranks. Totals below are unaffected — they are wall-clock from the
> wrapper, not derived from these phase values.
>
> | run | per-rank `Load weight end` | servekit | correct (max) |
> |---|---|---|---|
> | default | 466.20 / 466.31 / 466.33 / 466.81 | 466.20 | **466.81** |
> | preshard+shm+overlap | 3.72 / 4.06 / 5.81 / 6.19 | 3.72 | **6.19** |
>
> The error is negligible when loading is slow (all ranks bottleneck on the same
> storage and converge) and large when it is fast (per-rank variance dominates).
> It therefore biases *exactly* the configuration this experiment is promoting.

## Goal

Bristen (x86, A100) showed a ~4.4x cold-start win from TP-presharding + sliced
`/dev/shm` staging + overlapping the stage with startup. Every number was
bristen-only. Does it hold on Clariden (aarch64 Grace, GH200)?

## Method

Two configs, `n=1` each, one fresh node per run, `--exclusive`, engine version
held constant at `lmsysorg/sglang:v0.5.10` (multi-arch tag — the arm64 build
here, the x86 build on bristen). Llama-3.1-70B-Instruct, TP=4, ctx 32768,
mem-fraction 0.85. Scripts: `../preflight.sbatch`, `../baseline_mmap.sbatch`,
`../preshard_shm_overlap.sbatch`.

| job | config | node | partition |
|---|---|---|---|
| 2916234 | preflight | nid006619 | debug |
| 2916286 | **default** (mmap from capstor) | **nid007661** | debug |
| 2916421 | **preshard+shm+overlap** | **nid007585** | debug |

Different nodes, so preshard+shm+overlap could not have read the 141 GB from the
default run's page cache. `--cpus-per-task=288` (whole node) for both; bristen
used 64.

The presharded checkpoint was **reused from the bristen run**
(`/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded`,
28 shards, 141,115,552,320 B) — not regenerated. See Verdict for why that turned
out to be safe.

## Result

| | default | preshard+shm+overlap | bristen default | bristen preshard+shm+overlap |
|---|---|---|---|---|
| stage | — | **8.78 s** (17.02 GB/s, hidden) | — | 14.24 s (10.59 GB/s) |
| weight_loading (slowest rank) | **466.81 s** | **6.19 s** | 634 s (mean n=4) | 9.81 s |
| **total cold start** | **586.33 s** | **127.13 s** | 812 s (mean n=4) | 183.43 s |
| throughput | 822.9 tok/s | 797.7 tok/s | 401 tok/s | 401 tok/s |
| errors | 0 (64/64) | 0 (64/64) | 0 | 0 |

**Speedup on Clariden: 586.33 / 127.13 = 4.61x.**

Overlap validity gate: stage finished 8.78 s into the run, `weight_loading` began
at 39.53 s — **VALID, 30.75 s of slack**, `stage_rc=0`.

### SLURM wall-clock cross-check

The job accounting independently corroborates the servekit numbers:

| | SLURM elapsed | servekit cold start | bench | residual |
|---|---|---|---|---|
| default (2916286) | 621 s | 586.33 s | 9.96 s | **24.7 s** |
| preshard+shm+overlap (2916421) | 164 s | 127.13 s | 10.27 s | **26.6 s** |

The residual — enroot mount, `pip install -e`, teardown — is ~25 s in both and
agrees to within 2 s, which is what makes the two measurements consistent.

**Job-level speedup is 3.79x** (621/164) against **4.61x** for cold start. Both
are correct and measure different things: 4.61x is process-launch → serving,
this project's stated scope; 3.79x additionally carries the ~25 s fixed job
overhead and a 10 s benchmark that is not cold start at all. Do not conflate
them.

### Node effects were negligible here

Apertus-8B on this same cluster showed that non-loading phases can vary up to
1.7x across nodes, enough to fake a config effect (see
`../apertus-8b/results.md`). That did **not** happen here: subtracting weight
loading from each total leaves **119.52 s (default) vs 120.87 s (preshard)**,
1.1% apart. Both nodes were comparable, so the 4.61x is a config effect, not a
node draw.

### Per-phase breakdown — the internal control

Every phase that is *not* weight loading is unchanged between the two configs.
That is the strongest evidence that the delta is real and attributable:

| phase | default | preshard+shm+overlap |
|---|---|---|
| process_startup | 16.13 | 17.95 |
| tp_worker_spawn | 15.21 | 14.32 |
| torch_distributed_init | 5.68 | 5.91 |
| unknown | 1.18 | 1.28 |
| **weight_loading** (servekit, fastest rank) | **466.20** | **3.72** |
| **weight_loading** (slowest rank — the real one) | **466.81** | **6.19** |
| kv_cache_alloc | 1.21 | 3.07 |
| cuda_graph_capture | 22.15 | 22.74 |
| piecewise_cuda_graph_capture | 46.41 | 46.11 |
| http_bind | 1.33 | 1.34 |
| warmup_request(JIT) | 10.83 | 10.64 |
| TOTAL | 586.33 | 127.06 |

The servekit row is kept because the other phase durations are derived from it
and would not sum without it; the slowest-rank row is the one to quote. TOTAL is
wall-clock and independent of both.

### Correctness

All **6/6** greedy correctness probes produced byte-identical text across the two
configs. This is what actually validates reusing the bristen-produced shards: the
`sharded_state` checkpoint written by SGLang on x86/A100 loads and serves
correctly on aarch64/GH200.

## Verdict

The technique reproduces on Clariden and the win is **bigger** than on bristen:

- **The stage got faster**: 14.24 s → 8.78 s, 10.59 → 17.02 GB/s. Expected —
  the sliced stager is CPU-bound and Clariden gave it 288 cores instead of 64.
- **`weight_loading` got much faster**: 9.81 s → 6.19 s (slowest rank).
- **The 4-socket NUMA risk is NOT ruled out.** An earlier version of this file
  claimed it "did not materialise", reasoning that 3.72 s for 141 GB left no room
  for a cross-socket penalty. That rested on the fastest rank. The per-rank
  spread is in fact **3.72 / 4.06 / 5.81 / 6.19 s — a 1.7x spread across the four
  ranks**, which is exactly the shape a NUMA asymmetry would produce: `/dev/shm`
  pages land on whichever socket's `dd` wrote them, and a rank reading from a
  remote socket pays for it. Not proven — rank-to-socket placement was not
  recorded — but it is now an open question, not a closed one. A `numactl`-pinned
  stage is the obvious follow-up. Apertus-8B, whose ranks agree to within 0.03 s
  (0.90/0.90/0.88/0.88), does not show it, consistent with a per-byte effect.
- **Presharded checkpoints are portable across architectures** — reusing the
  bristen artifact saved a 1 h GPU job and produced identical outputs.

**The next bottleneck is now graph capture**, exactly as on bristen but more
starkly: `piecewise_cuda_graph_capture` (46.11) + `cuda_graph_capture` (22.74)
= 68.85 s, which is **54% of the 127 s**. Weight loading is no longer worth
optimising on this path; it is 3% of the remaining cold start.

## Caveats

- **n=1 per config.** Bristen's mmap runs ranged 430–939 s over 4 repeats, so the
  *default* is the noisy term and the 4.61x figure carries roughly that band.
  The direction and order of magnitude are solid; the third significant figure
  is not. preshard+shm+overlap is intrinsically far more reproducible (tmpfs,
  not Lustre).
- The default at 586 s sits near the *bottom* of bristen's mmap range, so this
  comparison is, if anything, conservative — a slower baseline draw would have
  made Clariden look better still.
- **CPU count is a second changed variable**: 288 here vs 64 on bristen. It
  favours preshard+shm+overlap (the stager is CPU-bound), so the
  bristen→Clariden stage improvement is partly hardware and partly core count,
  not cleanly separable from these two runs. It does *not* affect the
  within-Clariden comparison, which is what the 4.61x is.
- Throughput differs 3% between the two (822.9 vs 797.7 tok/s) where bristen saw
  401 vs 401. **Investigated: not a loading defect.** The latency percentiles
  localise it to the tail, not to decode speed:

  | | p50 | mean | p99 | max |
  |---|---|---|---|---|
  | default | 2.480 | 2.488 | 2.517 | 2.518 |
  | preshard+shm+overlap | 2.504 | 2.567 | 2.766 | 2.766 |

  The median request is within **1.0%** — decode is the same speed. The gap is
  entirely in the tail, and `max == p99` in both, so it is one slow request out
  of 64 (concurrency 16 = 4 waves; one straggler wave ≈ 0.25 s ≈ the whole 3% of
  `wall_s`). Given the default's tail, the preshard run lands at ~819 tok/s. A
  loading defect would instead show wrong outputs (6/6 greedy probes are
  byte-identical) or a systematic p50 shift (there is none). Host RAM pressure
  from the 141 GB left in `/dev/shm` during serving was considered and rejected:
  ~700 GB of the node's 870 GB is still free.
- No raw `dd` bandwidth probe was run in-job (deliberately, to keep the dir
  minimal), so a slow-OST draw on either node would not be directly visible.
  The 17.02 GB/s stage argues against it for that run.
- `/dev/shm` on these nodes is **334 GB**, not the ~435 GB a naive 50%-of-870 GB
  estimate suggests. Ample for 141 GB, but worth knowing before targeting a
  larger model.

## Raw artifacts

`clariden-preflight-2916234.out`, `clariden-baseline-mmap-2916286.out`,
`clariden-preshard-shm-overlap-2916421.out`,
`baseline-mmap-2916286-nid007661-profile.json`,
`preshard-shm-overlap-2916421-nid007585-{profile.json,stage.txt,timing.txt}`.
