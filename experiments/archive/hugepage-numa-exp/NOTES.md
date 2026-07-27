# hugepage-numa-exp — running log

**Question:** does exp2's NUMA correction compose with the hugepage
register-and-DMA loader? Target: beat **7.03 s** gated `weight_loading`.

## Why the two parents should compose

They fix different things and neither ever saw the other.

`hugepage-sharded-loading-exp` reached 7.03 s (job 76269) by staging the
checkpoint into one `memfd(MFD_HUGETLB)` and DMAing straight off the
registered buffer — no bounce at all. It ran at `--cpus-per-task=64`.

`shm-weight-loading-exp2` found that this exact setting silently disables
SGLang's own NUMA binding, on every run of every prior experiment including
that 7.03 s one. Bristen is a single-socket EPYC 7713 in **NPS4**: 4 NUMA
nodes, 16 cores and only 2 DDR4 channels each, and the GPU→node map is
**reversed** (GPU0→node3 … GPU3→node0). SGLang v0.5.10's `srt/utils/
numa_utils.py` already handles this — `numactl --cpunodebind=N --membind=N`
per TP worker, N from NVML — but `_is_numa_available()` bails when the
process's CPU affinity mask is already constrained, and `--cpus-per-task=64`
on a 128-logical-CPU node constrains it. Every historical log, all 4 ranks:
`NUMA affinity is already constrained ... skipping`.

The effect splits into two independent halves, and exp2 measured both on the
tmpfs bounce path (11.24 → 9.41 → 8.47 s):

| half | what it places | fix | worth on tmpfs |
|---|---|---|---|
| **process** | rank's cores + its own buffers | `--cpus-per-task=128` | 1.8 s |
| **page** | the source bytes themselves | NUMA-aware stager | 0.9 s |

The page half is decided by whoever **first touches** a page, i.e. the
stager, long before any rank exists — so binding the reader alone cannot fix
it.

## What this round tests, and what it deliberately does not

**Process half only.** It is a one-line change, and its size on this path
tells us whether the page half is worth building. That matters because on
hugepages the page half is *not* the same job as on tmpfs: exp2's
`stage_to_shm_numa.sh` runs each rank's writer under `numactl --membind`,
but `hugepage_stage_daemon.py` stages **all four ranks' files into one
memfd** from a single process with one big `ThreadPoolExecutor`
(`hugepage_stager.py:133`), and hugetlbfs pages come from per-node pools
allocated at fault time. Doing it properly needs `mbind()` on each file's
region before first touch, or four node-bound stager processes and a broker
that serves four fds. Deferred until the process-half number justifies it.

Two arms, differing in exactly one variable:

| arm | srun `--cpus-per-task` | binding |
|---|---|---|
| `ctl64` (76390) | 64 | disabled, as in 76269 |
| `numa128` (76391) | 128 | engages |

**`ctl64` is not redundant with 76269.** This harness pins
`LOOKAHEAD_CHUNKS=1` and adds the bit-exactness gate, so it is not
byte-identical to that run and 7.03 s cannot simply be assumed to carry over
— the delta has to come from a paired control, not from history.

The lookahead pin is deliberate. The client's lookahead design was rewritten
*after* the chunk-size sweep, so its current default of 3 combined with 1 GiB
chunks is an untested configuration that NOTES flags as "next step #1".
Leaving it at 3 would move two variables at once. It gets its own job.

## Measurement, inherited from exp2

- **Gated timing.** servekit reports the first `Load weight end` it sees,
  i.e. whichever rank finished first. The gating number is the *last* rank —
  no rank starts cuda-graph capture until all four have loaded. This
  flattered the hugepage path worse than anything else in the archive: job
  76161 reported 5.34 s against a true 10.07 s. `gated_weight_loading.py`
  re-scores from the per-rank `elapsed=` lines.
- **Bit-exactness, every run.** `exp2_verify.patch` asserts every parameter
  byte-identical to the source, hooked in immediately *after* the
  `Load weight end` log line so it cannot inflate the number it validates.
  The failure it exists to catch — a pipelined async H2D recycling a staging
  buffer while a copy still drains — is nondeterministic and passes a
  throughput benchmark. This path pipelines registration against in-flight
  DMA by design, so it is exactly the path that needs the gate.

## Jobs 76390/76391 — the process half, and it goes the wrong way

| arm | CPUs | per-rank `elapsed=` | **gated** | bit-exact | outcome |
|---|---|---|---|---|---|
| `ctl64` (76390) | 64 | 7.44 7.55 7.76 7.81 | **7.81 s** | 4/4 PASS | 401.2 tok/s, 64/64, 0 err |
| `numa128` (76391) | 128 | 9.07 9.41 9.49 9.51 | **9.51 s** | 4/4 PASS | **OOM-killed before serving** |

Binding engaged exactly as intended on `numa128` — all four
`mp.set_executable ... numactl --cpunodebind=N --membind=N` wrappers present,
in rank order 3/2/1/0, matching the reversed topology. It made the load
**1.7 s slower**.

`ctl64` re-anchors the incumbent at **7.81 s** under this harness (vs 7.03 s
historical for 76269 — the difference is lookahead=1 in the rewritten
continuous design plus the verify gate; this is the number the other arms are
compared against, not 7.03).

**Why the process half helps on tmpfs and hurts here.** On the tmpfs bounce
path the rank does a host gather: it reads tmpfs and writes a pinned buffer,
so binding makes both the copy threads and the destination buffer local, and
that is where exp2's 1.8 s came from. The hugepage path has **no host gather
at all** — the GPU's DMA engine reads the registered memfd directly. There is
no host-side destination to make local, so binding has nothing to win, and it
still pays the full cost of confining the rank to 16 cores and 2 memory
channels while its source pages remain spread across all four nodes. Cost
without the benefit, because the page half was never applied.

`numa128` also died: rank 1 was SIGKILLed during init (`exit code: -9`) after
loading. `--membind=N` is a hard confinement of *all* the rank's host
allocations to one 128 GB node, and the verify gate's readback pushed it over.
The weight-loading number is still valid — all four ranks logged
`Load weight end` and passed the bit-exact gate before the kill — but the arm
never reached serving.

This inverts the plan's assumption. The page half is not the optional
follow-on here; it is the only half that can pay.

## Jobs 76395-76400 — mbind does not do what it says on surplus hugepages

Placing the pages needs `mbind(2)` on each file's region (one shared buffer,
so exp2's `numactl`-around-a-writer has the wrong granularity, and hugetlb
placement is decided at fault time). `mbind_gate.py` / `mbind_gate2.py` are
~1-minute jobs that gate the plumbing before spending an e2e run, and they
earned their keep immediately:

| attempt | result |
|---|---|
| `MPOL_PREFERRED` | returns 0, `/proc/self/numa_maps` shows `prefer:0/1/2/3` on the four VMAs — and **every page lands on one node**: node0 with 16 CPUs held, node3 with 128 |
| first-touch from a thread pinned to the target node | 3/4 slices wrong |
| **`MPOL_BIND`** | **4/4 correct** |

The cause is in `/proc/meminfo`: `HugePages_Total: 0`, `HugePages_Surp: 128`,
`nr_overcommit_hugepages: 257505`. The persistent pool is empty, so every
page is a **surplus** hugetlb allocation, and that path does not honour a soft
policy — placement follows the faulting thread instead. Only a hard nodemask
survives to it.

So `hpnuma_stage_daemon.py` uses `MPOL_BIND`, accepting the SIGBUS-under-
pressure risk knowingly (each node is ~128 GB and holds one rank's ~35 GB),
and verifies with `move_pages(2)` on every run rather than trusting the
return code — an mbind that returns 0 has been shown here to place nothing.

## Jobs 76401/76402 — the page half, and the full 2×2

Placement verified on both runs: 28/28 files landed on the intended node by
`move_pages`, with the correct reversed mapping (rank 0 → node3, rank 3 →
node0).

| | source pages unplaced | placed on GPU-local node |
|---|---|---|
| **rank unbound (64 CPUs)** | **7.81 s** (76390) | 8.52 s (76401) |
| **rank bound (128 CPUs)** | 9.51 s (76391) | 8.59 s (76402) |

All four 4/4 bit-exact. Both 64-CPU arms served (401 tok/s, 64/64, 0 err);
both 128-CPU arms were OOM-killed before serving.

**NUMA locality loses on this path, and the 2×2 says why.** Placement helps
when the rank is bound (9.51 → 8.59) and hurts when it is not (7.81 → 8.52),
and the best cell is the one with no NUMA work at all. Concentrating a rank's
35.3 GB on one node caps its read at that node's **2 DDR4 channels**
(~51 GB/s peak, shared with everything else the rank does). The default —
pages scattered by 448 unbound staging threads — instead spreads every rank's
source across all 4 nodes, so each rank draws on 8 channels and the four
ranks' reads interleave across the whole machine. Since the DMA engine reads
the registered buffer directly, with no host gather to make local, locality
buys nothing to offset that.

This is the same wall exp2 hit from the other side, and it had already
guessed the shape of it: its own "directions worth trying" list proposes
*spreading* each rank's pages across 2 NUMA nodes to get 4 channels instead
of 2. On the hugepage path the accidental default is already the spread
version, and doing NUMA "properly" is what breaks it.

## Jobs 76403-76406 — the register-cost arms

With NUMA settled, the remaining candidates all attacked `cudaHostRegister`,
which dominates this path (DMA off an already-registered buffer is 1.33 s for
a rank's 35.3 GB; the phase is ~7.8 s). All run at the winning 64-CPU
unplaced baseline, so each moves exactly one variable off 7.81 s.

| arm | change | gated | vs 7.81 s |
|---|---|---|---|
| `la3` (76403) | lookahead 1 → 3 | 8.55 s | worse |
| `ilv` (76406) | explicit `MPOL_INTERLEAVE` | 9.20 s | worse |
| `ro` (76404) | `cudaHostRegisterReadOnly` | — | **`rc=801` cudaErrorNotSupported** |
| `ro_la3` (76405) | both | — | same failure |

`la3` closes out the hugepage NOTES' "next step #1": 1 GiB chunks with the
continuous 3-chunk lookahead, the one untested combination, is **not** better
— consistent with that experiment's own conclusion that per-call registration
overhead, not blocking, is what costs.

`ilv` is the sharpest result of the three: a deliberate, perfectly even
spread is *worse* than the accidental one (9.20 vs 7.81). So the default is
not winning by being evenly spread — MPOL_INTERLEAVE round-robins pages at
page granularity, which splits every large sequential read across all four
memory controllers, while first-touch by 448 threads leaves long contiguous
runs on one node. Contiguity of a rank's read matters more than balance.

`cudaHostRegisterReadOnly` is simply unavailable here (`cudaErrorNotSupported`
on chunk 0, every rank, both arms) — not a tuning result, a platform fact.

## Conclusion: no improvement. 7.81 s stands, unbeaten by 7 arms.

| config | gated |
|---|---|
| **`ctl64` — hugepage design, no NUMA, lookahead=1** | **7.81 s** |
| + process-side binding | 9.51 s |
| + page-side placement | 8.52 s |
| + both | 8.59 s |
| + lookahead=3 | 8.55 s |
| + MPOL_INTERLEAVE | 9.20 s |
| + registerReadOnly | unsupported |

**The premise of this experiment was wrong, and it is worth being precise
about why.** exp2's NUMA correction was real and large (11.24 → 8.47 s) — on
a path whose cost is a *host* gather: CPU threads reading tmpfs and writing a
pinned buffer, where making cores, source and destination node-local directly
cuts fabric traffic. The hugepage path deleted that gather. Its cost is
`cudaHostRegister`, which is serialised in the driver and indifferent to
where the pages live, plus a DMA that runs at PCIe line rate from anywhere.
A fix aimed at host-memory locality has nothing to grip on, and the one thing
NUMA binding reliably does — confine a rank to 2 memory channels — is pure
cost. **The two parents do not compose; they are alternatives, and the
hugepage path is both faster and already the one that does not care about
NUMA.**

What this round did produce, all of it negative but load-bearing for anyone
who picks this up:

- `mbind(MPOL_PREFERRED)` is silently ignored for surplus hugetlb pages here
  (returns 0, records `prefer:N` in `numa_maps`, places nothing). Only
  `MPOL_BIND` works. Never trust an mbind return code on this hardware —
  verify with `move_pages(2)`.
- `cudaHostRegisterReadOnly` is unsupported on this platform.
- 1 GiB chunks + continuous 3-chunk lookahead, the last untested cell of the
  hugepage sweep, is worse. That sweep is now closed.
- Contiguity beats balance for the source layout, which is a hypothesis worth
  keeping: it points at *larger contiguous per-rank runs*, not more even
  spread, if anyone revisits placement.

Anything further should target registration itself — the only term with real
headroom (10 s of registration hiding a 1.33 s transfer). exp2's own list
already names the two candidates that survive this result: hoisting
registration out of the weight-loading window entirely (it needs no model),
or a staging daemon that registers once and outlives the server. Both change
*when* registration happens rather than making it cheaper, which is the only
lever the evidence still supports.
