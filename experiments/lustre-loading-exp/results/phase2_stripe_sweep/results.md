# Phase 2 (2a) — raw dd sweep on striped copies vs native

**Status:** done — **striping is a dead end**

## Goal

Phase 1.1 found the model's 30 shards already scatter across 24 distinct
OSTs at `stripe_count=1`. Test whether per-file striping (spreading a single
shard's bytes across multiple OSTs) adds anything on top of that
shard-level concurrency.

## Method

Copy the model onto iopsstor scratch under several explicit
`lfs setstripe -c {4,8,16} -S {4M,16M}` layouts (layout set on the target dir
before copy — `scripts/lib/make_striped_copy.sh`), then re-run the Phase 1.2
dd O_DIRECT sweep (`scripts/phase2_stripe_sweep/phase2_stripe_sweep.sbatch`)
on each, plus an in-job native-layout probe bracketing each striped run (the
Phase 1.2 methodology fix — same job, same node, minutes apart).

## Result

Aggregate GB/s, full 141 GB model, O_DIRECT (native @N=30 pulled from the
Phase-3 in-job probes; this phase's own native probes only ran up to N=16,
which is what hid the conclusion below for a while):

| layout | N=8 | N=16 | N=30 | N=60 |
|---|---|---|---|---|
| **native** | 1.6–2.3 | **3.6–5.5** | **6.7–8.6** | — |
| striped c8_s16M | 2.14 | 3.77 | 7.35 | 7.80 |
| striped c16_s4M | 1.64 | 3.53 | 6.43 | 6.08 |
| striped c8_s4M | 1.92 | 3.26 | 5.86 | 4.96 |
| striped c4_s4M | 2.27 | 4.43 | 4.30 | 5.17 |

Raw data: `phase2_c{4,8,16}_s{4M,16M}-73645-nid002317.csv`,
`phase2_nativeprobe-73645-nid002317.csv`.

## Verdict

**Native is as fast as, or faster than, every striped layout at N=30.** Phase
1.1 already explains why: the 30 shards independently scatter across 24
OSTs, so concurrent multi-shard reads already exploit the whole filesystem
width. Per-file striping adds nothing on top of shard-level concurrency —
confirmed here at 2b's end-to-end level too (see
[`../phase2_e2e/results.md`](../phase2_e2e/results.md)).

**Do not restripe.** It requires copying the production model (141 GB per
layout), it is not plug-and-play, and it measurably buys nothing.

## Caveats

- All striped-layout points here ran on a single node (nid002317) in one job
  — no fresh-node repeats for this sub-phase specifically. The conclusion is
  still solid because it's a *negative* result (striping doesn't help) that
  agrees with the independent Phase 1.1 mechanism and the Phase 2b end-to-end
  numbers, not a single fragile positive claim.
