# Phase 6 — pre-sharded checkpoint + `/dev/shm` staging

**Status:** done — **win if overlapped** (the "if" answered by Phase 7)

## Goal

Port the pre-sharding treatment from the sibling `shm-weight-loading-exp`
repo into this one, and quantify it end-to-end on the 70B target: does
doing the TP de-stride once, offline, remove enough weight-loading cost to
be worth the extra staging step?

## Method

One-time offline job (`scripts/phase6_preshard_shm/save_sharded_ckpt.sbatch`)
does the TP de-stride ONCE via `--load-format sharded_state`, writing one
file per TP rank (already contiguous, final layout, no cross-rank exchange)
to `/capstor/store/cscs/swissai/infra01/cold-start-experiments/
llama70b-tp4-sharded` (131 GB, regenerated job 76434).

The e2e run (`scripts/phase6_preshard_shm/phase6_preshard_shm.sbatch`, job
76435, nid002281, 64 CPUs — Phase 4b's 128-CPU stage fix **not** applied
here) stages those shards into `/dev/shm` with the Phase 4b sliced stager,
then serves off tmpfs with `--load-format sharded_state`.

## Result

| stage | weight_loading | stage + weight_loading | total cold start |
|---|---|---|---|
| 11.89 s | 9.64 s | **21.53 s** | 180.70 s |

Reference: Phase 1.3's mmap baseline, straight off capstor, no shm —
`weight_loading` alone = 20.51 s (job's own reported figure for the same TP=4
config at that point in the sweep, distinct from Phase 1.3's own bracketed
665–939 s mmap numbers, which used the *un-sharded* checkpoint — the smaller
20.51 s reference here is the un-sharded loader's number on the same node
class as this Phase 6 run, used purely as this section's own point of
comparison).

Correctness: 64/64 requests, 0 errors, 401 tok/s.

## Verdict

**Pre-sharding cleanly cuts `weight_loading`** (20.51 s → 9.64 s, matching
the ~10.6 s seen in the sibling `shm-weight-loading-exp` repo). But the
stage cost (11.89 s) **only cancels that gain if counted serially**, as this
run does — `phase6_preshard_shm.sbatch` runs the `/dev/shm` copy *before*
launching `sglang.launch_server`, so this run's 21.53 s is stage-then-load,
back to back.

Nothing ties the shm stage to GPU/torch state — it's a plain host-side file
copy — so it doesn't have to sit in front of `weight_loading`. It could
instead run concurrently with `process_startup` (23.32 s) + `tp_worker_spawn`
(15.87 s) + `torch_distributed_init` (3.09 s), none of which touch
`/dev/shm/llama70b`, starting the stage as soon as the job lands on the node
rather than after the engine process is already up. Under that overlap the
stage's 11.89 s would mostly hide inside the ~42 s of init that already
happens first, and the effective win converges toward the full
`weight_loading` delta (~11 s). **Not measured in this phase** — only
established here as the open question; answered directly in
[`../phase7_overlap_stage/results.md`](../phase7_overlap_stage/results.md).

Whatever CPU-count fix from [`../phase4_shm/results.md`](../phase4_shm/results.md)
(Phase 4b) applies would also still apply to this stage leg (not applied
here — this run used 64 CPUs, not 128).

Separately: `cuda_graph_capture` (28.35 s) + `piecewise_cuda_graph_capture`
(79.95 s) already dwarf `weight_loading` under any arm here — graph capture
remains the bigger lever for total cold-start reduction, independent of
whatever wins the weight-loading race.

## Caveats

- The 20.51 s "un-sharded" reference used for the delta above is this run's
  own quoted comparison figure, not independently re-derived here from a
  separate profile JSON — treat the ~11 s pre-sharding win as approximate,
  not to the same precision as the bracketed Phase 1.3/3 tables.
