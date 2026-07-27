# Phase 7 — overlapping the stage with startup hides it entirely

**Status:** done — **answers Phase 6's open question; not yet a shippable design**

## Goal

Phase 6 left `stage 11.89 s + weight_loading 9.64 s` back-to-back because
the `/dev/shm` stage ran to completion before `launch_server` was even
exec'd. Test whether backgrounding the stage and launching immediately
actually hides it inside the ~42 s of import/init that already runs first.

## Method

`scripts/phase7_overlap_stage/phase7_overlap_stage.sbatch` (job 76436,
nid002281, 64 CPUs — same node and CPU count as Phase 6's control, job
76435, so overlap is the only variable) copies the 10 small config/tokenizer
files synchronously, then backgrounds the 28-shard sliced stage and launches
`sglang.launch_server` **immediately**, with no barrier and no engine patch.

Validity is checked post-hoc rather than synchronized: the stage's end
epoch must precede the absolute wall-clock time `weight_loading` began
(`profile.started_at` + durations of the phases ahead of it, computed and
printed by the job itself). A negative slack would mean SGLang opened a
partially staged file — the run would be discarded, not interpreted.

(Enabler: `stage_to_shm_sliced.sh` gained an optional `FILE_PATTERN` env,
default `*` — phases 4/6 behave exactly as before. Phase 7 passes
`'*.safetensors'` so the small files can be pre-copied outside the raced
window.)

## Result

| | phase 6 (serial, 76435) | phase 7 (overlapped, 76436) |
|---|---|---|
| stage | 11.89 s (before launch) | 14.24 s (concurrent) |
| weight_loading | 9.64 s | 9.81 s |
| servekit total | 180.70 s | 183.31 s |
| **TOTAL COLD START** | **192.59 s** | **183.43 s** |

The stage finished **14.24 s** into the run; the loader did not open a
weight file until **48.74 s** — **34.5 s of slack**, i.e. it passed the
post-hoc validity gate by a wide margin.

Per-phase delta vs 76435: `process_startup` **+4.25 s** (23.32 → 27.57 s),
`tp_worker_spawn` +0.47 s, everything else within ±1.5 s noise. The stage
itself also slowed slightly under contention for CPUs (11.89 → 14.24 s,
10.59 GB/s vs 11.87 GB/s) — the sliced stager is CPU-bound (Phase 4b), so
1704 concurrent `dd`s competing with Python imports for 64 CPUs tax the
import window, and the stage pays for it too.

Correctness held: 64/64 requests, 0 errors, 401.4 tok/s — identical to
Phase 6.

## Verdict

**The stage disappears from the critical path: −9.16 s total.** Net of
+2.61 s of startup interference and −11.89 s of removed serial stage time.
It's not merely hidden, it's hidden with 34.5 s of room to spare — the
stage could get 3.4× slower (well within the 2–6× Lustre contention swings
already documented in this repo) and still beat the loader on this node.

**Where this leaves the cold start**: `weight_loading` is now 9.81 s against
Phase 1.3's mmap baseline of 20.51 s, and the remaining ~183 s is dominated
by `piecewise_cuda_graph_capture` (78.42 s) + `cuda_graph_capture` (28.41 s)
= **58% of total**, with `process_startup` + `tp_worker_spawn` (43.91 s)
next. Weight loading is no longer the lever — graph capture is.

**This measures the ceiling, not a shippable design.** The run has no
synchronization by construction; the 34.5 s slack margin is what a
production version would need to survive a slow stage, not proof that a
barrier-free design is safe in general. Follow-up (documented, not yet
implemented): stage into a `.tmp` path, atomic `rename()` into the final
path, and a small loader-side wait (reusing the same clone-diff-verify patch
harness) right before the loader opens weight files — so a slow stage
degrades to serial instead of corrupting a partially-staged read. The 34.5 s
of slack is what makes that follow-up cheap: a barrier that never fires
costs nothing, and the design only has to handle the tail.

## Caveats

- Single run, one node — no fresh-node repeat of the overlap arm itself
  (Phase 6's control point on the same node is the only comparison).
- This result assumes the specific timing relationship measured here (stage
  finishes at 14.24 s, loader wants weights at 48.74 s) holds under
  production conditions; the whole point of the un-implemented follow-up is
  that this assumption isn't yet enforced.
