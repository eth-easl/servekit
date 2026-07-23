# Plan: import-startup-exp — quantify and test the redundant-import cost in process_startup/tp_worker_spawn

## Context

`CLAUDE.md`'s deferred-idea note flags `process_startup` + `tp_worker_spawn`
(~21% of the post-load-fix cold start, ~38-40s on the current Llama-3.1-70B
TP=4 sliced-shm baseline: 22-24s + 16-18s) as the next-largest chunk after
graph capture. There's already a prior, smaller-scale measurement
(`logs/apertus-8b-sglang-importtime-73350.out`, apertus-8B TP4, via `python -X
importtime` injected into the launch): torch/transformers/sglang imports
dominate both phases, and critically, **`tp_worker_spawn`'s cost sums the
same import cost across all 4 TP workers** (torch 15.7s + transformers 6.6s +
sglang 6.2s summed, ÷4 ≈ wall) — each rank independently re-imports the full
stack from scratch, rather than sharing already-imported modules.

I confirmed the mechanism directly in the vendored source: SGLang hard-codes
`mp.set_start_method("spawn", force=True)` at
`sglang/python/sglang/srt/entrypoints/engine.py:1172`, and each TP worker is
launched via plain `mp.Process(target=run_scheduler_process_func, ...)`
(`engine.py:558`). `spawn` re-execs a fresh Python interpreter per worker —
nothing from the parent's already-completed imports is inherited. `fork`
would give each child a copy-on-write snapshot of the parent's already-loaded
modules, avoiding the redundant re-import entirely — the standard fix for
this exact class of problem in other multi-process Python/CUDA frameworks.

The catch, and the reason this needs measurement rather than a one-line
patch: forking a process *after* it has touched CUDA silently produces a
broken CUDA context in the child. This plan's job is to (1) confirm the
apertus-8B import-cost finding reproduces at this model/TP scale, and (2)
determine whether the parent process reaches this `mp.Process(...)` call
before any CUDA initialization — if so, a `fork`-based patch is viable and
testable; if not, this is a dead end and should be recorded as such (same
verdict style as InstantTensor/striping elsewhere in this project).

## Working location

Existing (currently empty) scaffold `experiments/import-startup-exp/` at the repo root
— `scripts/`, `results/` already created in a prior session. Add `PLAN.md`
and `NOTES.md` to match the convention used by every other experiment dir
(`lustre-loading-exp/`, `experiments/graph-compile-cache-exp/`, `shm-weight-loading-exp/`).

## Phase 1 — reproduce the import-cost breakdown at this scale

- Reuse the same technique as the apertus-8B run: inject `python -X
  importtime` into the SGLang launch command (via
  `lustre-loading-exp/scripts/phase4_shm/phase4_shm_sliced_e2e.sbatch` as the
  base, sliced-shm baseline, unmodified apart from adding `-X importtime`),
  capture the merged stdout (import lines interleave with the normal
  timestamped phase-anchor log lines SGLang already emits and `servekit`
  already parses).
- Bucket each `import time:` line by which phase-anchor marker it falls
  between (same method as before: no PID/timestamp on importtime lines, so
  ordering in the interleaved stream is the only signal). Aggregate self-time
  by (phase, top-level package) — reuse/recreate the aggregation approach
  that produced `logs/collect-importtime-73552.out` (top packages by summed
  self time), since that script wasn't kept in the repo; write a small one
  under `experiments/import-startup-exp/scripts/`.
- Confirm at TP=4/70B: is `tp_worker_spawn` cost still ≈ single-worker import
  cost (roughly constant, since workers import in parallel) or does it grow
  with model size (would suggest per-rank weight-adjacent import work, not
  pure stdlib/torch import)?

## Phase 2 — confirm whether CUDA is touched before the scheduler `mp.Process` call

- Read forward from `engine.py`'s `_launch_subprocesses`/`run_scheduler_process_func`
  call chain (the site at `engine.py:558`) back through the parent's
  execution path up to that point, checking for any `torch.cuda.*` calls,
  device queries, or NCCL/driver init that would already have initialized a
  CUDA context in the parent before it forks.
- If confirmed clean (no CUDA touch pre-fork in the parent): fork is safe to
  test. If not: this rules out the fork approach cleanly, and the
  deliverable becomes "measured, structural blocker identified, not
  pursuing" rather than a working patch.

## Phase 3 (conditional on Phase 2) — test a fork-based patch

- Small patch (same clone-diff-verify harness already built and reused across
  this project: `lustre-loading-exp/scripts/lib/patch_sglang_in_container.sh`)
  changing `mp.set_start_method("spawn", force=True)` to `"fork"` (or using
  `mp.get_context("fork")` locally for just the scheduler-process launch,
  to avoid touching the DP-controller / detokenizer process paths which may
  have different constraints).
- Same fresh-node, bracketed-control methodology as every prior phase in this
  project: control (upstream, spawn) vs treatment (fork), on distinct fresh
  nodes, comparing `tp_worker_spawn` duration specifically (process_startup
  is single-process and shouldn't change).
- **Correctness check is non-negotiable given CUDA fork risk**: confirm 0
  bench errors, throughput unchanged (~402 tok/s), and greedy-decode outputs
  identical to the spawn baseline — a broken CUDA context after fork tends to
  fail loudly (CUDA errors) but must be explicitly ruled out, not assumed
  absent just because the process starts.

## Deliverable

`NOTES.md`: import-cost breakdown table (phase x package, self-time) at
70B/TP4 scale, confirming or revising the apertus-8B numbers; a clear
statement of whether the parent is CUDA-clean before the fork point; and, if
Phase 3 ran, a control-vs-treatment table for `tp_worker_spawn` plus the
correctness checks. Verdict: adopt (with patch details), or dead-end
(with the specific blocker), matching the recommendation style already used
for every other phase in this project.

## Verification

- Every run reaches "fired up and ready to roll", `servekit profile` emits a
  normal profile JSON.
- If Phase 3 runs: 0 bench errors, throughput matches baseline (~402 tok/s),
  outputs identical to the spawn control — required before calling a fork
  patch safe, not just faster.
- Success = either a materially lower `tp_worker_spawn` with correctness
  intact, or a documented, specific reason (CUDA pre-fork touch, or no
  measurable drop) why it doesn't help — no ambiguous "seems fine" verdicts.
