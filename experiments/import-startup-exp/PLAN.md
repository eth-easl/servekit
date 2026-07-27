# Plan: import-startup-exp — long import times in process_startup / tp_worker_spawn

## Observation

`process_startup` + `tp_worker_spawn` are ~21% of the post-load-fix cold start
— ~38-40s on the Llama-3.1-70B TP=4 sliced-shm baseline (22-24s + 16-18s),
the next-largest chunk after graph capture. A prior `-X importtime` run
(apertus-8B TP4, `logs/apertus-8b-sglang-importtime-73350.out`) shows both
phases are almost entirely **Python imports**:

- `process_startup` (~21s wall): torch 6.3s, transformers 5.3s, sglang 4.4s,
  sgl_kernel 1.0s — main process only.
- `tp_worker_spawn` (~13.5s wall): torch 15.7s + transformers 6.6s + sglang
  6.2s **summed across the 4 TP workers** (÷4 ≈ wall). Each worker re-imports
  the full stack from scratch.
- `transformers.utils.import_utils` alone = 3.68s self in one line.

"Import time" is not one thing. It is three, and each has a different fix:

1. **Compile** — parsing `.py` into bytecode, avoidable entirely with `.pyc`.
2. **Cold I/O** — reading thousands of small `.pyc`/`.so`/dist-info files out
   of a 30 GB squashfs on Lustre with an empty page cache.
3. **Module exec** — running each module's top-level code (torch's
   registrations, transformers' `importlib.metadata` probes). Irreducible by
   any cache; only avoidable by *not* importing, or by inheriting a process
   that already did.

Nothing so far tells us the split. The ideas below each attack a different
one of the three, so **the split decides which are worth pursuing** — this is
deliberately not a linear plan.

### Image audit (done, no jobs — `unsquashfs -l` on the EDF image)

`lmsysorg/sglang:v0.5.10`, python3.12, 30 GB squashfs at
`/iopsstor/scratch/cscs/yboughizane/.edf_imagestore/`:

| package | `.py` | `.pyc` |
|---|---:|---:|
| torch | 2018 | 2018 |
| transformers | 2169 | 2169 |
| sgl_kernel | 29 | 29 |
| **sglang** (`/sgl-workspace/sglang/python`) | **1663** | **0** |

sglang is an **editable install** (`__editable__.sglang-0.5.10.pth` + a
generated finder module in dist-packages); the image build never ran
`compileall` over that tree. torch/transformers ship fully precompiled — so
their cost is exec + I/O, *not* compile. sglang's is compile, in all five
processes (main + 4 TP workers), written into the ephemeral enroot overlay
and discarded at job end. Also relevant to idea 3: ~187k files in the image,
2236 dist-info dirs.

---

## Idea 0 (prerequisite) — split the 38-40s into compile / I/O / exec

Cheap: one node, one container, no sweep. Import `sglang.srt.server_args`
under `-X importtime`, then again in the same container. No model, no TP, no
server — the import graph is the workload, and its root cumulative was ~20s in
the prior in-launch run, i.e. effectively everything.

| run | page cache | `.pyc` present | isolates |
|---|---|---|---|
| A: first import, cold node | cold | no (sglang) | everything |
| B: repeat in same container | warm | yes (written by A) | exec only |
| C: repeat, `PYTHONDONTWRITEBYTECODE=1` and prior `__pycache__` removed | warm | no | exec + compile |

`C - B` = compile. `A - C` = cold I/O. `B` = exec floor. Headline numbers are
wall clock around each `python` call; the `-X importtime` self-time aggregation
only attributes each bucket to packages.

Only **run A can be cold, and only once per node** — page cache survives across
container runs ([[feedback_fresh_node_page_cache]]), so B and C follow in the
same `srun`, where the enroot overlay keeps A's `.pyc`. Three submissions on
distinct nodes (`--exclude` the ones already used); a node that already pulled
the 30 GB image reads as ~0 cold I/O, so take the consistent high A rather than
averaging.

Caveat to record, not design around: A also *writes* the 1663 `.pyc`, so that
write cost sits inside the cold-I/O bucket. It is the status-quo behaviour.

`scripts/idea0_split/split_imports.sbatch` → `results/idea0_split/`.

**DONE (2026-07-27, jobs 76482/76483/76484).** Cold I/O 15.7-21.0s, exec floor
8.90-9.01s, **compile 0.03-0.41s**. Full table and verdicts:
[`results/idea0_split/results.md`](results/idea0_split/results.md). This scopes
everything below: idea 3 is the whole story, idea 2 is closed, idea 1's ceiling
shrank to the exec floor.

---

## Idea 1 — spawn → fork, so TP workers inherit the parent's imports

**Targets: exec (and compile, and I/O) — but only the worker copies of it.**

SGLang hard-codes `mp.set_start_method("spawn", force=True)`
(`sglang/python/sglang/srt/entrypoints/engine.py:1172`) and launches each TP
worker via plain `mp.Process` (`engine.py:558`). `spawn` re-execs a fresh
interpreter per worker, inheriting nothing. `fork` gives each child a
copy-on-write snapshot of the parent's already-imported modules — it is the
only idea here that removes all three components at once, for the worker
processes.

Ceiling: the whole of `tp_worker_spawn`'s import cost (~13.5s wall), but
**nothing** in `process_startup` — the parent still imports normally.

Blockers, in order:
- Forking after the process has touched CUDA silently yields a broken CUDA
  context in the child. Must first read the parent's path up to `engine.py:558`
  for any `torch.cuda.*` / device query / NCCL init. If the parent is not
  CUDA-clean, this is a dead end — record it as such.
- Even if clean, it is a source patch (via
  `lustre-loading-exp/scripts/lib/patch_sglang_in_container.sh`), not
  plug-and-play, and carries the highest correctness risk of the three.
  Prefer `mp.get_context("fork")` for just the scheduler launch over flipping
  the global start method, to leave the DP-controller/detokenizer paths alone.
- Correctness is non-negotiable: 0 bench errors, throughput unchanged
  (~402 tok/s), greedy outputs identical to the spawn baseline.

**Do this last**, and only if Idea 0 shows the residual after ideas 2 and 3 is
exec-dominated.

Idea 0 update: the ceiling shrank. Workers' I/O is already largely free — they
start after the parent pulled the same files through page cache — so fork's
realistic prize is the ~9s exec floor per worker, overlapped across 4 ranks,
against the broken-CUDA-context risk. After idea 3, if the residual justifies it.

## Idea 2 — precompile sglang's bytecode — ❌ CLOSED by Idea 0

**Targets: compile — measured at 0.03-0.41s. Not where the time goes.**

The 1663-`.py`/0-`.pyc` audit below is correct but overestimates the cost: this
import touches only **117** of those modules and compiling them costs
milliseconds. Free and harmless to do anyway; not worth an experiment. The
reasoning is kept below because the audit still informs idea 3.

The image audit says every process compiles 1663 sglang modules from source,
five times per cold start, and throws the result away. Fix without touching
the image or the source:

```
PYTHONPYCACHEPREFIX=<persistent dir>          # in the EDF [env]
python -m compileall /sgl-workspace/sglang/python   # one-time, populates it
```

CPython reads *and* writes cached bytecode under the prefix, mirroring the
absolute source path, so a prepopulated dir is picked up by all five
processes with **no source patch and no engine flags** — the plug-and-play
shape this project wants ([[feedback_no_extra_flags]]).

Ceiling: the sglang compile slice only — roughly 4-6s of 38-40s if Idea 0
confirms sglang's 4.4s/6.2s is mostly compile. Real, free, low-risk, but not
the answer to the phase on its own.

Open questions:
- Does the prefix dir want to live on Lustre (persistent, but then it feeds
  straight into idea 3's problem) or be staged to `/dev/shm` per node?
- Invalidation mode: default timestamp mode stats each source file;
  `--invalidation-mode unchecked-hash` skips that stat entirely — worth
  measuring if metadata ops turn out to dominate.
- Does the editable-install finder interfere with the prefix? It uses a normal
  `SourceFileLoader`, so it should not, but verify rather than assume.

## Idea 3 — the I/O side of importing — ✅ PURSUE, confirmed dominant

**Targets: cold I/O — measured at 15.7-21.0s of a 25-30s cold import (~65-70%),
spread across torch/transformers/dist-info, not concentrated anywhere patchable.**

Measured as the largest of the three. Importing the stack touches
thousands of small files — `.pyc`, `.so`, and a `dist-info` scan across 2236
dirs for transformers' `importlib.metadata` probes — from a 30 GB squashfs on
Lustre with a cold page cache. That is small random reads at QD1, the exact
pathology this project already priced at 0.74-0.77 GB/s versus 19-20 GB/s for
the same bytes read with fan-out
(`lustre-loading-exp/scripts/phase4_shm/`). If a meaningful share of the
38-40s is this, it is addressable with machinery already built here and with
zero risk to correctness.

Sub-ideas, cheapest first:
- **Prewarm the image**: read the `.sqsh` (or just the python subtree) with
  the sliced parallel reader before launch, so imports hit page cache. Same
  trick as `stage_to_shm_sliced.sh`, different target. Costs a bulk
  sequential read at ~20 GB/s to avoid thousands of QD1 reads.
- **Fewer files**: a zipimport bundle of the pure-python tree turns ~30k
  opens into one. Native `.so` files must stay unpacked, so this is partial
  and more invasive — only if prewarming shows I/O is big and prewarming
  itself is too coarse.
- **Fewer probes**: `transformers.utils.import_utils`'s 3.68s is largely
  `importlib.metadata` walking dist-info. Check how much is metadata I/O
  versus exec before deciding whether env-level short-circuits are worth it.

---

## Deliverable

`NOTES.md`: the compile / I/O / exec split from Idea 0 at the current
70B TP4 scale, then a verdict per idea — adopt (with the exact env/patch), or
dead-end (with the specific blocker), matching the recommendation style used
elsewhere in this project. An idea whose target component turns out to be
small should be closed on the measurement, not attempted.

## Verification

- Every run reaches "fired up and ready to roll"; `servekit profile` emits a
  normal profile JSON.
- Any idea taken end-to-end: fresh-node control vs treatment, comparing
  `process_startup` and `tp_worker_spawn` specifically, plus 0 bench errors,
  throughput matching baseline (~402 tok/s), and identical greedy outputs.
- Success = a materially lower phase time with correctness intact, or a
  documented, specific reason why the lever does not apply — no ambiguous
  "seems fine" verdicts.
