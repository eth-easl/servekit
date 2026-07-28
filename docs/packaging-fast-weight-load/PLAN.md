# servekit fast model loading — plan

## Context

`experiments/lustre-loading-exp/` established that cold start for
Llama-3.1-70B on capstor Lustre is dominated by weight loading, and that
SGLang's default mmap loader is the worst option available (429–939 s weight
load, 812 s mean total). The best measured pipeline — TP-presharded
checkpoint + sliced staging into `/dev/shm` + staging overlapped with engine
startup — reaches **183 s total, ≈4.4×** faster, with identical outputs and
throughput.

That pipeline currently exists only as sbatch scripts with stale paths, a
stale servekit CLI invocation, and **no synchronisation between the stager
and the loader** — the stager pre-`truncate`s every destination to full size,
so a loader that reads too early gets zero-filled weights *silently*, with no
error. It measures a ceiling; it is not something anyone else can safely run.

The goal is to package it so the SwissAI serving folks can use it. Their
launcher is [swiss-ai/model-launch](https://github.com/swiss-ai/model-launch)
(`sml`) — a Python CLI that generates sbatch scripts from a model registry
(`/capstor/store/cscs/swissai/...` paths, Pyxis/EDF containers, vLLM and
SGLang, Bristen and Clariden). So the integration point is *a line inside a
generated job script*, which is what shapes the API below.

## The API

Two commands do the work; two more manage the RAM.

```bash
# once, offline (~5.5 min for 70B, needs the GPUs)
servekit prepare --model /capstor/store/.../Llama-3.1-70B-Instruct \
                 --sharded --tp 4 \
                 --out /capstor/store/.../llama70b-tp4

# every launch — prepared artifact
ARGS=$(servekit stage-model /capstor/store/.../llama70b-tp4)
#   reads servekit.json -> tp=4, sharded
#   -> --model-path /dev/shm/servekit/llama70b-tp4 --load-format sharded_state

# every launch — raw checkpoint, nobody ran prepare
ARGS=$(servekit stage-model /capstor/store/.../Llama-3.1-70B-Instruct)
#   no manifest -> stage as-is, default load format
#   -> --model-path /dev/shm/servekit/Llama-3.1-70B-Instruct

python -m sglang.launch_server $ARGS --tp 4

# RAM management, explicit only
servekit list
servekit free /dev/shm/servekit/llama70b-tp4
servekit free --all
```

Design rules this encodes, all chosen deliberately:

- **Sharding is a `prepare`-time decision.** `stage-model` takes one path and
  reads `servekit.json` to decide what to return. Nothing to keep in sync
  between the two commands.
- **`stage-model` returns engine args, not just a path** — one shell capture,
  and servekit can add a flag later without changing the job script.
- **It returns immediately.** The copy runs in the background and is
  overlapped with the engine's ~48 s of import/spawn/NCCL init. This is where
  the last 9 s comes from, and it is the part that needs the barrier.
- **No automatic cleanup.** Staged data outliving the job is the feature: a
  restart on the same node hits a warm copy and skips loading almost
  entirely. `free`/`list` are the escape hatch.
- **The barrier is invisible.** No eval, no PYTHONPATH, no wrapper — see
  below.

## The barrier (the part that does not exist yet)

servekit is `pip install`ed *inside the container* at job start (already how
the experiments do it). That lets it drop a `.pth` file into site-packages,
which CPython imports at every interpreter start — including every
`multiprocessing.spawn`ed TP worker. The hook is inert (a few ms) unless it
is actually needed.

Activation is **path-based, not env-based**: the hook wraps
`safetensors.safe_open`, and on each call walks up from the file to look for a
`.servekit/` directory. No env var, no argv parsing, works identically in the
main process and in spawned TP workers, and works for both the
`sharded_state` and default-mmap tiers (both go through `safe_open`).

Staged-directory layout:

```
/dev/shm/servekit/llama70b-tp4/
  model-rank-0-part-0.safetensors     # truncate()d to full size up front
  config.json, tokenizer.json, ...    # copied synchronously
  .servekit/state.json                # src, tier, pid, file list, tp_size
  .servekit/done/<filename>           # written as each file completes
  .servekit/complete                  # all files done
  .servekit/failed                    # stager died; hook aborts loudly
```

`stage-model` splits into a synchronous head and a background tail, which is
what makes the protocol safe:

- **synchronous, before returning:** free-RAM pre-flight, `mkdir`, copy the
  small metadata files, `truncate` every shard to full size, write
  `state.json`. This guarantees the loader's `glob()` always sees the complete
  file list — a file appearing late would otherwise be silently *missing*.
- **background:** the sliced parallel copy, then a `.done` marker per file,
  then `.servekit/complete`.

The hook blocks on `.servekit/done/<basename>` before letting `safe_open`
proceed, polling ~50 ms, with a timeout and a clear abort on
`.servekit/failed`. In the measured run the wait is 0 s — the loader reaches
the first shard 34.5 s after the stage finished. The barrier exists so that a
stage 3.4× slower than measured degrades to *waiting* instead of to *silently
wrong weights*.

**Warm-hit path:** if the destination already has `.servekit/complete` and its
`state.json` matches the requested source, `stage-model` returns the args
immediately and stages nothing. With no auto-cleanup, this makes a restart on
the same node nearly free — the largest win in the whole design, and it costs
almost no extra code.

## Files

New modules under `servekit/src/servekit/`, matching the existing style
(stdlib-only, `from __future__ import annotations`, dataclass + `to_dict()`,
pure functions separated from I/O, `[SERVEKIT]`-prefixed output):

- `manifest.py` — read/write `servekit.json`; the record is
  `{format, tp_size, engine, engine_version, dtype, source, source_bytes,
  num_files, created_at, servekit_version}`. **TP mismatch is a hard error**
  with an actionable message; everything else is informational.
- `stage.py` — the synchronous head + background tail above. Port
  `experiments/lustre-loading-exp/scripts/phase4_shm/stage_to_shm_sliced.sh`:
  keep `dd` subprocesses as the actual I/O (proven at 11.9 GB/s; O_DIRECT +
  per-file slicing is the whole trick — shards are `stripe_count=1`, so one
  reader per file means queue depth 1 per OST), with Python orchestrating.
  Defaults `SLICES=64`, `bs=16M`, `iflag=direct`. Add the `--cpus-per-task`
  observation to docs: 64→128 CPUs is 1.58× on the stage (job 75713), and
  neither phase 6 nor 7 took it — ~4.5 s still on the table.
- `prepare.py` — wraps
  `experiments/lustre-loading-exp/scripts/phase6_preshard_shm/save_sharded_state_fixed.py`
  (keep its nested-`params` RPC fix and its skip-directories fix), then writes
  the manifest. **Also fix the trap the experiment left in:** it copies the
  *original* `model.safetensors.index.json`, whose `weight_map` points at
  files that do not exist in the prepared dir. Exclude it.
- `hook.py` + `_servekit.pth` — the barrier. The `.pth` ships into
  site-packages via setuptools `data_files`; `servekit doctor` verifies it is
  actually active inside the container and prints how to fix it if not.
- `cli.py` — add `prepare`, `stage-model`, `free`, `list`, `doctor` alongside
  the existing `profile`/`bench`. `main()`'s dispatch (`cli.py:188-193`) and
  `USAGE` (`cli.py:13-26`) both need the entries; each subcommand keeps its
  own `argparse` parser, per existing convention.

Reuse, not rewrite: `servekit profile` already reports `weight_loading` as a
first-class phase (`profile.py:17`), so before/after needs no schema change.
`servekit bench` (`bench.py:11-17`) already captures greedy outputs verbatim
specifically to compare loaders — it is the correctness gate for this work.

## Risks to carry into the docs

- **TP-locked and engine-version-locked.** Prepared shards use SGLang's
  post-fusion parameter names (`qkv_proj`, `gate_up_proj`) at a specific TP.
  Manifest + hard TP check covers the first; an engine upgrade silently
  invalidating artifacts is a real operational hazard and only gets a warning.
- **131 GiB of tmpfs per 70B model**, and by choice nothing reclaims it. A
  crashed job leaks it until someone runs `free`. Pre-flight check refuses
  rather than filling the node.
- **Multi-node TP is unmeasured.** `stage-model` must run once per node
  (`srun --ntasks-per-node=1`) and each node stages a full copy. Not
  validated.
- **Clariden/GH200 is unmeasured.** Different memory topology and `/dev/shm`
  budget; the staging economics may not carry over from Bristen A100.
- **vLLM is unvalidated.** It has its own `sharded_state` loader; the mapping
  is plausible but no measurement backs it.
- **The fastsafetensors tier stays a documented recipe**, not a command — it
  needs `SGLANG_FST_FILES_PER_RANK`, which a child process cannot set in the
  parent shell, plus a 12-line patch
  (`scripts/phase3_loader_concurrency/fst_files_per_rank.patch`).

## Verification

1. **Unit** — extend `servekit/tests/`. Fake a staged dir of tiny files and
   assert the hook blocks until `.done` appears, aborts on `.failed`, and
   returns immediately on `complete`. Manifest round-trip + TP-mismatch error.
   Note `run_profile`'s subprocess path is currently untested (`profile.py:224`)
   — the new stage subprocess path should not repeat that gap.
2. **The barrier actually matters** — the load-bearing test. Stage with an
   artificially throttled copy so it finishes *after* the loader starts, run
   with the hook and without. Without: outputs are garbage. With: correct
   outputs, and `profile` shows the wait absorbed into `weight_loading`. This
   is the proof the phase-7 caveat is closed.
3. **End-to-end on Bristen** — one fresh node, TP=4, 70B, reproducing ~183 s
   via `servekit profile` + `servekit bench` (expect ~401 tok/s, 0 errors,
   matching outputs). Then a second launch on the same node to demonstrate the
   warm-hit path. Follow the experiment's rules: fresh node per cold
   measurement, bracket with a control, carry a dd probe in-job.
4. **Deliverable** — package + docs + that proven run, then a PR to
   `swiss-ai/model-launch` inserting `stage-model` into the generated sbatch.
