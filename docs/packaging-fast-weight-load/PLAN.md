# servekit fast model loading — plan

Supersedes `archived-PLAN.md`, which packaged the same pipeline behind a
`stage-model` command that printed engine args for the job script to capture.
The measurements and the risks carry over unchanged; **the API does not**. This
plan wraps the engine launch instead.

Delivered in four phases. **Phase 1 is a complete, useful tool on its own** —
one command, no engine hook, no new checkpoint format — and gets most of the
measured win. Each later phase buys back one specific thing Phase 1 gave up.

| | what lands | buys | costs |
|---|---|---|---|
| **1** | `servekit launch`, stage-then-start, free on ready | **4.64x on the 70B (measured)** | ~5 s of un-overlapped staging |
| **2** | `--overlap` + validity gate (shipped); the barrier (not built) | at most the ~5 s stage | a `sitecustomize` shim in the engine |
| **3** | `prepare` + manifest | no hand-run scripts, no manual flags | a TP- and version-locked artifact |
| **4** | vLLM, multi-node, earlier free, PR | production use | — |

**Phase 1 is done and measured** —
`experiments/servekit-fast-weight-load/phase-1-no-overlap/results.md`. Two of its
findings shrink what comes after: the stage measured 4.29–4.98 s rather than the
8.78 s this plan budgeted, which is all Phase 2 can win back; and presharding
buys only ~2 s once the bytes are in tmpfs (the stock checkpoint with no loader
flag gets 4.57x on its own), which is most of Phase 3's reason for existing.

## Context

`experiments/lustre-loading-exp/` (Bristen, A100) and
`experiments/clariden-loading-exp/` (GH200, Grace) both established that cold
start is dominated by weight loading and that the best pipeline —
TP-presharded checkpoint + sliced staging into `/dev/shm` + staging overlapped
with engine startup — removes essentially all of it:

| | weight_loading | total | speedup |
|---|---|---|---|
| Llama-3.1-70B, bristen | 634 s → 9.81 s | 812 → 183 s | 4.4x |
| Llama-3.1-70B, clariden | 466.8 → 6.19 s | 586 → 127 s | 4.61x |
| Apertus-8B, clariden | 81.4 → 0.90 s | 173 → 96 s | 1.81x |

That pipeline exists only as sbatch scripts, and with **no synchronisation
between the stager and the loader**: the stager pre-`truncate`s every
destination to full size, so a loader that reads too early gets zero-filled
weights *silently*. The experiments handle this with an after-the-fact validity
gate (`preshard_shm_overlap.sbatch`) — fine for measuring a ceiling, unusable by
anyone else.

The consumer is the SwissAI serving platform, whose launcher is
[swiss-ai/model-launch](https://github.com/swiss-ai/model-launch) (`sml`): a
Python CLI that generates sbatch scripts from a model registry. The integration
point is *a line inside a generated job script* — which is what shapes the API.

## The API

One command wraps the engine launch, and after Phase 3 one more prepares the
checkpoint offline:

```bash
# Phase 3, once, offline (~5.5 min for 70B, needs the GPUs)
servekit prepare --model /capstor/store/.../Llama-3.1-70B-Instruct \
                 --sharded --tp 4 \
                 --out /capstor/store/.../llama70b-tp4

# Phase 1, every launch: prepend `servekit launch --` to the existing command
servekit launch -- \
  python -m sglang.launch_server \
    --model-path /capstor/store/.../llama70b-tp4 \
    --tensor-parallel-size 4 --context-length 32768 ...
```

`servekit launch` behaves like the engine command it wraps: same stdout, signals
forwarded, exits with the child's exit code. In a job script it is a one-token
edit, and removing it is how you get the baseline back.

---

# Phase 1 — MVP

**`servekit launch -- <engine command>`: stage the model into `/dev/shm`, start
the engine against the copy, free the copy when the server reports ready.**

Sequential, not overlapped. The engine is not touched in any way — no hook, no
injected env, no rewritten flags beyond the model path. That is what makes this
phase small enough to trust: the only thing servekit does to the engine is hand
it a different directory.

```
scan argv for model path
  └─► run the stager, wait for it            ~8.8 s   (70B, Clariden)
        └─► rewrite model path → spawn engine via run_profile
              └─► on ready: free /dev/shm    ≈127 s   after the engine starts
```

Expected: **~136 s against a 586 s default, ~4.3x**. The overlap that Phase 2
adds is worth the ~8.8 s difference against Phase 1's ~127 s of engine startup.

### What it needs

- `_stage/stage_to_shm_sliced.sh` — the experiment's stager, byte-identical,
  shipped as package data (details below).
- `stage.py` — spawn the stager, wait, check rc, fold its summary line (wall
  time, GB/s) into the profile report as a `stage` phase.
- `engine_args.py` — **pure**: find the model path in argv, return argv with it
  replaced. Extends the existing `FrameworkSpec` (`profile.py:19-112`) with
  SGLang's `--model-path`, so engine knowledge stays in one table. Nothing else
  is rewritten in this phase: the user's `--load-format` and every other flag
  pass through untouched.
- `launch.py` — the ~100-line supervisor: stage, rewrite, `run_profile`,
  free in the `on_ready` callback (`cli.py:62` is already exactly that hook),
  exit code and signal passthrough. `free` lives here rather than in a
  `reclaim.py` of its own: it is fifteen lines with exactly one caller, and
  there is no `servekit free` subcommand for it to also serve (see below).
- `cli.py` — `launch` added to the dispatch (`cli.py:178-181`) and `USAGE`
  (`cli.py:13-26`). **One new subcommand, not three.**

`run_profile` (`profile.py:213`) already spawns the child, streams its output,
and forwards SIGTERM/SIGINT, which is most of `launch`'s process handling.
`profile` stays as-is — `launch --no-fast-load` in all but name, so the
experiment scripts keep working.

### Deliberately not in Phase 1

- **No overlap, therefore no barrier.** The stage completes before the engine
  starts, so a partially-written file cannot be read. This is the single biggest
  simplification: no `sitecustomize`, no `safe_open` wrapper, no done markers, no
  `doctor`. It costs the ~9 s Phase 2 buys back.
- **No manifest, no `prepare`.** `launch` stages whatever directory it is
  pointed at, as-is. To use a presharded checkpoint in this phase you produce it
  with the experiment's `save_sharded_state_fixed.py` and pass `--load-format
  sharded_state` yourself — the flags stay in your script, which is where they
  belong until servekit has something to check them against.
- **No leak recovery.** Free happens on ready and nowhere else.
- **No `servekit free`, no `servekit list`, no `--keep-staged`.** See below —
  all three exist only for cases where nothing is waiting on the RAM anyway.

### Freeing on ready

**Why the free exists at all: to give the RAM back to the job that is now
serving.** Not to tidy up after the job, and not to keep the node clean for
whoever lands there next. Once the weights are on the GPU the tmpfs copy is
131 GiB of a serving node's memory doing nothing, and the node wants it for its
own work — KV cache offloading (`--enable-hierarchical-cache` sizes a host KV
pool), page cache, everything else. Getting this backwards is what makes the
rest of this section look optional when it is not.

`run_profile` fires `on_ready` when the server announces itself, which is also
the point at which the weights are in GPU memory and the tmpfs copy has done its
job. So the RAM comes back *during* the run: for all but the first ~2 minutes of
a job that may serve for hours, the node looks as if servekit was never
involved.

Three things follow from that framing, and they are why the CLI has **one**
subcommand rather than three:

- **No `--keep-staged`.** Holding the copy for the serving lifetime of the model
  is exactly the thing this phase exists to avoid. It was proposed for measuring
  and debugging; the stage is already reported as a phase, and the staged bytes
  are already verified by the stager's own size gate and by unit tests, so it
  bought nothing that is not covered.
- **No `servekit free` / `servekit list`.** A copy is only ever left behind by a
  server that never reached ready (crash, OOM, bad flags) — and a server that is
  not serving has no workload waiting on that RAM. So the manual hatch is
  `rm -r /dev/shm/servekit/<name>`, and it does not need to be a subcommand. The
  plug-and-play promise is one token prepended to one command; two extra verbs in
  `USAGE` made the tool look like it had a cleanup workflow to learn.
- **Freeing earlier is the real improvement, not freeing manually.** Weight
  loading ends ~77 s before ready on the 70B (t+49 vs t+126), and SGLang
  allocates its host KV pool in between (`Scheduler.init_memory_pools`, before
  ready). Deferred on purpose for now: `Load weight end` fires once per TP rank,
  so servekit would have to count ranks rather than take the next marker, and
  the mmap measurement only proves mappings are dropped *by ready*, not at
  weight-load-end.

One caveat decides whether this works at all, and is therefore the first thing
to verify: **freeing is `unlink`, and tmpfs pages are only reclaimed once
nothing maps them.** With `sharded_state` the loader reads into GPU memory and
closes the files, so the RAM returns immediately. With the default mmap loader
the engine may still hold mappings at ready time, in which case the unlink
succeeds, `df` shows the space back, and the RAM does not actually return until
the engine drops them.

### The stager: the experiment's script, verbatim

`experiments/lustre-loading-exp/scripts/phase4_shm/stage_to_shm_sliced.sh`
(copied unchanged into `clariden-loading-exp/`) is what produced 11.9 GB/s on
Bristen and 17.0 GB/s on Clariden. It ships **as that file**, not as a Python
reimplementation, and `stage.py` invokes it. The measured code path stays
literally the measured code path, and stays diffable against the experiment dir.

Nothing in it is incidental:

- **64 slices per file** (`SLICES`, default 64), each a contiguous
  `dd skip=/seek=/count=` range, and *all slices of all files are in flight at
  once* (`xargs -P $NTASK` where `NTASK` is the total slice count — 28 shards ×
  64 = 1792 concurrent `dd`s for the 70B). The `-P 60` figure belongs to the
  sibling control `stage_to_shm.sh`, one `dd` per file over 60 files, and that
  one is the *slow* variant: shards are `stripe_count=1`, so one reader per file
  is queue depth 1 per OST and a single high-latency OST (~480 ms/RPC) caps the
  whole stage. Slicing took raw reads 0.72 → 18.9 GB/s on the same node and
  minute.
- **`iflag=direct`, `bs=16M`.** O_DIRECT keeps the stage honestly cold and stops
  it evicting the tmpfs copy it is creating; `READ_MODE=buffered` exists and is
  not the default for that reason.
- **`truncate` to full size before any writer starts**, plus `conv=notrunc` and
  per-slice `seek=`: N `dd`s racing to create one path is a bug, and a writer
  without `notrunc` shortens the file under its peers.
- **Size-parity gate** at the end, which the script itself flags as necessary
  but not sufficient — `truncate` already fixed the size, so only a checksum
  proves content. `--verify` runs that gate; off by default because it re-reads
  every byte.
- **Free-space pre-flight** (`df` + 10 GiB headroom), already in the script.
  Phase 1 needs no separate check.

servekit adds parameters, not logic: `SLICES` stays at the measured **64**, with
a `--slices` override. Scaling it with `os.cpu_count()` was considered and
dropped — 64 is what produced every number in `experiments/`, on both a
128-core and a 288-core node, and no measurement backs a scaling rule for a
value sitting in the critical path.

### Done when

1. **Unit** — `engine_args.py` table tests (model path found and replaced,
   every other flag preserved, unknown command rejected). Stager byte-identity
   against `experiments/`. Free on ready, and a run that never reaches ready
   leaves the dir intact. Note `run_profile`'s
   subprocess path is currently untested (`profile.py:224`); `launch` should not
   repeat that gap.
2. **The free actually returns the RAM** — on a real 70B run sample
   `MemAvailable` and `df /dev/shm` before staging, at ready, and a minute
   after, for both `sharded_state` and the default mmap loader. If the mmap case
   does not give memory back at ready, document that rather than implying
   otherwise. Throughput after the free must match a run without it — the check
   that nothing re-reads the weights later.
3. **End-to-end on Clariden** — one fresh node, TP=4, 70B, presharded
   checkpoint, `servekit launch` + `servekit bench`: expect ~136 s total, a
   stage of ~8.8 s at ~17 GB/s (the vendored stager lost nothing), ~800 tok/s,
   64/64, errors=0, outputs byte-identical to the default run. Fresh node per
   cold measurement, bracket with a control, `dd` probe in-job.

---

# Phase 2 — overlap

**Status: the barrier below is NOT built. Overlap ships as `--overlap`,
opt-in, guarded by an after-the-fact validity gate.**

The reasoning for deferring the barrier is sound and stands: the sequential
stage costs 4.29–4.98 s of the 70B's ~126 s (~4%), inside cross-node noise on
the measurements so far — a small, unconfirmed win against a real correctness
mechanism to build and maintain. Measure the win on one node before paying for
it.

What does **not** follow is making the unsafe path the default, which was tried
and reverted. Overlap without a barrier is not "fast with a caveat", it is
*silently wrong*: the stager `truncate`s every destination to full size before
writing a byte, so a loader that opens a file early reads zeros and nothing
raises. Demonstrated directly — 0.54 s into a stage the destination is at full
size with its first MiB entirely zero, `config.json` is already present so the
engine does not even hit a missing-file crash, and the final content is correct
so nothing is left to detect afterwards. A stderr warning does not mitigate a
failure mode whose defining property is that it produces no signal. Phase 1 was
safe by construction; defaulting to overlap traded that away for ~4% that the
same paragraph calls unconfirmed.

So: **sequential is the default, `--overlap` is opt-in**, and the thing that
makes `--overlap` usable is the **validity gate** ported from
`experiments/clariden-loading-exp/scripts/sglang/preshard_shm_overlap.sbatch`.
After the run, servekit reconstructs when the loader opened its first weight
file (`started_at` + every phase ahead of `weight_loading`) and compares it to
when the stage finished. Positive slack: VALID. Negative: it says loudly that
the run read partially-staged bytes, records `stage.valid = false` in the
report, **and exits non-zero**. That converts silent corruption into a run you
know to discard, which is exactly what let the experiments use overlap at all.
The gate is not a barrier — it detects after the fact, it does not prevent — but
shipping overlap without even the gate was strictly less safe than the sbatch
scripts this package replaces.

The barrier below is still what makes overlap safe *by construction*, and is
what to build if the same-node measurement shows the ~5 s is worth having.

Start the engine *at the same time* as the stager and let the copy hide inside
the engine's ~40 s of import / TP spawn / NCCL init, as the experiments did.
Worth ~9 s of the 70B's ~136 s; the whole cost of this phase is that the engine
must now be made to wait for weights that are not there yet.

**The hook.** Because servekit spawns the engine, it can inject
`PYTHONPATH=<pkgdir>/_boot:$PYTHONPATH` plus `SERVEKIT_STAGE_DIR=<dest>`. The
`_boot` dir contains a `sitecustomize.py` that CPython imports at every
interpreter start — including `multiprocessing.spawn`ed TP workers, which
inherit the environment. The shim is inert unless `SERVEKIT_STAGE_DIR` is set,
and it re-imports any pre-existing `sitecustomize` it shadowed (find the next on
`sys.path`; do not silently disable a container's own).

Active, it wraps `safetensors.safe_open` — the single funnel for both the
`sharded_state` and default-mmap tiers, in both engines — and blocks until
`.servekit/complete` exists. ~50 ms poll, a timeout, loud abort on
`.servekit/failed`.

**All-or-nothing, not per-file**, which follows from keeping the stager
verbatim: with every slice of every file in flight simultaneously, files do not
complete in any useful order, and per-file `.done` markers would mean
restructuring the proven `emit | xargs` core. The measured wait is 0 s in every
run (30.75 s of slack on Clariden's 70B, 34.5 s on Bristen), so finer
granularity buys nothing today; it is the escalation if a stage ever runs long
enough for the loader to catch it.

```
/dev/shm/servekit/llama70b-tp4/
  model-rank-0-part-0.safetensors     # truncate()d to full size up front
  config.json, tokenizer.json, ...    # copied first, read early by the engine
  .servekit/state.json                # src, file list
  .servekit/complete                  # stager rc=0 and size parity held
  .servekit/failed                    # stager died; hook aborts loudly
```

One ordering constraint the barrier cannot cover: the loader's `glob()` must see
the complete file list, or a late-appearing file is silently *missing* rather
than blocked on. The stager's `truncate` pass gives this for free — every
destination exists at full size within milliseconds of `t0`, before any `dd`
runs — which is another reason it goes first and unmodified. The small metadata
files (config.json, tokenizer) are copied ahead of the shards via the stager's
existing `FILE_PATTERN`, since the engine reads them seconds in.

**Rejected: directory-level atomic rename.** config.json and the tokenizer are
read long before the shards, so a whole-directory swap either blocks the engine
at startup or is not atomic where it needs to be.

**Fallback.** `--no-overlap` is Phase 1's behaviour, kept as a flag: always
correct, costs the stage wall time. `servekit doctor` reports whether the hook
is actually live inside the container and says to use `--no-overlap` if not.

### Done when

1. **Unit** — fake a staged dir; the hook blocks until `complete`, aborts on
   `failed`, returns immediately when already complete; inert without the env
   var; composes with a pre-existing `sitecustomize`.
2. **The barrier actually matters** — the load-bearing test. Throttle the copy
   so it finishes *after* the loader starts, run with the hook and without.
   Without: garbage outputs. With: correct outputs, and `profile` shows the wait
   absorbed into `weight_loading`. This is what closes the experiments'
   validity-gate caveat.
3. **End-to-end** — the Phase 1 run again, now ~127 s and 4.61x, with the stage
   invisible inside startup and outputs still byte-identical.

---

# Phase 3 — `prepare` and the manifest

Removes the hand-run script and the manually-passed flags. `prepare` writes a
presharded checkpoint plus a `servekit.json` next to it; `launch` reads the
manifest and derives what it previously had to be told.

- `prepare.py` — wraps
  `experiments/clariden-loading-exp/scripts/shared/save_sharded_state_fixed.py` (keep its
  nested-`params` RPC fix and its skip-directories fix), then writes the
  manifest. **Also fix the trap the experiment left in:** it copies the original
  `model.safetensors.index.json`, whose `weight_map` points at files that do not
  exist in the prepared dir. Exclude it.
- `manifest.py` — `{format, tp_size, engine, engine_version, dtype, source,
  source_bytes, num_files, created_at, servekit_version}`. **TP mismatch is a
  hard error** with an actionable message; engine-version drift is a warning.
- `engine_args.py` grows the rest of the rewrite: `--load-format sharded_state`
  when the manifest says sharded, and `--served-model-name` pinned to the
  original model id if the user did not set one — otherwise the API starts
  advertising a `/dev/shm/...` path as the model name. A conflicting
  user-supplied `--load-format` is a hard error, not a silent override.
- **No manifest → stage as-is**, exactly as Phase 1 behaved. A raw checkpoint
  still gets the `/dev/shm` win without anyone running `prepare`.

**Ordering: copy first, check alongside.** This phase is the first to add work
between `t0` and the first byte, and it must not. The only thing allowed before
the stage starts is the argv scan for the model path. Manifest read, TP check
and the load-format decision run *concurrently with the copy*; a TP mismatch
kills the stager and aborts a second in, having spent bandwidth rather than
time. Making every launch wait on a JSON read to protect a case that aborts
anyway is the wrong trade: staging unwanted bytes costs a second of Lustre
bandwidth once, delaying the stage costs a second of critical path forever.

### Done when

1. **Unit** — manifest round-trip; TP mismatch is a hard error; load-format
   conflict is a hard error; `--served-model-name` preserved when set, injected
   when not; no manifest → Phase 1 behaviour unchanged.
2. **The copy really starts first** — instrument `t0` and assert the stager's
   first `dd` precedes the manifest read and the engine spawn. Otherwise this
   ordering silently regresses the first time someone adds a check.
3. **End-to-end** — `prepare` then `launch` with no `--load-format` in the
   command, reproducing the Phase 2 numbers, and a deliberate TP mismatch
   failing loudly and fast.

---

# Phase 4 — hardening and rollout

Everything that Phases 1–3 knowingly deferred, in rough priority order:

- **vLLM.** `profile`/`bench` already cover it; staging does not. Its
  `sharded_state` loader and `prepare` path need a measurement before either is
  claimed to work, plus its own argv entries (positional model, `-tp`).
- **Freeing at weight-load-end instead of ready.** Worth ~77 s of held RAM on
  the 70B, and it moves the free *ahead* of SGLang's host KV pool allocation
  rather than after it. Needs rank counting (`Load weight end` fires per TP
  rank) and a measurement that the mmap loader's mappings are gone at that
  point, not merely by ready. This is the free-related work that pays; the two
  items below are not.
- **Leak recovery — probably never.** A copy is only left behind by a server
  that never reached ready, and nothing is waiting on that RAM. The design was
  `.servekit/owners/<pid>` refcounting plus a startup sweep, recording pid *and*
  process start time (`/proc/<pid>/stat` field 22) so a recycled pid cannot make
  a stale dir look live. Build it only if leaked copies are observed to hurt a
  *serving* node in practice; `rm -r` covers it until then.
- **Multi-node TP.** One `launch` per node (`--ntasks-per-node=1`), each staging
  and freeing its own copy. Currently unmeasured.
- **Warm-hit path — rejected.** A later `launch` reusing a copy left staged by
  an earlier one requires holding 131 GiB across the whole serving lifetime of
  the first model, which is precisely what freeing on ready exists to prevent.
  It trades a serving node's memory for start-up time it does not need; the
  stage is already ~5 s.
- **The fastsafetensors tier**, currently a documented recipe. The wrapper is
  what makes it promotable: it needs `SGLANG_FST_FILES_PER_RANK` in the engine's
  environment plus a 12-line patch
  (`experiments/lustre-loading-exp/scripts/phase3_loader_concurrency/fst_files_per_rank.patch`).
- **The PR** to `swiss-ai/model-launch`, prepending `servekit launch --` in the
  generated sbatch, with the measured before/after attached.

---

## Risks that outlive every phase

- **TP-locked and engine-version-locked** (from Phase 3 on). Prepared shards use
  the engine's post-fusion parameter names (`qkv_proj`, `gate_up_proj`) at a
  specific TP. Manifest + hard TP check covers the first; an engine upgrade
  silently invalidating artifacts is a real operational hazard and only gets a
  warning.
- **131 GiB of tmpfs per 70B model** while the copy is live. Free-on-ready keeps
  the window to ~2 minutes, but a node running two large models concurrently can
  still hit the wall; the stager's own `df` check refuses rather than filling it.
- **Unlink is not always reclaim.** Pages return when the last mapping goes,
  which for the default mmap loader may be later than the unlink.
  `sharded_state` is unaffected.
- **1792 concurrent `dd`s is not free.** It is what the measurements used, on
  `--exclusive` nodes with 128–288 cores. On a shared or small-CPU node the
  fan-out may need capping, and that has not been measured.
- **The wrapper is in the signal path.** Anything SLURM does to the job step now
  reaches the engine via servekit; exit codes, double-SIGTERM (freeing must not
  hang the shutdown) and orphaned engines need explicit tests.
- **`sitecustomize` is a shared slot** (from Phase 2 on). If the image ships
  one, or the site sets `PYTHONPATH`, the shim must compose rather than replace.
  `doctor` verifies the hook is live inside the container.
