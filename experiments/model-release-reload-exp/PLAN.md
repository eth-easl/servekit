# Plan: drop the model from a live SGLang server, put it back

## Context

The project goal is cold-start. A promising route is **checkpoint/restore of a
warm serving process** (`cuda-checkpoint-exp` is testing whether CRIU + CUDA
checkpoint works at all on Bristen). But checkpointing a 70B server *with* its
weights resident means dumping ~141 GB of GPU state — the checkpoint becomes as
expensive as the cold start it was meant to avoid.

The idea: **release the weights, checkpoint the thin process, restore it later,
reload the weights from disk.** The checkpoint stays small (CUDA context, graphs,
KV pool — no model) and the restore cost collapses to a weight load, which
`lustre-loading-exp` already got to ~38 s.

**This experiment does no checkpointing.** It establishes the prerequisite: that
`release_memory_occupation` / `resume_memory_occupation` /
`update_weights_from_disk` actually do what they claim, at 70B, with the fast
loader — and how long each takes. A "no" here kills the checkpoint idea before
anyone writes CRIU glue.

Setup is a clone of
`lustre-loading-exp/scripts/phase1.3_e2e/phase1_3_e2e_baseline_fastsafetensors.sbatch`
— Llama-3.1-70B-Instruct, TP=4, `--load-format fastsafetensors`, profiled by
`servekit profile`. Same anchor, one new variable. **No engine patch:** upstream
loader, `SGLANG_FST_FILES_PER_RANK` unset. The phase3 `fpr=8` tuning is a
separate variable and would confound this one.

## Verified against the pinned engine (`sglang/` @ v0.5.10 = the image's SHA)

| Fact | Where | Why it matters |
|---|---|---|
| `--enable-memory-saver` is **required** | `server_args.py:5561`; `torch_memory_saver_adapter.py` `create()` | Without it the adapter is a **Noop**: release returns 200, frees nothing, and only logs a warning. A green run proving nothing is the main trap → the grep gate in the sbatch. |
| `torch_memory_saver==0.0.9` is a **hard dep** | `sglang/python/pyproject.toml:65` | Already in the image; nothing to install, nothing added to the measured window. |
| `enable_weights_cpu_backup` defaults **False** | `model_runner.py:1148` | Release **truly frees** the weights instead of stashing them in host RAM → resume returns garbage pages → the reload is mandatory. This is why the sequence needs all three calls, and it is falsifiable: `VmRSS` must *not* grow ~35 GB/rank. |
| Tags are `weights`/`kv_cache`/`cuda_graph`; `None` = all | `constants.py`, `scheduler_update_weights_mixin.py:131` | We pass `["weights"]` explicitly. |
| Release **asserts `is_fully_idle()`** | `scheduler_update_weights_mixin.py:127` | The bench must be drained first — it is: the driver starts only once servekit has written the profile JSON, which happens after the bench returns. |
| `update_weights_from_disk` **only accepts `DefaultModelLoader`** | `model_runner.py:1353` | It hard-fails for anything else. `fastsafetensors` routes through it (`loader.py:510`), so the fast loader works for the reload. `instanttensor` would not. |

## servekit: the one change — `--keep-alive`

`--bench` terminates the server after benchmarking, which would kill the run
before the cycle starts. `--keep-alive` writes the report as soon as the bench
ends, then keeps draining until the server exits on its own.

servekit **stays alive rather than detaching**: it owns the child's stdout pipe,
so exiting would close the read end and SIGPIPE the server on its next log line.
The drain thread is also what keeps the pipe from filling under load.

## Layout

```
experiments/model-release-reload-exp/
  PLAN.md
  scripts/release_reload.sbatch   # one point: TAG, MEMORY_SAVER, DO_CYCLE, RELEASE_TAGS
  scripts/cycle_driver.py         # the post-ready sequence + memory snapshots
  results/                        # %x-%j.out, *-profile.json, *-cycle.json, *-ddprobe.csv
```

Reuses `lustre-loading-exp/scripts/lib/dd_read_sweep.sh` by path.

## The sequence (`cycle_driver.py`)

servekit writes the profile JSON on ready, so **its existence is the ready
signal** — no second health-check mechanism.

```
S1 post_bench -> release(weights) -> S2 -> resume(weights) -> S3
   -> update_weights_from_disk -> S4 -> re-bench
```

Every step timed. A snapshot = `nvidia-smi --query-compute-apps=pid,used_memory`
+ `--query-gpu=memory.used`, per-PID `VmRSS`/`VmPin`/`VmLck` from
`/proc/<pid>/status` and `smaps_rollup`, and `Cached` from `/proc/meminfo`.

**No `/generate` between S2 and S4:** the weight VAs are unmapped or garbage; a
query would segfault the worker, not return an error.

The re-bench calls `servekit.bench`'s own `run_benchmark` with the same prompts
and the same seeded workload the cold-start bench used, then compares against the
`benchmark` block already in the profile JSON — apples-to-apples by construction.
Comparison is **qualitative, not hash-equality**: SGLang isn't bit-deterministic
across runs, so exact equality would false-alarm (see `bench.py`'s docstring).

**Teardown kills the server by PID, never `pkill -f sglang.launch_server`** —
that pattern also matches servekit's own cmdline, since the wrapped command is in
its argv. Killing servekit's child makes servekit see EOF and exit by itself.

## Two runs, control first, on distinct nodes

| TAG | flags | purpose |
|---|---|---|
| `ctl_nosaver` | `MEMORY_SAVER=0 DO_CYCLE=0` | prices what `--enable-memory-saver` itself costs at cold start |
| `saver_cycle` | `MEMORY_SAVER=1 DO_CYCLE=1` | the full cycle |

```bash
sbatch --job-name=mrr-ctl_nosaver --export=ALL,TAG=ctl_nosaver,MEMORY_SAVER=0,DO_CYCLE=0 \
       experiments/model-release-reload-exp/scripts/release_reload.sbatch
# then, excluding the node the control landed on:
sbatch --exclude=<node> --job-name=mrr-saver_cycle \
       --export=ALL,TAG=saver_cycle,MEMORY_SAVER=1,DO_CYCLE=1 \
       experiments/model-release-reload-exp/scripts/release_reload.sbatch
```

The control must be **measured, not remembered**: the same config measured
69.5–86.1 s across nodes (21 % spread) — node/time variance is the dominant noise
source. `--exclusive` grants sole use of a node, **not a different node**, so the
second job explicitly excludes the first's node.

The dd probe runs **last** in each job, never before the load — reading 141 GB up
front warms the Lustre OSS-side cache and would flatter the numbers we measure.

## Verification

| Check | Pass condition |
|---|---|
| Saver actually on | no `will not save memory` warning — **gate everything on this** |
| Release frees the model | S2 per-PID GPU mem drops **~35 GB/rank** (141 GB / TP=4) vs S1 |
| Freed, not swapped to host | S2 `VmRSS` does **not** grow ~35 GB/rank |
| Resume restores allocation | S3 per-PID GPU mem ≈ S1 |
| Reload restores *weights* | S4 outputs coherent and consistent with the cold-start outputs |
| Serves again | S4 throughput ≈ cold start (~400 tok/s, **0 errors**) |
| Storage healthy | dd probe in the 6–9 GB/s band; a low number invalidates the reload timing |
| Saver's cold-start cost | `saver_cycle` weight-load vs `ctl_nosaver` weight-load |

Deliverable: wall-clock for release / resume / reload, and a yes/no on whether a
70B server can be reduced to a model-free resident process and brought back.

## Known caveat: the reload is warm-cache

The cold-start load reads all 141 GB through **buffered** `pread` (GDS is
unavailable here — `nvidia_fs` is not in `/proc/modules`), so the model is very
likely still in the node page cache when `update_weights_from_disk` runs minutes
later. The reload time is therefore a **warm-cache lower bound**, not a cold
restore, and we cannot drop caches without root.

So we measure it rather than hand-wave it: `Cached` is in every snapshot. A
reload materially faster than phase1.3's 69–88 s weight load is evidence of
caching, not of a fast path. Follow-up if the mechanism works: reload from a
never-read copy of the model.

## Out of scope

CRIU / `cuda-checkpoint` (that's `cuda-checkpoint-exp`), the `fpr=8` loader
patch, multi-model swapping, and any servekit change beyond `--keep-alive`.
