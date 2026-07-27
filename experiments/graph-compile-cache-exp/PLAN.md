# Plan: graph-compile-cache-exp — is graph capture just uncached JIT?

## Context

With weight loading solved (`lustre-loading-exp`, 812 s → 183.4 s), graph
capture is the largest remaining phase: `piecewise_cuda_graph_capture` 78.4 s +
`cuda_graph_capture` 28.4 s = **107 s, 58% of the 183.4 s phase-7 baseline**
(job 76436). It is near-constant across every loader and node measured, which
says fixed compute, not I/O.

Baseline for every run: presharded checkpoint + `/dev/shm`, i.e.
`experiments/lustre-loading-exp/scripts/phase7_overlap_stage/`. Staging happens
**once per job, outside the comparison** — this experiment compares phases
within a node, not end-to-end cold starts, so the overlap trick is irrelevant
here and is dropped for simplicity.

Out of scope (user decision): reducing the number of captured buckets.

## What the code says (checked, no jobs spent)

`piecewise_cuda_graph_capture` brackets exactly one statement,
`PiecewiseCudaGraphRunner(self)` (`model_runner.py:2563-2576`), which does:

| stage | code | job 76436 |
|---|---|---|
| dummy warmup + Dynamo trace / fx split | `piecewise_cuda_graph_runner.py:288-296` | ≈2 s |
| **a real forward pass at each of the 58 token buckets** | `:298-309` | **49 s** |
| `torch.cuda.CUDAGraph()` per bucket | `:317` → `capture()` | 28 s |

Critically, **no `torch.compile` codegen happens under the current flags**:
`piecewise_cuda_graph_compiler='eager'` (default), and `EagerAdapter.compile()`
returns the graph unchanged with `load()` raising `NotImplementedError`
(`compiler_interface.py:481-504`). `TORCHINDUCTOR_CACHE_DIR` is only ever set by
`InductorAdaptor.initialize_cache()` (`compiler_interface.py:190-195`), which
this path never reaches — **there is no Inductor cache in play, and persisting
one would be a no-op.**

So the 49 s is 58 genuine forwards through a 70B model, and what is expensive in
them is *kernel* JIT + autotune:

- **Triton** JIT per (kernel, shape signature) → `~/.triton/cache`
- **FlashInfer** — `attention_backend='flashinfer'`, `sampling_backend='flashinfer'`,
  `disable_flashinfer_autotune=False` — nvcc JIT + per-shape autotune → `~/.cache/flashinfer`
- **CUDA driver PTX JIT** for anything not shipped as sm_80 cubin → `~/.nv/ComputeCache`

The EDF pins `HOME=/root` in the container's ephemeral overlay, so **all of these
are destroyed at the end of every job, by construction.** That is the hypothesis:
graph capture is largely uncached JIT, and a persistent `HOME` recovers it. It
also predicts a drop in `warmup_request(JIT)` (15.2 s in job 76436), which hits
shapes the warmup loop never JIT'd.

## The experiment — wipe, cold, warm, cold-control (one job, one node)

`scripts/jit_cache/jit_cache_warm.sbatch`. `HOME` is the single knob — it catches
every cache above at once, and per-cache attribution comes free from `du`
afterwards rather than from guessing env-var names. `HOME` points at a
**node-local** dir seeded from / saved back to a persistent `/iopsstor` store;
see the deadlock note below for why it is not pointed at Lustre directly. Stage
the model to `/dev/shm` once, then three launches:

1. **cold** — cache root wiped immediately before. Populates.
2. **warm** — same root, now populated. The measurement.
3. **cold control** — a second, freshly-wiped root, same node, last.

Run 3 is not there for noise — `piecewise_cuda_graph_capture` measures 78.9–79.5 s
across every loader, node and job so far (<1% spread), the most stable number in
the project. It is there to separate the cache from *position in the job*: run 2
differs from run 1 in two ways, warm HOME **and** being the second launch on that
node. If run 3 (cold HOME, third position) matches run 1, position is inert and
the win is the cache; if it matches run 2, the win was never the cache.

Recorded per run: `piecewise_cuda_graph_capture`, `cuda_graph_capture`,
`warmup_request(JIT)`, total; plus `du -sh` and file counts of every cache
subdirectory after runs 1 and 2, which is what identifies *which* JIT cache
carries the win.

### Harness failures hit on the way (jobs 76440, 76444, 76446)

**`/dev/shm` cannot host the cache (76444).** The obvious node-local choice,
`/dev/shm/jit-home`, fails: `/dev/shm` is mounted `noexec`. Triton compiles
`cuda_utils.cpython-312-….so` into its own cache dir and `dlopen()`s it, which
dies with `failed to map segment from shared object`, taking the scheduler down
by sigquit before graph capture. `/dev/shm` holds the model (data) fine but not
the cache (executables). This is load-bearing for the eventual recommendation: a
prewarmed cache has to land somewhere local *and* exec-capable.

**Port 8080 race (76440, 76446).** Both runs died with `[Errno 98] address
already in use` — SGLang calls `get_free_port()` for `nccl_port`
(`server_args.py:6574`), the OS handed back 8080, and its own scheduler then held
the port Uvicorn needed. The server never reported ready and the run sat out its
wait-ready timeout in silence, which looks exactly like a hang. Phase 7 won the
same race by luck. `--nccl-port` is now pinned, removing the only random
allocation; nothing else about the engine config changes.

*(An earlier revision of this file attributed 76440 to a Lustre `flock` deadlock
in the JIT libraries. That was wrong — 76440's log contains the same `Errno 98`.
Whether `HOME` on Lustre works is simply untested; `HOME` is node-local here
because that is the deployable shape, not because Lustre was ruled out.)*

Per-run timeouts are 600 s (~3× a healthy run) so a future failure costs minutes,
not the allocation.

**Success**: run 2 drops materially below both runs 1 and 3.
**Null-result checks before concluding failure**: cache dirs empty (`HOME`
override not reaching the worker processes — TP workers are spawned, so confirm
the env propagates), or populated but not hit (keys salted by PID/tmp path).

## Deliverable

`results/jit_cache/results.md`: the 3-run × 3-phase table, the cache-directory
inventory, and a verdict — either graph capture is largely uncached JIT and a
persistent cache dir is worth building, or it is irreducible here and the next
lever is `import-startup-exp`.

**Left untested, deliberately**: whether the cache still hits on a *different*
node. That is the question a real deployment turns on, since cold starts land on
arbitrary nodes, and a same-node win could come from keys salted by host or by
paths that only match on the machine that wrote them. Any recommendation out of
this experiment carries that caveat until a cross-node run is done.

## Verification

Every run reaches "fired up and ready to roll", `servekit profile` emits a JSON,
`servekit bench` reports 0 errors at ~402 tok/s. A JIT cache must not change
outputs; a throughput change means something other than caching happened.
