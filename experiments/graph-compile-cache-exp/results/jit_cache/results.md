# jit_cache — is graph capture just uncached JIT?

Model: `Llama-3.1-70B-Instruct`, presharded + `/dev/shm`, TP=4, bristen A100.
Baseline: the phase-7 SOTA config from `lustre-loading-exp` (~183 s cold start).

SGLang's JIT caches (FlashInfer, Triton, tvm-ffi) live under `HOME`, which the
EDF pins to the container's ephemeral overlay — so they are destroyed every job,
by construction. Does persisting them cut the ~107 s of graph capture?

## Design

`scripts/jit_cache.sbatch`, one launch per job, one knob:

```
WARM=0 sbatch experiments/graph-compile-cache-exp/scripts/jit_cache.sbatch
WARM=1 --exclude=<cold node> sbatch experiments/graph-compile-cache-exp/scripts/jit_cache.sbatch
```

Every run is the *first* launch on its node. OS page cache survives across
container runs, so a second launch on the same node is faster in phases the JIT
cache cannot touch — one run per node removes that confound rather than
correcting for it. Submit serially with accumulated `--exclude`; never
`--dependency`, which hands back the same node.

`HOME` is node-local (`/root/jit-home`), seeded from a persistent `/iopsstor`
store before launch and saved back after. The cold run populates the store; the
warm run on a *different* node consumes it — a genuine cross-node test, which is
the question a deployment turns on, since cold starts land on arbitrary nodes
and cache keys could be salted by host or by write-time paths.

## Results

Cache built on **nid002312** (76476), consumed on **nid002313** (76477). Both
runs are the first launch on their node. Staging was 11.9 s / 11.8 GB/s in both,
outside the measured window.

| phase | cold (76476) | warm (76477) | delta |
|---|---:|---:|---:|
| process_startup | 21.86 | 22.55 | +0.69 |
| tp_worker_spawn | 17.44 | 17.65 | +0.21 |
| torch_distributed_init | 3.37 | 3.08 | −0.29 |
| unknown | 2.06 | 1.89 | −0.17 |
| weight_loading | 10.52 | 10.44 | −0.08 |
| kv_cache_alloc | 0.70 | 0.75 | +0.05 |
| **cuda_graph_capture** | 28.27 | 14.14 | **−14.13** |
| piecewise_cuda_graph_capture | 78.84 | 77.59 | −1.25 |
| http_bind | 1.61 | 1.73 | +0.12 |
| **warmup_request(JIT)** | 15.02 | 2.79 | **−12.23** |
| **TOTAL** | **179.69** | **152.61** | **−27.08** |
| throughput tok/s | 401.3 | 401.6 | — (0 errors both) |

## Verdict

**A persisted JIT cache is worth ~27 s, ~15% of cold start, and it transfers
across nodes.** A 3.6 MB artifact built on one machine gives the full win on a
machine that never compiled anything — so shipping a prewarmed cache dir
alongside the model is viable, not a same-node curiosity.

The entire win sits in two phases, exactly as predicted from the code:

- `cuda_graph_capture` −14.13 s (28.27 → 14.14) — decode graphs replay Triton and
  `sgl_kernel` kernels across 52 batch-size buckets
- `warmup_request(JIT)` −12.23 s (15.02 → 2.79) — FlashInfer JIT/autotune on the
  first real forward

Every phase unrelated to JIT is flat within ±1.3 s, which is the internal
control: `weight_loading` −0.08, `http_bind` +0.12, `tp_worker_spawn` +0.21.
Seeding the 3.6 MB from Lustre cost 0.44 s, so the win is ~60× its own setup
cost.

### Reproduction

These runs re-derive an earlier pair (jobs 76456/76457) made with a longer,
since-deleted script. The agreement is close enough to treat the effect as
settled, and gives n=2 per arm:

| | cold | warm | saving |
|---|---:|---:|---:|
| first pass (76456/76457) | 180.00 | 153.84 | 26.16 |
| this pass (76476/76477) | 179.69 | 152.61 | 27.08 |

Per-phase agreement on the JIT-sensitive phases is within noise across passes:
`cuda_graph_capture` cold 28.17 vs 28.27, `warmup_request` warm 2.80 vs 2.79.

### What is actually cached

3.6 MB after the cold run, 56 files:

| directory | size | contents | contributes? |
|---|---:|---|---|
| `.cache/tvm-ffi` | 1.4 MB, 32 files | `sgl_kernel` modules JIT-built and `dlopen()`ed (`fused_rope`, `kvcache`, `resolve_future_token_ids`, `clamp_position`) | likely |
| `.triton/cache` | 140 KB, 10 files | Triton-compiled kernels (cubin/PTX per signature) | likely |
| `.cache/flashinfer/0.6.7.post2` | 76 KB, 12 files | FlashInfer JIT + autotune | likely (`warmup_request`) |
| `.cache/sglang/torch_compile_cache` | 2.0 MB, 2 files | hash→handle map + `computation_graph_*.py` dumps | **no** — no Inductor artifacts exist under `eager` |

Not attributed per-directory: persisting each alone would identify which of the
three carries the win. Worth doing before packaging, since the mechanism should
persist those specific paths (`TRITON_CACHE_DIR`, `SGLANG_CACHE_DIR`,
FlashInfer's own) rather than relocating `HOME` wholesale as this experiment did.

## `piecewise_cuda_graph_capture` is not compilable cost

78.8 s here, and structurally uncacheable. With
`piecewise_cuda_graph_compiler='eager'` (the default), `EagerAdapter.compile()`
returns the fx graph unchanged and `load()` raises `NotImplementedError`
(`compiler_interface.py:481-504`) — **no Inductor codegen happens at all**, so
there is no compile artifact to cache for this path. The 78 s decomposes as ≈2 s
Dynamo trace + **49 s of 58 real forward passes**, one per token bucket up to
8192 tokens, + 28 s of CUDA graph capture. At ~0.85 s per forward on a 70B model
that is genuine compute. Only capturing fewer buckets would remove it, which is
out of scope by decision.

The cache inventory corroborates this — 3.6 MB total is orders of magnitude too
little to represent 49 s of compilation.

## Caches that live outside `HOME`

Two JIT caches are not under `HOME` and are therefore *not* captured by the
design above:

- **TorchInductor** — defaults to `/tmp/torchinductor_$USER`. SGLang only
  redirects it inside `InductorAdaptor.initialize_cache()`
  (`compiler_interface.py:192`), which the default `eager` piecewise compiler
  never reaches. It is nonetheless populated: ~31 SGLang modules decorate ops
  with `@torch.compile(backend=get_compiler_backend())`, and
  `get_compiler_backend()` returns `inductor` on CUDA (`utils/common.py:1915`),
  so rope, vocab embedding, repetition penalty and friends are Inductor-compiled
  every run, producing 5.5 MB / 324 files that go to `/tmp` and are discarded.
- **FlashAttention CUTE DSL** — opt-in, off unless
  `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`. Empty on this model: SGLang's only
  CUTE DSL path is `layers/moe/flashinfer_cutedsl_moe.py` (MoE grouped GEMM),
  which dense bf16 Llama-70B never touches.

**Measured, persisting them buys nothing.** An earlier arm (jobs 76459/76461,
data since deleted) pinned both into the cache tree via environment variables
and reached a warm total of 153.03 s against the equivalent 153.84 s — 0.81 s
apart, well inside the ~±4 s node variance on `cuda_graph_capture`. The 5.5 MB
of Inductor artifacts tripled the seed cost (2.03 s vs 0.33 s) and returned
nothing outside noise; those decorated ops are individually cheap to recompile.
That rules out a *large* Inductor win, not a small one.

The script therefore has no knob for it. **Recommendation: persist FlashInfer +
Triton + tvm-ffi only**; add `FLASH_ATTENTION_CUTE_DSL_CACHE_*` only for models
that exercise it.

## Constraints found

1. **The cache must be copied, not pointed at.** `/dev/shm` is mounted `noexec`
   and Triton `dlopen()`s a `.so` it compiles into its own cache dir — pointing
   `HOME` there kills the scheduler with `failed to map segment from shared
   object`. The cache has to land somewhere local *and* exec-capable; here
   `/root` on the container overlay.
2. **`cp` always exits non-zero on the seed.** It copies the file data fine,
   then fails setting permissions on the destination *directory*
   (`preserving permissions for '/root/jit-home/.': Invalid argument`) because
   GNU `cp` routes mode preservation through `copy_acl()`, which `EINVAL`s
   between Lustre and the container overlay. `--preserve=mode,timestamps` does
   not avoid it. The script tolerates the exit code and verifies the copy by
   file count instead — suppressing it blindly, as the first version did, would
   hide a genuinely incomplete seed.
3. **`du` under-reports on Lustre right after a write.** It reports pre-flush
   allocation: 111 KB for a directory that is really 3.7 MB. File counts are
   reliable; sizes need a settle or a `find -printf '%s'` sum.

4. **`--nccl-port` must be pinned.** Left to itself SGLang picks `nccl_port` via
   `get_free_port()` (`server_args.py:6574`), which handed back 8080 in jobs
   76440 and 76446 — its own scheduler then held the port Uvicorn needed, and
   the run sat out its wait-ready timeout in silence, indistinguishable from a
   hang. It is the only random allocation in the launch. The two runs above
   predate the flag being restored and happened to win the race, as phase 7 did;
   that is luck, not evidence the flag is unnecessary.
