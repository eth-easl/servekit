# graph-compile-cache-exp — summary

**Question.** After `lustre-loading-exp` cut weight loading (812 s → 183 s),
graph capture is the largest remaining phase: `piecewise_cuda_graph_capture`
78 s + `cuda_graph_capture` 28 s = **107 s, 58% of the 183 s baseline**. How much
of that is JIT compilation being thrown away every job, rather than real compute?

See `PLAN.md` for methodology. One sub-experiment so far.

## jit_cache — persist SGLang's JIT caches across jobs

`results/jit_cache/results.md` · `scripts/jit_cache.sbatch`

SGLang's FlashInfer, Triton and tvm-ffi caches live under `HOME`, which the EDF
pins to the container's ephemeral overlay — destroyed every job by construction.
Relocating `HOME` to a node-local dir seeded from `/iopsstor` makes them persist.

**Result: −27 s (~15% of cold start), and it transfers across nodes.**

| | cold (76476) | warm (76477) |
|---|---:|---:|
| cuda_graph_capture | 28.27 | 14.14 |
| warmup_request(JIT) | 15.02 | 2.79 |
| **TOTAL** | **179.69** | **152.61** |

Cache built on nid002312, consumed on nid002313 — a 3.6 MB artifact built on one
machine gives the full win on a machine that never compiled anything, so
shipping a prewarmed cache dir alongside the model is viable. Seeding costs
0.44 s, ~60× less than the win. Every non-JIT phase is flat within ±1.3 s.
Reproduced end-to-end (n=2 per arm, first pass 180.00/153.84).

**The other 78 s is not compilable cost.** Under the default
`piecewise_cuda_graph_compiler='eager'`, `EagerAdapter.compile()` is a no-op and
no Inductor codegen happens, so there is no artifact to cache. It decomposes as
~2 s Dynamo trace + 49 s of 58 real forward passes (one per token bucket) + 28 s
capture — genuine compute. Only capturing fewer buckets removes it, which is out
of scope by decision.

**TorchInductor and FA CUTE DSL are not worth persisting.** Inductor does run
(~31 `@torch.compile`-decorated ops) and writes 5.5 MB to `/tmp`, but persisting
it moved the warm total by 0.81 s — inside node variance — while tripling seed
cost. CUTE DSL is empty on dense Llama.

## Where this leaves the project

Persisting FlashInfer + Triton + tvm-ffi is the packageable win: small, portable,
cheap to seed. The next step before packaging is per-directory attribution —
persist each of the three alone to see which carries the ~27 s — so the mechanism
can pin those specific paths (`TRITON_CACHE_DIR`, `SGLANG_CACHE_DIR`,
FlashInfer's own) rather than relocating `HOME` wholesale as this experiment did.

Remaining cold-start budget after this win (~153 s): `piecewise_cuda_graph_capture`
78 s (out of scope), `process_startup` + `tp_worker_spawn` ~40 s (see the
deferred import-cost item in `CLAUDE.md`), `weight_loading` 10 s.
