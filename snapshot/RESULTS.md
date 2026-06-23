# `snapshot` — M1 results on beverin (MI300A / gfx942)

Validated 2026-06-19 on a CSCS beverin MI300A node (`gfx942:sramecc+:xnack-`),
ROCm via the pinned `vllm/vllm-openai-rocm` image, `--partition=mi300`.

## What was built

The CLI workload now drives the **real HIP path end to end** (it previously
operated only on a host-side struct model):

- `compile_synthetic_module` JITs a 3-kernel module with **hiprtc** → a real
  code object that is serialized into the snapshot and reloaded by a fresh
  process.
- Capture: deterministic VMM allocation at the fixed base → `memcpy_h2d` seed →
  **real `hipStreamBeginCapture`** around the kernel chain → `hipStreamEndCapture`
  → `introspect_graph` validates node count → `instantiate` + `launch`.
- Restore: parse → replay alloc log → **relocate embedded device pointers by Δ**
  → reload the code object → resolve functions → **`rebuild_graph`
  (`hipGraphAddKernelNode`)** → instantiate → launch. No stream capture.
- Bit-identical because the kernels use exact unsigned-integer arithmetic.

## Results

**ctest: 5/5 pass.** Host-only logic tests run on the stub backend; GPU tests
run on the device.

**M1.0 — fixed-base determinism (`probe-base`):**
```
requested_base=0x600000000000  returned_base=0x600000000000  honored=1
```
`hipMemAddressReserve` honors the fixed base on gfx942 → Foundry-style Δ=0 fast
path is viable. Allocation granularity = 4096.

**M1.3 — single-process bit-identical gate (`verify`), relocation exercised:**
```
captured_base=0x600000000000   restored_base=0x14cfa4c00000   (Δ≠0)
relocation_delta_nonzero=1     known_patches=6
capture_matches_reference=1    restore_matches_reference=1    restore_matches_capture=1
```
The capture region is kept mapped, so restore is forced to a different base;
all 6 embedded device pointers (A,B,C / C,OUT / OUT) are patched by the constant
Δ and the restored graph's output is bit-identical to both the captured output
and the host reference.

**M1.4 — two-process acceptance gate (`capture` then `restore` in a fresh
process):**
```
captured_base=0x600000000000   restored_base=0x600000000000   (Δ=0, fixed base honored)
nodes=3   bit_identical_vs_reference=1
```
A fresh process reconstructs a working, correct graph from the serialized file
alone. The fixed base is honored, so this exercises the Δ=0 fast path; the
single-process gate above covers the Δ≠0 relocation path.

**Timing (`bench --scaled`, 192-node graph, 20 iters):**
```
cold_capture_ms=666   warm_restore_ms=34   speedup=19.6x
```
The cold path includes 64 "warmup chains" that model the eager per-shape
warmup/profiling a real serving engine performs and a restored process skips.
The number therefore demonstrates the *principle* (restore avoids the expensive
capture/warmup) on a synthetic workload; it is **not** a measured vLLM cold-start
figure. The pure graph-reconstruction saving (capture vs rebuild, no warmup) is
smaller — wiring this into a real engine is future work.

## Build fixes required to compile the original scaffold on hardware

1. `cmake/SnapshotBackend.cmake`: define `__HIP_PLATFORM_AMD__=1` (HIP backend
   `.cpp` are compiled by g++, not hipcc) and link `libhiprtc`.
2. `hip_vmm.cpp`: `reinterpret_cast` (not `static_cast`) for the pointer-typed
   `hipMemGenericAllocationHandle_t`.
3. `hip_graph.cpp`: `const_cast` the param blob into HIP's `void*` launch config.
4. `backend_factory.cpp`: `make_hip_backend()` declared outside the anonymous
   namespace (else internal-linkage → undefined reference).

## Known issues / not done

- **Real VMM re-reserve-after-release bug:** `hipMemSetAccess` returns
  `invalid argument` when a region is reserved at a base that was *just released
  in the same process*. All workload flows avoid this (capture keeps its region
  mapped during a same-process restore; a two-process restore reserves cleanly
  in a fresh process), and `test_allocator` now runs on the stub backend, so no
  test hits it — but the underlying HIP VMM behavior is unexplained.
- **introspect_graph recovers structure only** (node count + types), not kernel
  identity or arguments — a captured node holds an opaque process-local function
  pointer. Launch identity is recorded at issue time (the synthetic workload acts
  as the interceptor). A real interceptor (LD_PRELOAD / vLLM hook) must do the
  same; pure graph introspection cannot recover entry names.
- **CUDA backend**: `compile_synthetic_module` is a stub (nvrtc TODO); the rest
  of the CUDA backend is unexercised. vLLM integration, the LD_PRELOAD
  interposer, and multi-GPU remain out of scope (future milestones).

## M2.0 — LD_PRELOAD HIP interception on AMD + real vLLM measurement

`libsnapshot_preload.so` interposes `hipStreamBeginCapture` / `hipStreamEndCapture`
and the kernel-launch entry points via `dlsym(RTLD_NEXT)`, forwarding to the real
HIP calls while logging each capture window's wall-time and node count. It does
not serialize anything yet — it proves the interception mechanism (the same one
Foundry relies on for NVIDIA) works on AMD, and measures the target cost.

**Proven on our own real-HIP binary** (`snapshot bench --scaled`): the interposer
saw exactly 1 capture window with 192 nodes and 192 launches inside it — matching
the workload.

**Attached to a real, unmodified vLLM cold start** (GLM-4.7-Flash, ROCm image,
TP=4, `--max-model-len 8192 --max-num-seqs 64`, `recipe/vllm_preload_probe.sbatch`):

```
total HIP graphs captured = 2065  (4 TP workers)
  pid 16473: 481 captures, 35.6s inside capture windows, max 21 nodes
  pid 16474: 528 captures, 34.1s inside capture windows, max 21 nodes
  pid 16475: 528 captures, 36.8s inside capture windows, max 21 nodes
  pid 16476: 528 captures, 35.0s inside capture windows, max 21 nodes
node-count histogram: 20 (×1440), 18 (×315), 21 (×180), 13/15/8/... (smaller)
```

Workers run in parallel, so wall-clock inside `hipStreamBeginCapture/EndCapture`
is **~35–37 s** for this (reduced) config. vLLM uses **piecewise** HIP graphs:
many small graphs (~18–21 nodes each), ~528 per worker, not a few big ones.

**Interpretation.** The pure capture-recording cost (~35 s here) is directly
eliminable by snapshot/restore. The full "capturing" phase in vLLM logs is larger
(~2–3 min at this config) — the difference is eager warmup/forward execution
*between* capture windows, which a restored process also skips if the kernels are
already disk-cached (they are; `VLLM_CACHE_ROOT` works). At prod config
(`--max-num-seqs 256 --max-model-len 131072`) this scales up toward the ~9 min
cold start previously characterized. A snapshot would need to serialize ~2065
graphs × ~20 nodes ≈ 41k kernel nodes (per replica) with their params + pointers.

**Not done (the hard part of vLLM integration):** making torch's allocations
deterministic so serialized device pointers are valid/relocatable on restore.
Our prototype solved this for *its own* VMM allocator (deterministic region +
relocation); for vLLM you would interpose `hipMalloc` and back torch's caching
allocator with deterministic VMM — substantially harder because torch
sub-allocates. That, plus recording launch identity at issue time and rebuilding
2065 graphs on restore, is the remaining work to convert the measured ~35 s
(reduced) / minutes (prod) into an actual restored cold start.

## M2.1 — Allocator-determinism spike (is vLLM snapshot/restore feasible?)

The interposer also logs every device allocation (`hipMalloc` / `hipFree` /
`hipMallocAsync` / `hipMemAddressReserve`, with size + returned pointer) to a
per-process file (`SNAPSHOT_PRELOAD_ALLOC_DIR`). We ran the same vLLM cold start
**twice** (`recipe/vllm_alloc_determinism.sbatch`) and diffed the busiest
worker's allocation sequence across the two runs.

**Result:**
- **Allocation size/order is deterministic.** The size/op sequence is a single
  diff hunk: the first **662 allocations are byte-identical** across both cold
  starts. Run2 only diverges by *appending* a repeating `M 293601280 B / F`
  loop (a ~280 MB KV-cache sizing probe) that run1 was killed just before. Net
  **live allocations are identical: 242 in both runs.**
- **Raw addresses are NOT reproducible.** Pointers land in entirely different VA
  ranges across runs (`0x14e4…` vs `0x151c…`), and the intra-run layout is not a
  constant offset — torch reserves many separate regions, each assigned a
  per-process VA by the driver.

**Conclusion (the feasibility verdict).** Snapshotting vLLM's *raw* addresses is
impossible (per-process VA randomization). But the necessary precondition —
**deterministic allocation size/order** — holds. So the path is to **redirect**
torch's allocations into one contiguous deterministic VMM region (the
`DeterministicAllocator` this prototype already implements): identical size/order
⟹ identical bump-allocated offsets ⟹ a single region base that may differ across
cold starts but a **constant Δ**, which our **already-proven on-hardware
relocation** patches. This validates the design's central bet that relocation
(not fixed-base) is the robust mechanism, and turns "vLLM integration" from an
open question into a concrete build:

1. Interpose `hipMalloc` / `hipMemAddressReserve` / `hipMemCreate` / `hipMemMap`
   and serve them from the deterministic region (the hard part — torch
   sub-allocates within segments, so the region must absorb whole segments).
2. Record kernel-launch identity at `hipModuleLaunchKernel` issue time (already
   interposed) to build the IR, since identity can't be recovered from a graph.
3. On the second cold start: replay the region, relocate by Δ, rebuild the
   ~2065 graphs from the snapshot, skip capture.

## M2.2 — Redirect prototype: deterministic addresses for unmodified apps (incl. PyTorch)

`libsnapshot_redirect.so` interposes `hipMalloc`/`hipFree` and serves allocations
from the prototype's single contiguous `DeterministicAllocator` region (in-region
`hipFree` is a no-op; the allocator's own VMM calls are not interposed, so there
is no recursion). This turns the M2.1 finding into a working capability: identical
size/order ⟹ identical addresses.

**Standalone raw-`hipMalloc` program (`redirect_smoke`):**
```
baseline:      A=0x155434600000  B=0x15542f800000  ... (driver-scattered)   verify OK
redirected #1: A=0x600000000000  B=0x600000400000  C=0x600000800000  OUT=0x600000c00000   verify OK
redirected #2: A=0x600000000000  B=0x600000400000  C=0x600000800000  OUT=0x600000c00000   verify OK
```
Addresses are byte-identical across runs and the result is bit-identical —
kernels and `hipMemcpy` work transparently on VMM-backed memory.

**Real PyTorch (`_torch_redirect_test.py`, a tiny tensor program):**
```
baseline:      data_ptr=0x14d323000000  in_region=False  correct=True
redirected #1: data_ptr=0x600000000000  in_region=True   correct=True
redirected #2: data_ptr=0x600000000000  in_region=True   correct=True
```
PyTorch's caching allocator uses `hipMalloc`, so the redirect catches it: the
tensor lands in the deterministic region at a **byte-identical address across two
cold starts**, and torch computes correctly. This proves — on the real target —
that redirecting torch's allocator yields reproducible device addresses, the
precondition for snapshotting its captured graphs.

**Caveats / remaining edges before full vLLM.** This is a tiny torch program; a
production redirect must handle: (1) the bump allocator never frees (fine for
capture-at-startup, but alloc/free churn in a long run grows the region — needs a
real sub-allocator or honored frees); (2) `expandable_segments` mode, where torch
allocates via `hipMemAddressReserve`/`hipMemCreate` instead of `hipMalloc` (must
redirect the VMM path too); (3) region sizing for multi-GB weights + KV cache;
(4) multi-GPU/TP. The mechanism is proven; scaling it to vLLM's full allocation
surface is the remaining build.

## M2.3 — Full vLLM under redirect: attempted, blocked by HIP VMM set_access

Goal: run a complete vLLM cold start with EVERY `hipMalloc` served from the
deterministic region (`recipe/vllm_redirect.sbatch` + `_vllm_redirect.sh`),
proving the redirect scales to the real engine. Added free-list reuse (so the
repeated KV-probe alloc/free loop doesn't grow the region) and a 64 GiB region.

**Result: vLLM does not yet run under redirect.** It crashes very early, in
`init_device` (the first `torch.zeros_like` on device), with
`torch.AcceleratorError: HIP error: invalid argument`. Verbose tracing pinned the
cause:

```
[redirect] region base=0x600000000000 size=64GiB gran=2097152 fixed_base_honored=1
[redirect] region alloc failed (hipMemSetAccess failed: invalid argument); real hipMalloc
[redirect] SUMMARY served=1 reused=0 passthrough=1 live=1 free=0
```

One of torch's first allocations completes `hipMemCreate` + `hipMemMap` but
**`hipMemSetAccess` returns `invalid argument`**, so that block falls back to real
`hipMalloc` — leaving torch with a broken mix of region and non-region memory,
and the next device op fails. This is the **same HIP VMM `set_access` fragility**
that also broke the single-process re-reserve path (M1). It is allocation-specific
(in the same process, one block's `set_access` succeeds and another's fails) and
correlates with torch interleaving its own HIP calls between our allocations.

Ruled out as causes (each tried on hardware):
- **Granularity** — rounding every block to 2 MiB (`gran=2097152`) did not help.
- **Device ordinal** — making the backend use `hipGetDevice()` instead of a
  hardcoded device 0 (a real latent multi-GPU bug, fixed and kept; single-GPU
  ctest still 5/5) did not change the TP=1 failure.

**Conclusion.** The redirect *mechanism* is proven (M2.2: standalone + real torch
get deterministic, correct, byte-identical addresses), but serving a full engine
from VMM-backed memory is blocked by `hipMemSetAccess` reliability on gfx942 —
a HIP-VMM-semantics issue that needs dedicated investigation (the exact
preconditions `hipMemSetAccess` requires here; whether to grant access over the
whole region in one call; or whether ROCm's expandable-segments / a different
mapping order avoids it). That is the concrete next blocker, not a vLLM- or
design-level problem. Until it's resolved, vLLM-scale snapshot/restore can't be
demonstrated end to end; the simple-case proofs (M1, M2.0–M2.2) stand.

## M2.4 — Full vLLM cold start under deterministic redirect: UNBLOCKED (arena), real end-to-end number

The M2.3 blocker was the VMM backing's `hipMemSetAccess`. The redirect only needs
**deterministic addresses**, not the fixed base `0x600000000000` — and relocation
(M1, proven) already handles a shifted base. So the redirect was repointed at a
plain **`hipMalloc` arena**: one real `hipMalloc` of the whole region at init,
then bump + free-list sub-allocation inside it. No `hipMemCreate`/`hipMemMap`/
`hipMemSetAccess` at all → the gfx942 set_access fragility is sidestepped
entirely. Toggle with `SNAPSHOT_REDIRECT_ARENA=1` (default); the VMM path is kept
for the M2.2 standalone demos.

**Result: vLLM runs a full cold start entirely on redirected, deterministic
memory, and serves correct tokens.** GLM-4.7-Flash, TP=1, `--max-model-len 8192
--max-num-seqs 64`, 64 GiB arena, `gpu-memory-utilization 0.60`
(`recipe/vllm_redirect.sbatch` + `_vllm_redirect.sh`, two cold starts back to
back). The arena `hipMalloc(64 GiB)` succeeds, the engine passes `init_device`
(the exact spot M2.3 crashed), loads 56.01 GiB of weights into the arena, sizes
KV, captures graphs, and reaches `/health` ready. A real completion confirms
correctness: prompt "The capital of France is" → "**Paris. The capital of the
United Kingdom**".

**Measured cold start to server-ready (warm `torch.compile` cache): `224 s`.**
Breakdown (warm run):
- weight load **93 s** (56 GiB, I/O-bound — *not* eliminable by graph snapshot)
- init engine (profile + KV + capture + warmup) **59.5 s**, of which compilation
  12.5 s and graph capture ≈15 s (19 PIECEWISE + 11 FULL capture sizes; the
  driver-level count is ~528 piecewise sub-graphs/worker per M2.0)
- the first cold start (cold compile cache) was slower: 112 s load + 75.9 s init
  with compilation **33.25 s** — i.e. `torch.compile`'s disk cache already saves
  ~21 s, independent of our work.

**Determinism (the snapshot precondition) — FULLY PROVEN.** Comparing the
per-worker allocation logs by **offset** (va − arena_base) across two warm cold
starts at the 72 GiB sweet spot below: **all 1323 allocations are byte-identical
by (op, size, rounded, offset)** — `1323/1323 FULLY DETERMINISTIC` — covering the
entire working set: 56 GiB weight load, KV, *and* the captured-graph pools, with
**passthrough = 0**. The two arenas landed at different bases (constant Δ) → the
relocation-not-fixed-base path is the one exercised, as designed. (An earlier
64 GiB / cold-vs-warm comparison matched only the first 320 allocs because the
arena spilled capture pools to passthrough and the two runs differed in
`torch.compile` cache state; the clean two-warm-run / 72 GiB comparison removes
both confounds and matches in full.) Both runs reached `/health` ready (222 s,
218 s) and served the correct completion.

**Sizing — the arena-vs-profiler window.** The arena is pre-grabbed with one real
`hipMalloc`, so vLLM's profiler sees it as used and computes
`available_kv ≈ gpu_mem_util·128 − arena`. That creates a window:
- too low `gpu_mem_util` (e.g. 0.52 at 72 GiB) → `available_kv ≤ 0` →
  `ValueError: No available memory for the cache blocks` (engine aborts at KV
  sizing, before capture);
- too high → KV grows until weights+KV+capture exceed the arena → capture pools
  spill to passthrough (non-deterministic).

For GLM-4.7-Flash (56 GiB weights) on a 128 GiB APU the sweet spot is
**arena = 72 GiB, `gpu_mem_util` = 0.60** (KV ≈ 4.8 GiB, 8.7× concurrency;
weights+KV+capture ≈ 60 GiB < 72 → passthrough = 0). The valid band is roughly
`gpu_mem_util ∈ (0.5625, 0.656)` at 72 GiB.

**Where this leaves the saving.** The engine now runs on a deterministic,
snapshot-ready address space (the hard precondition). The remaining work to turn
that into a cold-start *saving* is to skip graph capture on a second start by
restoring the serialized graphs — at this single-GPU config the eliminable
capture is ≈15 s of 224 s; the large (~9 min) capture cost the design targets is
at prod concurrency. The capture-skip restore is the next milestone.

## M2.5 — How much cold start can a graph persist/restore avoid? (measured)

Question: given that JIT/inductor kernels are already disk-cached (reused) and the
captured graphs can be persisted (M2.4 proved the addresses are deterministic),
how much of the cold start does CUDA-graph capture cost — i.e. what would a
persist/restore save? Measured by comparing a warm-cache cold start **with**
capture vs **`--enforce-eager`** (no capture), vanilla vLLM (no redirect),
GLM-4.7-Flash TP=1, `--max-num-seqs 64 --max-model-len 8192`
(`recipe/measure_init.sbatch`, `measure_repro.sbatch`, `_vllm_measure.sh`).

**The capture cost is dominated by KV concurrency, and at a realistic setting it
is the largest single cold-start cost.** At `gpu-memory-utilization 0.70`
(concurrency 76×):

| variant (warm cache) | cold start to ready | init engine | PIECEWISE capture |
|---|---|---|---|
| with CUDA graphs (run 1) | 527 s | 335 s (compile 11.6 s) | 4:45 (285 s) |
| with CUDA graphs (run 2) | 500 s | 318 s (compile 11.2 s) | 4:17 (257 s) |
| with CUDA graphs (run 3) | 564 s | 384 s (compile 11.1 s) | 5:15 (315 s) |
| **`--enforce-eager`** (no capture) | **227 s** | **41 s** | — |

- **Genuine, recurring capture — not one-time autotuning.** Capture reproduces at
  ~260–315 s across three runs, and the `[aiter]` MoE-autotune log line count is
  identical (27) every run → it is per-capture-step logging, not a cold/warm
  cache effect. This matches the prior production finding (~9 min capture at
  `max-num-seqs 256`, recurs every cold start).
- **Concurrency, not just batch count, drives it.** At `gpu-mem-util 0.60`
  (concurrency 8.7×) capture was only ~14 s; at 0.70 (76×) it is ~290 s. (For an
  MoE model, higher concurrency widens the per-expert token ranges captured.)
- **Decomposition (gmu 0.70):** init-engine 318 s = compile 11 s + capture ~264 s
  + profile/KV/warmup ~43 s. The `--enforce-eager` init-engine (41 s) ≈ that
  profile/KV/warmup remainder — confirming capture is the ~264 s delta.

**Conclusion.** With kernels already cached, **graph capture is the dominant
eliminable cold-start cost (~264 s here, ~9 min at prod concurrency).** Persisting
the captured graphs and restoring them (skipping capture) would cut this cold
start from **~530 s → ~240 s (eager 227 s + compile 11 s + restore overhead) — a
≈55% reduction.** The address-determinism precondition for that restore is
already proven (M2.4); building the persist/restore is the next milestone.

## M3a — de-risk graph persist/restore: identity recovery via the real interposer (RAN; identity partially recovered, gap precisely characterized)

M2.5 showed graph capture is the dominant eliminable cold-start cost (~264 s
here, ~9 min at prod); M2.4 proved the address-determinism precondition. The
remaining question before committing to a full vLLM persist/restore hook was
the one risk named in the design: *kernel-identity recovery*. **Reframed: it is
not recovery but recording.** Identity is established at clean, interceptable
HIP sites and correlated with the actual captured graph at `hipStreamEndCapture`
via `hipGraphKernelNodeGetParams` (the driver has already resolved every launch —
by whatever symbol — into nodes with `hipFunction_t` handles).

**Cluster results (GLM-4.7-Flash, TP=1, gmu 0.60, beverin mi300, ROCm 6.3):**

| step | gate | result |
|---|---|---|
| **M3a.2** real-interposer captures our synthetic workload → fresh-process `restore` | **bit-identical** vs reference; node count matches | **PASS** — 192/192 nodes, `bit_identical_vs_reference=1`, `MATCH_GATE=PASS` |
| **M3a.3** real vLLM under `LD_PRELOAD="redirect record"` | every node → known `(module, entry)` | **PARTIAL** — captured 4 real graphs (6–17 nodes each, 3 modules / 32 MiB HSACO); **~35–45 % of nodes resolve** via the module path, the rest are host-registered |
| **M3a.4** rebuild one real vLLM graph | `rebuild_graph` + instantiate succeed | **BLOCKED** — refuses rebuild of the unresolved nodes (correct/safe); unblocked exactly when identity is complete |

**The precise finding that de-risks the headline risk.** vLLM/PyTorch reach HIP
through *three* observably-distinct paths, and the probes (`pid=<gpu> FIRST
<symbol>`) show exactly which fired in the GPU worker:

1. `hipModuleLoad` (code objects loaded from **files**), `hipModuleGetFunction`,
   then captured → these nodes' `func` **correlate** with the load-time map →
   identity recovered. ~35–45 % of nodes. **Works.**
2. `hipStreamBeginCapture` / `hipStreamEndCapture` — capture delimiters seen ✓.
3. **`hipLaunchKernel` (`<<<>>>`, PyTorch/ATen host-registered kernels) — NOT
   `hipModuleLaunchKernel`.** These nodes' `func` is never obtained via
   `hipModuleGetFunction`, so the module map misses them → ~55–65 % of nodes,
   surfaced as `nodes_without_identity`. **This is the remaining gap.**

The gap is **one well-defined symbol**: interpose `__hipRegisterFunction`
(the host-symbol→device-function registration) to extend the identity map to
host-registered kernels — same `dlsym(RTLD_NEXT)` technique as every other
interposition, no new mechanism. The recorder's introspection-at-EndCapture
design means identity recovery is independent of which launch symbol was used;
only the *correlation map* needs the extra registration source.

**Conclusion:** the headline risk is de-risked. Identity recovery is feasible
and partially proven on real vLLM; the remaining work is mechanical (one more
interposition), not research. Bit-identical on a real vLLM graph remains M3b
(needs buffer-content snapshotting). Full design + run instructions in
`docs/M3A.md`.

## M3b — end-to-end vLLM graph recording (identity SOLVED; launch-time kernarg capture working; rebuild reaches AddKernelNode)

Goal: drive the recorder through a REAL vLLM cold start so it captures a graph,
names every kernel node, and `rebuild-check` reconstructs it. The session took
the run from "vLLM never idles, 2 names, no FLUSH" to a clean end-to-end capture
→ name → FLUSH → inspect → rebuild-check (no crash), and in doing so surfaced
and precisely characterized the two real blockers (one of which overturns an
M3a assumption). GLM-4.7-Flash, TP=1, gmu 0.60, 72 GiB arena, beverin MI300A.

**What was fixed (each a cluster job on beverin, 509272 → 509302):**

1. **`--cudagraph-capture-sizes 1` (the lever the design needed).** vLLM's default
   capture list (`[1,2,4] + range(8,256,8) …`) is ~34 sizes → ~528 driver-level
   piecewise sub-graphs, so the capture phase ran minutes and vLLM never reached
   the idle window the off-path namer needs. Clamping to one batch size collapses
   that to a single PIECEWISE + FULL capture (~3 s) and `/health` arrives at
   ~250 s. (`recipe/_vllm_record.sh`, `CAPTURE_SIZES` env.)
2. **Sentinel-gated name resolution.** The original background namer called
   `hipKernelNameRef` whenever the capture lock *looked* free — a check-then-call
   race that segfaulted when the next `BeginCapture` started mid-call. Now the
   recipe touches a sentinel file at `/health` and the recorder will not name
   until it exists (`SNAPSHOT_RECORD_DRAIN_MODE=idle`, `…_DRAIN_SENTINEL`).
3. **Main-thread inline naming.** `hipKernelNameRef` is **not thread-safe off the
   main HIP thread** (segfaults on its 2nd call from a background thread).
   `drain_naming_inline()` now runs on the inference thread: after `/health` the
   recipe sends one completion, whose `hipLaunchKernel`/`hipExtModuleLaunchKernel`
   calls drain the queue inline (one func per launch).
4. **Host-launched nodes named from the registration map (free, safe).** A
   captured `<<<>>>` node's `func` is the **host function pointer** (observed in
   the host VA range, e.g. `0x14c…`), which `__hipRegisterFunction` already gave
   us in `g_host_functions` (`device_name`). So those nodes are named from the
   map — no `hipKernelNameRef` — including a CK GEMM (`wvSplitK_hf_sml`) and an
   aiter kernel (`fused_qk_rmsnorm_kernel`).

**End-to-end result (one PIECEWISE decode graph, 6 kernel nodes):**

```
NAMER(main) func=0x39975b50  -> 'triton_red_fused__to_copy_embedding_rms_norm_0' [nameref]
NAMER(main) func=0x14fd887ad340 -> '_Z16wvSplitK_hf_sml_I14__hip_bfloat16...'     [host-map]
NAMER(main) func=0x14d033ece6c0 -> '_ZN5aiter23fused_qk_rmsnorm_kernel...'       [host-map]
NAMER(main) func=0x372c49a0  -> ''  [nameref:SKIPPED(cap)]   (device handle)
NAMER(main) func=0x3f745f30  -> ''  [nameref:SKIPPED(cap)]   (device handle)
[record] FLUSH idx=0 nodes=6 named=4 with_module=0 modules=0 wrote=1
nodes_without_identity=6   IDENTITY_GATE=FAIL
rebuild-check: refusing to rebuild: 6 node(s) have no recorded identity
```

4 of 6 nodes are named (across three kernel classes — Triton fused, CK/hipBLAS,
aiter), the snapshot is written, and the gates run without crashing.

**Both blockers are now SOLVED.** The remainder of this section records the
resolution; the final end-to-end result (job 509333, same config) is:

```
NAMER(main) func=0x... -> 'triton_red_fused__to_copy_embedding_rms_norm_0' [nameref:fork]
NAMER(main) func=0x... -> '_Z16wvSplitK_hf_sml_...'                       [host-map]
NAMER(main) func=0x... -> '_ZN5aiter23fused_qk_rmsnorm_kernel...'         [host-map]
NAMER(main) func=0x... -> 'triton_poi_fused_..._unsqueeze_1'             [nameref:fork]
NAMER(main) func=0x... -> '_Z16wvSplitK_hf_sml_...'                       [host-map]
NAMER(main) func=0x... -> 'triton_poi_fused_..._view_2'                  [nameref:fork]
[record] FLUSH idx=0 nodes=6 named=6 with_module=6 (syms=6) modules=5 wrote=1
            [tables mods=3 fatbins=4083 hsa=60 funcs=2 hostfuncs=135563]
IDENTITY_GATE=PASS   (nodes_with_identity=6 / 6)
hipModuleGetFunction ... triton_red...   rc=0   (function resolves)
hipModuleGetFunction ... wvSplitK_hf_sml rc=0
hipModuleGetFunction ... aiter fused_qk  rc=0
backend_error: hipGraphInstantiate failed: invalid argument
```

### (A) SOLVED — fork-isolated `hipKernelNameRef` names every device-handle node

The 2nd-handle segfault is **cumulative-state-dependent**, not handle-specific:
calling `hipKernelNameRef` once corrupts runtime state such that the next
call faults. The fix is to run EACH query in a fresh `fork()`ed child
(`kernel_name_via_fork`): `fork()` duplicates the address space, so the
`hipFunction_t` (a pointer into the HIP runtime's heap) stays valid in the
child, a metadata read needs no live GPU work, and a fault kills only the
child (the parent reads the name over a pipe, or notices and `SIGKILL`s it).
Two watchdogs: `alarm()` in the child (in case the runtime deadlocks on a lock
held at fork) and a `poll()` timeout in the parent. All 3 device-handle Triton
nodes now name cleanly; `SNAPSHOT_RECORD_NAMEREF_MODE=fork` (default) / `cap`
(legacy) / `off`. The in-process `siglongjmp` guard was insufficient because
PyTorch's `c10::SignalHandler` wins the SIGSEGV disposition — isolation is the
only robust answer.

### (B) SOLVED — `name → image` linkage via ELF symbols + HSA capture + arch filter

The gap was that `name → module_hash` was built only from `hipModuleGetFunction`,
which Triton/CK/aiter never call for these kernels. It is closed three ways:

1. **A global `name → image` map from ELF symbol tables.**
   `extract_elf_symbols` now walks every captured module + fatbin image's
   `.symtab`/`.dynsym` AND the AMDGPU **metadata note** (`.name:` in the
   `NT_AMDGPU_METADATA` YAML — the only place Triton kernel names live, since
   they are absent from `.symtab`). Any node whose name we learned (by
   `hipKernelNameRef`, `hipModuleGetFunction`, or `__hipRegisterFunction`) now
   links to a code object regardless of how its handle was obtained.
2. **HSA code-object capture.** Triton JIT does NOT load via the HIP module API
   at all (confirmed: no `hipModuleLoadData` FIRST probe, triton names absent
   from every HIP module/fatbin) — it registers HSACO directly through the ROCr
   runtime. `hsa_code_object_reader_create_from_memory` is interposed to copy
   the full HSACO (the API gives an authoritative size) once, at load — not on
   the hot `hsa_executable_freeze` path that previously stalled init. 60 Triton
   HSACOs captured per run.
3. **Per-arch bundle filtering.** A PyTorch fat binary bundles an amdgcn entry
   per target arch; without filtering, `wvSplitK` was matched to the **gfx1100**
   entry, so `hipModuleLoadData` rejected it on gfx942 ("no kernel image
   available"). `extract_amdgpu_elfs` and the FLUSH linker now keep only the
   entry whose triple matches `gcnArchName`'s gfx token (with a non-digit-after
   guard so `gfx942` ≠ `gfx9421`); `query_arch` retries until non-empty so a
   static-init-time call can't cache an empty token.

### Remaining step (new, precise): kernel-argument capture at LAUNCH time

The rebuild now loads all 5 modules, resolves all 3 distinct kernels
(`rc=0`), and builds the graph — previously failing only at the final
`hipGraphInstantiate`. The cause was proven: **`hipGraphKernelNodeGetParams`
returns `extra=NULL` on ROCm**, so the kernarg cannot be recovered from the
captured node (`extract_param_blob #0..5 extra=(nil)` for all 6 nodes).

### (C) SOLVED — launch-time kernarg capture (jobs 509884 → 509915)

The arg bytes are now snapshotted at LAUNCH time (while the caller's buffer is
still alive), gated on the capture window, in issue order, and merged into each
node's `param_blob` at FLUSH (`args_filled=N`). The rebuild decodes a tag byte
(`0`=buffer/`extra`, `1`=array/`kernelParams`) and replays the right HIP layout.

Three non-obvious problems were solved along the way:

1. **Over-read segfault (jobs 509884–509896).** A "generous 16 B per-arg slice"
faulted — real args are 4/8 B, and over-reading crosses page boundaries. Worse,
some launch sites (notably `wvSplitK` via `hipLaunchKernel`) leave a non-NULL
garbage value where the NULL terminator should be, so a plain NULL-scan
overcounts and pulls in device-range addresses (`args[12]=0x1000000040`).
2. **`write(/dev/null)` is NOT a readability probe.** `/dev/null`'s write
handler discards input *without touching the source buffer*, so it never
fault-tests — the following `memcpy` segfaulted. Replaced with
`process_vm_readv(getpid(), …)`, which fault-tests AND copies in one call
(returns a short count / `EFAULT` at the first unmapped page, never `SIGSEGV`).
3. **Readability-terminated scan.** The arg count is now taken as the first NULL
*or* first non-NULL-unreadable entry (a real arg pointer is always readable
host memory). This correctly caps `wvSplitK` at 12 args.

`SNAPSHOT_RECORD_CAPTURE_ARGS=1` (off by default; recipe `CAPTURE_KERNARG=1`).
Result: 3/6 nodes carry real args; the rebuild adds them with `rc=0`
(`hipGraphAddKernelNode` succeeds for the 3 arg-bearing nodes — a first).

### (C′) DONE — exact arg counts + pointer relocation (local-tested; pending cluster build)

Both precise items from the remaining step are now implemented and unit-tested
locally (pure C++, no HIP dependency):

* **Exact arg counts from AMDGPU MessagePack metadata.** A minimal big-endian
  msgpack walker (`extract_amdgpu_kernels` in `record.cpp`) parses the
  `NT_AMDGPU_METADATA` note from each module ELF, extracting every kernel's
  `.name`, `.kernarg_segment_size`, and per-arg `(.offset, .size, .value_kind)`.
  The `is_pointer` flag is set for `global_buffer` / `dynamic_shared_pointer` /
  etc. value-kinds. **Critical format detail**: on MI300A / ROCm 6.3, the
  metadata note uses **type 0x20 with name "AMDGPU\0"** (code-object v4+),
  NOT the older type 0x3a with name "AMD\0" (v3). The parser handles BOTH.
  Verified against the real snapshot: all 5 modules carry type-0x20 msgpack
  notes with `amdhsa.kernels` (e.g. `triton_red_fused__to_copy_embedding_rms_norm_0`
  has 9 args / 7 pointers / kernarg_sz=64; `triton_poi_fused_add_...` has
  7 args / 6 pointers). Tested against synthetic ELF+msgpack (15-arg kernel
  with 5 pointers, multi-kernel docs, str16 names, truncated/YAML/v3 safety,
  v4+ type-0x20 format). The rebuild's `decode_recorded_args` now pads to the
  **exact** declared count (not the blind 32-arg crash-safety pad) via
  `HipBackend::sig_by_name_`, populated during `load_module`. Empty blobs
  (child-graph-inlined nodes with no captured args) are also padded to the
  signature count so AddKernelNode gets a structurally-valid node.
* **Precise pointer relocation.** `tag1_blob_ptr_offsets(blob, sig)` computes
  the blob-relative byte offsets of pointer args for the array-format (tag-1)
  kernarg blob, walking the `{u32 len, bytes[len]}` packing. The rebuild-check
  populates `node.kernel.ptr_offsets` from these before `relocate_graph_ir`, so
  relocation patches **only** the args the metadata marks as pointers —
  replacing the blind 8-byte scan that could false-positive on scalar values.
  Nodes without a matching signature keep empty offsets and fall back to
  blind-scan.

**Net:** identity recovery is COMPLETE (6/6 nodes, `IDENTITY_GATE=PASS`), the
code-object pipeline is proven (load + resolve + graph-build), launch-time
kernarg capture works (3/6 nodes with args via hooked launch APIs), AND the
rebuild now has exact arg counts + precise pointer identification from parsed
metadata. The next end-to-end test (cluster build pending) verifies whether all
6 nodes pass `hipGraphAddKernelNode` + `hipGraphInstantiate`.

## Reproduce

```
rcc push
ssh beverin 'cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && \
  sbatch snapshot/recipe/build_snapshot.sbatch'
# full vLLM cold start on the deterministic arena + cold-start timing + determinism diff:
ssh beverin 'cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && \
  sbatch --export=ALL,GMU=0.52,REGION_GIB=72 snapshot/recipe/vllm_redirect.sbatch'
# M3b — record one real vLLM graph end-to-end (capture→name→FLUSH→rebuild-check),
#       with launch-time kernarg capture on (CAPTURE_KERNARG=1):
ssh beverin 'cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && \
  sbatch --export=ALL,MAX_GRAPHS=1,CAPTURE_SIZES=1,CAPTURE_KERNARG=1 snapshot/recipe/vllm_record.sbatch'
```
