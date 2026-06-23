# M3a — de-risk graph persist/restore: identity recovery + structural rebuild

## Why this milestone exists

M2.5 measured the prize: at realistic concurrency (`gmu 0.70`, 76×), graph
capture is **~264 s of a ~530 s cold start** — the single largest eliminable
cost, recurring every cold start. M2.4 proved the precondition (1323/1323
allocations byte-identical by offset across two cold starts on the deterministic
arena). The remaining work to convert that into a saving is **persist the
captured graphs on start #1, restore them on start #2 instead of capturing**.

Before committing to the full vLLM hook, M3a de-risks the cheap unknowns. It is
deliberately scoped to **identity recovery + structural rebuild**, not to a
bit-identical vLLM cold start (that is M3b, because it also needs to snapshot
buffer contents — weights/KV — which capture alone does not give us).

## The reframe that collapses the headline risk

The design listed the one genuine technical risk as *"kernel-identity recovery
(mapping captured graph nodes back to their kernel module/entry)."* Stated as
recovery it is impossible — a captured HIP node holds an opaque, process-local
`hipFunction_t`, and `introspect_graph` cannot reverse it (confirmed in M1).

**But no recovery is needed.** Identity flows into the system at exactly two
clean, interceptable sites, and we *record* it there:

| HIP call | gives us |
|---|---|
| `hipModuleLoadData(mod, image)` / `hipModuleLoad(mod, fname)` | module handle → **image bytes** (so we can serialize + reload it) |
| `hipModuleGetFunction(fn, mod, name)` | function handle → **(module hash, entry name)** |

Correlated with `hipModuleLaunchKernel(fn, grid, block, shared, stream, extra)`
inside a `hipStreamBeginCapture`/`hipStreamEndCapture` window, that is the
complete kernel-identity IR. The M2.0 interposer interposed launch + capture but
**not** the two calls above — that was the actual gap, and closing it is routine
symbol interposition, not a research question.

**One genuine sub-detail remains** — the launch API does not say which bytes of
the param blob are pointers, so we cannot record `ptr_offsets`. That is already
solved: record the raw blob; the proven M1 blind-scan relocation discovers the
pointers at restore time over the recorded region
`[region_base, region_base + region_size)`. No new mechanism.

## What was built

* `include/snapshot/record.hpp` + `csrc/core/record.cpp` — host-testable
  helpers: `elf_code_object_size` (recover an HSACO/ELF image length from its
  header — `hipModuleLoadData` takes no size), `assemble_recorded_snapshot`
  (recorded events → `SnapshotData`), `summarize_snapshot` (the gate counters).
* `csrc/preload/snapshot_record.cpp` → `libsnapshot_record.so` — the recorder
  interposer. Interposes module load/unload, `hipModuleGetFunction`, capture
  delimiters, `hipModuleLaunchKernel` (records identity + param blob extracted
  from the driver-style `extra` config), and the VMM calls
  (`hipMemAddressReserve`/`hipMemMap`) so the synthetic workload's region +
  buffers are captured too. Hot path (`hipModuleLaunchKernel`) is lock-free
  unless a capture is open and the graph cap is unmet. Region base/size come
  from the VMM reserve (synthetic) or env (vLLM-under-redirect).
* `snapshot inspect <file>` — host-only snapshot reader; prints the identity
  summary and the `IDENTITY_GATE` verdict (no GPU needed).
* `snapshot rebuild-check <file>` — GPU rebuild gate: loads recorded module
  images, resolves entries, `rebuild_graph` + `instantiate` (no launch).
* `tests/test_record_ir.cpp` — host unit test for the IR assembly + ELF parse
  (passes on the STUB build, runs in `ctest` on every backend).

## Gates (each independently checkable, so a failure localizes cheaply)

| Step | Gate | Where | Result |
|---|---|---|---|
| **M3a.1+2** Recorder captures our synthetic workload → fresh process `restore` | **bit-identical** vs host reference; recorder node count == workload node count | `record_synthetic.sbatch` | **PASS** — 192/192 nodes, `bit_identical_vs_reference=1` |
| **M3a.3** Recorder on real vLLM under redirect | every node → known `(module, entry)` / name | `vllm_record.sbatch` | **SOLVED (M3a.6)** — the launch-interception ceiling is real (~50–71 % named at issue time), but **`hipKernelNameRef(hipFunction_t)` names every captured node directly, off-path** — including the kernels no launch symbol reaches |
| **M3a.4** Rebuild one real vLLM graph | `rebuild_graph` + `hipGraphInstantiate` succeed | same job, `rebuild-check` | **UNBLOCKED in principle** — names now recoverable for all nodes; wiring names→modules into the snapshot before rebuild is the remaining engineering |

## Empirical findings on real vLLM (the actual de-risk)

The recorder's one-shot first-call probes (`[record] pid=<gpu> FIRST <symbol>`)
showed exactly which HIP entry points the GLM-4.7-Flash GPU worker reaches:

```
FIRST hipModuleLoad          ← code objects loaded from FILES (not LoadData!)
FIRST hipModuleGetFunction
FIRST hipStreamBeginCapture
FIRST hipStreamEndCapture
(hipModuleLaunchKernel NEVER fires)
```

Two things wrong with the original design hypothesis, both now fixed/understood:

1. **The launch hot path is `hipLaunchKernel` (`<<<>>>`), NOT `hipModuleLaunchKernel`.**
   PyTorch/ATen/torch-compiled kernels are host-registered; they are launched by
   host function pointer, never through the explicit module-launch API. So
   recording per-launch events via `hipModuleLaunchKernel` captures *nothing*
   on real vLLM (it captured 192/192 on our synthetic workload, which does use
   the module-launch path — hence M3a.2 passed).
2. **The fix is to introspect the actual captured graph at `hipStreamEndCapture`.**
   By then the driver has resolved every launch — regardless of symbol — into
   kernel nodes whose `func` (`hipFunction_t`, via `hipGraphKernelNodeGetParams`)
   is correlatable against the load-time `(handle → module, entry)` map built at
   `hipModuleGetFunction` time. This is authoritative and launch-API-agnostic.

With introspection, 4 real vLLM graphs were captured (6–17 kernel nodes each,
3 modules / 32 MiB of HSACO — the Triton/CK/torch-compiled code objects):

| graph | nodes | identity | unknown | known % |
|---|---|---|---|---|
| 0 | 6  | 2 | 4  | 33 % |
| 1 | 11 | 5 | 6  | 45 % |
| 2 | 17 | 6 | 11 | 35 % |
| 3 | 17 | 6 | 11 | 35 % |

**The ~55–65 % "unknown" nodes are host-registered kernels** (launched via
`hipLaunchKernel`): their `func` was never produced by `hipModuleGetFunction`,
so the module map misses them. They are registered through a *different* clean,
interceptable site — `__hipRegisterFunction` (host-symbol → device-function) —
which the recorder does not yet interpose.

## What this session actually established (supersedes the table above)

The "interpose `__hipRegisterFunction`" plan above was tried and **does not
close the gap** — and chasing it surfaced the real, deeper finding. The
investigation (jobs 509179–509241 on beverin, GLM-4.7-Flash, gmu 0.60, 72 GiB
arena, 4 graphs/run) ruled out hypotheses one by one:

1. **`__hipRegisterFunction` works** (136 k host kernels registered) but a
   captured node's `func` for a host kernel is an **opaque device
   `hipFunction_t` handle** (e.g. `0x39ae2f30`), **not** the host pointer
   `__hipRegisterFunction` was given (`0x151b…`). Keying identity by the host
   pointer matches nothing.
2. **`hipGetFuncBySymbol(hostFunc)` works** and returns a device handle, but a
   *different* handle (`0x37…`) than the one stored in the captured node
   (`0x39…`) — the canonical handle **aliases** the captured one. Handle
   correlation is therefore unreliable. (Resolving the full 136 k registry
   eagerly inside `hipStreamEndCapture` also stalled capture past the deadline —
   wrong place, abandoned.)
3. **Identity must be recorded at issue time, not recovered from the handle.**
   So the recorder now records each launch's identity *in issue order* and
   index-aligns it to the introspected nodes (`reconcile_identity_by_position`).
   The launch APIs that actually fire on this stack are exactly two:
   * **`hipExtModuleLaunchKernel`** (`hip_ext.h`) — Tensile/hipBLASLt GEMMs.
     `f` is a `hipFunction_t` from `hipModuleGetFunction` → identifiable.
   * **`hipLaunchKernel`** — PyTorch/ATen host kernels → identifiable via the
     `__hipRegisterFunction` host-pointer map.
   Neither `hipModuleLaunchKernel`, the cooperative variants, `hipLaunchByPtr`,
   `hipLaunchKernelExC`, `hipDrvLaunchKernelEx`, `hipExtLaunchKernel`, nor any
   `hipGraph*` construction API ever fires (verified by forward-only probes).

## The launch-path interception ceiling (the real M3a finding)

With both firing launch APIs recorded, the per-graph result is stable and
reproducible:

| graph | kernel nodes | recorded launches (identifiable) | nodes with **no** interceptable launch |
|---|---|---|---|
| 0 | 6  | 3  | 3 |
| 1 | 11 | 6  | 5 |
| 2 | 17 | 12 | 5 |
| 3 | 17 | 12 | 5 |

Instrumented `record_issue` counters prove this is **not** a recording bug:
`dropped=0`, `fallback=0`, every recorded launch resolved to a non-zero identity
(`issue zero=0`), and the recorded count equals the raw count of
`hipExtModuleLaunchKernel`+`hipLaunchKernel` calls during capture. So **~30–50 %
of the captured kernel nodes enter the graph without any HIP launch symbol that
LD_PRELOAD can interpose** — the residual `0x39–0x3e` handles. On a MoE model
these are almost certainly the **aiter / Composable-Kernel** MoE-expert and
attention (FA) kernels, which launch by calling the HIP runtime directly /
through a statically-linked copy, bypassing the dynamic-symbol PLT entries the
shim hooks.

**Consequence:** a pure-LD_PRELOAD recorder has a hard ceiling on this stack. It
cleanly recovers GEMM + PyTorch-host kernel identity (~50–71 % of nodes,
trending higher on larger graphs), but cannot name the aiter/CK kernels at issue
time. The positional fallback can make `nodes_without_identity` hit 0 by
consuming the recorded ids, but that **double-assigns** identities and is not
trustworthy per-node — so M3a.4 (a *correct* rebuild) stays blocked.

## Options to break the ceiling (for M3b)

1. **Interpose one layer down — ROCr/HSA** (`libhsa-runtime64.so`): kernel
   dispatch goes through HSA AQL packets and `hsa_executable_symbol_*`
   regardless of how HIP is linked, so the code object + kernel name are
   recoverable there. Biggest change, but vendor-position-equivalent to where
   Foundry sits on NVIDIA, and the most likely complete fix.
2. **aiter / framework cooperation**: a vLLM- or aiter-side hook that reports
   the `(hipFunction_t → name, code object)` for the kernels it launches.
3. **Hybrid persistence**: persist only the identifiable subgraph and re-capture
   the aiter kernels — reduces but does not eliminate the cold-start saving.

Recommendation: prototype the HSA-layer interposer (option 1) as M3a.6 before
committing to the full M3b integration — it is the load-bearing unknown.

## M3a.6 — the ceiling breaks (and the "aiter" guess was wrong)

The HSA-layer prototype was built (interpose `hsa_executable_freeze`, iterate
each frozen executable's kernel symbols → `kernel_object → name` map). It works:
**9683 kernel symbols** captured. But the captured node's opaque func is **not**
an HSA `kernel_object` (`hsa_hits_total=0`) — it is a HIP `hipFunction_t`
*wrapper*. So HSA alone didn't bridge to the captured node.

The real bridge turned out to be a single HIP call. ROCm exports
**`const char* hipKernelNameRef(const hipFunction_t f)`** — a live-handle query
that returns a kernel's name from its `hipFunction_t`, regardless of how it was
launched or loaded. Validated against known handles
(`kname == hipKernelNameRef` for the Cijk GEMMs). The catch: calling it **inline
during introspection hangs** the capture thread (HIP holds a global lock across
stream capture). Resolved by a **background namer thread** that drains the
captured node funcs off the capture-critical path. Result:

```
NAMER func=0x399e7160 -> 'triton_red_fused__to_copy_embedding_rms_norm_0'
NAMER func=0x447b0250 -> 'Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x...'
```

The `0x39..` handle — the exact class that "no launch symbol reaches" — resolves
to **`triton_red_fused__to_copy_embedding_rms_norm_0`**. So the residual nodes
are **not** aiter/CK launched via a static HIP; they are **torch.compile /
inductor Triton fused kernels** (RMS-norm, embedding, elementwise fusions) whose
`hipFunction_t` is a perfectly valid handle that `hipKernelNameRef` names. The
"interception ceiling" is real for *issue-time* identity, but **identity itself
is fully recoverable from the captured handle**, post-hoc.

**Conclusion (supersedes the ceiling section): every captured node IS nameable.**
The de-risk succeeds. The corrected M3b plan:
1. During capture: record graph structure + node funcs (no naming inline).
2. Off-path (background thread / during serving idle): `hipKernelNameRef` each
   node func → name. The only constraint is timing — it must not run while a
   capture holds the HIP lock (it blocks, harmlessly, until the lock frees).
3. Rebuild: resolve each name against the captured module images (already
   recorded via `hipModuleLoadData`/`Ex`; `rebuild-check` already tries every
   module for an entry). HSA code-object capture
   (`hsa_code_object_reader_create_from_memory`) is the fallback for any name
   whose HSACO we did not see loaded through the HIP module API.

> **⚠ M3b correction (supersedes this conclusion).** Stress-testing the namer
> end-to-end (jobs 509282–509302, `recipe/vllm_record.sbatch` with
> `--cudagraph-capture-sizes 1` so vLLM actually reaches idle) showed
> `hipKernelNameRef` is **not** a complete solution: it segfaults on the 2nd
> distinct *device-handle* node (unrecoverably — PyTorch's `c10::SignalHandler`
> wins the signal), and named host-registered kernels are not yet tied to a
> recorded module image. 4/6 nodes of a decode graph name (1 via `hipKernelNameRef`,
> 2 via the `__hipRegisterFunction` host map, …); the rest are blocked. See the
> **M3b** section in `RESULTS.md` for the precise, current blockers and what it
> took to get the plumbing (`capture-sizes` clamp, sentinel-gated drain,
> main-thread inline naming) working end-to-end.
>
> **Update (resolved, jobs 509309–509333):** both identity blockers are now
> CLOSED and `IDENTITY_GATE=PASS` on a real vLLM decode graph (6/6 nodes).
> (A) the `hipKernelNameRef` 2nd-call crash was cumulative-state-dependent, so
> each query now runs in a `fork()`ed child (`kernel_name_via_fork`) — every
> device-handle node names, no parent crash. (B) `name → image` linkage comes
> from ELF symbol tables (incl. the AMDGPU metadata note) across captured
> modules AND a new `hsa_code_object_reader_create_from_memory` interpose
> (Triton loads via ROCr, not the HIP module API), filtered to the device's
> gfx arch. The rebuild now loads all modules + resolves all kernels, failing
> only at `hipGraphInstantiate` because `hipGraphKernelNodeGetParams` returns
> `extra=NULL` on ROCm — so kernarg must be captured at launch time (the
> deferred param-snapshotting work). Full detail in the **M3b** section of
> `RESULTS.md`.

**Throughput caveat (test harness only):** in these runs the namer drained only
2 funcs before the job killed vLLM — vLLM never reached idle in the window (the
recorder's per-freeze 9683-symbol HSA iteration slows init, and the capture lock
stays busy through the ~528-graph capture phase). In production the names
resolve while the model serves. For a fuller in-test sample, drop the HSA
iteration (it was only the de-risk probe) and let vLLM reach `/health` before
draining.

Bit-identical on a *real vLLM* graph is deliberately **not** an M3a gate — it
needs buffer-content snapshotting (weights/KV/activations), which is the M3b
integration.

## How to run (on beverin)

```
rcc push
# M3a.2 — controlled bit-identical gate (real interposer, our workload)
ssh beverin 'cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && \
  sbatch snapshot/recipe/record_synthetic.sbatch'

# M3a.3 + M3a.4 — real vLLM identity recovery + structural rebuild
ssh beverin 'cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && \
  sbatch snapshot/recipe/vllm_record.sbatch'
# one graph only:
ssh beverin 'cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && \
  sbatch --export=ALL,MAX_GRAPHS=1 snapshot/recipe/vllm_record.sbatch'
```

Outputs land in `snapshot/record-synthetic/` and `snapshot/record-vllm/`
(`graph-<pid>-<idx>.snap` per captured graph).

## What is validated locally vs on the cluster

* **Locally (STUB build, done):** `test_record_ir` (ELF parse + IR assembly +
  serialize round-trip + summary counters); `snapshot inspect` on a synthetic
  snapshot. These exercise all host-testable logic.
* **Cluster-only (HIP):** compiling/running `libsnapshot_record.so`;
  `rebuild-check` (real HIP backend); M3a.2 synthetic bit-identical
  (`snapshot capture` under preload, then `restore`); M3a.3/M3a.4 on real vLLM.

The recorder mirrors the existing `snapshot_preload`/`snapshot_redirect`
`dlsym(RTLD_NEXT)` pattern exactly and passes a `-fsyntax-only` check against a
minimal HIP header stub; HIP ABI risk is low but only a cluster build confirms it.

## If M3a passes → M3b (the full integration)

The remaining M3b work, in dependency order:

1. **Allocator events for vLLM** — record the redirect arena's allocations (the
   redirect already serves them; expose its offset map to the recorder) so a
   restored process can replay the working set.
2. **Buffer contents** — d2h-copy the inputs each captured graph reads at
   capture time (weights are reloaded from disk, not snapshotted; KV/activations
   must be captured), so a restored graph launches against correct data. *This
   is the step that makes bit-identical possible on vLLM.*
3. **vLLM-side hook** — install the rebuilt execs into vLLM's graph pool,
   skipping `capture_model`. Requires finding the right seam in vLLM's
   `CUDAGraphRunner` (the ROCm equivalent).
4. **Multi-graph / multi-GPU** — ~2065 graphs/replica, TP>1.

## If M3a.3 fails (the informative outcome)

A non-zero `nodes_without_identity` would mean some kernels launch through a
path that bypasses `hipModuleGetFunction` (e.g. a runtime-resolved symbol, or
`hipLaunchKernel` with a function pointer not obtained via the module API). The
recorder already interposes `hipModuleLaunchKernel`; we'd add
`hipLaunchKernel` / `hipModuleLaunchCooperativeKernel` and re-measure. That is a
cheap, well-defined fix — exactly the kind of blocker M3a is meant to surface
before the full integration depends on it.
