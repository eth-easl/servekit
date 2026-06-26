# Design — Porting `snapshot` to NVIDIA / A100 (bristen)

Date: 2026-06-26
Status: approved design, pending implementation plan
Scope owner: Xiaozhe Yao

## 1. Goal & context

The `snapshot` prototype reduces serving-engine **cold-start time** by persisting
captured GPU graphs across restarts. It is fully built and validated on **AMD**
(beverin, MI300A / gfx942, ROCm, vLLM): the M1→M3j arc proved deterministic
fixed-base VMM, capture→relocate→rebuild→bit-identical, real-vLLM identity
recovery, a ~24% projected cold-start win with **zero steady-state serving
overhead**, and a simpler "skip-capture" path that won ~57% on AMD.

This design ports that capability to **NVIDIA / A100** while keeping AMD a
first-class, never-regressed target.

### Decisions (settled with the user)

- **Outcome:** a measured **cold-start win on a real engine** (not just a
  mechanism proof).
- **Engine:** **vLLM on CUDA**, for direct comparability with the beverin
  numbers (bristen currently runs SGLang only — vLLM-CUDA needs a new
  EDF/recipe).
- **Win strategy:** **both** — implement the simpler **skip-capture** first
  (fast win + clean baseline), then the full **snapshot/restore** mechanism, and
  measure both against eager + baseline on A100.
- **Model/config:** **GLM-4.7-Flash, TP=4** on a 4× A100-80GB node. Higher KV
  concurrency makes the capture cost we eliminate larger and the win more
  visible.
- **Dual-vendor:** the system must **support both AMD and NVIDIA** from one
  source tree (see §3).

### Target environment (bristen)

- 4× A100-SXM4-80GB per node, NVLink, x86_64; SLURM `-A a-infra02`, `normal`
  partition; enroot/pyxis containers (`.sqsh` + EDF); shared `/capstor` +
  `/iopsstor` (same `.sqsh`/weights visible from beverin).
- Container CUDA = **12.9.1** (from the pinned SGLang image; a vLLM image will be
  pinned similarly). CUDA 12.9 provides the full VMM API
  (`cuMemAddressReserve` fixed-base + `cuMemSetAccess`), the full CUDA Graph API
  (`cuGraphAddKernelNode`, `cuGraphKernelNodeGetParams`, `cuGraphInstantiate`,
  `cuGraphLaunch`), nvrtc, and `cuFuncGetParamInfo`/`cuFuncGetName`.
- Login nodes have **no GPU** — all GPU validation runs on compute nodes via
  `srun`/`sbatch`. Code sync via `rcc --profile bristen push`.

## 2. Approach

Three approaches were weighed:

- **A1 — Mirror AMD 1:1.** Swap only the C-API. Max reuse, fastest, every
  milestone has a known-good AMD analog; but inherits AMD design choices
  (per-node rebuild vs direct replay).
- **A2 — CUDA-native rewrite.** Foundry-style direct graph replay,
  `cudaGraphExecUpdate`. Potentially bigger win; much more research risk;
  hard to validate incrementally.
- **A3 — Hybrid (chosen).** Port 1:1 for a working, directly-comparable result
  fast, but bake in the three hard-won AMD lessons from day one — **eager
  interposer gate**, **fixed-base Δ=0 VMM**, **skip-capture-first** — then layer
  CUDA-native optimizations only where measurement justifies them.

### Why CUDA should be easier than AMD was

Every AMD primitive has a direct CUDA equivalent, and CUDA removes several of the
worst AMD pain points outright:

| AMD (what was painful) | CUDA equivalent | Effect |
|---|---|---|
| `hipMemSetAccess` fragility (M2.3/M3h blocker) | `cuMemSetAccess`, one-shot over whole region | Canonical path (Foundry is built on it); blocker likely disappears |
| Triton HSACO capture via `hsa_code_object_reader_*` | Triton-CUDA loads cubin via `cuModuleLoadData` | Whole HSA layer gone — one load path |
| `hipKernelNameRef` segfaults → fork-isolation hack | `cuFuncGetName` (12.3+) / cubin symtab | Fork hack gone |
| AMDGPU MessagePack metadata parsing for arg sigs | cubin `.nv.info` / `cuFuncGetParamInfo` (12.4+) | msgpack walker gone |
| `hipGraphKernelNodeGetParams` returns `extra=NULL` | populates `kernelParams` | Arg recovery works directly |
| `gcnArchName` multi-arch ELF filtering | single `sm_80` cubin | Simpler |

## 3. Dual-vendor structure (the "support both" invariant)

**Interpretation:** one source tree and one `GpuBackend` abstraction support both
vendors, **selected at build time** per target/container
(`SNAPSHOT_BACKEND=HIP` on a ROCm node, `=CUDA` on bristen). A single fat binary
linking both ROCm and CUDA is impractical (different runtimes, different
containers) and unnecessary — the `Vendor` enum, `vendor_from_name`, and the
`AUTO` probe (hipcc → nvcc → stub) already embody build-time selection.

Structure — maximize shared code, isolate the mechanism:

- **Vendor-neutral core stays shared and untouched.** `csrc/core/*` (IR,
  serialize, relocate, hashing, allocator-bump) and `include/snapshot/*` are
  already vendor-neutral; both backends and both interposer sets link
  `snapshot_core`. CUDA work adds only *additive* changes here (e.g. a cubin
  `.nv.info` signature parser sibling to the AMDGPU-msgpack one).
- **Layer 1 backend: parallel files** (`backends/cuda/cuda_*.cpp`, already
  scaffolded as stubs) vs `backends/hip/hip_*.cpp` — thin driver-API wrappers
  behind `GpuBackend`, no sharing needed.
- **Layer 2 interposers: parallel files, NOT a refactor of the HIP ones.** Add
  `snapshot_redirect_cuda.cpp` / `snapshot_record_cuda.cpp` as CUDA siblings and
  leave the proven HIP interposers **byte-for-byte untouched**. They share the
  genuinely-common logic through `snapshot_core`. This duplicates only the thin
  hook glue, so the fragile AMD path carries **zero regression risk** from CUDA
  work. Extracting a common interposer-policy core is a *later* cleanup, done
  with the AMD path as a regression oracle — not now.
- **Build:** extend `SnapshotBackend.cmake`'s CUDA branch to build the CUDA
  preload libs + a `cuda_redirect_smoke`, mirroring the HIP branch (which builds
  them today only under `if(HIP)`). Add `libnvrtc` to `snapshot_configure_cuda`.
- **Regression invariant (enforced every milestone):** the HIP build must still
  configure, compile, and pass `ctest` (5/5) and its on-cluster gates on
  beverin. Because HIP sources are untouched this holds by construction, but it
  is **checked, not assumed**.

## 4. Layer 1 — `CudaBackend` (foundation)

Fill the existing stubs in `backends/cuda/{cuda_vmm,cuda_graph,cuda_backend}.cpp`
with the **CUDA driver API** (`cu*`, link `libcuda`) — the runtime API is too
high-level for VMM / module-by-image / graph-introspection. nvrtc for the
synthetic module. Mirrors `hip_*.cpp` one-to-one:

- **`cuda_vmm.cpp`:** granularity → `cuMemGetAllocationGranularity`;
  `reserve_address` → `cuMemAddressReserve(size, align, requested_base, 0)`
  (`addr` arg = fixed-base hint); `create_physical` →
  `cuMemCreate(CU_MEM_ALLOCATION_TYPE_PINNED, current device)`; `map`/`set_access`
  → `cuMemMap`/`cuMemSetAccess`; releases → `cuMemUnmap` / `cuMemAddressFree` /
  `cuMemRelease`; `arch` → CC major/minor → `"sm_80"`.
- **`cuda_graph.cpp`:** `cuModuleLoadData` / `cuModuleGetFunction`;
  `cuStreamBeginCapture` / `cuStreamEndCapture`; introspect → `cuGraphGetNodes` +
  `cuGraphNodeGetType` + `cuGraphKernelNodeGetParams` (CUDA populates
  `kernelParams` → real arg recovery); rebuild → `cuGraphCreate` +
  `cuGraphAddKernelNode` per node + edges; `cuGraphInstantiate` / `cuGraphLaunch`;
  `cuLaunchKernel`; `cuMemcpyHtoD` / `cuMemcpyDtoH`.
- **`cuda_backend.cpp`:** replace the nvrtc stub → `nvrtcCreateProgram` +
  `nvrtcCompileProgram("--gpu-architecture=sm_80")` + `nvrtcGetCUBIN`. Use the
  **same 3-kernel exact-uint workload** as HIP so output is bit-identical and the
  relocation gate is meaningful.

**Gate (M1 analog on A100):** `probe-base` (fixed base honored), `verify`
(single-process Δ≠0 relocation, bit-identical), `capture`→`restore` (two-process
fresh-process rebuild, bit-identical), `bench --scaled`, plus `ctest`
(`test_graph_capture_gpu` / `test_e2e_roundtrip_gpu` now exercise CUDA).

## 5. Layer 2 — CUDA interposers

### `snapshot_redirect_cuda.cpp` — deterministic device addresses

- Hook `cudaMalloc` / `cudaFree` (runtime) **and** `cuMemAlloc` / `cuMemFree`
  (driver): torch's caching allocator uses `cudaMalloc`; `expandable_segments`
  uses the driver VMM path — hook both, as on AMD.
- **Fixed-base Δ=0 VMM arena from day one** (M3h lesson): one
  `cuMemAddressReserve(0x600000000000)` + one `cuMemCreate(whole)` + one
  `cuMemMap` + **one** `cuMemSetAccess(whole region)`; sub-allocate by
  bump + free-list inside it. Keep a plain-`cudaMalloc` arena fallback toggle
  (`SNAPSHOT_REDIRECT_ARENA`) for diagnosis, but fixed-base is the default since
  both win strategies and Δ=0 correctness depend on it.
- **TP=4** → 4 worker processes, each with its own arena at the same fixed base
  (separate address spaces — fine). NCCL/comms buffers also flow through
  `cudaMalloc` → land in the arena; size the region for weight-shard + KV +
  capture + comms per GPU (80 GB is roomy vs the APU).

### `snapshot_record_cuda.cpp` — identity + capture + restore

- **Eager interposer gate from day one** (M3i +271 s lesson): never scan fatbins
  on the hot import path; eagerly disable the heavy import-path hooks (fatbin
  register, function register, module load) until the first real capture window
  (or keep them disabled entirely in restore mode).
- **Identity (much simpler than AMD):** `__cudaRegisterFunction` (host-fn →
  device-symbol map) + `__cudaRegisterFatBinary` (cubin registry) +
  `cuModuleLoadData` / `cuModuleGetFunction` (Triton / explicitly-loaded
  modules). No HSA layer, no fork-isolated naming, no msgpack — names from cubin
  symtab / `cuFuncGetName`.
- **Capture / kernarg:** hook `cuStreamBeginCapture` / `cuStreamEndCapture` (+
  `cudaStreamBeginCapture`); at EndCapture introspect via
  `cuGraphKernelNodeGetParams` (`kernelParams` populated → arg pointers
  directly). Arg counts/offsets from cubin `.nv.info` (`EIATTR_KPARAM_INFO`) or
  `cuFuncGetParamInfo` (12.4+) → precise pointer relocation, replacing the
  AMDGPU-msgpack path.
- **Restore:** load `.snap` graphs, rebuild via `cuGraphAddKernelNode`, relocate
  by Δ (zero-op under fixed-base Δ=0), shim `cuStreamBeginCapture` /
  `cuStreamIsCapturing` / `cuStreamEndCapture` to return pre-built graphs (the
  M3g/M3i restore-shim design), fall through to real capture when the queue is
  exhausted.

## 6. Layer 3 — vLLM-CUDA deploy + two win strategies

- **3a Deploy:** new EDF + recipe under `deploy/glm-47-flash-bristen-vllm/`
  (only SGLang exists on bristen today). Pin a `vllm/vllm-openai` CUDA image
  (12.x / sm_80) with the same `glm4_moe_lite` transformers overlay the SGLang
  README documents (cheap enroot pip-overlay, not a full rebuild). GLM-4.7-Flash
  **TP=4**. Establish a clean baseline cold start: `READY` time, per-phase
  breakdown, capture-phase cost at TP=4 concurrency.
- **3b Skip-capture (simpler win, first):** port
  `snapshot/recipe/cginst_skip/cg_skip.py` → suppress vLLM's capture phase, run
  eager, lazy-capture during serving. Needs *no* interposer determinism → fastest
  path to a real NVIDIA cold-start number and a clean baseline for 3c.
- **3c Snapshot/restore (the mechanism):** record graphs on cold start #1 under
  `redirect_cuda` (fixed-base) + `record_cuda`; restore on #2 (rebuild + relocate,
  skip capture). Clamp capture sizes for the record drain (the AMD
  `CAPTURE_SIZES` lever). Verify Δ=0 pointer correctness (the "fundamental wall"
  is closed by fixed base) + correct inference.

## 7. Layer 4 — measurement & comparison

A/B/C — **baseline** vs **skip-capture** vs **snapshot-restore**:

- Cold-start `READY` time + per-phase decomposition (startup, weight load,
  compile, profile/KV, capture/rebuild).
- **Interposer process-startup overhead** — watch for the M3i +271 s; the eager
  gate should keep it < 10 s.
- **Steady-state serving overhead** — benchmaker A/B (like M3j): single-stream
  ITL/tokens-per-s + saturation throughput; target ~0 overhead.
- Output a `snapshot/RESULTS.md` "**NVIDIA / A100**" section directly comparable
  to the AMD one, reporting the **eliminable-fraction decomposition** honestly,
  not just a headline number.

## 8. Milestones & gates

| # | Milestone | Gate (acceptance) | Risk |
|---|---|---|---|
| **N1** | `CudaBackend` VMM + nvrtc + graph; CLI gates on A100 | probe-base / verify / capture→restore / bench pass; `ctest` green; **HIP build still green** | Low |
| **N2** | `redirect_cuda` fixed-base on raw program + real torch | byte-identical device addresses across 2 cold runs; correct compute; startup overhead < 10 s | Low–Med |
| **N3** | vLLM-CUDA TP=4 deploy + baseline cold start | clean `READY`; per-phase breakdown; capture-phase cost characterized | Med |
| **N4** | Skip-capture win | `READY` reduction vs baseline; correct tokens | Low |
| **N5** | Snapshot/restore win | Δ=0 pointer correctness (reloc known=0 blind=0); capture skipped; `READY` reduction; serving overhead ~0 | Med–High |

Each milestone is a cluster job on bristen (`rcc --profile bristen push` →
`sbatch -A a-infra02`), mirroring the beverin recipe shape.

## 9. Cross-cutting risks & mitigations

1. **Interposer import overhead** (the M3i killer on AMD) — mitigated by the
   eager gate baked into N2; verified by the N2 startup-overhead gate.
2. **vLLM-CUDA TP=4 capture structure** (FULL vs PIECEWISE; multi-process) — the
   record drain and restore-shim must handle vLLM's CUDA-graph mode and 4 worker
   processes; clamp capture sizes for a tractable record drain.
3. **Honest cold-start accounting** — the AMD arc found weight-load I/O often
   dominates and capture cost is config-dependent. TP=4's higher capture cost is
   *why* this config makes the win visible; the report decomposes the eliminable
   fraction truthfully.
4. **vLLM-CUDA image / `glm4_moe_lite` support** — same transformers-version
   issue the SGLang README hit; resolved by a cheap enroot pip-overlay, pinned by
   digest.
5. **`cuMemSetAccess` reliability** — expected reliable (canonical CUDA path),
   but N2 explicitly gates on it; the `cudaMalloc`-arena fallback toggle exists
   if a surprise appears.

## 10. Out of scope (this design)

- Foundry-style direct graph archive/replay (no per-node rebuild) — a possible
  A2-style optimization considered only if N5 measurement justifies it.
- Multi-node / cross-node TP (bristen is Ethernet; single-node TP=4 only).
- SGLang integration (engine chosen = vLLM-CUDA).
- Extracting a shared interposer-policy core across HIP/CUDA — deferred cleanup.

## 11. Files (new / changed)

New:
- `snapshot/csrc/preload/snapshot_redirect_cuda.cpp`
- `snapshot/csrc/preload/snapshot_record_cuda.cpp`
- `snapshot/csrc/cli/cuda_redirect_smoke.cpp` (a clean CUDA sibling of the HIP
  `redirect_smoke.cpp`; the HIP demo is left untouched per the regression
  invariant)
- CUDA cubin signature parser (sibling to the AMDGPU-msgpack parser in
  `core/record.cpp`) + `tests/test_cuda_cubin.cpp`
- `deploy/glm-47-flash-bristen-vllm/` (EDF, recipe, README)
- `snapshot/recipe/` CUDA siblings of the vLLM record / restore / measure /
  serve-bench sbatch + driver scripts

Changed (additive only):
- `snapshot/csrc/backends/cuda/cuda_vmm.cpp`, `cuda_graph.cpp`,
  `cuda_backend.cpp` (fill stubs)
- `snapshot/cmake/SnapshotBackend.cmake` (CUDA branch: nvrtc; preload libs)
- `snapshot/CMakeLists.txt` (CUDA branch: build preload libs + smoke + cubin test)
- `snapshot/RESULTS.md` (new NVIDIA/A100 section)

Untouched (regression invariant): all `backends/hip/*`, all existing
`preload/snapshot_*.cpp` HIP interposers, all `core/*` except additive.
