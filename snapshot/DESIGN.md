# `snapshot` — cross-vendor GPU compute-graph snapshot prototype (design)

## Context

vLLM cold start on beverin (CSCS MI300A / ROCm) is dominated by **CUDA/HIP graph capture**: ~9 min per fresh start (51 PIECEWISE + 35 FULL graphs at `--max-num-seqs 256`), re-done every launch because captured graphs hold live GPU addresses and cannot be serialized. The torch.compile disk cache already works (verified: prod hits key `59fd8c6226`, ~11 s warm) — graph capture is the remaining bottleneck and nothing today persists it.

[Foundry](https://github.com/foundry-org/foundry) solves exactly this on NVIDIA: it `LD_PRELOAD`-intercepts the CUDA driver, forces a deterministic VMM layout, and serializes captured graphs + kernel modules to disk so a fresh process restores instead of re-capturing. It is **NVIDIA-only** (cu* driver API, requires fixed-address `cuMemAddressReserve`) and won't run on MI300A.

`snapshot/` is a greenfield, cross-vendor reimplementation of Foundry's core idea behind a hardware-abstraction layer (HAL), so the same mechanism works on AMD (HIP) and NVIDIA (CUDA) and is extensible to future hardware. **Milestone 1 (this doc) = standalone library + CLI only**, validated on AMD/beverin first, CUDA backend implemented behind the HAL and validated later on Clariden/Bristen. **vLLM integration, the LD_PRELOAD interposer, and multi-GPU/TP are out of scope** (future milestones) — but the HAL is designed to reach them.

### The one hard problem
Foundry's correctness rests on **byte-identical device addresses across save/restore** (fixed VMM base `0x600000000000`; absolute device pointers are embedded in serialized kernel-node params). **`hipMemAddressReserve` does not guarantee honoring a requested fixed base** (HIP docs: validate the returned pointer). So the design makes **relocation** a first-class core feature: reserve one contiguous region, record allocations as `base+offset`; on restore, if the region lands at a different base, patch embedded device pointers by the constant delta `Δ`. Fixed-base (Foundry fast path, `Δ=0`) becomes an optimization; relocation is the correctness guarantee. This is the central research contribution and the riskiest unknown — it gets validated first.

## Directory tree (`serving-stack/snapshot/`)

```
snapshot/
├── CMakeLists.txt                 # backend autodetect (hipcc/ROCM_PATH vs nvcc/CUDA), C++17 core
├── cmake/SnapshotBackend.cmake    # sets SNAPSHOT_BACKEND=HIP|CUDA, guards GPU langs
├── include/snapshot/
│   ├── gpu_backend.hpp            # abstract GpuBackend interface (the HAL) + POD param structs
│   ├── handles.hpp               # opaque handles (Module/Function/Graph/GraphExec/Mem/Stream)
│   ├── allocator.hpp             # DeterministicAllocator: region reserve + bump cursor + event log
│   ├── relocation.hpp            # pointer-relocation scan (patch device ptrs by Δ)
│   ├── snapshot_format.hpp       # on-disk container structs, magic/version, section tags
│   ├── capture.hpp / restore.hpp # CaptureSession / RestoreSession
│   └── status.hpp                # Status enum (no exceptions across HAL boundary)
├── csrc/core/                     # VENDOR-NEUTRAL (no hip*/cu* symbols)
│   ├── allocator.cpp  relocation.cpp  serialize.cpp
│   ├── capture.cpp    restore.cpp     graph_model.cpp  hashing.cpp
├── csrc/backends/
│   ├── backend_factory.cpp        # make_backend() -> compiled-in backend
│   ├── hip/  {hip_backend,hip_graph,hip_vmm}.cpp     # libamdhip64.so (hip*)
│   └── cuda/ {cuda_backend,cuda_graph,cuda_vmm}.cpp  # libcuda.so   (cu* driver)
├── csrc/cli/ {main,workload,workload_kernels}.cpp    # `snapshot` CLI + synthetic workload
├── tests/   {test_allocator,test_relocation,test_serialize_roundtrip}.cpp  # host-only
│           {test_graph_capture_gpu,test_e2e_roundtrip_gpu}.cpp             # GPU
├── recipe/  snapshot-rocm.toml  build_snapshot.sbatch  bench_snapshot.sbatch
└── docs/    DESIGN.md  DETERMINISM.md  FORMAT.md
```

## The HAL (`include/snapshot/gpu_backend.hpp`)

Semantic-level abstract `GpuBackend` so the core never touches a vendor type; all methods return `Status`, outputs via opaque handles. Methods:
- **Identity:** `vendor()`, `arch(out)` — written to header; restore refuses mismatch (kernel binaries are vendor+arch specific, not portable).
- **VMM:** `get_allocation_granularity`, `reserve_address(size,align,requested_base,out_base)` (hint = fixed-base fast path; `out_base` may differ → relocation), `release_address`, `create_physical`, `release_physical`, `map(va,size,offset,handle)`, `unmap`, `set_access`.
- **Modules:** `load_module(image,n,out)` (HSACO for AMD, cubin/fatbin for NVIDIA), `unload_module`, `get_function`.
- **Graphs:** `stream_create/destroy`, `begin_capture`, `end_capture(out_graph)`, `introspect_graph(g,out_IR)`, `rebuild_graph(IR,out)`, `instantiate`, `exec_set_kernel_node_params` (post-instantiation param update — the relocation apply point and the future LD_PRELOAD seam), `launch`, `synchronize`.
- **Direct (workload outside capture):** `launch_kernel`, `memcpy_h2d/d2h`.

HIP impl uses `hipMemAddressReserve/hipMemCreate/hipMemMap/hipMemSetAccess`, `hipModuleLoadData`, `hipStreamBeginCapture/hipGraphInstantiate/hipGraphExecKernelNodeSetParams`. CUDA impl mirrors with `cu*` driver calls; on NVIDIA the base hint is honored so `Δ=0`.

## Deterministic allocator + relocation

`AllocatorState{ region_base, requested_base, region_size, cursor, granularity, vector<AllocEvent{offset,size,tag}> }`.
1. Query granularity; `reserve_address(region_size, granularity, 0x600000000000, out=region_base)`; record `fixed_base_honored = (region_base==requested_base)`.
2. `alloc(size,tag)`: `aligned=round_up(size,granularity)`; `off=cursor; cursor+=aligned`; `create_physical→map(region_base+off,aligned,off,h)→set_access`; append `AllocEvent`. Event list is **deterministic** (same workload → same offsets/sizes regardless of base) → relocation is a pure constant Δ.

**Relocation** (`relocation.cpp`), restore-only when `Δ=restored_base−captured_base ≠ 0`: for each kernel-node `param_blob`, prefer known `ptr_offsets` from capture (zero false positives); else blind 8-byte-window scan patching values in `[captured_base, captured_base+region_size)`. Memcpy/memset addresses relocated the same way. High unused base + known-offset patching makes false positives astronomically unlikely; the two-process bit-identical gate catches any corruption.

## Capture and restore flows (HAL calls)

**Capture:** `make_backend` → allocator init → run workload allocations (event log) → `stream_create`/`begin_capture` → workload kernels → `end_capture` → `introspect_graph→GraphIR` (nodes, edges, per-kernel `KernelLaunchParams`+ptr_offsets) → build module registry keyed by `hash(image)` → `serialize` (header records captured_base, region_size, vendor, arch).

**Restore:** parse header (refuse vendor/arch mismatch) → `reserve_address(region_size,granularity,captured_base,out=restored_base)` → replay AllocEvent log (`create_physical`/`map`/`set_access` at each offset) → compute Δ; relocate IR params if Δ≠0 → `load_module` each registry entry + `get_function` → `rebuild_graph→instantiate` → `launch`/`synchronize`/`memcpy_d2h`. **No stream capture** — that's the eliminated cost.

## Serialization format (little-endian, section-tagged, per-section CRC32)

`HEADER`(magic, version, vendor, arch, captured_base, region_size, fixed_base_honored, granularity, section_count) · `SEC_ALLOC_LOG`(offset,size,tag) · `SEC_MODULES`(hash,image_bytes,entry_names) · `SEC_GRAPH_NODES`(node_id,type,module_hash,entry,grid,block,shmem,param_bytes,ptr_offsets) · `SEC_GRAPH_EDGES`(from,to) · `FOOTER`(crc,bytes).

## Synthetic CLI workload + verification

`cli/workload*`: deterministic pipeline over VMM-backed buffers A,B,C,OUT (4×16 MiB). Host seeds A,B (`A[i]=i`, `B[i]=2i+1`), `memcpy_h2d`; chain a few kernels on the captured stream (`C=A*B+bias` → `OUT=relu(C)+offset` → in-place transform that reads its own base pointer). Integer / exact-FP-representable arithmetic only → **bit-identical by construction**; correctness depends on relocation being right.

CLI verbs: `capture <file>`, `restore <file>`, `verify <file>` (one process: fresh reference vs restore-replay, assert `memcmp==0`), `bench <file> [--iters N] [--scaled]` (time capture path vs restore path; `--scaled` issues ~200 kernel nodes to make capture cost visible and probe the HIP-graph node-count risk).

## Build / run on beverin

CMake `SNAPSHOT_BACKEND=AUTO|HIP|CUDA`; core is plain C++17 (no vendor symbols); only `backends/<vendor>` + kernel TU use the GPU compiler; ctest gates GPU tests on a runtime device check. `recipe/snapshot-rocm.toml` mirrors `deploy/glm-47-flash-beverin/glm-47-flash-rocm.toml` (same pinned image `vllm/vllm-openai-rocm@sha256:3813…`, mounts `/capstor`,`/users/xyao`,`/iopsstor`, `ROCM_PATH=/opt/rocm`, `HSA_NO_SCRATCH_RECLAIM=1`). `recipe/build_snapshot.sbatch` mirrors `probe_vllm.sbatch` header (`--partition=mi300 --account=root --nodes=1 --gpus-per-node=1 --time=00:30:00`) then `srun --environment=snapshot-rocm` → `cmake -B build -DSNAPSHOT_BACKEND=HIP && cmake --build && ctest`. `bench_snapshot.sbatch` runs `capture` then a **separate** `srun` for `restore`+`verify`+`bench`. Sync/submit via `rcc`; add `snapshot/build/` to `.rcc/rccignore`.

## Milestones (de-risk determinism first)

- **M1.0 Determinism spike (day 1):** ~50-line container test — query granularity, `hipMemAddressReserve` at `0x600000000000`, print whether returned base == request. Answers the single riskiest question before architecture depends on it.
- **M1.1 Host-only core:** allocator + relocation + serialize/format; unit tests pass on any machine (locks the determinism math, no GPU).
- **M1.2 HIP backend:** VMM + modules + direct launch; `test_graph_capture_gpu` captures the workload graph, verify nodes/params via `introspect_graph`.
- **M1.3 Single-process round-trip:** `verify` verb + `test_e2e_roundtrip_gpu`; exercises relocation against a real shifted base.
- **M1.4 Two-process round-trip + bench (ACCEPTANCE GATE):** `capture` in one process, `restore` in a fresh one, bit-identical vs from-scratch reference; `bench` prints capture-vs-restore saving.
- **M1.5 CUDA backend behind the HAL:** implement `CudaBackend` (expect `Δ=0` fast path); validate later on Clariden GH200 / Bristen A100. Reuses 100% of core.

## Verification

- **Host-only (always):** allocator granularity/cursor/event-order; relocation (known-offset + blind-window, Δ=0 no-op, region-edge pointers); serialize byte-exact round-trip + CRC + cross-vendor-refusal.
- **GPU integration:** capture introspection matches issued kernels; single-process restore == reference (`memcmp==0`).
- **Acceptance gate (M1.4):** two-process capture/restore == from-scratch reference, bit-identical, with relocation actually exercised (assert `restored_base != captured_base` was hit at least once on AMD, else warn that the relocation path is untested).
- **Timing:** `bench` reports wall time of full path (incl. `begin/end_capture`+`instantiate`) vs restore path (load+relocate+rebuild+instantiate, no capture); `--scaled` ~200 nodes.

## Out of scope (future milestones)
vLLM integration (route torch's GPU calls through the core) · LD_PRELOAD symbol interposer (`libcuda.so` cu* / `libamdhip64.so` hip*) · multi-GPU/tensor-parallel/multi-rank · Foundry-style two-pass profiling · cross-vendor/arch portable snapshots (intentionally refused).

## Top risks → mitigation
1. **AMD won't honor fixed VA base → stale pointers.** THE core risk. Relocation is a first-class core feature; **M1.0** validates real `hipMemAddressReserve` behavior before code depends on it; **M1.1** locks relocation math host-only.
2. **Relocation false positives.** Prefer known `ptr_offsets` (zero FP), bound scan to the region, high unused base; M1.4 bit-identical gate catches corruption.
3. **HIP graph beta instability (stale ptrs at 200+ nodes, gfx1201).** MI300 is CDNA3 gfx942 (likely unaffected); `--scaled` bench deliberately issues ~200 nodes to surface it early; `exec_set_kernel_node_params` is an alternative node-level patch path.

---

*Provenance: Foundry is NVIDIA-only (intercepts `libcuda.so`, requires CUDA Driver 12.0+ VMM + fixed-address reservation). ROCm 6.x exposes the needed primitives — HIP VMM (`hipMemCreate/AddressReserve/Map/SetAccess`, beta), HIP graphs (kernel/memcpy/memset + `hipGraphExecKernelNodeSetParams`, beta) — but `hipMemAddressReserve` does not guarantee a fixed base, which is why relocation is core. vLLM's `CuMemAllocator` (sleep mode) does not use VMM on ROCm, so it cannot be reused for the deterministic-address layout.*
