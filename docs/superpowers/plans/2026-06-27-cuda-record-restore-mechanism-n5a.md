# N5a — `record_cuda` snapshot/restore mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `record_cuda` LD_PRELOAD interposer and prove the CUDA graph **record → restore** mechanism on a CLI synthetic workload — record captured graphs on cold start #1, restore them (rebuild + skip real capture) on cold start #2, **bit-identical** output, under fixed-base Δ=0.

**Architecture:** A parallel `*_cuda` interposer (sibling of the proven N2 `snapshot_redirect_cuda.cpp`; the HIP `snapshot_record.cpp` is never touched). It composes *with* `redirect_cuda` (fixed-base Δ=0 addresses) and has two env-selected modes: `record` (hook `cuStreamEndCapture`, walk the graph, persist kernel nodes + verbatim kernarg blobs + function identity to `.snap`) and `restore` (shim the capture APIs to rebuild and hand back pre-built graphs via N1's `cuGraphAddKernelNode` path, skipping real capture). A two-kernel (static `nvcc` + nvrtc) CLI smoke validates both identity paths and all gates on A100.

**Tech Stack:** CUDA driver API (`cuStream*Capture`, `cuGraph*`, `cuFuncGetParamInfo`, `cuFuncGetName`, `cuModule*`) + runtime registration (`__cudaRegisterFatBinary`/`Function`), C++17, LD_PRELOAD/`dlsym(RTLD_NEXT)`, CMake (`SNAPSHOT_BACKEND=CUDA`), enroot/pyxis on CSCS **bristen** (4× A100-80GB, `-A a-infra02`, partition `normal`), `rcc --profile bristen-snapshot push`.

## Global Constraints

- **Verbatim kernarg replay, no relocation, no cubin parser.** Under fixed-base Δ=0 the recorded kernarg device-pointers stay valid at restore; record the kernarg blob and replay it byte-for-byte. `cuFuncGetParamInfo` supplies only the blob *size*. Do NOT build a cubin `.nv.info` parser or any Δ≠0 relocation.
- **Full identity machinery, vLLM-ready:** `__cudaRegisterFatBinary` + `__cudaRegisterFunction` (+ `__cudaRegisterFatBinaryEnd`) for static kernels, `cuModuleLoadData`/`cuModuleGetFunction` + `cuFuncGetName` for nvrtc/explicit modules. Identity is recorded **by name/symbol** (+ a module content hash), never by raw pointer.
- **Eager interposer gate (M3i +271s lesson):** registration hooks record only lightweight identity; never scan fatbins on the hot import path; in `restore` mode the registration hooks may stay passive.
- **`core/` is byte-untouched.** The `record_cuda` `.snap` (de)serialization lives in `preload/` (inline in `snapshot_record_cuda.cpp` or `preload/record_cuda_format.{hpp,cpp}`), never under `core/`.
- **Dual-vendor additive:** only NEW `*_cuda` files + an additive CMake CUDA-record branch. The existing HIP files (`backends/hip/*`, `preload/snapshot_record.cpp`, `preload/snapshot_redirect.cpp`), all of `core/`, and the N1/N2 CUDA files (`backends/cuda/*`, `preload/snapshot_redirect_cuda.cpp`) stay byte-identical. **HIP build must still be green.**
- **Build/run on bristen A100**, mirroring the N1 build harness (CUDA dev container) — never the login node. Gates are cluster jobs.
- **The `.so` links libcuda + libcudart only, `-static-libstdc++ -static-libgcc`** (the N2 cross-container recipe), so N5b can later load it into the vLLM-CUDA image. (`libcudart` is needed for the `__cudaRegister*` runtime entry points; keep the interposer otherwise driver-API based.)

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `snapshot/csrc/cli/cuda_record_smoke.cpp` | Two-kernel (static nvcc + nvrtc) CLI smoke: alloc → capture graph → launch → checksum | Create |
| `snapshot/csrc/preload/snapshot_record_cuda.cpp` | The record/restore interposer (identity + capture/record + restore-shim) | Create |
| `snapshot/csrc/preload/record_cuda_format.hpp` | `.snap` schema + (de)serialization for the CUDA verbatim format | Create |
| `snapshot/CMakeLists.txt` | Additive CUDA branch: build `snapshot_record_cuda.so` + `cuda_record_smoke` | Modify (additive only) |
| `snapshot/recipe/cuda_record_smoke.sbatch` | Build + run #1 (record) + run #2 (restore) + assert gates on bristen | Create |
| `snapshot/recipe/cuda_record_build.sbatch` (or reuse N1's) | Build harness in the CUDA dev container | Create/reuse |
| `snapshot/RESULTS.md` | Append an N5a section | Modify |

---

## Task 1: Two-kernel CLI smoke target (the controllable workload)

Build the target the interposer will record/restore. No interposer yet — prove the smoke captures a graph over both a static and an nvrtc kernel, launches it, and prints a deterministic checksum.

**Files:**
- Create: `snapshot/csrc/cli/cuda_record_smoke.cpp`
- Modify: `snapshot/CMakeLists.txt` (add the `cuda_record_smoke` target under the CUDA branch)
- Create: `snapshot/recipe/cuda_record_smoke.sbatch` (build + run; extended in Task 5)

**Interfaces:**
- Produces: an executable `cuda_record_smoke` that, given `SMOKE_OUT=<path>`, writes a deterministic checksum line `CHECKSUM=<hex>` to stdout and exits 0. Reads no recording in this task.

- [ ] **Step 1: Read the reference patterns**

Read `snapshot/csrc/cli/cuda_redirect_smoke.cpp` (N2's CLI smoke — device alloc, error checking, the `main()` shape) and `snapshot/csrc/backends/cuda/cuda_graph.cpp` (N1 — how it builds/launches a `CUgraph` via `cuGraphAddKernelNode`, and its `cuFuncGetParamInfo` exact-kernarg handling). Mirror their CUDA error-check macro and structure.

- [ ] **Step 2: Implement the two kernels + graph capture**

`cuda_record_smoke.cpp` does, in `main()`:
1. `cuInit(0)`, get device 0, create a context (or use the primary context, matching `cuda_redirect_smoke.cpp`).
2. Allocate two device buffers `a`, `out` (e.g. 1024 `int32`) via `cuMemAlloc` (these land at fixed-base addresses when run under `redirect_cuda`; standalone in Task 1 they're driver-chosen — fine, Task 1 doesn't record).
3. **Static kernel** (compiled by `nvcc` in the same TU): `__global__ void k_add(int* a, int* out, int n)` that writes `out[i] = a[i] + 1`. Launch-capable via the runtime launch (`<<<>>>`/`cudaLaunchKernel`) so `__cudaRegisterFunction` covers it.
4. **nvrtc kernel**: compile `extern "C" __global__ void k_mul(int* out, int n){ out[i]*=2; }` to PTX with nvrtc, `cuModuleLoadData` → `cuModuleGetFunction` → `f_mul`.
5. Initialize `a` to a known pattern (`a[i]=i`) via `cuMemcpyHtoD`.
6. **Capture a graph** on a non-default stream: `cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)` → launch `k_add` (runtime launch on the captured stream) → launch `f_mul` via `cuLaunchKernel(f_mul, ...)` → `cuStreamEndCapture(stream, &graph)`.
7. `cuGraphInstantiate(&exec, graph, 0)`; `cuGraphLaunch(exec, stream)`; `cuStreamSynchronize`.
8. `cuMemcpyDtoH` `out`; compute `CHECKSUM` = a stable hash (e.g. FNV-1a over the bytes) and print `CHECKSUM=<hex>`.

Expected logical result: `out[i] = (i + 1) * 2`, so the checksum is deterministic across runs.

- [ ] **Step 3: Add the CMake target (additive)**

Under the existing CUDA branch of `snapshot/CMakeLists.txt` (the `SNAPSHOT_BACKEND_CUDA` block N1/N2 added), add:
```cmake
# N5a: CLI smoke for record/restore (CUDA). nvcc compiles the static kernel TU;
# links libcuda + nvrtc + cudart. Additive — does not touch the HIP targets.
if(SNAPSHOT_BACKEND STREQUAL "CUDA")
  add_executable(cuda_record_smoke csrc/cli/cuda_record_smoke.cpp)
  set_source_files_properties(csrc/cli/cuda_record_smoke.cpp PROPERTIES LANGUAGE CUDA)
  target_link_libraries(cuda_record_smoke PRIVATE CUDA::cuda_driver CUDA::nvrtc CUDA::cudart)
endif()
```
(Match the exact target/option names N1/N2 used — read the existing CUDA branch first. If the project uses raw `find_library` instead of `CUDA::` imported targets, mirror that.)

- [ ] **Step 4: Write the build+run sbatch (skeleton)**

`snapshot/recipe/cuda_record_smoke.sbatch`: mirror N1's build harness (same CUDA dev container EDF/`--environment`, same `-A a-infra02 --partition=normal --gpus-per-node=1`, `--nodes=1`). Build with `cmake -DSNAPSHOT_BACKEND=CUDA` + `cmake --build`, then run `./cuda_record_smoke` once and echo its `CHECKSUM=` line. (Record/restore runs added in Task 5.)

- [ ] **Step 5: Gate — build + run on bristen**

```bash
rcc --profile bristen-snapshot push
ssh bristen 'cd /capstor/scratch/cscs/xyao/snapshot-cuda && sbatch snapshot/recipe/cuda_record_smoke.sbatch'
ssh bristen 'tail -n 40 <the job .out>'
```
Expected: clean build, a `CHECKSUM=<hex>` line, exit 0. Run it **twice** and confirm the checksum is identical (determinism precondition for the later bit-identical gate).

- [ ] **Step 6: Commit**
```bash
git add snapshot/csrc/cli/cuda_record_smoke.cpp snapshot/CMakeLists.txt snapshot/recipe/cuda_record_smoke.sbatch
git commit -m "snapshot(cuda): N5a Task 1 — two-kernel CLI smoke (static + nvrtc graph capture)"
```

---

## Task 2: `record_cuda` interposer skeleton + eager-gated identity

Create the interposer with LD_PRELOAD plumbing, env mode selection, the eager gate, and the full identity hooks. No record/restore yet — prove identity is captured for **both** functions.

**Files:**
- Create: `snapshot/csrc/preload/snapshot_record_cuda.cpp`
- Modify: `snapshot/CMakeLists.txt` (add the `snapshot_record_cuda` shared lib under the CUDA branch)

**Interfaces:**
- Consumes: nothing from Task 1 at link time (it's LD_PRELOADed into the smoke process).
- Produces: an env contract — `SNAPSHOT_RECORD_CUDA_MODE=record|restore` (default `record`), `SNAPSHOT_RECORD_CUDA_DIR=<dir>` (`.snap` location); a `[record-cuda] … SUMMARY` line at exit (mirroring redirect's SUMMARY) reporting mode, identity count, recorded/restored graph count, and `fallthrough=N`. An internal `IdentityMap` mapping a recorded id `{kind, name, module_hash}` → `CUfunction`, used by Tasks 3–4.

- [ ] **Step 1: Read the N2 interposer for the plumbing pattern**

Read `snapshot/csrc/preload/snapshot_redirect_cuda.cpp` — copy its exact patterns for: `dlsym(RTLD_NEXT, …)` lazy resolution of real symbols, the env parsing, the `-static-libstdc++` posture, the constructor/`__attribute__((constructor))` init, and the atexit SUMMARY line. Mirror these; do not reinvent.

- [ ] **Step 2: Implement the identity hooks (eager-gated)**

In `snapshot_record_cuda.cpp` interpose:
- `void** __cudaRegisterFatBinary(void* fatCubin)` → call real, then store `{handle, fatCubin ptr}` in a fatbin registry. Lightweight only.
- `void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun, const char* deviceName, …)` → call real, then record `hostFun → deviceName` (the symbol). Lightweight only.
- `__cudaRegisterFatBinaryEnd(void**)` → pass-through (record completion).
- `CUresult cuModuleLoadData(CUmodule* mod, const void* image)` → call real; compute a content hash of `image` (size-bounded FNV-1a); store `*mod → image_hash`.
- `CUresult cuModuleGetFunction(CUfunction* f, CUmodule mod, const char* name)` → call real; on success record `{kind=module, name, module_hash=hash(mod)} → *f` into `IdentityMap`.
- For static kernels, the host-stub → `CUfunction` link is resolved lazily: when a kernel node references a function, resolve its name via `cuFuncGetName(&name, func)` (12.3+) and match against the `__cudaRegisterFunction` symbol table; record `{kind=fatbin, name} → func`.

**Eager gate:** none of these hooks enumerate cubin symbols or do heavy work on the import path — they store pointers/handles/hashes only.

- [ ] **Step 3: Add the CMake shared-lib target (additive)**
```cmake
if(SNAPSHOT_BACKEND STREQUAL "CUDA")
  add_library(snapshot_record_cuda SHARED csrc/preload/snapshot_record_cuda.cpp)
  target_link_libraries(snapshot_record_cuda PRIVATE CUDA::cuda_driver CUDA::cudart ${CMAKE_DL_LIBS})
  target_link_options(snapshot_record_cuda PRIVATE -static-libstdc++ -static-libgcc)
endif()
```
(Mirror the exact `snapshot_redirect_cuda` target definition; keep linkage libcuda+libcudart only.)

- [ ] **Step 4: Gate — identity captured for both kernels**

Add to the sbatch a record-mode run: `LD_PRELOAD=<build>/libsnapshot_record_cuda.so SNAPSHOT_RECORD_CUDA_MODE=record ./cuda_record_smoke`. Expected: the SUMMARY logs **identity for both** `k_add` (fatbin/static) and `k_mul` (module/nvrtc) — e.g. `identity: 2 functions (1 fatbin, 1 module)`. The smoke still runs and prints its normal `CHECKSUM=`. Startup overhead negligible (no +271s regression).

- [ ] **Step 5: Commit**
```bash
git add snapshot/csrc/preload/snapshot_record_cuda.cpp snapshot/CMakeLists.txt snapshot/recipe/cuda_record_smoke.sbatch
git commit -m "snapshot(cuda): N5a Task 2 — record_cuda interposer skeleton + eager-gated identity (both paths)"
```

---

## Task 3: Capture / record path + `.snap` format

Hook the capture-end APIs, walk the captured graph, and persist kernel nodes (function identity + verbatim kernarg blob + structure) to `.snap`. Run under `redirect_cuda` so addresses are fixed-base.

**Files:**
- Create: `snapshot/csrc/preload/record_cuda_format.hpp`
- Modify: `snapshot/csrc/preload/snapshot_record_cuda.cpp` (add capture/record)

**Interfaces:**
- Produces: the `.snap` schema (consumed by Task 4). Per graph: a header `{magic, version, node_count}`, then per kernel node: `{func_id (the IdentityMap key: kind+name+module_hash), gridDimX/Y/Z, blockDimX/Y/Z, sharedMemBytes, kernarg_size, kernarg_blob[kernarg_size], dep_count, dep_indices[dep_count]}`. Node indices are the `cuGraphGetNodes` order; deps reference earlier indices.

- [ ] **Step 1: Define the `.snap` format**

`record_cuda_format.hpp`: POD structs for the header + per-node record above, plus `serialize_graph(const RecordedGraph&, path)` and `RecordedGraph deserialize_graph(path)`. Keep it a plain binary format (length-prefixed blobs); no msgpack, no dependency on `core/record.cpp`.

- [ ] **Step 2: Hook the capture-end APIs (record mode)**

In `snapshot_record_cuda.cpp`, interpose `cuStreamEndCapture(CUstream stream, CUgraph* phGraph)` (and `cudaStreamEndCapture`): call the **real** end-capture first (so the caller gets a working graph and run #1 executes normally), then if `MODE==record`, walk `*phGraph`:
```cpp
size_t n = 0; cuGraphGetNodes(graph, nullptr, &n);
std::vector<CUgraphNode> nodes(n); cuGraphGetNodes(graph, nodes.data(), &n);
for (size_t i = 0; i < n; ++i) {
  CUgraphNodeType t; cuGraphNodeGetType(nodes[i], &t);
  if (t != CU_GRAPH_NODE_TYPE_KERNEL) continue;          // N5a: kernel nodes only
  CUDA_KERNEL_NODE_PARAMS p{}; cuGraphKernelNodeGetParams(nodes[i], &p);
  // func identity:
  RecordedFuncId id = identity_for(p.func);               // IdentityMap reverse-lookup; cuFuncGetName fallback for static
  // kernarg blob size = sum over params of (offset+size); last param's offset+size:
  size_t ksize = kernarg_size(p.func);                    // via cuFuncGetParamInfo loop (see N1 cuda_graph.cpp)
  // kernelParams is an array of void* each pointing at one arg value; copy them
  // back into a single contiguous blob using per-param offset+size from cuFuncGetParamInfo:
  std::vector<uint8_t> blob = pack_kernarg(p.func, p.kernelParams, ksize);
  record_node(graph_rec, id, p /*dims*/, blob);
  // deps:
  size_t dn = 0; cuGraphNodeGetDependencies(nodes[i], nullptr, &dn);
  std::vector<CUgraphNode> deps(dn); cuGraphNodeGetDependencies(nodes[i], deps.data(), &dn);
  record_deps(graph_rec, i, deps, nodes);                 // map dep handles → indices
}
serialize_graph(graph_rec, snap_path(graph_index++));
```
`pack_kernarg`: for each param `j`, `cuFuncGetParamInfo(func, j, &offset, &size)`; `memcpy(blob.data()+offset, p.kernelParams[j], size)`. This reconstitutes the contiguous kernarg buffer the kernel expects (the same exact-kernarg approach N1 already uses). Stop when `cuFuncGetParamInfo` returns `CUDA_ERROR_INVALID_VALUE` (past the last param).

- [ ] **Step 3: `identity_for(CUfunction)`**

Reverse-lookup the `IdentityMap` built in Task 2 (module functions are keyed at `cuModuleGetFunction`). For a static-kernel `CUfunction` not in the module map, call `cuFuncGetName(&name, func)` and emit `{kind=fatbin, name}`. If neither resolves → mark the node **blind** and increment a blind counter (G4 must end at 0).

- [ ] **Step 4: Gate — `.snap` written, smoke still correct, Δ=0**

Run under `LD_PRELOAD=libsnapshot_redirect_cuda.so:libsnapshot_record_cuda.so` with `SNAPSHOT_RECORD_CUDA_MODE=record` and the redirect fixed-base env (mirror N2's torch/raw recipe env). Expected: a `.snap` file written with **2 kernel nodes** (k_add, k_mul) and a dependency edge (k_mul after k_add); the smoke prints its normal `CHECKSUM=`; the redirect SUMMARY shows `passthrough=0` (Δ=0 held); the record SUMMARY shows `recorded: 1 graph, 2 nodes, blind=0`.

- [ ] **Step 5: Commit**
```bash
git add snapshot/csrc/preload/record_cuda_format.hpp snapshot/csrc/preload/snapshot_record_cuda.cpp snapshot/recipe/cuda_record_smoke.sbatch
git commit -m "snapshot(cuda): N5a Task 3 — capture/record path + record_cuda .snap format (verbatim kernarg, Δ=0)"
```

---

## Task 4: Restore-shim path + bit-identical restore

In `restore` mode, load the `.snap`, shim the capture APIs to rebuild and return pre-built graphs (reusing N1's `cuGraphAddKernelNode` rebuild), skip real capture, fall through on exhaustion.

**Files:**
- Modify: `snapshot/csrc/preload/snapshot_record_cuda.cpp` (add restore-shim)

**Interfaces:**
- Consumes: the `.snap` format (Task 3), the `IdentityMap` (Task 2), N1's rebuild approach (`backends/cuda/cuda_graph.cpp`).
- Produces: the restore behavior gated by `SNAPSHOT_RECORD_CUDA_MODE=restore`; SUMMARY reports `restored: N graphs, fallthrough: M`.

- [ ] **Step 1: Load `.snap`s into a restore queue at init**

In `restore` mode, at the constructor (or first capture call), `deserialize_graph` every `.snap` in `SNAPSHOT_RECORD_CUDA_DIR` into an ordered queue (the same order the record run produced them — encode an index in the filename).

- [ ] **Step 2: Shim the capture APIs**

Interpose:
- `cuStreamBeginCapture(stream, mode)` → in restore mode, mark `stream` as "shim-capturing" in a thread-safe set; return `CUDA_SUCCESS` **without** calling the real begin-capture.
- `cuStreamIsCapturing(stream, *status)` → if shim-capturing, set `*status = CU_STREAM_CAPTURE_STATUS_ACTIVE`, return success.
- `cuStreamEndCapture(stream, *phGraph)` → if shim-capturing: pop the next `.snap` from the queue and **rebuild** it (Step 3) into a fresh `CUgraph`; set `*phGraph` to it; clear the shim-capturing flag; return success. The caller then `cuGraphInstantiate`/`cuGraphLaunch` the rebuilt graph as usual.
- **Fall-through:** if the queue is empty when a shim-capture ends, call the **real** begin/end capture for that window instead (record `fallthrough++`), so a short recording degrades gracefully. (Requires beginning the real capture lazily — simplest: if queue empty at `cuStreamBeginCapture`, do NOT shim that stream at all; let real capture run.)

- [ ] **Step 3: Rebuild a graph from a `.snap`**

Mirror N1 `backends/cuda/cuda_graph.cpp`'s rebuild:
```cpp
CUgraph g; cuGraphCreate(&g, 0);
std::vector<CUgraphNode> built(rec.nodes.size());
for (i, node_rec in rec.nodes) {
  CUfunction func = resolve(node_rec.func_id);     // IdentityMap; blind→abort (must be 0 by G4)
  CUDA_KERNEL_NODE_PARAMS p{};
  p.func = func; p.gridDimX = node_rec.gx; /* …dims… */ p.sharedMemBytes = node_rec.smem;
  // verbatim kernarg: point kernelParams at the recorded blob via per-param offsets,
  // OR (simpler, matches N1) pass the contiguous blob through `p.extra` using
  // CU_LAUNCH_PARAM_BUFFER_POINTER/SIZE so no per-arg pointer array is needed:
  void* extra[] = { CU_LAUNCH_PARAM_BUFFER_POINTER, blob_ptr,
                    CU_LAUNCH_PARAM_BUFFER_SIZE, &blob_size,
                    CU_LAUNCH_PARAM_END };
  p.kernelParams = nullptr; p.extra = extra;
  std::vector<CUgraphNode> deps; for (d in node_rec.dep_indices) deps.push_back(built[d]);
  cuGraphAddKernelNode(&built[i], g, deps.data(), deps.size(), &p);
}
return g;
```
The kernarg blob is replayed **byte-for-byte** (Δ=0 → its device pointers are valid). The `p.extra` buffer form is exactly the N1 mechanism (N1 records a flat padded kernarg blob and launches via the driver `extra` format) — reuse it so kernarg-size exactness (the N1 CUDA finding) is preserved.

- [ ] **Step 4: Gate — bit-identical restore (G1–G4)**

Run #2 under `LD_PRELOAD=redirect_cuda:record_cuda`, `SNAPSHOT_RECORD_CUDA_MODE=restore`, same fixed-base env, same `.snap` dir. Expected:
- **G1:** `CHECKSUM=` identical to the Task 3 record run.
- **G2:** restore SUMMARY `restored: 1 graph, fallthrough: 0`; real-capture count 0.
- **G3:** redirect SUMMARY `passthrough=0` (Δ=0).
- **G4:** `blind=0` (both funcs resolved at restore).

- [ ] **Step 5: Commit**
```bash
git add snapshot/csrc/preload/snapshot_record_cuda.cpp snapshot/recipe/cuda_record_smoke.sbatch
git commit -m "snapshot(cuda): N5a Task 4 — restore-shim (rebuild + skip capture), bit-identical restore on A100"
```

---

## Task 5: Record→restore driver + full gate run + RESULTS + regression

Finalize the two-run driver, run the full gate sequence, document, and confirm additivity.

**Files:**
- Modify: `snapshot/recipe/cuda_record_smoke.sbatch` (the full record→restore sequence + gate assertions)
- Modify: `snapshot/RESULTS.md` (append N5a section)

- [ ] **Step 1: Finalize the driver sbatch**

The sbatch: build (CUDA dev container) → **run #1 record** (`redirect_cuda:record_cuda`, `MODE=record`) capturing CHECKSUM_1 + `.snap`s → **run #2 restore** (`MODE=restore`) capturing CHECKSUM_2 → assert `CHECKSUM_1 == CHECKSUM_2` (G1), grep the record/restore/redirect SUMMARY lines for `blind=0`, `fallthrough=0`, `passthrough=0`, real-capture `=0`. Print a `N5A_GATES: G1=.. G2=.. G3=.. G4=..` summary line.

- [ ] **Step 2: Gate — full sequence green on bristen**
```bash
rcc --profile bristen-snapshot push
ssh bristen 'cd /capstor/scratch/cscs/xyao/snapshot-cuda && sbatch snapshot/recipe/cuda_record_smoke.sbatch'
ssh bristen 'tail -n 60 <job .out>'
```
Expected: `N5A_GATES: G1=PASS G2=PASS G3=PASS G4=PASS`.

- [ ] **Step 3: Regression check (G5)**
```bash
git diff --stat "$(git merge-base main HEAD)" -- \
  snapshot/csrc/backends/hip snapshot/csrc/preload/snapshot_record.cpp \
  snapshot/csrc/preload/snapshot_redirect.cpp snapshot/csrc/core \
  snapshot/csrc/backends/cuda snapshot/csrc/preload/snapshot_redirect_cuda.cpp
# expected: (empty)
```
And confirm the **HIP build is still green**: build once with `-DSNAPSHOT_BACKEND=HIP` (mirror the N1/N2 HIP regression job) — or, if no AMD node is available this session, assert no HIP source/CMake-HIP-branch bytes changed (the diff above) and note that the HIP build was last green at the N2 gate.

- [ ] **Step 4: Append the N5a RESULTS section**

Document: the bristen environment, the two-kernel smoke, the record→restore sequence, the gate results (G1 bit-identical CHECKSUM, G2 capture skipped, G3 Δ=0/passthrough=0, G4 identity resolved both paths, G5 additive), the `.snap` node count, and a one-line "N5b (vLLM-CUDA TP=4 record/restore + A/B/C) is next." Match the N1/N2/N3 RESULTS tone; decompose honestly (this is a *mechanism* proof on the CLI, not yet a vLLM win).

- [ ] **Step 5: Commit**
```bash
git add snapshot/recipe/cuda_record_smoke.sbatch snapshot/RESULTS.md
git commit -m "snapshot(cuda): N5a complete — record/restore mechanism bit-identical on A100, HIP additive"
```

---

## Self-Review

**Spec coverage (against the N5a design):**
- Verbatim kernarg, no parser → Task 3 (`pack_kernarg`/`extra` blob) + Task 4 (byte-for-byte replay); Global Constraints ✓
- Full identity (`__cudaRegister*` + `cuModuleLoadData`/`cuFuncGetName`) → Task 2; both paths exercised by the two-kernel smoke (Task 1) ✓
- Eager interposer gate → Task 2 Step 2 (lightweight hooks) + Task 2 Step 4 (no startup regression) ✓
- Capture/record + `.snap` → Task 3 ✓
- Restore-shim (`cuStreamBeginCapture`/`IsCapturing`/`EndCapture`, rebuild via N1 path, fall-through) → Task 4 ✓
- `record_cuda` format in `preload/`, `core/` untouched → File Structure + Task 3 ✓
- Gates G1–G5 → Tasks 4–5 ✓
- Dual-vendor additive, HIP green → Global Constraints + Task 5 Step 3 ✓

**Placeholder scan:** the only `<…>` are run-specific job-output paths in gate commands (resolved at run time) — not design placeholders. The exact CUDA struct field names (`CUDA_KERNEL_NODE_PARAMS`) and imported-target names (`CUDA::cuda_driver`) must be bound by the implementer from `cuda.h` / the existing CUDA CMake branch — flagged in Task 1 Step 1 and Task 2 Step 1 (read N1/N2 first). No TODO/TBD.

**Interface consistency:** the `.snap` schema (Task 3 Interfaces) is the exact structure rebuilt in Task 4 Step 3; the `IdentityMap` key `{kind, name, module_hash}` is written in Task 2, used in Task 3 `identity_for`, and resolved in Task 4 `resolve`. The env contract (`SNAPSHOT_RECORD_CUDA_MODE`/`_DIR`) and the SUMMARY counters (`blind`, `fallthrough`, `passthrough`, real-capture) are consistent across Tasks 2–5 and the gates.

**Scope:** N5a only — mechanism on the CLI. vLLM TP=4, FULL+PIECEWISE, 4-process, serving overhead, and the cubin parser are N5b/deferred (design §6 Out).

---

## Execution Handoff

Notes specific to N5a before execution:
1. **Reference-first:** Tasks 1–4 each begin by reading the proven N1 (`backends/cuda/cuda_graph.cpp` rebuild + `cuFuncGetParamInfo`) and N2 (`snapshot_redirect_cuda.cpp` plumbing) so the implementer binds exact API signatures and mirrors the cross-container link posture rather than guessing.
2. **The large uncommitted AMD WIP lives in `main`'s working tree** (M3k + `csrc/*` + recipes). Execute in an isolated worktree off local `main` HEAD (the N2/N3 pattern) so N5a's commits stay clean; N5a touches only new `*_cuda` files so there is no overlap with the AMD WIP this time.
3. Every gate is a bristen cluster job (`rcc --profile bristen-snapshot push` → `sbatch -A a-infra02`); builds run in the N1 CUDA dev container. The HIP-build-green check (Task 5 Step 3) needs an AMD node — if unavailable this session, the byte-unchanged diff stands in and is noted.
