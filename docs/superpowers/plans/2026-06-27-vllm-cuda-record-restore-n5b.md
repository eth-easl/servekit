# N5b — vLLM-CUDA TP=4 record/restore + cold-start measurement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the CLI-proven `snapshot_record_cuda` interposer into vLLM-CUDA (GLM-4.7-Flash, TP=4) on bristen so cold start #1 records the CUDA graphs and cold start #2 restores them (skip capture) serving **token-identical**, and measure the A/B/C cold-start win + serving overhead.

**Architecture:** Two cooperating layers. (1) The **C-interposer** (`snapshot_record_cuda.cpp`, additively extended) records/rebuilds the CUDA graphs — now via the **runtime** capture APIs PyTorch uses, with non-kernel nodes, per-rank dirs, and a `region_base` export. (2) A **Python layer** (`cg_meta_cuda.py`, ported from the AMD `cg_meta.py`) reconstructs vLLM's `CUDAGraphWrapper.entry.output` (which the C layer cannot create) and skips the model forward. Fixed-base Δ=0 (N2/N5a) makes the recorded device pointers valid as-is in run #2 — no relocation.

**Tech Stack:** CUDA driver+runtime API (sm_80, CUDA 12.x), nvrtc, C++17 LD_PRELOAD interposer, vLLM 0.23.0 (`vllm/vllm-openai`), PyTorch `torch.cuda.CUDAGraph`, Python `sitecustomize.py` + ctypes/DLPack, SLURM/enroot on bristen (`-A a-infra02`), `rcc` deploy.

## Global Constraints

- **Dual-vendor additive.** Change ONLY new `*_cuda`/`*_cuda.py` files plus the additively-extended `snapshot/csrc/preload/snapshot_record_cuda.cpp`. HIP (`backends/hip/*`, `preload/snapshot_record.cpp`, `preload/snapshot_redirect.cpp`), all `snapshot/csrc/core/*`, and N1/N2 CUDA (`backends/cuda/*`, `preload/snapshot_redirect_cuda.cpp`) are **byte-unchanged**. The existing HIP `snapshot/recipe/cginst_skip/*` prototypes are **references, not edits** — the CUDA port is a sibling dir.
- **Verbatim Δ=0 replay.** Recorded kernarg blobs and `entry.output` device pointers are replayed/reconstructed byte-for-byte; `live_base == snap_base` so there is NO pointer relocation and NO cubin/PTX parser.
- **`.so` linkage.** The record/redirect `.so`s link libcuda + libcudart + libdl only, `-static-libstdc++ -static-libgcc` (N2/N5a posture, for cross-container load).
- **N5a CLI gates stay green.** Every C-interposer change must keep `cuda_record_smoke.sbatch`'s existing G1–G4 PASS (job pattern from N5a: `N5A_GATES: G1=PASS …`).
- **Build AND run on bristen.** A task is not done until its cluster gate passes: `rcc --profile <profile> push` → `rcc --profile <profile> run 'sbatch <recipe>'` → poll `squeue`/log. CLI gates use profile `bristen-snapshot` (`/capstor/scratch/cscs/xyao/snapshot-cuda`); vLLM gates use profile `glm-47-flash-bristen-vllm` (`/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda`). Account `a-infra02`, partition `normal`, 4×A100 sm_80, **one `vllm serve` per `srun` step** (enroot holds CUDA IPC shmem at container level).
- **Honest measurement accounting.** vLLM's `init engine … took` already *includes* capture (do not double-count it as a separate additive phase); per-graph `VLLM_CG_INSTRUMENT` sums are 4-TP-worker aggregates (≈ work, not wall-clock). Use the wall-clock cold-start-to-ready delta for eliminable time.
- **Reference files (read first, do not edit):** `snapshot/csrc/preload/snapshot_record_cuda.cpp` (N5a interposer), `snapshot/csrc/preload/record_cuda_format.hpp` (`.snap` format), `snapshot/csrc/cli/cuda_record_smoke.cpp` + `snapshot/recipe/cuda_record_smoke.sbatch` (N5a CLI gate), `snapshot/csrc/backends/cuda/cuda_graph.cpp` (N1 rebuild + `cuFuncGetParamInfo`), `snapshot/recipe/cginst_skip/cg_meta.py` + `cg_skip.py` (AMD Python prototypes to port), `snapshot/recipe/cginst/sitecustomize.py` (N3 instrumentation loader), `snapshot/recipe/vllm_coldstart_cuda.sbatch` + `_vllm_coldstart_cuda.sh` + `_vllm_measure_cuda.sh` (N3 deploy/measure), `deploy/glm-47-flash-bristen-vllm/glm-47-flash-vllm-cuda.toml` (EDF).

---

## File Structure

- `snapshot/csrc/preload/snapshot_record_cuda.cpp` — **modified** (additive): runtime capture shims + dedupe-by-`CUgraph`; non-kernel (memcpy/memset) record+rebuild; `extra`-buffer kernargs; per-rank dir resolution; `snapshot_record_cuda_region_base()` export; scale hardening (prebuilt reverse maps, bounded lifetime, `cuModuleUnload` eviction, `cuModuleLoadDataEx`).
- `snapshot/csrc/preload/record_cuda_format.hpp` — **modified** only if non-kernel node fields are added (bump format version).
- `snapshot/csrc/cli/cuda_record_runtime_smoke.cpp` — **new**: CLI smoke capturing via the **runtime** `cudaStreamBeginCapture`/`EndCapture` (Task 1 gate).
- `snapshot/csrc/cli/cuda_record_full_smoke.cpp` — **new**: FULL-like graph (many kernel + memcpy/memset nodes + deep/branching deps) record→restore (Task 2 gate, G1).
- `snapshot/recipe/cuda_record_smoke.sbatch` — **modified** (additive): build+run the two new smokes' gates alongside the N5a gates.
- `snapshot/recipe/cginst_cuda/cg_meta_cuda.py` + `sitecustomize.py` — **new**: the CUDA Python record/restore layer (port of `cginst_skip/cg_meta.py`).
- `snapshot/csrc/cli/test_cg_meta_cuda_wrap.cpp` **or** `snapshot/recipe/cginst_cuda/_test_wrap.py` — **new**: host-side unit test of the DLPack device-pointer wrap + metadata round-trip (Task 4 gate).
- `snapshot/recipe/vllm_record_cuda.sbatch` + `_vllm_record_cuda.sh` — **new**: TP=4 record-mode cold start (Task 6 gate, G2).
- `snapshot/recipe/vllm_restore_cuda.sbatch` + `_vllm_restore_cuda.sh` — **new**: TP=4 restore-mode cold start + correctness (Task 7 gate, G3/G4).
- `snapshot/recipe/_vllm_abc_cuda.sh` + `vllm_abc_cuda.sbatch` — **new**: A/B/C cold-start + focused serving-overhead measurement (Task 8 gate, G5).
- `snapshot/RESULTS.md` — **modified**: append §N5b.

---

## Task 1: Runtime capture shims + dedupe-by-`CUgraph`

**Files:**
- Create: `snapshot/csrc/cli/cuda_record_runtime_smoke.cpp`
- Modify: `snapshot/csrc/preload/snapshot_record_cuda.cpp` (add runtime hooks)
- Modify: `snapshot/recipe/cuda_record_smoke.sbatch` (add the runtime-smoke gate)

**Interfaces:**
- Consumes: the N5a interposer's record/restore internals (`record_captured_graph`, `rebuild_graph`, the shim-capturing stream set, `g_mode()`), and `record_cuda_format.hpp`.
- Produces: interposed `cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode)`, `cudaStreamEndCapture(cudaStream_t, cudaGraph_t*)`, `cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus*)` (+ `cudaStreamGetCaptureInfo` passthrough) that route to the SAME record/restore logic as the N5a driver hooks, recording/restoring each `CUgraph` exactly once (dedupe set keyed by the underlying `CUgraph` handle).

- [ ] **Step 1: Write the failing CLI gate — runtime-API capture smoke**

`cuda_record_runtime_smoke.cpp`: mirror `cuda_record_smoke.cpp` (cudaMalloc buffers; static `k_add` + nvrtc `k_mul`) but drive capture with the **runtime** API:
```cpp
cudaStream_t stream;  cudaStreamCreate(&stream);
cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
k_add<<<grid,block,0,stream>>>(d_a, d_out, n);            // runtime launch
cuLaunchKernel(f_mul, grid,1,1, block,1,1, 0, stream, mul_args, nullptr);
cudaGraph_t graph;  cudaStreamEndCapture(stream, &graph);
cudaGraphExec_t exec;  cudaGraphInstantiate(&exec, graph, 0);
cudaGraphLaunch(exec, stream);  cudaStreamSynchronize(stream);
// readback + CHECKSUM (fnv1a32), expect (i+1)*2 == aa44e26d (same kernels)
```
Add a build+run block to `cuda_record_smoke.sbatch` after the N5a gates: build `cuda_record_runtime_smoke` + `snapshot_record_cuda`; run record→restore into a fresh per-run dir; assert `RT_CHECKSUM` record==restore, `restored=1 fallthrough=0 real_begin=0 suppressed=2 blind=0` in the SUMMARY, and that it routed through the **runtime** path (a new SUMMARY field `rt_capture=1`).

- [ ] **Step 2: Run the gate — confirm it FAILS (runtime hooks absent)**

`rcc --profile bristen-snapshot push && rcc --profile bristen-snapshot run 'sbatch snapshot/recipe/cuda_record_smoke.sbatch'`; poll, read the log.
Expected: the runtime-smoke record→restore does NOT skip capture (no `[record-cuda]` runtime engagement / `real_begin>0`) → gate FAILS. (Proves PyTorch's runtime path is not yet intercepted.)

- [ ] **Step 3: Add the runtime capture hooks**

In `snapshot_record_cuda.cpp`, add `extern "C"` interposers for `cudaStreamBeginCapture` / `cudaStreamEndCapture` / `cudaStreamIsCapturing` mirroring the N5a driver hooks (same shim-capturing stream set; same `record_captured_graph`/`rebuild_graph`). `cudaGraph_t` IS `CUgraph` (same `CUgraph_st*`), so pass through to the shared logic. Maintain a `std::set<CUgraph> g_walked_graphs` under `g_mu`: in record, before walking a graph in `*Stream*EndCapture`, skip if already walked (the runtime may delegate to the interposed driver `cuStreamEndCapture` → would otherwise double-walk). Increment a `g_rt_captures` counter and print `rt_capture=<n>` in the SUMMARY. Real-symbol resolution via `dlsym(RTLD_NEXT, …)` thunks as in N5a.

- [ ] **Step 4: Run the gate — confirm PASS**

Re-push + re-run. Expected: runtime-smoke `RT_CHECKSUM` record==restore, `restored=1 fallthrough=0 real_begin=0 suppressed=2 blind=0 rt_capture=1`, AND the N5a gates (`N5A_GATES: G1=PASS…`) still green (driver path unaffected). No double-walk (`recorded graphs=1`, not 2).

- [ ] **Step 5: Commit**
```bash
git add snapshot/csrc/cli/cuda_record_runtime_smoke.cpp snapshot/csrc/preload/snapshot_record_cuda.cpp snapshot/recipe/cuda_record_smoke.sbatch snapshot/CMakeLists.txt
git commit -m "snapshot(cuda): N5b Task 1 — runtime cudaStream*Capture shims + dedupe-by-CUgraph"
```

---

## Task 2: FULL-like graph — non-kernel nodes + `extra` kernargs + deep deps (G1)

**Files:**
- Create: `snapshot/csrc/cli/cuda_record_full_smoke.cpp`
- Modify: `snapshot/csrc/preload/snapshot_record_cuda.cpp` (record+rebuild non-kernel nodes; `extra` kernargs)
- Modify: `snapshot/csrc/preload/record_cuda_format.hpp` (add node-type + memcpy/memset fields; bump version to 2)
- Modify: `snapshot/recipe/cuda_record_smoke.sbatch` (add the FULL-smoke gate)

**Interfaces:**
- Consumes: Task 1's runtime hooks; the N5a graph walk (`cuGraphGetNodes`/`cuGraphNodeGetType`) and create-then-link rebuild.
- Produces: record+rebuild support for `CU_GRAPH_NODE_TYPE_MEMCPY` and `CU_GRAPH_NODE_TYPE_MEMSET` nodes (via `cuGraphMemcpyNodeGetParams`/`cuGraphMemsetNodeGetParams` + `cuGraphAddMemcpyNode`/`cuGraphAddMemsetNode`); `extra`-buffer kernarg record/replay (`CU_LAUNCH_PARAM_BUFFER_POINTER`); a `blind_reason` per unsupported node (no silent edge-drop). `.snap` format v2.

- [ ] **Step 1: Write the failing CLI gate — FULL-like graph**

`cuda_record_full_smoke.cpp`: build a graph with breadth + node-type variety to mimic a real FULL/PIECEWISE graph: e.g. memset `d_out=0` (memset node) → `k_add` (kernel) → memcpy D2D `d_tmp←d_out` (memcpy node) → `k_mul` on `d_tmp` (kernel, launched via the `extra` buffer form) → memcpy `d_out←d_tmp`, with a branch (two independent `k_add` chains merging into a final kernel) so dependencies are non-linear. Capture via the runtime API (Task 1). Deterministic CHECKSUM. Add a build+run gate to the sbatch: record→restore into a fresh dir; assert `FULL_CHECKSUM` record==restore, the recorded node count matches (kernel+memcpy+memset), `blind=0`, byte-identical `.snap` across two record runs.

- [ ] **Step 2: Run the gate — confirm it FAILS (non-kernel nodes blind/dropped)**

Push+run. Expected: FAIL — memcpy/memset nodes are blind or their dep edges drop (N5a was kernel-only), so restore mismatches or `blind>0`.

- [ ] **Step 3: Record + rebuild non-kernel nodes and `extra` kernargs**

In `snapshot_record_cuda.cpp`: in the graph walk, dispatch on `cuGraphNodeGetType` — kernel (existing), MEMCPY (`cuGraphMemcpyNodeGetParams` → record the `CUDA_MEMCPY3D` fields + the Δ=0 device pointers verbatim), MEMSET (`cuGraphMemsetNodeGetParams` → value/width/height/pitch/dst). In rebuild, create each via `cuGraphAddMemcpyNode`/`cuGraphAddMemsetNode` in pass 1, then link deps in pass 2 (create-then-link, as N5a). Include non-kernel nodes in `rec_index` so kernel→memcpy edges resolve (fixes the N5a non-kernel edge-drop). Add `extra`-buffer kernarg record/replay: when `p.kernelParams==nullptr && p.extra!=nullptr`, read the `CU_LAUNCH_PARAM_BUFFER_POINTER`/`SIZE` blob and record it verbatim (replay via the same `extra` form). Any still-unsupported node type → record a `blind_reason` string, increment blind. Bump `record_cuda_format.hpp` to v2 (add `node_type` + a per-type union/fields); keep v1 readers rejecting v2 cleanly (the smoke always writes the current version, so this only matters for stale dirs — assert version on read).

- [ ] **Step 4: Run the gate — confirm PASS (G1)**

Push+run. Expected: `FULL_CHECKSUM` record==restore, recorded node count == built count, `blind=0`, `.snap` byte-identical across two record runs; N5a + Task-1 gates still green.

- [ ] **Step 5: Commit**
```bash
git add snapshot/csrc/cli/cuda_record_full_smoke.cpp snapshot/csrc/preload/snapshot_record_cuda.cpp snapshot/csrc/preload/record_cuda_format.hpp snapshot/recipe/cuda_record_smoke.sbatch snapshot/CMakeLists.txt
git commit -m "snapshot(cuda): N5b Task 2 — FULL-like rebuild: non-kernel nodes + extra kernargs + deep deps (G1)"
```

---

## Task 3: Per-rank dirs + `region_base` export + scale hardening

**Files:**
- Modify: `snapshot/csrc/preload/snapshot_record_cuda.cpp`
- Modify: `snapshot/recipe/cuda_record_smoke.sbatch` (assert the export + per-rank resolution)

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces: `extern "C" uint64_t snapshot_record_cuda_region_base(void)` (returns the fixed VMM base, or 0 if redirect inactive); per-rank dir resolution — if `SNAPSHOT_RECORD_CUDA_DIR` contains `%r`, substitute the rank from `RANK`/`LOCAL_RANK`/`VLLM_DP_RANK`/`SLURM_PROCID` (first set wins); prebuilt reverse-identity maps (`{name,module_hash}→CUfunction`, `deviceName→hostFun`) replacing the O(N) scans; a `cuModuleUnload` hook evicting `g_module_identity`/`g_module_hashes`; an interposed `cuModuleLoadDataEx`; a bound on retained rebuilt-graph side-data (configurable, default large enough for vLLM).

- [ ] **Step 1: Write the failing gate — region_base export + per-rank dir**

Extend the sbatch: after building `snapshot_record_cuda`, assert the symbol exists (`nm -D libsnapshot_record_cuda.so | grep snapshot_record_cuda_region_base`) and, in a record run with `SNAPSHOT_RECORD_CUDA_DIR=…/rank%r` and `RANK=2` set, assert the `.snap`s land under `…/rank2/`. Add a tiny C check (or reuse a smoke) that `dlsym`s `snapshot_record_cuda_region_base()` under redirect and prints a non-zero `0x6…` base.

- [ ] **Step 2: Run — confirm FAIL (symbol/per-rank absent)**

Push+run. Expected: `nm` finds no symbol / `%r` not substituted → FAIL.

- [ ] **Step 3: Implement exports + per-rank + scale hardening**

Add the exported `snapshot_record_cuda_region_base()` (read the redirect base — resolve `snapshot_redirect_region_base` via `dlsym(RTLD_DEFAULT,…)` if the redirect `.so` exports it, else parse the fixed base constant; return 0 if absent). Add `%r` substitution in `g_snap_dir()`. Replace the reverse-resolution linear scans with `std::map` reverse indices built at registration/module-get time. Add the `cuModuleUnload` eviction + `cuModuleLoadDataEx` hooks. Keep the SUMMARY fields stable.

- [ ] **Step 4: Run — confirm PASS + N5a/Task-1/Task-2 gates green**

Push+run. Expected: symbol present, per-rank dir honored, base non-zero, all prior CLI gates still PASS.

- [ ] **Step 5: Commit**
```bash
git add snapshot/csrc/preload/snapshot_record_cuda.cpp snapshot/recipe/cuda_record_smoke.sbatch
git commit -m "snapshot(cuda): N5b Task 3 — per-rank dirs + region_base export + scale hardening"
```

---

## Task 4: `cg_meta_cuda.py` — port + host-side unit test

**Files:**
- Create: `snapshot/recipe/cginst_cuda/cg_meta_cuda.py`
- Create: `snapshot/recipe/cginst_cuda/sitecustomize.py`
- Create: `snapshot/recipe/cginst_cuda/_test_wrap.py` (host unit test)
- Modify: `snapshot/recipe/cuda_record_smoke.sbatch` (run the host unit test under the CUDA image)

**Interfaces:**
- Consumes: `snapshot_record_cuda_region_base()` (Task 3) via ctypes; the AMD `cginst_skip/cg_meta.py` (port source).
- Produces: `cg_meta_cuda` with `wrap_device_ptr(ptr, shape, dtype) -> torch.Tensor` (DLPack `kDLCUDA`=2), `_serialize/_reconstruct` (Δ=0: `_reconstruct` uses `region_base + off`, delta 0), and the `CUDAGraphWrapper.__call__` RECORD/RESTORE hooks + `GPUModelRunner._warmup_and_capture` skip, env-selected by `VLLM_CG_RECORD_META`/`VLLM_CG_RESTORE_META`. `sitecustomize.py` loads it (mirrors `cginst/sitecustomize.py`).

- [ ] **Step 1: Write the failing host unit test**

`_test_wrap.py`: allocate a CUDA tensor `t0 = torch.arange(16, device="cuda", dtype=torch.bfloat16)`; serialize it with `_serialize(t0, region_base=0)`; `wrap_device_ptr(t0.data_ptr(), [16], torch.bfloat16)` → `t1`; assert `t1.data_ptr()==t0.data_ptr()`, `t1.shape==t0.shape`, `t1.dtype==t0.dtype`, and `torch.equal(t1, t0)`. Round-trip a nested `{ "a": [t0, t0] }` through `_serialize`/JSON/`_reconstruct(live_base = t0.data_ptr()-off)` and assert tensors view the same pointers. Print `CG_META_WRAP_OK=1`.

- [ ] **Step 2: Run — confirm FAIL (module absent)**

Add to the sbatch a step that runs `python3 snapshot/recipe/cginst_cuda/_test_wrap.py` under the `snapshot-torch-cuda` EDF (torch present). Expected: ImportError / FAIL.

- [ ] **Step 3: Port `cg_meta.py` → `cg_meta_cuda.py`**

Copy `cginst_skip/cg_meta.py`; change: DLPack `device_device_type = 2` (`kDLCUDA`) not 10; `_region_base()` resolves `snapshot_record_cuda_region_base` (not the HIP symbol); `_reconstruct` keeps `region_base + off` (Δ=0 → equals recorded `data_ptr`, so it is also correct to assert delta==0 and log it). Keep the RECORD incremental-flush (SIGKILL-safe) and the RESTORE empty-`torch.cuda.graph()` + `_warmup_and_capture` skip. Re-bind against vLLM 0.23.0 symbols (`vllm.compilation.cuda_graph.CUDAGraphWrapper`, `vllm.v1.worker.gpu_model_runner.GPUModelRunner`, `vllm.forward_context`, `vllm.config.CUDAGraphMode`, `concrete_cudagraph_entries`) — the same surface `cginst/sitecustomize.py` patches. `sitecustomize.py`: load `cg_meta_cuda` when either env var is set (mirror `cginst/sitecustomize.py`).

- [ ] **Step 4: Run — confirm PASS**

Push+run. Expected: `CG_META_WRAP_OK=1`.

- [ ] **Step 5: Commit**
```bash
git add snapshot/recipe/cginst_cuda/ snapshot/recipe/cuda_record_smoke.sbatch
git commit -m "snapshot(cuda): N5b Task 4 — cg_meta_cuda port (kDLCUDA, Δ=0) + host wrap unit test"
```

---

## Task 5: Cross-container `.so` + vLLM record/restore recipe scaffold

**Files:**
- Create: `snapshot/recipe/_vllm_record_cuda.sh`, `snapshot/recipe/vllm_record_cuda.sbatch`
- Create: `snapshot/recipe/_vllm_restore_cuda.sh`, `snapshot/recipe/vllm_restore_cuda.sbatch`

**Interfaces:**
- Consumes: the built `libsnapshot_redirect_cuda.so` + `libsnapshot_record_cuda.so`; `cginst_cuda/`; the N3 deploy (`_vllm_coldstart_cuda.sh`, the EDF, the rcc profile `glm-47-flash-bristen-vllm`).
- Produces: a record-mode launcher that sets `LD_PRELOAD=<redirect>:<record>`, `SNAPSHOT_RECORD_CUDA_MODE=record`, `SNAPSHOT_RECORD_CUDA_DIR=…/snap/rank%r`, `PYTHONPATH` including `cginst_cuda/`, `VLLM_CG_RECORD_META=…/meta/rank${RANK}.json`, launches `vllm serve` GLM-4.7-Flash TP=4, waits for ready, sends one request, `kill -9`. A restore launcher mirrors it with `MODE=restore`/`VLLM_CG_RESTORE_META`.

- [ ] **Step 1: Cross-container glibc check (gate)**

In `vllm_record_cuda.sbatch`, before serving: build (or copy) the two `.so`s and run `ldd libsnapshot_redirect_cuda.so` / `libsnapshot_record_cuda.so | grep "not found"` **inside** the `vllm/vllm-openai` EDF. Gate: empty (no missing libs). If non-empty, the plan's fallback is to build the `.so`s against the vLLM image (note it in the log and stop).

- [ ] **Step 2: Record-launch smoke (gate) — vLLM serves under the interposer**

`_vllm_record_cuda.sh`: mirror `_vllm_coldstart_cuda.sh` but add the LD_PRELOAD + env above. One serve per `srun` step. Wait for the `Application startup complete`/ready log; send one `/v1/completions` request (greedy, fixed prompt); capture the tokens; `kill -9` the server; container restart frees the GPU. Gate: server reaches ready AND returns a non-empty completion AND each rank's `[redirect-cuda]` SUMMARY shows `fixed_base_honored=1` (interposer engaged, serving works in record mode).

- [ ] **Step 3: Run the gate**

`rcc --profile glm-47-flash-bristen-vllm push && rcc --profile glm-47-flash-bristen-vllm run 'sbatch snapshot/recipe/vllm_record_cuda.sbatch'`; poll; read log. Expected: ldd clean, ready reached, completion returned, redirect engaged per rank.

- [ ] **Step 4: Commit**
```bash
git add snapshot/recipe/_vllm_record_cuda.sh snapshot/recipe/vllm_record_cuda.sbatch snapshot/recipe/_vllm_restore_cuda.sh snapshot/recipe/vllm_restore_cuda.sbatch
git commit -m "snapshot(cuda): N5b Task 5 — cross-container .so check + vLLM record/restore recipe scaffold"
```

---

## Task 6: Record path under vLLM-CUDA TP=4 (G2)

**Files:**
- Modify: `snapshot/recipe/_vllm_record_cuda.sh`, `snapshot/recipe/vllm_record_cuda.sbatch` (full record gate)

**Interfaces:**
- Consumes: Task 5 scaffold; `VLLM_CG_RECORD_META`; the interposer record mode.
- Produces: a complete per-rank `.snap` set + `entry.output` meta JSON for every `CAPTURE_SIZES` entry on all 4 ranks.

- [ ] **Step 1: Add the record-completeness gate**

In the record sbatch, after the server reaches ready (capture done) and before `kill -9`: for each rank dir, assert `.snap` files exist and the record SUMMARY shows `blind=0` and `recorded graphs>=1`; assert each rank's `meta/rank${R}.json` exists with `entries` count == the captured-graph count for that rank (cross-check against the `cginst` `VLLM_CG_INSTRUMENT` capture count if enabled). Print `N5B_RECORD: ranks=4 blind=0 complete=1`.

- [ ] **Step 2: Run the gate (G2)**

Push+run. Expected: `N5B_RECORD: ranks=4 blind=0 complete=1`; per-rank `.snap` + meta present; recording covers `CAPTURE_SIZES`.

- [ ] **Step 3: Commit**
```bash
git add snapshot/recipe/_vllm_record_cuda.sh snapshot/recipe/vllm_record_cuda.sbatch
git commit -m "snapshot(cuda): N5b Task 6 — record path under vLLM-CUDA TP=4, per-rank complete (G2)"
```

---

## Task 7: Restore + FULL-rebuild decision (G3) + bit-identical serving (G4)

**Files:**
- Modify: `snapshot/recipe/_vllm_restore_cuda.sh`, `snapshot/recipe/vllm_restore_cuda.sbatch`
- Possibly modify: `snapshot/recipe/cginst_cuda/cg_meta_cuda.py` (FULL/PIECEWISE branch) and `snapshot/csrc/preload/snapshot_record_cuda.cpp` (only if a FULL-rebuild fault needs a fix)

**Interfaces:**
- Consumes: Task 6's recorded snaps+meta; the restore launcher; the interposer restore mode.
- Produces: token-identical TP=4 serving from restored graphs; the FULL-vs-PIECEWISE decision recorded.

- [ ] **Step 1: Restore gate — rebuild ALL graphs (attempt FULL)**

`_vllm_restore_cuda.sh`: restore mode, same snap+meta dirs. Gate: server reaches ready WITHOUT a real capture phase (restore SUMMARY per rank `restored>=1 fallthrough=0 blind=0`, redirect `passthrough=0 fixed_base_honored=1`), then send the SAME fixed prompts as the record run and assert **token-identical** completions across all 4 ranks vs a baseline (no-interposer) reference captured once. Print `N5B_RESTORE: full_rebuild=<1|0> token_identical=<1|0> fallthrough=0`.

- [ ] **Step 2: Run — FULL-rebuild de-risk (G3)**

Push+run. **If FULL graphs rebuild + serve token-identical** → `full_rebuild=1`, keep rebuild-both; record the evidence. **If a FULL-rebuild fault appears** (hang / wrong tokens isolated to FULL-mode graphs): diagnose at the node level (mirror the M3k method); if not quickly fixable, **switch to the fallback** — set the record run to PIECEWISE-only (skip FULL capture recording, `cg_skip`-style `record_pw`) and the restore to live-capture FULL (`shim_pw` shape: run the real forward for FULL, rebuild PIECEWISE from snapshot). Re-run until `token_identical=1 fallthrough=0`.

- [ ] **Step 3: Lock the strategy + bit-identical gate (G4)**

With the chosen strategy, confirm the final gate: `N5B_RESTORE: token_identical=1 fallthrough=0` on all 4 ranks, per-rank `passthrough=0 fixed_base_honored=1`. Record `full_rebuild=0|1` (which path was taken).

- [ ] **Step 4: Commit**
```bash
git add snapshot/recipe/_vllm_restore_cuda.sh snapshot/recipe/vllm_restore_cuda.sbatch snapshot/recipe/cginst_cuda/cg_meta_cuda.py snapshot/csrc/preload/snapshot_record_cuda.cpp
git commit -m "snapshot(cuda): N5b Task 7 — restore + FULL-rebuild decision (G3) + bit-identical TP=4 serving (G4)"
```

---

## Task 8: A/B/C cold-start + serving-overhead measurement (G5) + RESULTS + regression (G6)

**Files:**
- Create: `snapshot/recipe/_vllm_abc_cuda.sh`, `snapshot/recipe/vllm_abc_cuda.sbatch`
- Modify: `snapshot/RESULTS.md` (append §N5b)

**Interfaces:**
- Consumes: the record (A producer) + restore (B) launchers; the N3 eager path (C); `_vllm_measure_cuda.sh`.
- Produces: the A/B/C cold-start-to-ready table + focused serving-overhead numbers + §N5b RESULTS.

- [ ] **Step 1: A/B/C + serving-overhead driver**

`_vllm_abc_cuda.sh`: warm-weight-cache cold-start-to-ready wall-clock for **A** = baseline graph (no interposer, N3 path), **B** = restore (Task 7 strategy), **C** = eager (`--enforce-eager`/no-cudagraph). One serve per `srun` step. Then a focused serving check: throughput + p50/p99 at one concurrency, **B vs A**, on a fixed request set; assert B within a small tolerance of A (identical graphs → ~no overhead). Honest accounting per Global Constraints (wall-clock delta; don't double-count capture). Emit `N5B_ABC: A=<s> B=<s> C=<s> elim=<A-B>s overhead_pct=<x>`.

- [ ] **Step 2: Run the gate (G5)**

Push+run. Expected: the A/B/C table + overhead; B < A by ≈ the eliminated capture (minus live-FULL if the fallback path), B serving ≈ A.

- [ ] **Step 3: Regression check (G6)**
```bash
git diff --stat "$(git merge-base main HEAD)" HEAD -- \
  snapshot/csrc/backends/hip snapshot/csrc/preload/snapshot_record.cpp \
  snapshot/csrc/preload/snapshot_redirect.cpp snapshot/csrc/core \
  snapshot/csrc/backends/cuda snapshot/csrc/preload/snapshot_redirect_cuda.cpp
# expected: (empty) — N5b additive; only snapshot_record_cuda.cpp (CUDA) extended
```
And confirm the N5a CLI gate (`cuda_record_smoke.sbatch`) still `N5A_GATES: G1=PASS…` plus the new Task-1/2/3 gates.

- [ ] **Step 4: Append §N5b to RESULTS.md**

Document: the two-layer architecture, the FULL-rebuild outcome (rebuild-both or PIECEWISE+live-FULL, with evidence), the per-rank record/restore, the gates G1–G6, the A/B/C table + serving overhead (honestly framed — capture-phase elimination on a fast-capture engine, ~32s envelope), and a one-line "next" (full benchmaker sweep / other models deferred). Match the §N1–§N5a tone.

- [ ] **Step 5: Commit**
```bash
git add snapshot/recipe/_vllm_abc_cuda.sh snapshot/recipe/vllm_abc_cuda.sbatch snapshot/RESULTS.md
git commit -m "snapshot(cuda): N5b complete — vLLM-CUDA TP=4 record/restore, A/B/C measured, HIP additive (G5/G6)"
```

---

## Self-Review

**Spec coverage:** §2.1 C-interposer → Tasks 1–3; §2.2 Python `cg_meta_cuda` → Task 4 (+ vLLM hooks exercised in 6–7); §2.3 record→restore flow → Tasks 6–7; §3 FULL de-risk → Task 2 (CLI/3a) + Task 7 (vLLM/3b + branch); §4 TP=4 plumbing → Tasks 5–7 (per-rank, one-serve-per-srun, completeness); §5 measurement → Task 8; §6 cross-container → Task 5 Step 1; gates G1→T2, G2→T6, G3/G4→T7, G5/G6→T8. All covered.

**Placeholder scan:** code steps reference concrete existing files to port/extend (not vague "handle edge cases"); the one genuinely conditional step is Task 7 Step 2 (FULL works vs fault) — both branches are spelled out with the exact fallback (`record_pw`/`shim_pw` shape), which is a real decision gate, not a placeholder.

**Type/name consistency:** `snapshot_record_cuda_region_base()` (Task 3) is consumed by `cg_meta_cuda._region_base()` (Task 4) and the recipes (Task 5); `SNAPSHOT_RECORD_CUDA_DIR=…/rank%r` (Task 3) ↔ recipe per-rank dirs (Task 5); `VLLM_CG_RECORD_META`/`VLLM_CG_RESTORE_META` (Task 4) ↔ recipes (Tasks 5–7); SUMMARY fields (`rt_capture`, `blind`, `fallthrough`, `passthrough`, `fixed_base_honored`) consistent across the interposer and the gate greps. `.snap` format v2 (Task 2) read by restore (Task 7).
