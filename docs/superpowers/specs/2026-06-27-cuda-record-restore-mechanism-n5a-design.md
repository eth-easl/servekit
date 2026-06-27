# Design — N5a: `record_cuda` snapshot/restore mechanism, CLI-validated (bristen, A100)

Date: 2026-06-27
Status: approved design, pending implementation plan
Scope owner: Xiaozhe Yao
Parent design: `docs/superpowers/specs/2026-06-26-nvidia-cuda-snapshot-port-design.md` (§5 `snapshot_record_cuda.cpp`, §6 3c, milestone **N5**)

## 1. Goal & context

The `snapshot` NVIDIA port has a working CUDA backend (**N1**), a deterministic
fixed-base `redirect_cuda` interposer (**N2**), and a characterized vLLM-CUDA
baseline cold start (**N3**: warm graph 110s, eager 78s, eliminable capture
~32s). N5 (snapshot/restore — the real mechanism that skips capture *and* keeps
graph-mode speed) was **decomposed** into:

- **N5a (this design):** build the `record_cuda` interposer (identity + capture
  introspection + restore-shim) and prove the **record → restore** mechanism on
  the controllable CLI synthetic workload — record captured CUDA graphs on cold
  start #1, restore them (rebuild + skip capture) on cold start #2, **bit-identical**
  output, under fixed-base Δ=0.
- **N5b (later):** wire the proven interposer into vLLM-CUDA TP=4 (FULL+PIECEWISE
  graphs across 4 worker processes), record on run #1 / restore on run #2, and
  measure the A/B/C cold-start win + serving overhead.

N5a mirrors the proven AMD M1 arc (capture/restore bit-identical on the CLI
before any engine integration) and the N1/N2 discipline (CLI gates on A100,
dual-vendor additive, HIP untouched).

### Decisions (settled with the user, 2026-06-27)

- **Verbatim kernarg replay, no cubin parser.** Under fixed-base Δ=0 (N2-proven),
  recorded kernarg device-pointers are *still valid* at restore (addresses are
  byte-identical across cold starts), so relocation is a no-op. N5a records the
  kernarg blobs + graph structure and **replays them verbatim** — no cubin
  `.nv.info` (`EIATTR_KPARAM_INFO`) parser, no pointer identification, no Δ≠0
  relocation machinery. `cuFuncGetParamInfo` (already used in N1) supplies only
  the blob *size*. The gate becomes the stronger end-to-end **bit-identical
  restored output** + an asserted `passthrough=0`/Δ=0, not AMD-style pointer
  counting. The cubin parser is deferred to N5b *only if* a Triton kernel ever
  needs it.
- **Full identity machinery, vLLM-ready now.** N5a builds the complete identity
  path — `__cudaRegisterFunction` (host-fn → device-symbol) + `__cudaRegisterFatBinary`
  (cubin registry) + `cuModuleLoadData`/`cuModuleGetFunction` (nvrtc/explicit
  modules) + `cuFuncGetName` — so the interposer is ready for N5b's vLLM kernels.
  To avoid shipping the fatbin path untested, the CLI smoke exercises **both** a
  static `nvcc`-compiled kernel (→ `__cudaRegisterFunction`/fatbin) and an nvrtc +
  `cuModuleLoadData` kernel.
- **Separate `record_cuda` record format; `core/` byte-untouched.**
  The dual-vendor invariant forbids refactoring the proven HIP code, and the N2
  regression gate asserts the whole `snapshot/csrc/core` directory is unchanged.
  The CUDA verbatim-replay format (graph structure + kernarg blobs + function
  identity) is simpler than AMD's msgpack-signature format, so there is little to
  reuse — N5a's record/restore serialization lives **alongside the interposer**
  (inline in `snapshot_record_cuda.cpp` or a `snapshot/csrc/preload/record_cuda_format.{hpp,cpp}`
  sibling), **never under `core/`**, so `core/` stays clean for the regression check.

### Target environment (bristen)

1× node, 4× A100-SXM4-80GB (`sm_80`), x86_64; SLURM `-A a-infra02`, partition
`normal`; enroot/pyxis. N5a is single-process (the CLI smoke), not TP — the
4-process concern is N5b. The interposer `.so` is built static-libstdc++ /
libcuda-only (the N2 cross-container recipe) so N5b can later load it into the
vLLM-CUDA image. Code sync via `rcc --profile bristen-snapshot push` (the
existing N1/N2 snapshot profile → `/capstor/scratch/cscs/xyao/snapshot-cuda`).

## 2. The interposer — `snapshot_record_cuda.cpp`

LD_PRELOAD shim, a parallel sibling of `snapshot_redirect_cuda.cpp` (the HIP
`snapshot_record.cpp` is never touched). It **composes with `redirect_cuda`**:
redirect provides fixed-base Δ=0 device addresses; record/restore captures and
rebuilds the graphs at those addresses. Two modes (env-selected, like the AMD
record interposer): `record` (capture #1 → `.snap`) and `restore` (`.snap` →
rebuild + shim, skip capture).

### 2.1 Eager interposer gate (the M3i +271s lesson)

The heavy import-path hooks — `__cudaRegisterFatBinary`, `__cudaRegisterFunction`,
`cuModuleLoadData` — must **never scan fatbins on the hot import path**. They
record lightweight identity (pointers, names, handles) only, and any expensive
work (symbol enumeration) is deferred to the first capture window. In `restore`
mode the registration hooks can stay fully passive (identity is resolved lazily
at rebuild time). The CLI smoke asserts process-startup overhead stays small
(the N2 `<10s` style gate), guarding against an M3i-style regression early.

### 2.2 Identity (full machinery)

Builds a `recorded-id → CUfunction` map across both kernel-provenance paths:

- **Statically-linked CUDA (fatbin):** `__cudaRegisterFatBinary` registers each
  cubin blob; `__cudaRegisterFunction` maps the host stub pointer → device symbol
  name. At record, a kernel node's function resolves to a `(fatbin-id, symbol)`
  identity; at restore, the same registrations (deterministic in a re-run)
  re-establish the map, and `cuFuncGetName` confirms the name.
- **nvrtc / explicitly-loaded modules:** `cuModuleLoadData`/`cuModuleGetFunction`
  hooks capture `(module-content-hash, function-name) → CUfunction`. The CLI
  workload's nvrtc PTX recompiles deterministically, so the same content-hash +
  name resolves at restore.

Identity is recorded **by name/symbol** (+ a content hash for module
disambiguation), never by raw pointer (pointers differ across processes; names
do not). A node whose function cannot be resolved at restore is a **blind**
node — G4 requires zero.

### 2.3 Capture / record

Hook `cuStreamBeginCapture`/`cuStreamEndCapture` (+ the `cudaStream*` variants).
At `EndCapture`, before returning the graph to the caller, walk it:

- `cuGraphGetNodes` → for each `CU_GRAPH_NODE_TYPE_KERNEL` node,
  `cuGraphKernelNodeGetParams` → `{func, gridDim, blockDim, sharedMem, kernelParams}`.
- The kernarg **blob** = the `kernelParams` bytes, total size from
  `cuFuncGetParamInfo` (sum of param offset+size). Recorded **verbatim** (the
  device pointers inside it are valid as-is under Δ=0).
- Node **structure**: node type, dependency edges (`cuGraphNodeGetDependencies`),
  and the launch config. (N5a targets kernel nodes; memcpy/memset/host nodes, if
  the smoke produces any, are recorded structurally too — but the smoke is kept
  kernel-only to stay focused.)
- The node's **function identity** (§2.2).

Persist a `.snap` per graph via the `record_cuda` format. Recording happens once
per graph at the capture window; the real graph is still returned to the caller
so run #1 executes normally.

### 2.4 Restore (restore-shim)

In `restore` mode, at process start the recorded `.snap` graphs are loaded into a
restore queue. The capture APIs are shimmed:

- `cuStreamBeginCapture` → mark the stream "capturing" but do **not** begin a real
  capture.
- `cuStreamIsCapturing` → report capturing (so the caller's logic proceeds).
- `cuStreamEndCapture` → pop the next `.snap` from the queue, **rebuild** it via
  `cuGraphAddKernelNode` (resolving each node's function via §2.2 identity,
  replaying the recorded kernarg blob **verbatim** — Δ=0, no relocation, no
  per-node patching), and return the rebuilt `CUgraph`. Real capture is skipped.
- **Fall-through:** if the restore queue is exhausted (more captures than
  recorded), fall back to real capture (so a partial recording degrades
  gracefully rather than crashing) — and log it (a non-empty fall-through count
  is a recording-completeness signal for N5b).

## 3. CLI validation — `cuda_record_smoke.cpp`

A clean CUDA sibling of `cuda_redirect_smoke.cpp` (the HIP `redirect_smoke.cpp`
demo is left untouched per the regression invariant). It:

1. Allocates device buffers (under `redirect_cuda` → fixed-base addresses).
2. Defines **two** kernels to cover both identity paths:
   - a **static `nvcc`-compiled** `__global__` kernel (→ `__cudaRegisterFunction`/
     `__cudaRegisterFatBinary`),
   - an **nvrtc**-compiled kernel loaded via `cuModuleLoadData`/`cuModuleGetFunction`.
3. Captures a CUDA graph that launches both kernels (with known input → known
   output), instantiates, and launches it.
4. Prints a deterministic checksum of the output buffers.

Driven by an sbatch (`cuda_record_smoke.sbatch`) that runs it **twice** under
`LD_PRELOAD=redirect_cuda:record_cuda`:
- **Run #1 — `record`:** capture the graph, write `.snap`(s), print checksum.
- **Run #2 — `restore`:** load `.snap`(s), rebuild + shim (skip capture), launch,
  print checksum.

## 4. Gates (acceptance — bristen A100)

| # | Gate |
|---|---|
| **G1** | Run #2 (restore) output checksum **bit-identical** to run #1 (record). |
| **G2** | Capture genuinely **skipped** in run #2: the restore-shim served every graph; real `cuStreamBeginCapture`-driven capture count = 0 (restore queue not exhausted, fall-through = 0). |
| **G3** | Δ=0 / `passthrough=0` asserted (the redirect SUMMARY line; fixed-base held, so verbatim replay is valid — the "fundamental wall" closed). |
| **G4** | Identity resolved for **both** the static and the nvrtc functions — 0 blind/unresolved nodes at restore. |
| **G5** | **Regression:** HIP build still green; the existing HIP files (`backends/hip/*`, `preload/snapshot_record.cpp`, `preload/snapshot_redirect.cpp`), all of `core/`, and the N1/N2 CUDA files (`backends/cuda/*`, `preload/snapshot_redirect_cuda.cpp`) are **byte-unchanged** (verified by `git diff` on those paths). N5a is purely additive: `snapshot_record_cuda.cpp`, the `record_cuda` format, `cuda_record_smoke.cpp`, the CMake CUDA-record branch, the recipe. |

Each gate is a bristen cluster job (`rcc --profile bristen-snapshot push` →
`sbatch -A a-infra02`).

## 5. Files (new / changed)

New:
- `snapshot/csrc/preload/snapshot_record_cuda.cpp` — the record/restore interposer.
- `snapshot/csrc/cli/cuda_record_smoke.cpp` — the two-kernel CLI smoke target.
- the `record_cuda` `.snap` (de)serialization — inline in `snapshot_record_cuda.cpp`
  or a `snapshot/csrc/preload/record_cuda_format.{hpp,cpp}` sibling (NOT under `core/`).
- `snapshot/recipe/cuda_record_smoke.sbatch` — the record→restore driver.
- CMake CUDA-record branch (additive, mirroring the N2 `redirect_cuda` branch).
- tests as warranted (e.g. a host-side `.snap` round-trip unit test).

Untouched (regression invariant): all HIP `backends/hip/*`, `snapshot_record.cpp`,
`snapshot_redirect.cpp`, `core/record.cpp`, `core/relocation.cpp`, N1
`backends/cuda/*`, N2 `snapshot_redirect_cuda.cpp`, and the existing CLI/demo.

## 6. Scope

**In:** the `record_cuda` interposer (eager gate + full identity + verbatim
capture/record + restore-shim), the two-kernel CLI smoke, the `record_cuda`
format, record→restore gates on A100.

**Out (N5b / later):**
- vLLM-CUDA TP=4 integration: FULL+PIECEWISE graph modes, 4 worker processes,
  recording drain with the `CAPTURE_SIZES` clamp, the A/B/C cold-start win, and
  serving-overhead (benchmaker) measurement.
- The cubin `.nv.info` parser + Δ≠0 relocation (deferred per the Δ=0 decision).
- Non-kernel graph nodes beyond what the smoke needs; `cudaMallocAsync` aliasing.

## 7. Risks & mitigations

1. **Identity for static kernels under re-run determinism** — `__cudaRegisterFunction`
   ordering must be stable across the two runs. Mitigation: identify by symbol
   name (+ fatbin content hash), not registration order; G4 asserts resolution.
2. **`cuStreamEndCapture` shim correctness** — returning a rebuilt graph in place
   of a real capture must satisfy the caller's expectations (handle, status).
   Mitigation: the smoke is the controlled first proof; the shim mirrors the
   proven AMD M3g/M3i restore-shim semantics.
3. **Verbatim replay depends on Δ=0** — an undersized redirect region → passthrough
   → driver-chosen VA → Δ≠0 → stale pointers. Mitigation: G3 asserts
   `passthrough=0`; the smoke's allocations are small and fully covered.
4. **Eager-gate regression (M3i +271s)** — Mitigation: registration hooks record
   only lightweight identity; a startup-overhead check in the smoke.
5. **Cross-container `.so` (for N5b)** — build static-libstdc++/libcuda-only (N2
   recipe); re-check `ldd … | grep "not found"` inside the vLLM-CUDA image before
   N5b loads it. Not exercised in N5a (CLI runs in the dev/N1 container).

## 8. Out of scope (this design)

Everything in the parent design's N5b/measurement layer; the A2-style direct
graph archive/replay optimization; multi-node TP; SGLang.
