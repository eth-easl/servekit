# Design — N5b: vLLM-CUDA TP=4 record/restore + cold-start measurement (bristen, A100)

Date: 2026-06-27
Status: approved design, pending implementation plan
Scope owner: Xiaozhe Yao
Parent design: `docs/superpowers/specs/2026-06-26-nvidia-cuda-snapshot-port-design.md` (milestone **N5**)
Predecessor: `docs/superpowers/specs/2026-06-27-cuda-record-restore-mechanism-n5a-design.md` (N5a — the CLI-validated record/restore mechanism this milestone integrates)

## 1. Goal & context

N5a built and CLI-validated `snapshot_record_cuda` — an LD_PRELOAD interposer that records captured CUDA graphs to `.snap` on cold start #1 and rebuilds them (skip real capture) on cold start #2, **bit-identical** under fixed-base Δ=0 (merged to main, `b477337`). **N5b wires that proven mechanism into the real engine**: vLLM-CUDA serving **GLM-4.7-Flash, TP=4** on bristen — record the CUDA graphs on cold start #1, restore them on cold start #2, serve **bit-identical** tokens, and **measure** the A/B/C cold-start win + serving overhead.

The honest framing (carried from N3): on CUDA the eliminable capture is modest — vLLM 0.23.0 graph cold start ≈ **110s**, eager ≈ **78s**, eliminable capture ≈ **32s** (~29%), of which the tqdm capture loop is ~23s. So N5b's value is **(a)** demonstrating the snapshot/restore mechanism end-to-end on a real CUDA engine (the research contribution — restore buys back most of the capture phase while keeping graph-mode serving speed), and **(b)** a measured, honestly-accounted A/B/C result — NOT a dramatic second count. The win is a *capture-phase* elimination, not a weight-load (20.4s, non-eliminable Lustre I/O) one.

### Decisions (settled with the user, 2026-06-27)

- **One milestone** — integration *and* measurement together (not decomposed).
- **Two-layer restore, Δ=0-simplified.** vLLM drives CUDA-graph capture through PyTorch's `torch.cuda.CUDAGraph` (the **runtime** capture APIs), wrapped in vLLM's `CUDAGraphWrapper`. Restore therefore requires two cooperating layers (neither sufficient alone):
  1. **C-interposer** (`snapshot_record_cuda.cpp`, extended) — rebuilds the CUDA graph from `.snap` and substitutes it at capture-end.
  2. **Python** (`cg_meta_cuda.py`, ported from the AMD `cg_meta.py`) — reconstructs `CUDAGraphWrapper.entry.output` (a Python `torch.Tensor` the C layer cannot create) from recorded metadata, and skips the model forward. **Δ=0 makes the CUDA port simpler than AMD's**: `live_base == snap_base` so there is **no pointer relocation** (the recorded `data_ptr` is directly valid), and the DLPack device type is `kDLCUDA` (2), not `kDLROCM` (10).
- **Test CUDA FULL-graph rebuild first (de-risk gate).** The AMD FULL-graph rebuild fault is unsolved (the M3k WIP) and was worked around with PIECEWISE-only snapshots + live-captured FULL. CUDA's graph/VMM APIs were more robust than HIP throughout N1–N5a, so N5b **tests CUDA FULL-graph rebuild directly** (CLI then under vLLM) and branches on the result — rebuild-both if clean, PIECEWISE-from-snapshot + live-capture-FULL fallback if it faults. A fault cannot stall the milestone.
- **Real-capture record.** The record run (cold start #1) is a *normal* graph-mode cold start with the interposer recording each graph; it runs once. No dummy-forward trickery on the record side (that is a restore-side concern).
- **Focused serving-overhead check** (not a full benchmaker sweep) — throughput/latency at one representative concurrency, restored vs baseline-graph, plus token-identical correctness on a fixed prompt set. The graphs are identical, so the expectation is ~zero overhead; the check confirms it.

### Target environment (bristen)

1× node, 4× A100-SXM4-80GB (`sm_80`), x86_64; SLURM `-A a-infra02`, partition `normal`; enroot/pyxis. Engine: vLLM 0.23.0 (`vllm/vllm-openai@sha256:6d8429e3…`), GLM-4.7-Flash **TP=4**, the N3 deploy (rcc profile `glm-47-flash-bristen-vllm` → `/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda`, EDF `deploy/glm-47-flash-bristen-vllm/glm-47-flash-vllm-cuda.toml`, recipes `snapshot/recipe/{_vllm_coldstart_cuda.sh,_vllm_measure_cuda.sh,vllm_coldstart_cuda.sbatch}`). The redirect+record `.so`s are `-static-libstdc++`/libcuda+cudart+dl-only (N2/N5a) for cross-container load.

## 2. Architecture

### 2.1 C-interposer extensions (`snapshot_record_cuda.cpp`)

Additive to the N5a interposer; the N5a CLI gates must stay green. New for N5b:

- **Runtime capture shims.** Interpose the **runtime** `cudaStreamBeginCapture` / `cudaStreamEndCapture` / `cudaStreamIsCapturing` (+ `cudaStreamGetCaptureInfo` as vLLM/PyTorch require), mirroring the N5a driver-level shims. **Dedupe by `CUgraph`**: if the runtime internally delegates to the (already-interposed) driver `cuStreamEndCapture`, walk/record/restore each graph exactly once.
- **Non-kernel graph nodes.** FULL/PIECEWISE graphs contain memcpy/memset (and possibly child/event) nodes. Record them structurally (type + params) and rebuild them, or — for node types not yet supported — **blind-mark with an explicit reason** (no silent edge-drop; the N5a `rec_index` kernel-only miss is fixed here). G2 requires `blind=0` for the chosen strategy's graphs.
- **`extra`-buffer kernargs.** Handle `CU_LAUNCH_PARAM_BUFFER_POINTER` launches (vLLM/Triton may use them) in record + replay, not just the `kernelParams` array.
- **Per-rank snapshot dirs.** `SNAPSHOT_RECORD_CUDA_DIR` resolves per worker rank (e.g. `…/rank${RANK}`), so each of the 4 TP workers records/restores its own `.snap` set without collision.
- **Exports for the Python layer.** `snapshot_record_cuda_region_base()` (the fixed VMM base, for `entry.output` reconstruction) and a restore-queue/SUMMARY hook so Python can confirm engagement.
- **Scale hardening** (N5a carry-forwards now load-bearing): O(N) reverse-resolution scans → prebuilt maps; widen/dynamic kernel-name buffer (long mangled Triton/template names); bound process-lifetime growth of rebuilt graphs/NodeConfig (hundreds of graphs × thousands of nodes); add `cuModuleUnload` eviction + interpose `cuModuleLoadDataEx`.

### 2.2 Python layer (`cg_meta_cuda.py`)

Ported from `snapshot/recipe/cginst_skip/cg_meta.py`, loaded via a `sitecustomize.py` (the N3 `cginst` mechanism). Two env-selected modes:

- **RECORD** (`VLLM_CG_RECORD_META=<per-rank path>`): hook `CUDAGraphWrapper.__call__`; on each *real* capture (mode matches the wrapper's `runtime_mode` and `entry.cudagraph is None`), serialize `entry.output` to JSON — `{kind, offset = data_ptr - region_base, shape, dtype}` recursively over tensors/lists/dicts. **Incremental flush per capture** (SIGKILL-safe — the harness kills the server with `kill -9`).
- **RESTORE** (`VLLM_CG_RESTORE_META=<per-rank path>`): hook `CUDAGraphWrapper.__call__`; dispatch exactly like the original but override the CAPTURE branch — run an empty `with torch.cuda.graph(cudagraph, pool=…, stream=…): pass` (the C-interposer fakes begin/end and attaches the next pre-built graph at EndCapture), **skip the forward**, and reconstruct `entry.output` via a hand-built DLPack capsule (`kDLCUDA`=2) pointing at `region_base + offset` (Δ=0 → equals the recorded `data_ptr`). Set `entry.cudagraph`/`entry.output`. Also patch `GPUModelRunner._warmup_and_capture` to skip warmup `_dummy_run`s (only the capture path reaches the wrapper).

**vLLM 0.23.0 repair.** The AMD prototypes drifted on 0.23.0 (capture moved into `_warmup_and_capture`/`_capture_cudagraphs`; `CUDAGraphWrapper`/`concrete_cudagraph_entries`/`forward_context` shapes). N5b re-binds against the live 0.23.0 symbols (the same surface the N3 `cginst` instrumentation already patches successfully).

### 2.3 The record → restore flow (per rank)

- **Cold start #1 (record):** `LD_PRELOAD=redirect_cuda:record_cuda`, `SNAPSHOT_RECORD_CUDA_MODE=record`, `VLLM_CG_RECORD_META=…/rank${RANK}.json`. vLLM captures normally; the C-interposer writes `…/rank${RANK}/graph-*.snap`; Python writes the `entry.output` JSON. **Completeness:** every `CAPTURE_SIZES` entry must produce a `.snap`+meta.
- **Cold start #2 (restore):** same preload, `MODE=restore`, `VLLM_CG_RESTORE_META=…/rank${RANK}.json`. Capture is skipped; graphs are rebuilt + output reconstructed; serving proceeds on the rebuilt graphs. **Gate:** `fallthrough=0` (every graph served from snapshot), `blind=0`, bit-identical tokens.

## 3. The FULL-graph de-risk gate ("test FULL first")

A proven fallback guarantees completion regardless of the FULL result.

- **3a — CLI (cheap, isolates the C-interposer).** Extend the smoke (or add `cuda_record_full_smoke.cpp`) to a FULL-like graph: many kernel nodes **+ non-kernel nodes (memcpy/memset) + deep/branching deps**. Verify record→restore rebuilds it bit-identical. Catches the known N5a gaps (non-kernel-dep edge-drop, name truncation) with fast iteration before vLLM.
- **3b — under vLLM (definitive).** Record the real graphs, then attempt to **rebuild/restore the FULL graphs**. Gate: rebuild without the AMD M3k fault **and** serve bit-identical. (The real FULL decode graph may have structure a synthetic CLI graph doesn't, so 3b is the authority; 3a is the fast early signal.)
- **Branch:** FULL rebuilds cleanly → **rebuild both FULL + PIECEWISE** from snapshot (maximum elimination; proves what AMD couldn't). FULL faults → **fall back to PIECEWISE-from-snapshot + live-capture-FULL** (the AMD `record_pw`/`shim_pw` shape; FULL is cheap to live-capture on CUDA). The chosen strategy is recorded in RESULTS with the FULL-rebuild evidence either way.

## 4. TP=4 + multi-graph plumbing

- **4 worker processes**, each redirect-pinned (Δ=0 per rank — verify each rank's `passthrough=0`/`fixed_base_honored=1`), each with its own per-rank `.snap` dir + `entry.output` JSON.
- **Recording completeness:** record run captures every `CAPTURE_SIZES` entry on every rank; restore asserts a matching `.snap`+meta per (rank, size); non-zero `fallthrough` = incomplete recording (a signal, with graceful live-capture fallback).
- **One serve per `srun` step** (enroot holds CUDA IPC shmem at the container level — the N3 lesson; multiple serves in one step leave the GPU stuck). The image has `python3` only (no `python`); `nvidia-smi` is absent.

## 5. Measurement (the deliverable)

- **Cold start A/B/C** (warm weight cache, to isolate capture): **A** = baseline graph (~110s, N3) · **B** = restore (skip capture) · **C** = eager (~78s, N3). The pitch: **B = graph-mode serving speed at a cold start near eager's** — restore reclaims most of the eliminable ~32s (minus the live-captured FULL portion, if the fallback is taken) while keeping fast graph-mode inference (which C sacrifices). Account honestly: `init engine … took` already *includes* capture (do not double-count); per-graph instrument sums are 4-worker aggregates, not wall-clock — use the wall-clock cold-start-to-ready delta.
- **Serving overhead (focused):** benchmaker throughput + p50/p99 latency at one representative concurrency, **restored vs baseline-graph** — expected ~equal (identical graphs).
- **Correctness:** restored serving emits **token-identical** output to baseline on a fixed prompt set (greedy/seeded), across all 4 ranks.

## 6. Cross-container `.so`

Before loading the interposers under vLLM: build the redirect+record `.so`s for the target image and run `ldd …redirect_cuda.so` / `…record_cuda.so | grep "not found"` **inside** the `vllm/vllm-openai` image (glibc check — N3 baseline ran clean with no interposer, so this is not yet exercised there). If the image glibc is older than the build container's, build the `.so`s in/against the image.

## 7. Gates (acceptance — bristen A100×4)

| # | Gate |
|---|---|
| **G1** | CLI FULL-like graph (many kernel + non-kernel nodes + deep deps) records → restores **bit-identical** (extends the N5a CLI gate). |
| **G2** | Record path under vLLM-CUDA TP=4 captures **every** graph on **every** rank — per-rank `.snap` + `entry.output` JSON written, `blind=0` for the chosen strategy's graphs, recording complete over `CAPTURE_SIZES`. |
| **G3** | **FULL-rebuild de-risk resolved with data** — CUDA FULL-graph rebuild either works (→ rebuild-both) or the PIECEWISE+live-FULL fallback is taken; the decision + evidence are in RESULTS. |
| **G4** | Restore serves **token-identical** to baseline across all 4 ranks; `fallthrough=0`, `passthrough=0`/`fixed_base_honored=1` per rank (Δ=0 held under vLLM). |
| **G5** | Measured **A/B/C cold-start** (honestly accounted) + **serving overhead** (restored vs baseline-graph) reported in RESULTS. |
| **G6** | **Regression / additive:** HIP (`backends/hip/*`, `snapshot_record.cpp`, `snapshot_redirect.cpp`), all `core/`, and N1–N5a CUDA files except the additively-extended `snapshot_record_cuda.cpp` are **byte-unchanged**; new files only (`cg_meta_cuda.py`, the FULL smoke, recipes). N5a CLI gates still green. |

Each gate is a bristen cluster job (`rcc --profile … push` → `sbatch -A a-infra02`).

## 8. Files (new / changed)

New:
- `snapshot/recipe/cginst_cuda/cg_meta_cuda.py` + `sitecustomize.py` — the CUDA Python record/restore layer (ported from `cginst_skip/cg_meta.py`).
- `snapshot/csrc/cli/cuda_record_full_smoke.cpp` (or an extension of `cuda_record_smoke.cpp`) — the FULL-like CLI rebuild test (G1).
- `snapshot/recipe/vllm_record_cuda.sbatch` + `vllm_restore_cuda.sbatch` (+ `_*.sh`) — the TP=4 record / restore + measurement drivers.
- RESULTS §N5b.

Changed (additive):
- `snapshot/csrc/preload/snapshot_record_cuda.cpp` — runtime capture shims, non-kernel nodes, `extra` kernargs, per-rank dirs, `region_base` export, scale hardening. (`record_cuda_format.hpp` extended only if new node types need new record fields — versioned.)
- `snapshot/CMakeLists.txt` — additive only.

Untouched (regression invariant — G6): all HIP `backends/hip/*`, `snapshot_record.cpp`, `snapshot_redirect.cpp`, `core/*`, N1 `backends/cuda/*`, N2 `snapshot_redirect_cuda.cpp`, and the existing HIP `cginst_skip/*` prototypes (the CUDA port is a sibling, not an edit).

## 9. Scope

**In:** the C-interposer extensions for real vLLM graphs; the CUDA Python `cg_meta` layer; the FULL-graph de-risk gate; TP=4 per-rank record/restore; bit-identical serving; the A/B/C cold-start + focused serving-overhead measurement.

**Out (later):**
- The cubin `.nv.info` parser + Δ≠0 relocation (Δ=0 makes verbatim replay valid — deferred unless a Δ≠0 case appears).
- A full serving-benchmaker sweep across concurrencies/sequence lengths.
- SGLang; multi-node; other models/quantizations.
- The A2-style direct graph archive/replay optimization.

## 10. Risks & mitigations

1. **vLLM 0.23.0 capture-path drift** (the AMD prototypes broke here). *Mitigation:* re-bind against the live 0.23.0 symbols — the same surface the N3 `cginst` instrumentation already patches; the restore shim is at the driver/runtime level (below the Python loop), robust to internal reshuffles.
2. **FULL-graph rebuild fault on CUDA** (unknown; unsolved on AMD). *Mitigation:* §3 de-risk gate with the PIECEWISE+live-FULL fallback — completion is guaranteed either way.
3. **Runtime↔driver capture double-walk** (record/restore a graph twice). *Mitigation:* dedupe by `CUgraph` handle in the interposer.
4. **`entry.output` reconstruction fidelity** — a wrong shape/dtype/offset yields wrong tokens. *Mitigation:* G4 token-identical gate; Δ=0 removes relocation error; incremental-flush record is SIGKILL-safe.
5. **Per-rank Δ=0 under TP=4** — an undersized redirect region on any rank → `passthrough>0` → stale pointers. *Mitigation:* per-rank `passthrough=0` assertion (G4); size the region for weights+KV+capture+comms per GPU.
6. **Cross-container glibc** (the N3/N5a carry). *Mitigation:* §6 `ldd … | grep "not found"` inside the vLLM image before load.
7. **Scale** (hundreds of graphs × thousands of nodes vs N5a's one). *Mitigation:* §2.1 hardening (prebuilt maps, bounded lifetime, eviction) before the vLLM run.

## 11. Out of scope (this design)

Everything in the parent design's post-N5 layer; the cubin parser / Δ≠0 path; multi-node TP; SGLang; non-GLM models.
