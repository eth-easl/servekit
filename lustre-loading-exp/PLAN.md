# Plan: Lustre-aware model loading — side experiments (Llama-3.1-70B)

## Context

Cold-start weight loading is a top bottleneck: the real Apertus-8B profile
(`logs/apertus-8b-sglang-73540-profile.json`) shows `weight_loading` = 74 s of
182 s total (~40%). For the 70B target the phase will be larger still. Models
currently load straight from Lustre with a naive layout, so there is real
headroom in (a) Lustre striping, (b) read concurrency, (c) avoiding mmap,
(d) staging to node-local RAM.

This plan is **pure experimentation, on the side**. It adds **nothing to
servekit** and modifies no package code. The deliverable is a set of scratch
scripts + a measured comparison of two strategies (striped Lustre copy vs
`/dev/shm` staging) against the current baseline, plus a written
recommendation that will *later* inform a servekit feature (out of scope here).

## Environment (verified)

- **Model**: `/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct`
  — ~132 GB across **30 `.safetensors` shards**, capstor is **Lustre**.
- **Layout today**: every shard `stripe_count=1, stripe_size=1M`; shards are
  independently placed (shard 1 → OST 117), so they already scatter across
  ~30 OSTs. **Concurrent multi-shard reads already exploit many OSTs at
  `stripe_count=1`** — the experiments must isolate what per-file striping adds
  *on top of* shard-level concurrency, not assume striping alone is the win.
- **iopsstor** scratch: Lustre, 20 OSTs, default dir layout `stripe_count=1`.
- **Node**: bristen A100/sm_80, 4 GPU, TP=4, 32 CPU, `--exclusive`.
- **Serving**: `srun --environment=<toml>` container (`lmsysorg/sglang:v0.5.10`),
  mounts `/capstor` + `/iopsstor`, `HOME=/root` (ephemeral → cold HF/Triton
  cache, but **NOT** cold OS page cache).
- Reference launch: `profile/llama-3.1-70b-bristen/serve_llama70b_sglang.sbatch`
  (calls `servekit profile -- python -m sglang.launch_server --model-path $MODEL
  --tensor-parallel-size 4 ...`).

## Working location

New experiments dir **inside the repo but outside the package**:
`/iopsstor/scratch/cscs/yboughizane/simple-serving-stack/lustre-loading-exp/`
containing `scripts/`, `results/` (fio JSON, servekit profile JSON), `NOTES.md`
(running log + findings). Nothing under `servekit/` is touched.

## Experimental principles

- **Isolate storage from the engine first** with a raw read benchmark before
  touching SGLang — a `layout × concurrency` sweep answers the physical questions
  in minutes instead of full server restarts. Tool: `dd iflag=direct` (fio is
  unavailable on these nodes).
- **Constant workload = the whole model.** Every concurrency level reads all 30
  shards (the full ~132 GB), not a subset — a pool of N parallel workers drains
  the shard list (`xargs -P N`). So `wall_s` is directly "time to load the whole
  model at parallelism N" and rows are comparable (same bytes), and N can exceed
  the shard count to exercise hot OSTs.
- **Cold reads / page cache — HARD RULE**: OS page cache is kernel-level and
  survives across container runs on the same node, so a reused node silently
  serves later reads from RAM and inflates results. We cannot `drop_caches`
  (needs root). Therefore:
  - **Raw storage numbers** use `dd iflag=direct` (O_DIRECT) — bypasses page
    cache on every read, so different concurrency levels within one run are each
    genuinely cold and can share a node safely. (fio unavailable on these nodes.)
  - **Every end-to-end / buffered cold-start measurement runs on its own fresh
    `--exclusive` allocation — one node per run, never reused across runs.** No
    two cold-start data points share a node. Baseline and each treatment get
    separate allocations; repeat on ≥2 distinct nodes to confirm.
  - Record the node id (`hostname`/`nid`) with every result so cache reuse is
    auditable after the fact.
- **mmap is an explicit test axis, not a fixed choice**: every end-to-end
  strategy is run **both with mmap (default) and with `--weight-loader-disable-mmap`**,
  so we measure the mmap effect on each layout/storage rather than assuming
  no-mmap wins. (fio can't exercise mmap directly; the mmap comparison lives in
  the end-to-end SGLang runs.)
- **Fair comparison**: hold TP, context-length, mem-fraction constant except for
  the one variable under test (layout, concurrency, mmap, or storage location).
  Record engine-reported `weight_loading` **and** total wall-clock (measured
  differently).

## Phase 1 — Baseline + storage characterization

1. **Shard→OST map** of the model as-is: `lfs getstripe` over all 30 shards →
   confirm the spread (how many distinct OSTs, any hot OST with multiple shards).
2. **Raw ceiling** with `dd iflag=direct` (O_DIRECT, page cache bypassed):
   constant full-model workload, worker-pool size N ∈ {1,4,8,16,30,60}, large
   block size (16M) → full-model load time + effective GB/s vs concurrency on the
   native layout (`scripts/dd_read_sweep.sh`). Establishes the achievable ceiling
   before any restriping.
3. **End-to-end baseline**: run the existing 70B sbatch on the native layout,
   capture `weight_loading` + total from the servekit profile JSON — **both with
   mmap (default) and with `--weight-loader-disable-mmap`** so we have the mmap
   effect at the baseline layout too. These are the numbers every treatment is
   compared against.

## Phase 2 — Layout sweep (does striping add anything?)

Create restriped copies on iopsstor scratch and re-run the Phase-1 `fio` sweep
on each:
- `lfs setstripe -c {4,8,16} -S {1M,4M,16M} <dir>` then copy shards in (layout is
  set on the target dir *before* copy; cannot restripe in place without
  `lfs migrate`).
- Compare GB/s at matched concurrency: native-30-shard vs striped. Expected
  learning: striping helps most at low reader counts / when shards cluster on
  few OSTs; may add little once concurrency already ≥ shard count. Pick the best
  layout(s) to carry forward.

## Phase 3 — Strategy A: striped Lustre copy + loader flags (end-to-end)

- Point `--model-path` at the best striped copy from Phase 2.
- Enable in-tree parallel loader:
  `--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": N}'`
  (PR sgl #7277 — **verify present in v0.5.10**).
- **Run each layout twice: with mmap (default) and with `--weight-loader-disable-mmap`**
  (maps to `server_args.weight_loader_disable_mmap`, confirmed in vendored
  `sglang/python/sglang/srt/server_args.py`) → isolates the mmap effect on a
  striped layout.
- Sweep `num_threads` around the Phase-2 sweet spot. Also test
  `--load-format fastsafetensors` if available (shard-aware, per-TP-rank reads).
- Measure `weight_loading` + total vs baseline for each {layout, mmap} cell.

## Phase 4 — Strategy B: `/dev/shm` parallel staging (end-to-end)

- Standalone parallel copy script (e.g. GNU `parallel`/xargs of `cp`, or `dd`
  with O_DIRECT read) capstor → `/dev/shm/llama70b`, timed independently.
- Launch SGLang with `--model-path /dev/shm/llama70b`, **both with mmap and with
  `--weight-loader-disable-mmap`** (mmap over tmpfs has no page-fault-from-disk
  cost, so this quantifies whether mmap-vs-explicit-read still matters once the
  bytes are already in RAM).
- **Risk to size first**: check compute-node `/dev/shm` capacity — 132 GB must
  fit alongside the SGLang process RAM on an A100 node. If it doesn't fit,
  fall back to staging a subset / document the RAM ceiling as a finding.
- Measure staging time + `weight_loading` + total. Note the
  **overlap opportunity**: staging can run concurrently with import/CUDA-init;
  record both serial and (if scripted) overlapped timings.

## Phase 7 — Hide the stage: overlap staging with SGLang startup

Phase 6 left `stage 11.89 s + weight_loading 9.64 s` back-to-back because the
stage ran to completion before `launch_server` was even exec'd. The stage is a
host-side file copy with no dependency on GPU/torch state, and ~42 s of
`process_startup` + `tp_worker_spawn` + `torch_distributed_init` runs before the
loader opens a weight file — so it should fit entirely inside that window.

`scripts/phase7_overlap_stage/phase7_overlap_stage.sbatch`, one arm, 64 CPUs
(matching phase 6, so overlap is the only variable vs job 76435):

- Non-safetensors files (config/tokenizer, read within the first seconds of
  startup) copied **synchronously** first. `stage_to_shm_sliced.sh` grew a
  `FILE_PATTERN` env (default `*`, no behaviour change) so only the 28
  `*.safetensors` (131 GB) go into the background stage.
- Stage backgrounded, `launch_server` started immediately — **no barrier, no
  engine patch**. This measures the ceiling, not a shippable design.
- **Post-hoc validity gate** instead of synchronization: the stage's end epoch
  must precede the absolute time `weight_loading` began (`profile.started_at` +
  durations of preceding phases). Negative slack ⇒ the run read partially
  staged bytes and is discarded, not interpreted.
- Headline metric is `ready_at − stage_start`, compared against phase 6's
  192.59 s. Serving correctness (64/64, 0 errors) remains the gate.
- Expect the stage itself to take **longer** than 11.89 s: Phase 4b showed it is
  CPU-bound, and it now contends with `process_startup`/`tp_worker_spawn` for the
  same 64 CPUs. That interference is a result, not noise.

Production version (sentinel + atomic rename + a loader-side wait) is the
follow-up at the end of this file, only worth building if this arm pays.

## Phase 5 — Comparison + recommendation

Table in `NOTES.md`: baseline vs A vs B, **each split by mmap on/off**, on
{raw GB/s, staging cost, `weight_loading` s, total cold-start s, RAM cost,
persistence, engine-flag coupling, plug-and-play fit}. Written recommendation on
which to package into servekit later, including the two cross-cutting questions:
shard-level concurrency vs per-file striping, and whether disabling mmap helps
on Lustre and/or on tmpfs.

## Reuse / references

- `servekit profile` unchanged as the end-to-end measurement tool (JSON schema:
  `phases[].{name,duration_s,source}`, `weight_loading` from engine `elapsed=`).
- SGLang flags in `sglang/python/sglang/srt/server_args.py`: `weight_loader_disable_mmap`,
  `load_format` (incl. `sharded_state`, `runai_streamer`, fastsafetensors),
  `model_loader_extra_config`, `download_dir`.
- Existing launch template: `profile/llama-3.1-70b-bristen/serve_llama70b_sglang.sbatch`
  + `.toml` — copy into the scratch dir and edit `--model-path` / add flags per phase.

## Verification

- Phase 1–2: `fio` JSON output shows monotonic, plausible GB/s curves; shard→OST
  map matches `lfs df` OST list.
- Phase 3–4: each SGLang run reaches "fired up and ready to roll", servekit emits
  a profile JSON, and served requests return 200 (sanity `curl /generate`).
- Success = a treatment reduces `weight_loading` (and total) vs baseline with the
  server still serving correctly; numbers reproduced across ≥2 fresh allocations.

## Open risks / to confirm

- `enable_multithread_load` presence in vendored SGLang v0.5.10 (else use fork/PR).
- `/dev/shm` capacity on the A100 compute node vs 132 GB model.
- Residual OS page-cache contamination between end-to-end runs on a reused node.
- 263 GB `du` vs ~132 GB safetensors: confirm SGLang only reads the safetensors
  (ignore any `original/` pth checkpoint) so staging copies the right subset.

## Follow-up (not yet implemented): staging/startup overlap correctness

Overlapping the sliced `/dev/shm` stage with SGLang's import window is free
when staging finishes first, but under Lustre contention (2-6x swings,
already documented) staging could outlast the import window. Safe design:
atomic `rename()` from a `.tmp` staging path into the final path, plus a
small SGLang patch (reuse `scripts/lib/patch_sglang_in_container.sh`'s
clone-diff-verify harness) that waits on the final path's existence right
before the loader opens weight files. Not needed until overlap is actually
implemented.
