# Design — N3: vLLM-CUDA cold-start baseline on bristen (A100)

Date: 2026-06-26
Status: approved design, pending implementation plan
Scope owner: Xiaozhe Yao
Parent design: `docs/superpowers/specs/2026-06-26-nvidia-cuda-snapshot-port-design.md` (§6 "Layer 3 — vLLM-CUDA deploy", milestone **N3**)

## 1. Goal & context

The `snapshot` cold-start tool's NVIDIA port has a working CUDA backend (**N1**)
and a deterministic-address `redirect_cuda` interposer (**N2**). **N3** stands up
**vLLM-CUDA serving GLM-4.7-Flash at TP=4** on bristen — which today runs SGLang
only — and establishes a clean, honestly-decomposed **baseline cold start**: the
`READY` time, a per-phase breakdown, and the **characterized eliminable capture
fraction** that N4 (skip-capture) and N5 (snapshot/restore) will be measured
against. N3 only *measures* the capture cost; it does not eliminate it.

The cold-start measurement stack proven on AMD/beverin is **vendor-neutral**
(Python/log-based — `/health` polling, vLLM log-grep phase breakdown, a
`torch.cuda.CUDAGraph` monkeypatch, a vLLM-API-level skip-capture probe), so N3
ports it to CUDA rather than re-deriving it.

### Decisions (settled with the user, 2026-06-26)

- **Depth = full instrumentation.** Serving + `READY` + log-grep phases, **plus**
  porting the AMD capture instrumentation to CUDA: `VLLM_CG_INSTRUMENT`
  (per-graph capture timing), `cg_skip.py measure` mode (total capture-phase
  cost), and an `--enforce-eager` A/B. This precisely quantifies the eliminable
  capture fraction and validates the instrumentation ports to CUDA, de-risking
  N4/N5.
- **Baseline config = production-realistic.** TP=4, `gpu-memory-utilization 0.90`,
  `max-num-seqs 256`, `max-model-len 131072`, default cudagraph capture sizes.
  56 GiB weights / 4 GPUs ≈ 14 GiB/GPU leaves ~58 GiB/GPU for KV → very high
  concurrency → a large, honest capture phase (the design's "higher KV
  concurrency makes the win more visible"). Recipes parameterize these via env
  vars for sweeps; these are the headline defaults.
- **Approach = A3 hybrid.** Mirror the AMD recipes 1:1 (port the vendor-neutral
  instrumentation verbatim), but bake in the N1/N2 de-risking discipline:
  probe-first, image pinned by digest + transformers overlay, a dedicated
  vLLM-CUDA cache dir (separate from SGLang's, warmed within N3's runs), and a
  warm-cache A/B protocol with `PYTHONHASHSEED=0`.

### Target environment (bristen)

- 1× node, 4× A100-SXM4-80GB, NVLink, x86_64; SLURM `-A a-infra02`, partition
  `normal`; enroot/pyxis (`.sqsh` + EDF). Model on shared `/capstor`:
  `/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash`.
  Login nodes have no GPU — all runs via `srun`/`sbatch`; code sync via `rcc`
  using a new `glm-47-flash-bristen-vllm` profile (remote dir
  `/capstor/scratch/cscs/xyao/glm-47-flash-vllm`), added in the plan.

## 2. Deploy — `deploy/glm-47-flash-bristen-vllm/`

Mirrors the shape of `deploy/glm-47-flash-bristen/` (SGLang) but with vLLM-CUDA.

- **EDF:** pin a `vllm/vllm-openai` **CUDA** image (sm_80 / CUDA 12.x) **by
  digest** (resolved empirically — see §3), with a startup
  `pip install transformers==5.12.1` overlay (the `glm4_moe_lite` fix the SGLang
  README documents; `--trust-remote-code` alone does not help — the checkpoint
  ships no custom modeling code). Persistent pip + Triton + vLLM caches under the
  deploy dir on `/capstor`. Mounts `/capstor`, `/users/xyao`, `/iopsstor`.
- **Serve recipe:** GLM-4.7-Flash, single node / single replica, **TP=4**,
  `--gpu-memory-utilization 0.90 --max-num-seqs 256 --max-model-len 131072
  --trust-remote-code --enable-prefix-caching --tool-call-parser glm47
  --reasoning-parser glm45 --enable-auto-tool-choice`. Env-overridable.
- **README** documenting the image pin, the transformers overlay, the model
  path, the cache layout, and the run commands.

## 3. Probe-first de-risking

The single biggest N3 risk is whether vLLM-CUDA serves `glm4_moe_lite` **at
all**. So the first build step is a **probe job** (mirroring N1's toolchain
probe) that, against a candidate `vllm/vllm-openai` digest + the transformers
overlay: launches `vllm serve` at TP=4, polls `/health` to `READY`, and fires a
correctness completion ("The capital of France is" → "Paris"). Only once the
probe is green is the image digest pinned into the EDF and the measurement stack
built. If the candidate image lacks `glm4_moe_lite` support, the probe surfaces
it immediately (escalate: try a newer vLLM tag).

## 4. Cold-start measurement (CUDA siblings of the AMD recipes)

Port verbatim where vendor-neutral; swap only the container/image/env.

- **READY timer** (`_vllm_coldstart.sh` analog): fork `vllm serve`, poll
  `/health` every 2 s from `t0`, emit `READY at Ns`; fire one correctness probe;
  clean up.
- **Phase breakdown** (`_vllm_measure.sh` analog): grep the vLLM log for the same
  lines that appear on CUDA — `Model loading took`, `init engine … took`,
  `Maximum concurrency`, `Capturing CUDA graphs (…): 100% … [MM:SS`,
  `Application startup complete` — to decompose startup / weight-load / compile /
  profile+KV / capture.
- **Per-graph capture timing:** port `cginst/sitecustomize.py` (hooks
  `torch.cuda.CUDAGraph.{capture_begin,capture_end,replay}` — the same class on
  CUDA), activated by `PYTHONPATH` + `VLLM_CG_INSTRUMENT=<csv>`. Emits per-graph
  `capture_wall_ms` and origin classification.
- **Total capture-phase cost:** port `cginst_skip/cg_skip.py measure` mode
  (no-ops the warmup/capture forward, times the whole capture phase, prints
  `CAPTURE_PHASE_S`). Vendor-neutral (vLLM Python API only).
- **`--enforce-eager` A/B:** the non-capture baseline; capture cost ≈
  with-graphs `READY` − enforce-eager `READY`. The per-graph total, the
  `measure`-mode total, and the A/B delta should cross-agree.
- **Warm-cache protocol:** because the snapshot win targets warm-cache *restart*
  (not first-ever JIT compile), a warm-up run first populates the shared
  `TRITON_CACHE_DIR` / `VLLM_CACHE_ROOT` on `/capstor`; the measured run is
  warm-cache. `PYTHONHASHSEED=0` pins inductor fusion order for reproducibility.
  First-run (cold-cache) compile is reported separately, not as the baseline.

## 5. Output

A `snapshot/RESULTS.md` **N3** section: the bristen environment (image digest,
CUDA version, sm_80, config), the clean warm-cache `READY`, the per-phase
decomposition (with weight-load I/O called out as **non-eliminable** and capture
as the **eliminable target**), the characterized capture-phase cost (per-graph +
`measure`-total + enforce-eager delta cross-agreeing), and a note that N4
(skip-capture) is next. Directly comparable to the AMD arc's decomposition.

## 6. Gates (acceptance)

| # | Gate |
|---|---|
| G1 | Probe: vLLM-CUDA serves GLM-4.7-Flash TP=4 → `/health` READY + correct "Paris" completion |
| G2 | Clean baseline `READY` (warm cache), reproducible across 2 runs (same config/seed) |
| G3 | Per-phase breakdown sums to ~`READY`; weight-load and capture both quantified |
| G4 | Capture-phase cost characterized: per-graph total ≈ `measure`-mode total ≈ (with-graphs − enforce-eager) delta |
| G5 | RESULTS N3 section written; honest eliminable-fraction accounting |

Each gate is a bristen cluster job (`rcc push` → `sbatch -A a-infra02`).

## 7. Scope

**In:** the vLLM-CUDA deploy, probe-first serving confirmation, the ported
cold-start measurement + capture instrumentation, the warm-cache A/B, the
RESULTS N3 section.

**Out (later milestones):**
- The skip-capture **win** (N4) — N3 only *measures* capture cost.
- Snapshot/restore (N5); the `redirect_cuda` / `record_cuda` interposers — the
  N3 baseline runs **clean** (no interposer), so its numbers are uncontaminated
  (the N2 +0-overhead and the eager-gate concerns belong to N5).
- Steady-state serving benchmark (`benchmaker` A/B) — that's N5's
  serving-overhead check.
- Multi-node / multi-replica — N3 is 1 node, TP=4 (the SGLang deploy's 5-replica
  topology is not needed for a single cold-start measurement).

## 8. Risks & mitigations

1. **`glm4_moe_lite` under vLLM-CUDA** — mitigated by probe-first (G1) +
   transformers 5.12.1 overlay; if the pinned vLLM version lacks support, bump to
   a newer `vllm/vllm-openai` tag (the probe makes this a fast, early failure).
2. **Image/version drift** — pin by digest (the `:latest` JIT-cache-invalidation
   lesson from the SGLang deploy); record the resolved digest in the EDF.
3. **Honest capture accounting** — weight-load I/O (56 GiB / 48 shards on Lustre,
   80–150 s on AMD) is non-eliminable and often dominates; the report decomposes
   truthfully and identifies capture as the eliminable target, not a headline
   total.
4. **Capture cost is config-dependent** — the production-realistic config (gmu
   0.90, max-num-seqs 256) is chosen precisely to make the capture phase large
   and the eventual win visible; the report states the config alongside the
   numbers.

## 9. Files (new)

- `deploy/glm-47-flash-bristen-vllm/` — EDF (`*.toml`), serve recipe (`*.sbatch`),
  probe recipe, README.
- `snapshot/recipe/` CUDA siblings of the AMD cold-start recipes: a vLLM-CUDA
  `READY`/measure sbatch + driver script, and CUDA-side copies (or reuse, via
  env/container swap) of `cginst/sitecustomize.py`, `cginst_skip/cg_skip.py`
  (`measure` mode), and the phase-grep logic.
- `snapshot/RESULTS.md` — new N3 section.

Untouched: all `backends/*`, `preload/*`, `core/*`, the N2 redirect recipes, and
the SGLang `deploy/glm-47-flash-bristen/`.
