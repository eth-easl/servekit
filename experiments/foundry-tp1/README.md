# Foundry CUDA-graph caching — Apertus-8B, DP=4

## Why

CUDA graph capture is one of the largest cold-start costs in SGLang.
[Foundry](https://github.com/foundry-org/foundry) persists captured graphs plus their
execution context during a one-time SAVE run, and restores them on LOAD — skipping
capture entirely. The paper reports Llama3-8B **28s → 1.3s** on the capture phase.

This measures that on Apertus-8B on bristen. Three timed runs, **DP=4 / TP=1**
(foundry has no TP support yet):

| Mode | Measures |
|---|---|
| `baseline` | today's cold start, no foundry |
| `save` | cost of the one-time capture + persist run |
| `load` | cold start with graphs restored — **the number we care about** |

## Key decision: no SGLang fork

Foundry ships an SGLang fork, but its entire diff is **47 lines across 4 files** — all
real logic lives in the pip package `foundry.integration.sglang`. Every anchor and all
three flags it sets (`disable_piecewise_cuda_graph`, `enable_profile_cuda_graph`,
`disable_flashinfer_autotune`) already exist in the **v0.5.10** the image ships. And
v0.5.10 pins **torch 2.9.1** while foundry needs 2.9.0+, so there is **no cu130 torch swap**.

So we patch v0.5.10 in place ([`apply_foundry_patch.py`](apply_foundry_patch.py)) instead
of swapping the engine. Because `apply_server_args` returns early when
`--foundry-graph-extension-config-path` is absent, the patch is **inert for the baseline** —
one image serves all three runs and the only variable is that single flag. No
engine-build confound.

## Files

```
experiments/foundry-tp1/
├── Containerfile            FROM lmsysorg/sglang:v0.5.10 + boost + foundry + patch
├── apply_foundry_patch.py   the 4 direct edits, anchored on source text
├── build_image.sh           podman build + enroot import -> .sqsh   (run once)
├── apertus-8b-foundry.toml  EDF, = the apertus EDF with the new image
├── foundry_save.toml        mode="save"
├── foundry_load.toml        mode="load"   (all other fields identical)
├── apertus8b_dp4.sbatch     ONE script, MODE=baseline|save|load
└── results/
```

`apply_foundry_patch.py` anchors on source text rather than line numbers, so it survives
drift between the pip wheel and the git tag, and it asserts every edit landed. Verified
offline against v0.5.10: all 5 anchors unique, all 4 patched files compile.

## Run

```bash
bash experiments/foundry-tp1/build_image.sh                                    # once, login node

sbatch --export=ALL,MODE=baseline experiments/foundry-tp1/apertus8b_dp4.sbatch
sbatch --export=ALL,MODE=save     experiments/foundry-tp1/apertus8b_dp4.sbatch
sbatch --export=ALL,MODE=load --exclude=<save's node> experiments/foundry-tp1/apertus8b_dp4.sbatch
```

`--exclude` the SAVE node so the ~4 GB archive and the weights are genuinely cold on
LOAD — page cache survives container runs. The SAVE job prints the exact next command
with its node filled in.

All three runs are wrapped in the same `servekit profile --bench`. Without `--keep-alive`,
servekit tears the server down once the bench ends, which is precisely the "wait for
startup complete, then SIGTERM" that the SAVE recipe asks for.

## Results (2026-07-22, Apertus-8B DP=4, A100-80GB, sglang v0.5.11)

All seconds. `baseline_nopw` = baseline + `--disable-piecewise-cuda-graph`, no foundry.

| Phase | baseline 75642 | baseline rpt 75656 | baseline_nopw 75655 | SAVE 75652 | LOAD 75654 |
|---|---|---|---|---|---|
| **total** | 203.44 | **191.65** | **150.28** | 181.87 | **155.51** |
| process_startup | 16.47 | 7.59 | 7.61 | 7.67 | 7.59 |
| imports | 28.47 | 27.19 | 26.61 | 30.10 | 30.78 |
| weight_loading | 72.22 | 69.58 | 70.57 | 81.02 | 74.36 |
| cuda_graph_capture | 22.72 | 22.72 | 22.31 | 35.18 | **2.97** |
| piecewise_capture | 39.95 | 41.15 | — | — | — |
| warmup(JIT) | 14.95 | 14.77 | 15.07 | 18.15 | **32.30** |
| throughput tok/s | 1157.1 | 1155.3 | **1161.7** | — | **1104.9** |

SAVE produced 144 graphs (36/rank x 4), 7.4 GB. LOAD emitted zero `Saved SGLang CUDA
graph` lines and its correctness outputs match baseline exactly, so restored graphs
replay correctly.

### Reading it

Use run **75656** as the baseline: 75642's 16.47 s `process_startup` is a first-run
artifact (every later run lands at ~7.6 s).

Foundry forces `disable_piecewise_cuda_graph=True`, so a naive baseline->LOAD delta
(191.65 -> 155.51 = -36 s) credits foundry with the piecewise phase it merely switched
off. `baseline_nopw` isolates that, and shows **disabling piecewise is free**: warmup
and throughput are unchanged (15.07 s, 1161.7 tok/s). So the like-for-like comparison
is `baseline_nopw` vs `LOAD` -- both with piecewise off:

| | baseline_nopw | LOAD | delta |
|---|---|---|---|
| cuda_graph_capture | 22.31 | 2.97 | **-19.34** |
| warmup(JIT) | 15.07 | 32.30 | **+17.23** |
| total | 150.28 | 155.51 | +5.23 |
| throughput | 1161.7 | 1104.9 | **-4.9%** |

**Foundry's graph restore is real (7.5x on capture) but it returns almost all of the
saving as warmup cost, and costs ~5% steady-state throughput.** On this workload plain
`--disable-piecewise-cuda-graph` reaches ready at least as fast as foundry, for one
flag and no throughput loss.

Caveat: single runs; `weight_loading` alone spans 69.6-81.0 s, so the 5.23 s total gap
is within noise -- read it as LOAD ~= baseline_nopw. The phase shifts (-19.3/+17.2 s)
and the throughput drop are large and systematic.

### Why piecewise is off under foundry

Not a bug and not temporary. Foundry's SGLang integration hooks only the centralized
full-graph seam (`CudaGraphRunner.capture`); piecewise **is** the `torch.compile` path
and goes through a runner foundry never intercepts. Every piecewise implementation
reference in the repo lives in `integration/vllm/graph_ops.py` -- there is none in the
SGLang integration, and `ROADMAP.md` does not list it even as planned (it lists `EP on
SGLang` and `TP on SGLang` as open). Foundry's stated reason, in `foundry_shim.py`, is
to "keep phase-1 SAVE/LOAD deterministic". So on SGLang, foundry and piecewise CUDA
graphs are mutually exclusive by design.

### Where the +17.2 s warmup comes from (traced)

Two separate forced behaviours, both confirmed in the logs:

**1. Graph capture is what compiles the kernels, and foundry replaces it.** Not
`kernel_warmup` -- that is under a second here (`Load weight end` 14:50:32 ->
`Capture begin` 14:50:33). The CUTLASS/CuTe JIT is triggered by capture's own warmup
forwards. Log timelines:

| | baseline_nopw 75655 | LOAD 75654 |
|---|---|---|
| Capture begin | 14:50:33 | 14:43:47 |
| **CUTE_DSL import** | **14:50:36 (inside capture)** | *(absent)* |
| Capture end | 14:50:55 (22.31 s) | 14:43:50 (2.97 s) |
| Uvicorn ready | | 14:43:54 |
| **CUTE_DSL import** | | **14:44:03 (after ready)** |
| warmup POST 200 | 14:51:15 | 14:44:26 |

On LOAD no forwards run, so the JIT never fires during capture -- it fires on the first
real request, *after* the ready signal. The two phases must therefore be read as a pair:

| | capture | warmup(JIT) | sum |
|---|---|---|---|
| baseline_nopw | 22.31 | 15.07 | **37.38** |
| LOAD | 2.97 | 32.30 | **35.27** |
| delta | -19.34 | +17.23 | **-2.11** |

**foundry's real saving on this workload is ~2 s, not ~19 s.** The 19 s capture saving
is almost entirely JIT work relocated past the ready line, not eliminated -- and LOAD's
total is still 5.23 s slower than `baseline_nopw` because other phases vary by more
than 2 s.

**2. The -4.9% throughput is a forced flag, not graph restore.** Verified in the
server_args dumps: `disable_flashinfer_autotune=False` in 75655 vs `True` in 75654.
foundry's shim force-sets it, so FlashInfer serves with untuned kernels.

Both are structural to keeping SAVE<->LOAD allocations byte-identical, not incidental
bugs -- so neither is likely to be tuned away without upstream changes.

**What foundry's docs say about it.** The rationale is documented on the vLLM side only
(`docs/sglang/hooks.md` §4 is terse). `docs/vllm/memory-consistency.md` §3:

> "Rule: no-op on SAVE/LOAD. Their internal allocations don't survive a SAVE/LOAD cycle,
> so reproducing them on LOAD is a waste — and any non-determinism is forbidden."

and `docs/vllm/hooks.md` §2 adds that it skips "FlashInfer autotune, CUDA-graph-private
warmups, and other inductor cache-priming forwards whose allocations would diverge
between SAVE and LOAD". `docs/sglang/hooks.md` §4 confirms autotune is disabled twice:
"shut off both by this no-op and by the forced `disable_flashinfer_autotune` flag".

Our measurement contradicts the "is a waste" half. That claim holds for the *device-side*
kernel binaries, which foundry does restore (`fatbin_image_packed.img`), but not for the
*host-side* JIT -- CUTLASS/CuTe DSL import, FlashInfer module building -- which foundry
does not capture and which therefore still runs. The logs show it firing after Uvicorn,
inside the first request. So skipping `kernel_warmup` does not avoid the work; it moves
~17 s past the ready signal, which also moves it outside the cold-start metric.

## Verify

1. **Build** — `build_image.sh` asserts `libcuda_hook.so` exists, `sglang.srt.foundry_shim`
   imports, and `ServerArgs.foundry_graph_extension_config_path` is present.
2. **Baseline unaffected** — no `[Foundry]` lines in the log; a normal
   `Capture cuda graph end. Time elapsed: N s`.
3. **SAVE** — `[Foundry] SGLang hooks installed`, `kernel_warmup skipped in save mode`,
   a run of `Saved SGLang CUDA graph ...` lines, `Saved graph_manifest.json`,
   `final_alloc_offset=`. Archive has `warmup_state.json` + `rank_0..3/`, each with
   `graph_*.{json,cugraph}`, `graph_manifest.json`, `fatbin_image_packed.img`.
4. **LOAD** — `reused saved memory pool config`, **no** `Saved SGLang CUDA graph` lines,
   `cuda_graph_capture` collapses vs baseline. **The bench correctness probe must pass** —
   a graph that replays garbage would still report "ready".
5. **Compare** the three `*-profile.json`: `total_ready_s` and the `cuda_graph_capture` phase.

## Notes / risks

- **Deliberately not applied**: the foundry recipe pins `--disable-radix-cache
  --attention-backend flashinfer --cuda-graph-max-bs 512 --mem-fraction-static 0.6`.
  We keep the flags from `serve_apertus_8b_sglang.sbatch` instead. If LOAD faults or
  replays garbage, these are the **first lever** — especially
  `--attention-backend flashinfer`, the only backend foundry's SGLang path is validated on.
- **fastsafetensors is dropped — it is incompatible with DP in v0.5.10.** Found the hard
  way on the first baseline run (job 75557, OOM). `weight_utils.py:751` sets
  `device = torch.device(f"cuda:{rank}")` from the torch.distributed rank, but under DP
  each rank is its own process with `tp_size=1`, so `pg.rank()==0` in all four and every
  rank stages weights onto physical `cuda:0` (10.87+16.41+31.59+16.41 ≈ 75 of 79 GiB).
  It works under TP only because there `tp_rank` coincides with `gpu_id`. A one-line fix
  (`torch.cuda.current_device()`) is possible but we chose not to patch the engine further.
  **Consequence**: weight-load times here are not comparable to the phase1.3 fastsafetensors
  numbers. The image still has the package installed; it is simply unused.
- **DP=4 loads the full model per rank**, so weight reads are ~4× a TP=4 run. Inherent to
  DP and identical across all three modes, so the comparison holds.
- **Asymmetry, documented not fixed**: foundry force-sets `disable_piecewise_cuda_graph`
  and `disable_flashinfer_autotune` on SAVE/LOAD but not baseline, so the baseline→load
  delta is not *purely* graph restore.
- **DP=4 is an extrapolation** — upstream validated DP=2 (plus EP=2 and single-GPU).
- `Reserved address ... != requested base` at startup is a known non-deterministic VMM
  collision — just resubmit.
- **servekit under DP**: `seen_phases` dedupes, so per-phase rows reflect only the
  first-reporting rank. Total ready time is still exact.
