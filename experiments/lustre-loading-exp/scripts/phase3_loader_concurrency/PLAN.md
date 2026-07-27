# Phase 3 — is the fastsafetensors loader concurrency-starved?

## Context

For Llama-3.1-70B, `weight_loading` is the dominant cold-start phase. The best
loader today (`--load-format fastsafetensors`) takes **73.7 s** for 132 GB =
**1.78 GB/s**.

Two results say that number is a *loader* limit, not a *storage* limit:

- The Phase-2 stripe sweep ran a **contemporaneous native-layout dd probe**
  (`results/phase2_stripe_sweep/phase2_nativeprobe-*.csv`): capstor delivers
  **3.63–5.47 GB/s on the unmodified native layout** at high reader
  concurrency. The 1.75 GB/s "ceiling" recorded in NOTES Phase 1.2 was one
  badly-contended sample, not the ceiling.
- dd at N=8 gives 1.6–2.3 GB/s. The loader gives 1.78 GB/s. It lands exactly
  where a ~4–8-wide reader should land.

The mechanism is in `sglang/srt/model_loader/weight_utils.py`,
`fastsafetensors_weights_iterator`:

```python
weight_files_sub_lists = [hf_weights_files[i : i + pg.size()] ...]  # batches of 4
    rank_file_map = {i: [f] for i, f in enumerate(f_list)}          # ONE file per rank
```

With TP=4 and 30 shards: **8 serial batches, 4 files in flight**, each batch a
collective barrier gated by its slowest file. The header of
`scripts/phase2_e2e/phase2_e2e_layout.sbatch` already names this
("pg.size()=4 -> only ~4 concurrent readers") — this phase turns it into a knob
and measures it.

**Hypothesis**: files-in-flight is the lever. Raising it from 4 → 16/32 should
move `weight_loading` from ~74 s toward the 25–40 s implied by the dd curve,
**with no change to how the model is stored** (native layout, plug-and-play).

Intended outcome: a number that either confirms the loader is the bottleneck —
which makes InstantTensor (PR sgl#28506: *"tuned I/O size and concurrency,
pipelining and prefetching"*) the obvious next move — or refutes it and
redirects effort to graph capture (now 101 s of the 177 s `/dev/shm` run).

## Approach: hermetic in-container patch, no working-tree dependency

No reliance on the untracked `sglang/` tree at the repo root. Each job, inside
the container:

1. **Clone** `sgl-project/sglang` at the SHA pinned to the image
   (`lmsysorg/sglang:v0.5.10` → tag `v0.5.10` →
   `1519acf37c23f2189adb93f57ca9cd2db1bebf18`), shallow, into `$HOME` (`/root`,
   ephemeral overlay).
2. **Verify** the clone is byte-identical to the container's *installed* sglang
   for the file we touch (`diff` clone vs `site-packages`). **This diff is the
   version check** — if the pin has drifted from the image, the job aborts loud
   instead of silently running different code.
3. **Patch** the clone, then **copy the single patched file over
   site-packages**. Blast radius is one file; the running engine is otherwise
   the container's blessed install; it dies with the container.

This deliberately avoids a `PYTHONPATH` shadow of the whole tree: site-packages
is already proven writable (every job does `pip install -e servekit` into it),
the swap survives TP-worker spawn for free (it *is* the imported module), and
the env toml stays untouched — `PYTHONPATH = ""` keeps user venvs locked out.

## The patch

Env-gated, defaulting to **exactly upstream behavior**:

```python
fpr = int(os.getenv("SGLANG_FST_FILES_PER_RANK", "1"))
chunk = pg.size() * fpr
weight_files_sub_lists = [
    hf_weights_files[i : i + chunk] for i in range(0, len(hf_weights_files), chunk)
]
...
    rank_file_map = {
        i: files for i in range(pg.size()) if (files := f_list[i :: pg.size()])
    }
```

`rank_file_map` already accepts a **list** per rank, so this needs no
fastsafetensors change.

**At `fpr=1` this is provably identical to upstream**: `chunk == pg.size()`, so
the sub-lists are unchanged; `f_list[i::pg.size()]` on a ≤4-element list yields
`[f_list[i]]`; and a short final chunk leaves the trailing ranks' slices empty,
so the `if` omits them — exactly as upstream's `enumerate` does. That makes
`fpr=1` a **true null run**, which is the control for the whole mechanism.

Files in flight: `fpr` 1→4, 2→8, 4→16, 8→32 — deliberately straddling the dd
sweep's N=4/8/16/32 points so the two curves are directly comparable.

(Check `os` is already imported in `weight_utils.py`; add if not.)

## New files — `lustre-loading-exp/scripts/phase3_loader_concurrency/`

Follows the established convention exactly (one dir per phase; `.sbatch` =
submittable, `.sh` = helper; `set -euo pipefail`; why-comment header + literal
submit command; results dir mirrors scripts dir).

| file | role |
|---|---|
| `fst_files_per_rank.patch` | the sglang diff above (also the upstream PR) |
| `patch_sglang_in_container.sh` | clone@pinned-SHA → verify vs installed → apply → swap into site-packages → echo provenance |
| `phase3_loader_concurrency.sbatch` | one sweep point, parameterized by `TAG` + `FILES_PER_RANK` |
| `phase3_submit_chain.sh` | login-node driver; one job per point, `--dependency=afterok` |

Results → `lustre-loading-exp/results/phase3_loader_concurrency/`, named
`phase3_loader_concurrency-<tag>-<jobid>-<node>-profile.json` (tag must be the
last dash-segment before the job id or `analysis/summarize_e2e.py` mis-parses it).

### sbatch shape

Copy `phase2_e2e/phase2_e2e_layout.sbatch` verbatim (same header: `normal`,
`a-infra02`, 1 node, 32 cpus, 4 gpus, `--exclusive`, 30 min), changing only:

- `--output=lustre-loading-exp/results/phase3_loader_concurrency/%x-%j.out`
- `MODEL` fixed to the **native** dir
  (`/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct`)
- `TAG` / `FILES_PER_RANK` from `--export`
- `INNER_CMD` gains the patch step and exports the env var:

```bash
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
pip install fastsafetensors -q
bash "${EXP_DIR}/scripts/phase3_loader_concurrency/patch_sglang_in_container.sh"
export SGLANG_FST_FILES_PER_RANK=${FILES_PER_RANK}
servekit profile --out "${PROFILE_OUT}" --bench ... -- python -m sglang.launch_server ... --load-format fastsafetensors ...
bash "${EXP_DIR}/scripts/lib/dd_read_sweep.sh" "${MODEL}" 16M "${DD_CSV}" "30"
```

Two deliberate deviations from the phase2 template, both load-bearing:

- **Drop `exec` before `servekit`** so the dd probe can run afterwards
  (`--bench` terminates the server itself, so the node is free).
- **The dd probe runs AFTER the profiled launch, not before.** Reading all
  141 GB with `dd` immediately *before* the measured load would warm the
  Lustre **OSS-side** cache and flatter the very number we're measuring
  (O_DIRECT only bypasses the *client* page cache).
  `scripts/probes/oss_cache_probe.sh` exists to test that assumption if the
  post-probe numbers look suspicious.

The probe is the point: capstor drifts 2–6× over tens of minutes, so a 5-job
sweep spread over ~2.5 h is **uninterpretable without a per-job bandwidth
sample**. Reuse `scripts/lib/dd_read_sweep.sh` (a single N=30 point, ~25–40 s).

### Sweep order — bracketed, fresh node per point

Reuse the `phase2_submit_e2e_chain.sh` shape (`sbatch --parsable
--dependency=afterok:$prev --job-name=... --export=ALL,TAG=...`). One
submission per point = one fresh `--exclusive` allocation = the page-cache rule
holds. Bracket with the null run, exactly as phase2 brackets with `native`:

```
fpr1_first → fpr2 → fpr4 → fpr8 → fpr1_last
```

5 jobs × ~30 min ≈ **2.5 h**.

## Known risk: `fpr=8` may OOM — and that is a result

`copy_files_to_device()` buffers the batch **in GPU memory**. At `fpr=8` each
rank holds ~8 × 4.7 GB ≈ **38 GB** of shards on device *on top of* the 32.9 GB
of weights it is materializing. The mmap run logs `avail mem=44.83 GB` after
load on an 80 GB A100, so `fpr=8` is at or past the edge.

**Do not compensate by lowering `--mem-fraction-static`** — everything except
`fpr` stays constant. If it OOMs, that is the memory ceiling of this approach,
it is worth knowing, and it is precisely the argument for a *pipelined* loader
(bounded buffer, N reads in flight) over a bigger batch — i.e. for InstantTensor
rather than this patch. `fpr=4` (≈19 GB buffer) should be safe and already gives
16 files in flight.

## Verification

1. **Mechanism** — `patch_sglang_in_container.sh` aborts unless the clone's
   `weight_utils.py` is byte-identical to the installed one. Job log records
   `git rev-parse HEAD` and `sglang.__version__`. A passing diff proves the pin
   matches the image.
2. **Null run (the critical control)** — `fpr1` must reproduce the Phase-1.3
   fastsafetensors baseline (**weight_loading 73.7 s, total 237 s**) within
   noise, and `fpr1_last` must bracket it. If `fpr1` ≠ baseline, the
   clone/patch/swap harness is changing behavior → **stop and fix the harness
   before reading any other number.**
3. **Correctness** — servekit's `--bench` correctness probe: greedy outputs must
   stay character-for-character identical across all `fpr` values (as they
   already are across mmap/nommap/fastsafetensors). Different batching must not
   change a single weight.
4. **Signal** — plot `weight_loading` against files-in-flight (4/8/16/32) beside
   the same job's dd probe. Hypothesis confirmed if `weight_loading` falls
   materially at `fpr ≥ 2` and tracks the dd concurrency curve toward
   3.5–6 GB/s.
5. **Sanity** — every run reaches "fired up and ready to roll", emits a profile
   JSON, and reports 0 bench errors with throughput unchanged (~402 tok/s, p50
   5.09 s). The loader must move time-to-serving only, never runtime perf.

Success = `weight_loading` drops materially below 73.7 s on the **native,
unmodified** model dir, from a source change alone. That is the plug-and-play
result, and it sizes the prize for adopting InstantTensor.

## Follow-ups (explicitly out of scope here)

- Backport InstantTensor (PR sgl#28506) onto this same harness — one more patch
  file, same clone/verify/swap script.
- Graph capture: `cuda_graph` 25 s + `piecewise` 76 s = **101 s of the 177 s**
  `/dev/shm` run. Once the loader is fixed, this is the largest remaining phase.
- Fold the Phase-1.2 correction into NOTES.md — its "contention-dominated
  ceiling" finding is actively misleading now that the native probe shows
  3.6–5.5 GB/s.
