# fastsafetensors `max_threads` — is the internal read/copy pipeline also starved?

## Context

Phase 3 (`scripts/phase3_loader_concurrency/`) fixed fastsafetensors' *file*-level
concurrency starvation: patching `fastsafetensors_weights_iterator` to accept
`SGLANG_FST_FILES_PER_RANK` raised files-in-flight from 4 (upstream) to 32
(`fpr=8`), taking `weight_loading` from 69–88 s to **37–43 s**. But NOTES.md's
own follow-up analysis found that plateau is *not* purely a read limit: at
`fpr8`, only ~13–18 s of the 38.2 s is actually reading 132 GB (≈7–10 GB/s,
already near the raw `dd` ceiling); the rest is fixed, non-overlapped
GPU-side work (H2D copy, tensor materialization) that more files-in-flight
cannot touch.

`files_per_rank` controls **how many files are read across ranks**. It never
touched fastsafetensors' own internal thread pool, which pipelines *each
file's* `pread` + `cudaMemcpy` into device memory — that pool size is
`SafeTensorsFileLoader`'s own `max_threads` kwarg, hard-defaulted to 16, and
sglang never passes it through. Confirmed by cloning both repos at the
versions actually in play:

- `fastsafetensors/fastsafetensors/loader.py`,
  `SafeTensorsFileLoader.__init__(..., max_threads: int = 16, bbuf_size_kb:
  int = 16*1024, ...)`.
- `fastsafetensors/fastsafetensors/cpp/ext.cpp`
  (`nogds_file_reader::submit_read`, ~line 755): the **pinned host bounce
  buffer is `bbuf_size_kb * 1024 * max_threads` bytes**, one `cudaHostAlloc`
  per `SafeTensorsFileLoader` instance — the "pinned buffer" is not an
  independent knob, it *is* this product. At the default `bbuf_size_kb=16384`
  (16 MB), raising `max_threads` 16→64 grows the pinned buffer 256 MB→**1 GB**
  automatically, with no separate code change needed for that scaling.
- sglang v0.5.10 `python/sglang/srt/model_loader/weight_utils.py:768`
  (SHA `1519acf37c23f2189adb93f57ca9cd2db1bebf18`, verified byte-identical to
  the `lmsysorg/sglang:v0.5.10` image) calls `SafeTensorsFileLoader(pg,
  device)` with **no `max_threads` kwarg** — stuck at 16 threads / 256 MB
  pinned regardless of `--load-format` flags.

**Decision**: leave `bbuf_size_kb` at its 16 MB default and only raise
`max_threads` — the pinned buffer scaling to 1 GB is the intended/accepted
side effect of the knob, not something to compensate for.

`max_threads` is an axis independent of `files_per_rank`: it speeds up (or
doesn't) the per-file read+copy pipeline itself, which is part of the
~13–18 s read portion of the `fpr8` plateau — it cannot move the fixed
GPU-side floor Phase 4 already measured (~20 s). Expect some additional
shaving of the read portion at best, not an outsized further win.

## Working location

`lustre-loading-exp/scripts/fastsafetensor_many_threads/` +
`lustre-loading-exp/results/fastsafetensor_many_threads/`. Self-contained,
same one-dir-per-phase convention as phase3/phase5: its patch is a fresh diff
against the same pinned SHA (not a diff-on-a-diff against phase3's patch), so
this phase's harness has no cross-phase dependency.

## The patch — `fst_max_threads.patch`

One diff against `weight_utils.py`'s `fastsafetensors_weights_iterator`,
carrying phase3's `files_per_rank` knob forward unchanged and adding a second
env-gated knob:

```python
files_per_rank = int(os.getenv("SGLANG_FST_FILES_PER_RANK", "1"))
chunk = pg.size() * files_per_rank

max_threads = int(os.getenv("SGLANG_FST_MAX_THREADS", "16"))
...
loader = SafeTensorsFileLoader(pg, device, max_threads=max_threads)
```

Both knobs default to exactly upstream behavior (`fpr=1` reduces to the
original per-rank slicing per phase3's own null-run proof; `max_threads=16`
is the library's own default). Verified to `git apply --check` cleanly
against a fresh clone of `sgl-project/sglang@v0.5.10` and to `py_compile`.
Applied via the existing hermetic harness, unmodified:
`scripts/lib/patch_sglang_in_container.sh fst_max_threads.patch` (clone @
pinned tag → assert SHA → diff clone vs. installed as the version check →
apply → swap into site-packages → compile-check).

## Sweep — fresh node per point, bracketed nulls

Reusing `phase3_submit_chain.sh`'s shape: submit one job at a time with
`--wait`, read its node back from `sacct`, grow a `--exclude` list so no two
points ever land on the same (page-cache-contaminated) node.

| tag | files_per_rank | max_threads | purpose |
|---|---|---|---|
| `mt16_first` | 1 | 16 | **null run** — both knobs at upstream default. Must reproduce Phase 3's `fpr1` baseline (weight_loading 69–88 s, total 242–258 s). Proves the harness changes nothing at defaults. |
| `mt64` | 1 | 64 | **run alone** — max_threads effect in isolation, upstream file concurrency. |
| `mt64_fpr8` | 8 | 64 | **with fpr8** — combined with Phase 3's best file-concurrency point. |
| `mt16_last` | 1 | 16 | closing bracket, catches capstor drift across the sweep (same rationale as phase3/phase5). |

4 jobs (+ 1 preflight). Phase 3's already-measured `fpr8`-alone result
(37–43 s) is reused as the comparison baseline for `mt64_fpr8` — no need to
rerun it here.

## Files

| file | role |
|---|---|
| `fst_max_threads.patch` | the sglang diff (files_per_rank + max_threads knobs) |
| `fst_max_threads_preflight.sbatch` | clone/verify/patch/compile check, no GPUs — asserts both knobs are present and default correctly |
| `fst_max_threads_e2e.sbatch` | one sweep point, parameterized by `TAG`/`FILES_PER_RANK`/`MAX_THREADS` via `--export` — same launch flags and post-hoc `dd_read_sweep.sh` probe / `--bench` correctness check as phase3's e2e sbatch |
| `fst_max_threads_submit_chain.sh` | login-node driver: preflight → `mt16_first` → `mt64` → `mt64_fpr8` → `mt16_last`, `--wait` + growing `--exclude` |

## Risk to note (new vs. Phase 3)

Phase 3's known risk (`fpr8` GPU device-buffer OOM, `copy_files_to_device()`)
is unrelated to this knob — `max_threads` only sizes the **host-pinned**
bounce buffer (`cudaHostAlloc`), not the GPU-side buffer. At
`max_threads=64` that's 1 GB pinned host memory per active loader per rank
(freed at the end of each batch, not multiplied across all batches at once)
— trivial against a bristen A100 node's host RAM, but worth watching in case
`cudaHostAlloc` behaves unexpectedly under the container.

## Verification

1. **Harness** — preflight asserts both `os.getenv(...)` knobs are present in
   the installed module and default to `"1"` / `"16"`.
2. **Null run** — `mt16_first`/`mt16_last` must reproduce Phase 3's `fpr1`
   baseline within noise; if not, stop and fix the harness before reading
   `mt64`/`mt64_fpr8`.
3. **Correctness** — `servekit --bench` greedy-output check must stay
   character-for-character identical across all four runs.
4. **Signal** — compare `weight_loading` for `mt64` vs. the `mt16` bracket
   (isolated max_threads effect), and `mt64_fpr8` vs. Phase 3's existing
   `fpr8` result (37–43 s) (combined effect). Record the contemporaneous
   `dd_read_sweep.sh` bandwidth sample from each job, since capstor
   throughput drifts 2–6× over the sweep.
5. Append the comparison table + a written finding to `NOTES.md`, same
   format as the existing Phase 3/5 write-ups.

## Submit

```
bash lustre-loading-exp/scripts/fastsafetensor_many_threads/fst_max_threads_submit_chain.sh
```
