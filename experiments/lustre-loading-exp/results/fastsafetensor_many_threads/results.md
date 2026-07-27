# fastsafetensors `max_threads` — is the internal read/copy pipeline also starved?

**Status:** done — **no additional win; negative result**

> This experiment ran but was never written up in NOTES.md. Reconstructed
> here from `scripts/fastsafetensor_many_threads/PLAN.md` and the raw
> `.out`/`-profile.json`/`.csv` files in this directory.

## Goal

Phase 3 fixed fastsafetensors' *file*-level concurrency starvation
(`SGLANG_FST_FILES_PER_RANK`, 4 → 32 files in flight, `weight_loading` 69–88 s
→ 37–43 s). But `files_per_rank` controls how many **files** are read across
ranks — it never touches fastsafetensors' own internal thread pool that
pipelines each file's `pread` + `cudaMemcpy`. That pool size is
`SafeTensorsFileLoader`'s own `max_threads` kwarg, hard-defaulted to 16 in the
fastsafetensors library, and sglang v0.5.10 never passes it through
(confirmed by cloning both repos at the pinned versions:
`fastsafetensors/loader.py` and `weight_utils.py:768` at SHA
`1519acf37c23f2189adb93f57ca9cd2db1bebf18`, byte-identical to the
`lmsysorg/sglang:v0.5.10` image).

Also relevant: the pinned host bounce buffer is `bbuf_size_kb * 1024 *
max_threads` bytes (one `cudaHostAlloc` per loader instance,
`fastsafetensors/cpp/ext.cpp:~755`) — raising `max_threads` 16→64 grows it
256 MB → 1 GB automatically. Decision made up front: leave `bbuf_size_kb` at
its 16 MB default and only vary `max_threads`, accepting the buffer growth as
the intended side effect, not something to compensate for.

## Method

Second env-gated knob, `SGLANG_FST_MAX_THREADS` (default 16 = upstream),
added via a fresh diff against `weight_utils.py`
(`scripts/fastsafetensor_many_threads/fst_max_threads.patch`, carrying
Phase 3's `files_per_rank` knob forward unchanged). Same hermetic
clone-diff-verify-patch harness as Phase 3
(`scripts/lib/patch_sglang_in_container.sh`). Fresh node per point,
`--exclude`-accumulating chain (`fst_max_threads_submit_chain.sh`), bracketed
by a null run at both ends:

| tag | files_per_rank | max_threads | purpose |
|---|---|---|---|
| `mt16_first` | 1 | 16 | null run — both knobs at upstream default |
| `mt64` | 1 | 64 | max_threads effect in isolation |
| `mt64_fpr8` | 8 | 64 | combined with Phase 3's best files_per_rank point |
| `mt16_last` | 1 | 16 | closing bracket, catches capstor drift |
| `mt128_fpr8` | 8 | 128 | extra point beyond the original plan, pushed further |

## Result

`weight_loading` per point (from `*-profile.json`, this repo's job IDs
74750–74754 renamed to `75688`–`75696` for this sweep):

| tag | node | job | files_per_rank | max_threads | weight_loading (s) | dd probe same-job (GB/s) |
|---|---|---|---|---|---|---|
| mt16_first | nid002281 | 75688 | 1 | 16 | **61.06** | 5.04 |
| mt64 | nid002296 | 75690 | 1 | 64 | **64.04** | 6.92 |
| mt64_fpr8 | nid002285 | 75691 | 8 | 64 | **36.79** | 2.34 |
| mt16_last | nid002288 | 75694 | 1 | 16 | **56.75** | — |
| mt128_fpr8 | nid002289 | 75696 | 8 | 128 | **35.83** | — |

Correctness held throughout: 64/64 requests, 0 errors, 401–402 tok/s on
every point (from the `--bench` sections of the `.out` files).

## Verdict

**`max_threads` alone buys nothing**: `mt64` (64.04 s) is not better than —
arguably slightly worse than — the `mt16` upstream bracket (56.75–61.06 s).
Raising the internal read/copy thread pool without also raising
files-in-flight doesn't move the needle.

**Combined with `files_per_rank=8`, it's statistically indistinguishable
from Phase 3's `fpr8`-alone result**: `mt64_fpr8` (36.79 s) and `mt128_fpr8`
(35.83 s) land inside Phase 3's own `fpr8` range (37–43 s), not below it.
Doubling `max_threads` again (64→128) moved nothing (36.79 → 35.83 s, within
noise).

This matches the mechanism Phase 3 already identified: at `fpr8`, only
~13–18 s of the ~38 s is actual reading; the rest is fixed, non-overlapped
GPU-side work (H2D copy, tensor materialization) that no read-side knob —
file concurrency or internal thread count — can touch. `max_threads` was
worth checking because it's a distinct axis from `files_per_rank`, but the
read side was already close to the raw-dd ceiling by Phase 3, so there was
limited room for it to matter. **Not adopted**: no measured benefit over the
simpler `SGLANG_FST_FILES_PER_RANK=8`-only recommendation.

## Caveats

- No repeated/bracketed measurement of `mt64_fpr8`/`mt128_fpr8` against a
  fresh `fpr8`-only control in *this* sweep — the comparison leans on Phase
  3's separately-measured `fpr8` range (37–43 s), collected on different
  nodes at a different time. Given the result is negative (no improvement)
  and consistent with the already-understood GPU-side-floor mechanism, this
  is treated as sufficient, not as tightly bracketed as Phase 3 itself.
- One dd probe sample only for `mt64_fpr8` (2.34 GB/s) is well below this
  repo's typical native-layout range (6–8 GB/s) — plausibly a contended
  sample; it does not change the `weight_loading` conclusion since that's an
  engine-reported figure, not derived from the probe.
