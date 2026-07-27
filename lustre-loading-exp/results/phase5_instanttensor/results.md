# Phase 5 — InstantTensor (PR sgl#28506)

**Status:** done — **dead end on this platform, structural not configurational**

## Goal

Phase 3 fixed fastsafetensors' file-level read concurrency; the residual
~20–25 s of `weight_loading` is non-overlapped H2D + NCCL broadcast + param
copy (priced by Phase 4's tmpfs run). InstantTensor's pitch is exactly
"pipelining and prefetching" — overlap the read of batch N+1 with the
broadcast of batch N — which targets precisely that residual. Test whether
it delivers.

## Method

Backported PR sgl#28506 to v0.5.10
(`scripts/phase5_instanttensor/instanttensor_backport.patch`, 4 files:
`LoadFormat.INSTANTTENSOR`, `instanttensor_weights_iterator`, the
`loader.py` branch, the `--load-format` allow-list), verified by the same
clone-and-diff harness as Phase 3, generalized to multi-file patches. The PR
itself passes no I/O knobs (library defaults only); the backport adds
optional `SGLANG_IT_CONCURRENCY` / `SGLANG_IT_IO_DEPTH` pass-through
(default `None`, so an unset run reproduces the PR exactly).

**5a**: bracketed A/B (fastsafetensors `fpr=8` control vs `--load-format
instanttensor` as written), one variable, 4 distinct nodes, n=2 per arm.

**5b**: knob sweep (`concurrency` ∈ {16, 32, 64}, `io_depth=64` at the top
point) under a separately-contended capstor, read only against its own
bracket.

**Preceding probe**: whether GPUDirect Storage (GDS) is even available on
bristen, since InstantTensor's headline 35–45 GB/s numbers are GDS-specific
— see [`../probes/results.md`](../probes/results.md). Answer: no.

## Result

### 5a — PR as written

| point | node | weight | eff BW | total | capstor same-job | tok/s | errs |
|---|---|---|---|---|---|---|---|
| ctl (fst, fpr=8) | nid002292 | **36.9 s** | 3.58 GB/s | 208.4 s | 5.62 | 402 | 0 |
| it_default | nid002324 | **352.0 s** | 0.375 GB/s | 510.1 s | 6.18 | 401 | 0 |
| it_default2 | nid002289 | **376.2 s** | 0.351 GB/s | 542.1 s | 7.28 | 401 | 0 |
| ctl (fst, fpr=8) | nid002293 | **42.7 s** | 3.09 GB/s | 211.6 s | 1.54 | 401 | 0 |

Reproduces on two nodes; capstor was healthy (6.2/7.3 GB/s) during both
InstantTensor runs, so it isn't contention. Outputs/throughput fine — the
weights load correctly, it's purely slow. **A ~9× regression vs the
fastsafetensors+patch control.**

### 5b — knob sweep

| point | knobs | weight | eff BW |
|---|---|---|---|
| ctl (fst, fpr=8) | — | **62.6 s** | 2.11 GB/s |
| it_default | none (PR as written) | 352–376 s | 0.36 |
| it_c16 | `concurrency=16` | **244.0 s** | 0.54 |
| it_c32 | `concurrency=32` | **245.0 s** | 0.54 |
| it_c64_d64 | `concurrency=64, io_depth=64` | **276.7 s** | 0.48 |
| ctl (fst, fpr=8) | — | **58.6 s** | 2.25 GB/s |

Control bracket tight (58.6/62.6 s) → comparison sound. Concurrency buys a
one-time ~30% (352 → 244 s) then flatlines (16/32/64 indistinguishable).
**Still ~4× slower than patched fastsafetensors, fully tuned.**

## Verdict — root cause, and why tuning can't fix it

**5a's first-pass diagnosis (0.35 GB/s ≈ single-stream dd, "it's reading
with one stream") was a red herring, corrected in 5b.** The real tell is in
InstantTensor's own progress bar: it iterates **723 individual tensors**
(not 30 files) at ~2–4 it/s. 723 ÷ 3 ≈ 240 s — exactly what's measured at
*every* concurrency setting. The bottleneck is a fixed **~0.34 s
per-tensor** cost, almost certainly a per-tensor cross-rank
broadcast/sync on the TP=4 path.

fastsafetensors yields per **file** (30, i.e. 8 serial batches at TP=4).
InstantTensor yields per **tensor** (723 collectives instead of 8). Read
concurrency knobs cannot touch a per-tensor sync — which is exactly why
turning `concurrency`/`io_depth` up does almost nothing.

**Why the headline numbers (35–45 GB/s, 10-32×) don't transfer**: those are
GPUDirect Storage on local NVMe. GDS is unavailable on bristen (`nvidia_fs`
not loaded — see `../probes/results.md`); it falls back to `Backend.URING`
(confirmed via the library's own auto-selection, not assumed). With the
read path unaccelerated *and* a fixed per-tensor sync cost that concurrency
can't touch, there is nothing left to carry the pitch.

**VERDICT: structural, not configurational. Do not adopt sgl#28506 on this
platform.** Merging it and setting `--load-format instanttensor` here would
be a ~4–9× cold-start regression relative to the fastsafetensors+patch
recommendation, not an improvement.

## Caveats

- `it_c64` (5b) was killed by an external SIGTERM at 14:36:40, not a
  timeout or OOM, cause unattributed. Doesn't change the verdict:
  `it_c32` (245 s) and `it_c64_d64` (277 s) straddle it, and the plateau is
  already established by `c16 ≈ c32`.
- `safe_open(..., load_now=False)` segfaults on teardown in this library
  version — the PR path uses `load_now=True` (the default) and is
  unaffected, but worth flagging if anyone extends this backport.
- A hand-rolled `CUfileDrvProps` ctypes struct is not trustworthy for
  probing GDS status — `cufile.h` nests `size_t` fields, so an all-`c_uint`
  layout is misaligned and silently prints garbage. Ask the library which
  backend it picked (`Backend.URING` etc.) instead of parsing driver
  properties by hand.
