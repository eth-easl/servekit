# Baseline — true-cold Apertus-8B launch (TP1)

**Status:** done

## Goal

Establish the number that a checkpoint/restore has to beat: how long does a genuinely
cold Apertus-8B SGLang launch take on this box, phase by phase? Without this, "18 s to
restore" has nothing to compare against — and a naive "just launch it again" measurement
would be dishonestly fast, because a second launch reuses whatever the first launch left
warm in the page cache and the JIT/compile cache.

## Method

`scripts/baseline.sh` wraps the same launch command the cluster uses
(`profile/apertus-8b-bristen/serve_apertus_8b_sglang.sbatch`, knobs reduced to fit a
24 GB RTX 3090, TP1) with `servekit profile` (phase breakdown) + `servekit bench`
(correctness + throughput), merged into one report JSON.

`--cold` controls **two** confounds that a lazily-repeated launch would otherwise hide:

1. **Page cache (weights)** — warm weights make `weight_loading` ~3x too fast. Evicted
   via `scripts/cache_tools.py` (unprivileged `posix_fadvise` + `mincore` verify).
2. **JIT/compile cache** — the *first* launch ever compiles flashinfer/triton/sglang
   kernels (`~/.cache/*`), which dominates `cuda_graph_capture` + `warmup` (~38 s the
   first time, ~3 s once cached). The cluster neutralizes this with an ephemeral
   `HOME=/root` per job; here, `--cold` redirects `FLASHINFER_WORKSPACE_BASE`,
   `SGLANG_CACHE_DIR`, `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and
   `XDG_CACHE_HOME` to a wiped throwaway dir (`.cold-compile-cache/`) instead of
   touching the user's real `~/.cache`.

A true cold node (fresh deploy) has **both** confounds cold — that's what `--cold`
reproduces, and what a checkpoint/restore must be honestly compared against.

## Result

Raw data: `baseline-*-profile.json`, `baseline_*.log`. Phase breakdown, TP1, ctx 8192:

```
process_startup + imports   ~14.2 s
weight_loading                6.3 s   (cold read, 16 GB from NVMe)
cuda_graph_capture           17.0 s   (mostly first-touch flashinfer kernel COMPILE)
piecewise_cuda_graph_capture 17.9 s   (torch.compile eager, runs every launch)
warmup(JIT)                   1.7 s
────────────────────────────────────
total (ready)                59.3 s   |  throughput ~278 tok/s
```

Reproducible to ±0.3% across repeated `--cold` runs.

## Verdict

**True-cold reference: 59.3 s.** Note `cuda_graph_capture` is dominated by kernel
*compilation*, not graph capture itself — warm-JIT it collapses to ~2 s, which is why
controlling the compile-cache confound (not just the page cache) matters. This is the
number [`results/gate3b_server_checkpoint_restore/results.md`](../gate3b_server_checkpoint_restore/results.md)
compares its 18.2 s restore against.

## Caveats

TP1 only (single 24 GiB GPU); the cluster runs TP4 for this model — phase magnitudes
(especially NCCL/distributed-init cost, folded into `process_startup` here at TP1) are
not assumed to carry over 1:1 to TP4.
