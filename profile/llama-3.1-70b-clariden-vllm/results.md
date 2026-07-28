# vLLM cold start — Llama-3.1-70B, TP=4, Clariden (GH200)

First vLLM profile with servekit. Config mirrors `experiments/clariden-loading-exp`
(TP=4, 32768 context, 0.85 memory fraction, 256 concurrent requests) so it sits
next to the SGLang baseline for the same model on the same hardware.

Headline: **353.34 s to ready, 820.8 tok/s** (job 2918061, nid006183).

## The image is the finding

The obvious choice does not work. `vllm/vllm-openai:v0.25.0` is a multi-arch
manifest whose arm64 half imports, passes an arch check and loads the model —
then dies at CUDA graph capture, at exactly 32/51 PIECEWISE captures, twice:

```
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED
  when calling cublasGemmEx(... CUDA_R_16BF ...)
```

in an inductor-compiled kernel (`extern_kernels.mm`) during the warmup forward
that precedes capture. The `illegal memory access` that follows is the shutdown
symptom, not the cause — chasing it to `--disable-custom-all-reduce` (job 2917951)
changed nothing.

`nvcr.io#nvidia/vllm:26.07-py3` — NVIDIA's own container, built and validated on
Grace-Hopper — runs the identical command to completion. **On GH200, use the NGC
image.** Note it ships vLLM 0.24.0.dev, i.e. *older* than the upstream tag that
fails, so this is a build/toolchain difference, not a version fix.

## Phases

| phase | vLLM 2918061 | SGLang 2917854 | note |
|---|---|---|---|
| process_startup | 14.80 | 14.84 | — |
| worker spawn + dist init | 45.42 | 20.10 | SGLang splits these (13.69 + 6.41) |
| weight_loading | **233.24** | **510.27** | engine-reported, max over 4 ranks |
| compile | 35.70 | — | vLLM only (`torch.compile`) |
| graph capture | 11.00 | 65.38 | SGLang: 19.42 full + 45.96 piecewise |
| kv_cache_alloc | 4.24 | 0.57 | — |
| api_server_startup | 6.65 | — | SGLang has no equivalent gap |
| http_bind + warmup request | — | 12.14 | vLLM issues no warmup request |
| unattributed gap | 2.22 | 1.28 | left `unknown` rather than assumed |
| **total** | **353.34** | **624.58** | |
| non-load (total − weight_load) | 120.03 | 114.31 | |
| throughput | 820.8 tok/s | 811.5 tok/s | 64/64 reqs, 0 errors |

## Reading it

**Do not read 353 vs 625 as an engine result.** The gap is almost entirely
`weight_loading`, and that phase's spread on this storage is enormous — the
bristen reference for the same SGLang config ranged 430–939 s across 4 runs.
n=1 per engine here.

**The comparable number is non-load: 120.03 vs 114.31 s** — within ~5%. Two
engines, same hardware, same model, near-identical fixed cost. But the
composition differs sharply:

- vLLM spends **45.42 s** on worker spawn + distributed init against SGLang's
  20.10 s. This is the largest genuine engine difference in the table, and it
  points at the same suspect as the deferred import-startup work in `CLAUDE.md`.
- SGLang spends **65.38 s** capturing graphs; vLLM spends **46.70 s** on
  compile + capture combined.

**The totals are not measured to the same boundary.** SGLang announces ready
only after its own warmup request (10.83 s of the SGLang total is first-call
JIT). vLLM never issues one, so its total stops earlier. `ready_wait_s` closes
the gap: 0.076 s for vLLM, i.e. it really was serving at that point — vLLM's
lazy init is genuinely cheaper here, not merely deferred past the boundary.

**Hypothesis, untested:** vLLM 0.24 prefetches checkpoint files into page cache
in background threads (38 such log lines, 8 threads, 16 MiB blocks, finishing in
9–11 s) while loading. That is a weak form of what the preshard/`/dev/shm`
experiment does deliberately, and it is a candidate explanation for the 233 s
load sitting below SGLang's entire observed range. Confirming it needs repeated
runs on fresh nodes, which this single run cannot support.

## Parser notes

vLLM's log has no milestone servekit can time between HTTP bind and ready: its
uvicorn never prints `Uvicorn running on`, and `Application startup complete` is
the last line of startup. The `http_bind` / `first_request` milestones were
therefore removed from the vLLM spec rather than left unable to fire.

The 6.65 s before ready is **`api_server_startup`** — everything after the
engine is up: Triton JIT monitor activation, the `init engine … took 56.09 s`
summary, async scheduling and fusion config, chat-template detection, then
`Starting vLLM server on http://0.0.0.0:8080` and route registration. Named off
those last two markers.

The remaining 2.22 s (last rank's weight-load report → the compile timer
starting) stays `unknown` on purpose. The only line in it is `Using FlashInfer
for top-p & top-k sampling`, which is conditional on FlashInfer being present
and does not plausibly account for the whole interval — one weak marker is not
evidence, so it is not named.

That `init engine (profile, create kv cache, warmup model) took 56.09 s` line is
also why `engine_init` is not a phase: it spans kv_cache_alloc + compile +
capture (4.24 + 35.70 + 11.00 = 50.94 plus gaps), so counting it would
double-count phases already measured.

The trimmed log is `servekit/tests/fixtures/llama70b-vllm-ngc.log`, asserted in
`test_parses_real_vllm_log`.

## Raw artifacts

`logs/llama-70b-vllm-ngc-2918061.out`,
`logs/llama-70b-vllm-ngc-2918061-nid006183-profile.json`. Failed upstream-image
runs: jobs 2917848, 2917951.

## Reproducing

```bash
sbatch profile/llama-3.1-70b-clariden-vllm/preflight_vllm.sbatch     # arch gate
sbatch --export=ALL,VLLM_EDF=$PWD/profile/llama-3.1-70b-clariden-vllm/llama-3.1-70b-vllm-ngc.toml \
       --job-name=llama-70b-vllm-ngc \
       profile/llama-3.1-70b-clariden-vllm/serve_llama70b_vllm.sbatch
```

One fresh node per run: the page cache survives container runs.
