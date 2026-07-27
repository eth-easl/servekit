# Reference — Apertus-8B cold-start benchmark

**Status:** done

> Not a phase in its own right — a reference data point NOTES.md mentions
> only in passing ("Compare Apertus-8B earlier: weight_loading 74 s").
> Reconstructed here directly from the raw profile JSON.

## Goal

Establish a smaller-model reference point (Apertus-8B, not the 70B target)
for cold-start phase breakdown, predating the Llama-3.1-70B investigation.
Used as a sanity check on the servekit profile tool and as a scale
comparison for how much larger weight_loading becomes at 70B.

## Method

`scripts/reference/test_bench_apertus8b.sbatch`, `servekit profile --
python -m sglang.launch_server` against Apertus-8B, TP presumably 1 (single
job, nid002313), same profiling + `--bench` methodology as the 70B
experiments.

## Result

Job 73619, nid002313:

| phase | s |
|---|---|
| process_startup | 18.53 |
| tp_worker_spawn | 15.70 |
| torch_distributed_init | 3.00 |
| unknown | 1.87 |
| **weight_loading** | **63.66** |
| kv_cache_alloc | 0.53 |
| cuda_graph_capture | 24.09 |
| piecewise_cuda_graph_capture | 29.37 |
| http_bind | 2.17 |
| warmup_request(JIT) | 13.72 |

Throughput: 64/64 requests, 0 errors, **1971.9 tok/s**, p50 1.04 s (far
higher than 70B's ~401 tok/s, as expected for an 8B model).

## Verdict

`weight_loading` ≈ 64–74 s for an 8B model (~16 GB) vs the 70B mmap baseline
of 665–939 s (Phase 1.3) — the 70B default-loader case is roughly **9× worse**
even accounting for the ~8× larger model, meaning mmap's demand-paging
penalty scales worse than linearly with model size on this storage. This
reference point is what first flagged weight loading as disproportionately
expensive at 70B and motivated the rest of this experiment directory.

## Caveats

- Single run, no fresh-node repeats or bracket — this is a scale reference,
  not a rigorously measured baseline like Phase 1.3's 70B numbers.
