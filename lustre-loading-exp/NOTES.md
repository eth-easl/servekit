# Lustre-aware model loading — experiment log

Model: `/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct`
(~132 GB, 30 safetensors shards, capstor Lustre). Node: bristen A100, TP=4.

## Phase 1.1 — shard→OST map (native layout)  ✅

Ran `scripts/shard_ost_map.sh` → `results/phase1_shard_ost_map.txt`.

- Every shard: `stripe_count=1, stripe_size=1M`.
- 30 shards land on **24 distinct OSTs** (capstor has ≥150 OSTs; indices up to 149).
- **Hot OSTs** (host >1 shard, become the concurrency tail):
  - OST 30 → **3** shards (~13 GB serialized on one OST)
  - OST 15, 46, 59, 93 → 2 shards each
- **Implication**: concurrent multi-shard reads already spread over 24 OSTs, so
  the win from restriping is NOT "use more OSTs" — it's about killing the
  straggler tail on hot OSTs (either per-file striping so each big shard spans
  multiple OSTs, or a balanced re-copy). Phase 2 must test this specifically.

## Phase 1.2 — raw read ceiling (dd O_DIRECT, full-model workload)  ⏳ running

Tool: `scripts/dd_read_sweep.sh` (fio unavailable → `dd iflag=direct`). Constant
workload = all 30 shards (141 GB); N = worker-pool size draining the shard list.

**Full curve (job 73586, nid002313, COMPLETED, CONC=1 4 8 16 32 64):**
| N | full-model wall_s | agg GB/s |
|---|---|---|
| 1 | 405 | 0.35 |
| 4 | 175 | 0.80 |
| 8 | 81 | **1.75** (peak) |
| 16 | 128 | 1.10 |
| 32 | 209 | 0.68 |
| 64 | 202 | 0.70 |

Earlier partial run (job 73582) hit 0.91 @N=8 and 2.42 @N=16 — totally different.

### KEY FINDING: capstor bandwidth is contention-dominated & non-reproducible
- Single-stream is stable (~0.35–0.41 GB/s) but **aggregate scaling is not**:
  this run *peaks at N=8 (1.75 GB/s) then declines* with more concurrency; the
  earlier run scaled to 5.26 @N=16. The difference is **whole-cluster load on
  the shared capstor OSTs**, which varies minute to minute.
- N=32/64 ≈ N=30 here: one dd worker per shard caps concurrency at the 30 shard
  count. Going beyond needs **striping** (one shard spanning multiple OSTs so a
  single reader drives several OSTs) → motivates Phase 2.
- **Consequences**: (1) don't trust a single dd curve as "the ceiling"; treat
  each as one contended sample. (2) The real lever may be **decoupling from
  capstor contention entirely** — i.e. stage once to node-local /dev/shm
  (Phase 4), after which reads are contention-immune. (3) Compare strategies by
  end-to-end time under realistic (contended) conditions, with repeats.

- Single-stream ≈ **0.38 GB/s** (matches the earlier per-shard run's 0.41).
- **Concurrency scaling collapsed to a ~0.9 GB/s ceiling** here, vs the earlier
  per-shard run that reached 5.26 GB/s at N=16. Single-stream agrees, so the
  difference is **capstor is shared production storage with time-varying
  contention** — the achievable aggregate depends on whole-cluster load.
  → **Methodology consequence**: repeat each layout across multiple submissions/
  times; treat a single run as one contended sample, not the ceiling.

## Phase 2 — feasibility of write location  ✅ (checked, nothing created)

- I'm in group **infra01**; `/capstor/store/cscs/swissai/infra01` is
  `drwxrws--x+` → group-writable + setgid. **Can create
  `infra01/cold-start-experiments/`** (inherits infra01 group). Doesn't exist yet.
- **Right place**: same Lustre FS as the model (clean striping comparison, not a
  capstor-vs-iopsstor difference). Capstor has ≥150 OSTs (shard indices up to
  149); iopsstor scratch only ~20.
- **TO VERIFY before creating copies**: infra01 space quota — `lfs quota`/`df`
  returned nothing usable. Each striped copy ≈ 141 GB × several layouts.
- Code ready (not run): `scripts/make_striped_copy.sh`,
  `scripts/phase2_stripe_sweep.sbatch` (make copy → sweep → delete per layout).

## Phase 1.3 — end-to-end baseline  (mmap on/off + fastsafetensors)

### mmap baseline (job 73605, nid002325) — the anchor
total cold start **840.6 s**; breakdown:
| phase | s | % |
|---|---|---|
| **weight_loading** | **664.7** | **79%** |
| piecewise_cuda_graph_capture | 79.0 | 9% |
| cuda_graph_capture | 29.4 | 4% |
| process_startup | 28.7 | 3% |
| tp_worker_spawn | 16.7 | 2% |
| warmup(JIT) | 15.0 | 2% |
| torch_distributed_init | 2.5 | - |

- **Weight loading = 79% of cold start** for 70B. Effective BW = 132 GB / 665 s
  = **0.20 GB/s** — *slower than one dd O_DIRECT stream* (0.35) and ~9× under the
  contended dd peak (1.75). The mmap demand-paging path over Lustre is the
  culprit. → nommap + multithread + fastsafetensors should win big.
- (Compare Apertus-8B earlier: weight_loading 74 s. 70B mmap is ~9× worse.)

### three-way weight_loading comparison (native layout, TP=4)
| variant | weight_loading | eff BW | total | notes |
|---|---|---|---|---|
| mmap (73605) | 664.7 s | 0.20 GB/s | 840.7 s | nid002325 |
| no-mmap (73606) | 162.5 s | 0.81 GB/s | 331.2 s | nid002313 |
| fastsafetensors (73611) | 81.6 s | 1.62 GB/s | 242.7 s | nid002313, warm, ran last |
| **fastsafetensors (73612)** | **88.1 s** | **1.50 GB/s** | 258.5 s | **nid002804, COLD, ran FIRST** |

**Ranking is robust.** Re-running fastsafetensors on a cold node, first in time
(73612: 88 s) matched the warm/last run (73611: 82 s) within noise → the win is
NOT a page-cache or contention-order artifact. Confirmed:
**fastsafetensors ≈85 s  <  no-mmap ≈163 s  <  mmap ≈665 s.**

- fastsafetensors = **~8× faster than mmap**, ~2× faster than no-mmap on weights.
- Total 70B cold start 840 → ~258 s just by changing the loader (no striping,
  no staging yet). mmap default is by far the worst thing here.
- Even fastsafetensors' ~1.5 GB/s is below the dd contended peak (1.75) and far
  below what /dev/shm could give → Phase 2/4 headroom remains.
- Note: with the loader fixed, the graph-capture phases (cuda_graph 28 +
  piecewise 79 ≈ 107 s) become the *next* biggest chunk of cold start.

### REVERSE-ORDER RERUN + post-ready benchmark (servekit `--bench`)
Run order reversed (fastsafetensors FIRST, mmap LAST), each pinned to a distinct
COLD node, serialized so nothing contends on capstor.

| loader | node | weight_s | total_s | tok/s | p50 s | errs |
|---|---|---|---|---|---|---|
| fastsafetensors (73620) | nid002805 | **73.7** | **237.2** | 402.3 | 5.09 | 0 |
| nommap (73621) | nid002808 | 161.9 | 348.2 | 401.3 | 5.10 | 0 |
| mmap (73622) | nid002809 | **939.0** | 1123.2 | 402.0 | 5.09 | 0 |

**Conclusions (all three confounders now ruled out):**
1. **Ordering/cache confound eliminated.** fastsafetensors ran first (worst
   position if contention eases over time) and still won; mmap ran last and got
   *worse* (939 s vs 664.7 s). Gap is now **12.7×** on the weight phase.
2. **Correctness verified — no weight corruption.** Greedy outputs are
   character-for-character identical across all three loaders (e.g. "The capital
   of France is" and "def fibonacci(n):" produce the same continuations).
   fastsafetensors loads bit-equivalent weights.
3. **Throughput identical (~402 tok/s, p50 5.09 s, 0 errors) across all three.**
   The loader affects only time-to-serving, not runtime performance.
4. **Reproducibility differs sharply**: nommap 162.5 → 161.9 s (rock steady);
   mmap 664.7 → 939.0 s (wild). Demand-paged mmap is the most
   contention-sensitive access pattern on Lustre — slow *and* unpredictable.

**Recommendation so far**: `--load-format fastsafetensors` is a one-flag,
plug-and-play change taking 70B cold start from ~1123 s (mmap worst case) to
~237 s (**4.7× total**), with identical outputs and throughput. Never use the
mmap default on Lustre.

## Phase 2 — layout sweep  ⏳
## Phase 3 — Strategy A (striped copy)  ⏳
## Phase 4 — Strategy B (/dev/shm)  ⏳
## Phase 5 — comparison + recommendation  ⏳
