# PP loading experiment — summary

**Question:** the weight-loading technique from `clariden-loading-exp` was
measured only at TP=4, PP=1. Does it carry over to **pipeline parallelism**?

**Answer: yes, but not with the loader that round used.** SGLang's
`ShardedStateLoader` keys files on the TP rank alone and silently corrupts at
PP>1. Upstream's four-day-old `PreshardedModelLoader` handles it correctly, and
on Clariden GH200 it takes weight loading to **~1 s** at both TP=1/PP=4 and
TP=2/PP=2, serving byte-identical output.

Apertus-8B, 4× GH200, one node, `lmsysorg/sglang:nightly-dev-20260729-16a52bff`:

| config | arm | node | total | weight_load | non-load | tok/s |
|---|---|---|---|---|---|---|
| **TP=2 PP=2** | default | nid007666 | 136.39 s | 43.42 s | 92.97 s | 1429.6 |
| | presharded+shm+overlap | nid007661 | **96.39 s** | **0.89 s** | 95.41 s | 1457.7 |
| | | | **1.41x** | **49x** | *flat (2.6%)* | — |
| **TP=1 PP=4** | default | nid006457 | 187.80 s | 46.41 s | 141.39 s | 481.1 |
| | presharded+shm+overlap | nid007032 | 82.19 s | **1.05 s** | 81.05 s | 1047.0 |
| | | | *see below* | **44x** | *not flat* | — |

**Only the TP=2 PP=2 row supports a total-cold-start claim.** Its `non-load` is
flat across arms (92.97 vs 95.41, 2.6% apart), which is the check that the
technique moved only the phase it targets. The TP=1 PP=4 pair failed that check
badly — 141.39 vs 81.05 — because *every* phase was ~2x faster on nid007032,
including `process_startup` (5.17 → 2.02) and `cuda_graph_capture`
(23.09 → 16.71), which no loader can influence. Across the three TP=1 PP=4 runs
`non-load` was 141.39 / 81.05 / 63.67 s on three different nodes: a 2.2x spread,
wider than the effect being measured. So its 44x weight-loading result stands,
its 2.29x total does not, and the 481 → 1047 tok/s is node variance — the same
trap `clariden-loading-exp` had to correct.

## What this settles

- **`sharded_state` cannot do PP, and fails silently.** `loader.py:1415,1470`
  both key on `get_tensor_model_parallel_rank()`, identical on every pipeline
  stage, against a `DEFAULT_PATTERN` with one `{rank}` field. The stages write
  disjoint tensors to one filename (last writer wins) and the load then fails on
  keys the stage does not own. Nothing in `server_args.py` rejects it.
- **`PreshardedModelLoader` does it correctly, at both TP=1 PP=4 and TP=2 PP=2.**
  Verified from `checksum.json`, not inferred: the stages tile the layers
  exactly, with no overlap and no duplication.

  ```
  TP=1 PP=4   4 stages: [0-7] [8-15] [16-23] [24-31]      16.11 GB
  TP=2 PP=2   2 stages x 2 TP peers: [0-15] [16-31]       16.11 GB
  ```
  Both equal the 16 GB model, so no stage redundantly stores another's layers.
- **The technique composes with PP unchanged.** `/dev/shm` staging plus overlap
  works as it did at TP=4: stage 1.25–1.36 s at 14.3–14.5 GB/s, hidden entirely
  inside startup, gate VALID with 52–68 s of slack both times.
- **Output is bit-exact.** 6 greedy probes byte-identical between the HF loader
  and the presharded loader, in both parallelism configs.
- **Content dedup does real work.** 32 identical `act_fn` scalars collapse to one
  98-byte tensor; `lm_head.weight` is materialised on all four stages and stored
  once.

## Costs and caveats

- **The loader is main-only.** Absent from v0.5.10–v0.5.16; merged 2026-07-25,
  ~20 h after v0.5.16 was cut. Using it means tracking a nightly.
- **Two upstream defects block the arm64 nightly out of the box** (details in
  `results/apertus-8b/results.md`): the aarch64 `sgl_kernel` ships no FA3 while
  the engine auto-selects `fa3`; and `triton_backend.py:203` hardcodes
  `get_value_buffer(0)`, so at PP>1 every stage but the first raises
  `IndexError`. This round runs `--attention-backend flashinfer` for that
  reason, which also means throughput here is not comparable to
  `clariden-loading-exp`.
- **`n=1` per config**, and node variance on this partition is large enough to
  swamp a 1.4x effect. Anything finer-grained needs repeats.
- **`-common` files are read by every rank**, so a multi-node staging plan must
  replicate them rather than partition them — 1.07 GB of 16 GB here.
- **Not tested:** multi-node PP, and the 70B. Both deliberately out of scope.

## servekit fix landed here

SGLang renamed the graph-capture log line between v0.5.10 and these nightlies
(`Capture cuda graph end. Time elapsed:` → `Capture target decode CUDA graph
end. elapsed=`), so ~16 s of a 121 s cold start was silently falling into the
unaccounted gap. Both spellings are now in `profile.py:46-54`, verified not to
cross-match, so `clariden-loading-exp`'s profiles still parse. The preflight's
regex check is what caught it.

Per-run detail: `results/apertus-8b/results.md`. Design: `PLAN.md`.
