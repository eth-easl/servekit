# Apertus-8B, TP=1 PP=4 — preflight

## Run 1 — job 2922510, nid006751, `nightly-dev-20260729-16a52bff`

**FAILED, on the attention backend, not on anything this round is testing.**

All four schedulers died immediately after weight loading:

```
[PP0..PP3] Scheduler hit an exception:
    from sgl_kernel import flash_ops
ImportError: cannot import name 'flash_ops' from 'sgl_kernel'
ImportError: Can not import FA3 in sgl_kernel. Please check your installation.
```

The engine had chosen it itself — `Attention backend not specified. Use fa3
backend by default.` → `attention_backend='fa3'`. So the aarch64 `sgl_kernel`
wheel in this nightly does not ship FA3, and the default backend selection does
not check that before committing to it.

### What passed before it died

Worth keeping, because none of it is in doubt any more:

| check | result |
|---|---|
| image runs on GH200 | yes — aarch64, sm_90, 4× GH200 120GB, torch 2.11.0+cu130 |
| `LoadFormat.PRESHARDED` present | yes; `PreshardedModelLoader` imports |
| PP=4 takes effect | `pp_size=4`, `disable_overlap_schedule=True`, stages log as `PP0`–`PP3` |
| weight loading at PP=4 | completed: 24.40 / 26.51 / 28.46 s per stage (PP3 not reached before the crash) |
| servekit regexes still match | yes — `Init torch distributed ends. elapsed=` and `Load weight end. elapsed=` both parsed against this build |
| `BASH_ENV` | **set**, to `/usr/share/lmod/8.7.34/init/bash` — the stager must run under `env -u BASH_ENV` |

`Parameter model.embed_tokens.weight not found in params_dict` on PP1/PP2/PP3
and `model.norm.weight not found` on PP0/PP1/PP2 are expected: only the first
stage owns the embedding and only the last owns the final norm. The default
loader warns and skips.

### Also found: a flaw in this experiment's own scripts

The job stayed RUNNING for six minutes after the server was dead, because
`servekit bench --wait-ready 3000` kept polling a server that would never come
up — and it held the single debug-partition slot while doing so. All three
sbatch scripts now wait for either the profile report or the profile process to
exit, and skip the bench when `success` is false.

## Run 2 — job 2922525, nid007049, `nightly-dev-cu12-20260729-16a52bff`

Narrow question: is the missing FA3 specific to the CUDA-13 build, or true of
the aarch64 kernel wheel generally? Imports only, no model.

**Not specific to CUDA 13.** The cu12 build of the same commit (torch
2.11.0+cu129) has the same gap:

```
has flash_ops: False
  fa3           import OK
  flashinfer    import OK
  triton        import OK
  torch_native  import OK
```

So FA3 is absent from the aarch64 `sgl_kernel` wheel in both CUDA variants of
this nightly, and switching image variant does not help. Note the `fa3` line is
misleading on its own: the backend *module* imports fine, because
`from sgl_kernel import flash_ops` happens deeper, at backend construction —
which is why job 2922510 got all the way through weight loading before dying.
`has flash_ops: False` is the load-bearing line.

The other three backends are importable and are the only way this round
proceeds on a nightly.

## Run 3 — job 2922719, `--attention-backend triton`

**FAILED, and this one is a genuine PP bug in SGLang.** Got much further: past
backend selection, through weight loading (32.73 s), into CUDA graph capture.
Then PP2 and PP3 died at scheduler construction:

```
triton_backend.py:203  self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[...]
memory_pool.py:2253    return self.v_buffer[local_layer_id]
IndexError: list index out of range
```

`get_value_buffer(0)` is a **hardcoded layer 0**, evaluated once at backend
construction. Only pipeline stage 0 owns layer 0; stage 3 owns layers 24–31 and
maps global 0 to a negative local index. PP0 survived and reached graph capture;
the other stages did not.

`triton_backend.py:203` is the only hardcoded layer id across all four attention
backends on main — every other call site uses `layer.layer_id`, which the KV
pool maps correctly. So this is specific to Triton, and flashinfer should be
unaffected.

## Run 4 — job 2922770, `--attention-backend flashinfer` — **PREFLIGHT PASS**

```
phases: process_startup=0.46, unknown=2.92, torch_distributed_init=7.05,
        weight_loading=57.66, http_bind=20.52, warmup_request(JIT)=6.61
total = 121.33 s
bench: 304.0 tok/s, 8/8 ok, errors=0
```

PP=4 serves correctly on the nightly. The gate is cleared; the measured runs can
proceed. `piecewise_cuda_graph_capture` is absent as predicted for `pp_size>1`.

### servekit fix this preflight earned

The phases summed to 95.2 s against a 121.3 s total. SGLang renamed the graph
capture line between v0.5.10 and these nightlies:

```
v0.5.10   Capture cuda graph end. Time elapsed: 22.74 s
nightly   Capture target decode CUDA graph end. elapsed=16.40 s
```

so ~16 s was silently falling into the unaccounted gap — exactly the failure the
preflight's regex check exists to catch, and one that would have been invisible
in the results. Both spellings are now in
`servekit/src/servekit/profile.py:46-49`, verified to match their own format and
not each other, so `clariden-loading-exp`'s profiles still parse.

## Run 5 — job 2922836, dump — **DUMP CHECK PASS**

**The round's core question, answered: `PreshardedModelLoader` shards correctly
across pipeline stages.**

```
subfolder: TP-1-sig-bf6babb08ad0b807
world_size=4  files=7  total=16.11 GB
  rank 0:  81 private,  3 shared, writes 5.64 GB
  rank 1:  80 private,  3 shared, writes 3.49 GB
  rank 2:  80 private,  3 shared, writes 3.49 GB
  rank 3:  81 private,  3 shared, writes 3.49 GB
OK: all 4 stages hold disjoint private tensor sets
```

16.11 GB total against a 16 GB model, so no stage redundantly stores another's
layers — the thing `sharded_state` structurally cannot do at PP>1.

```
READY                              70 B
checksum.json                     164 KB
model-00000-rank-000.safetensor   4.56 GB
model-00001-common.safetensor     1.07 GB
model-00002-common.safetensor       98 B
model-00003-common.safetensor       98 B
model-00004-rank-001.safetensor   3.49 GB
model-00005-rank-002.safetensor   3.49 GB
model-00006-rank-003.safetensor   3.49 GB
```

Reading `rank_to_names` out of the plan confirms the split is real, not just
disjoint bookkeeping:

- **layer assignment is exactly the PP split** — `act_fn` params land on rank 0
  for layers 0–7, rank 1 for 8–15, rank 2 for 16–23, rank 3 for 24–31.
- **content dedup is doing real work.** Those 32 `act_fn.beta` values are
  identical, so all 32 collapse to one 98-byte stored tensor referenced by 8
  names per stage. Same for `act_fn.eps`.
- **`lm_head.weight` is materialised on all four stages**, under that name on
  each — not only the last. 1.07 GB logically replicated ×4, stored once.

### Consequence for the deferred multi-node case

`-common` files are read by every world rank, so a per-node staging plan must
place them on **every** node, not partition them. Here that is 1.07 GB against a
16 GB model. `checksum.json`'s `rank_to_reads` gives the exact per-rank file
list, so the split is computable — but it is not a clean partition, and any
multi-node stager has to special-case the common set.

## Runs 6–7 — TP=1 PP=4 measured arms

| | default (2922859, nid006457) | presharded (2922893, nid007032) |
|---|---|---|
| weight_loading | 46.41 s | **1.05 s** |
| total | 187.80 s | 82.19 s |
| non-load | 141.39 s | 81.05 s |
| tok/s | 481.1 | 1047.0 |

Gate VALID (READY beat the loader by 52.51 s; stage 1.25 s @ 14.51 GB/s), 64/64,
errors=0, **probes byte-identical**.

**The total is not attributable to the loader.** Every phase was ~2x faster on
nid007032, including ones the loader cannot touch:

| phase | nid006457 | nid007032 | ratio |
|---|---|---|---|
| process_startup | 5.17 | 2.02 | 2.6x |
| unknown (imports+spawn) | 82.45 | 41.76 | 2.0x |
| torch_distributed_init | 12.24 | 6.99 | 1.8x |
| kv_cache_alloc | 7.81 | 3.59 | 2.2x |
| cuda_graph_capture | 23.09 | 16.71 | 1.4x |
| weight_loading | 46.41 | 1.05 | 44x |

`non-load` across the three TP=1 PP=4 runs on three nodes: 141.39 / 81.05 /
63.67 s — a 2.2x spread. So the 44x on weight_loading stands, the 2.29x total
does not, and 481 → 1047 tok/s is node variance, not staging.

## Runs 8–9 — TP=2 PP=2, the mixed case

Dump (job 2924462) then both arms.

| | default (2925092, nid007666) | presharded (2926018, nid007661) |
|---|---|---|
| weight_loading | 43.42 s | **0.89 s** (49x) |
| total | 136.39 s | **96.39 s** (1.41x) |
| non-load | 92.97 s | 95.41 s (**flat, 2.6%**) |
| tok/s | 1429.6 | 1457.7 |

Gate VALID (68.31 s slack; stage 1.36 s @ 14.25 GB/s), 64/64, errors=0,
**probes byte-identical**.

Unlike the TP=1 PP=4 pair, `non-load` is flat here, so **this row does support a
total-cold-start claim**: 1.41x, with the gain confined to weight loading.

This is the harder test of the two. At TP=1 only the PP rank varied, so a loader
keyed on either field alone could still look right. At TP=2 PP=2 the four world
ranks are `(tp0,pp0) (tp1,pp0) (tp0,pp1) (tp1,pp1)` — every rank shares a
coordinate with two others, so single-field keying collides.

### A bug in this experiment's checker, not in the loader

The first TP=2 PP=2 dump check reported `DUMP CHECK FAIL` on three counts. All
three were mine:

- expected world size was passed as `PP_SIZE` (2) instead of `TP*PP` (4);
- "no two ranks may share a tensor" is only valid at TP=1. With TP>1 the TP
  peers inside a stage hold identical copies of every TP-**replicated** param
  (layernorms), which content dedup correctly stores once — reported as
  `rank 0 and rank 1 share 96 tensors`. The `{0,2}`/`{1,3}` single-tensor
  overlaps are the TP-split `lm_head` halves, replicated across stages.

`inspect_dump.py` now checks the invariant that actually matters for PP — that
ranks group into exactly `pp` distinct layer sets of `tp` ranks each, tiling
`0..L-1` without overlap — and derives the grouping from the data rather than
assuming SGLang's world-rank-to-`(pp,tp)` mapping. Both dumps pass it.

## Upstream defects

Two, both found on the way to the actual question, neither in the loader:

1. **The aarch64 nightly defaults to an attention backend it cannot run.**
   `Attention backend not specified. Use fa3 backend by default.` →
   `ImportError: cannot import name 'flash_ops'` after weight loading. The
   default selection does not check FA3 is present in the installed
   `sgl_kernel`. Reproduces on both `nightly-dev-20260729-16a52bff` and
   `nightly-dev-cu12-20260729-16a52bff`, aarch64, GH200. Makes every arm64
   nightly fail out of the box.
2. **The Triton attention backend is broken at `pp_size > 1`.**
   `triton_backend.py:203` hardcodes `get_value_buffer(0)` at construction, so
   every pipeline stage but the first raises `IndexError`. Independent of
   architecture and of this round's loader.

Both are worth reporting upstream. The second is the more interesting one for
this project: it is the kind of defect that exists because PP is under-exercised
upstream, which is the same reason `PreshardedModelLoader` ships with no
end-to-end PP test.
