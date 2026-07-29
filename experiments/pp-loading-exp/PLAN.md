# PP loading experiment — can the technique shard across pipeline stages?

**Question:** `clariden-loading-exp` removed essentially all of weight loading
at **TP=4, PP=1** (70B 4.61x, 8B 1.81x). Does that carry over to **pipeline
parallelism**, and what does it buy?

## Why it does not carry over for free

SGLang's `ShardedStateLoader` — the loader every result in the last round used —
keys both save and load on the TP rank alone:

```
loader.py:1323  DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"
loader.py:1415  rank = get_tensor_model_parallel_rank()   # load
loader.py:1470  rank = get_tensor_model_parallel_rank()   # save
```

`get_tensor_model_parallel_rank()` is `get_tp_group().rank_in_group`, identical
on every pipeline stage. Each stage holds a different set of layers (global
indices, `PPMissingLayer` padding — `utils/common.py:642`), so at PP>1 the
stages write disjoint tensors to *one* filename: the dump races and the last
writer wins, and the load then fails on keys the stage does not own. Nothing in
`server_args.py` rejects the combination, so it fails silently.

## What this round uses instead

Upstream fixed it four days before this experiment.
`PreshardedModelLoader` ([PR #24256](https://github.com/sgl-project/sglang/pull/24256),
merged 2026-07-25) keys on `get_world_group().rank_in_group` and all-gathers the
structural signature so the cache key agrees across stages that legitimately
differ. It is in **no release** — absent from v0.5.10 through v0.5.16 — so this
round runs a pinned nightly, `lmsysorg/sglang:nightly-dev-20260729-16a52bff`
(arm64, commit verified to contain `LoadFormat.PRESHARDED`).

That makes the engine version a second variable, so the baseline is **PP=4 on
the same nightly**, not last round's v0.5.10 numbers. Nothing here is meant to
be compared against 586.33 → 127.06 s.

`presharded` also does more than `sharded_state`: it caches state *after*
`process_weights_after_loading`, so a reload skips source-shard iteration,
`post_load_weights` and quant reshape.

## Design

| | |
|---|---|
| parallelism | `TP=1 PP=4`, 4 GPUs, one node |
| models | `apertus8b` to shake out the path, `llama70b` for the number |
| baseline | `--load-format auto`, PP=4, same image, fresh node |
| treatment | `--load-format presharded` from `/dev/shm`, staged overlapped, fresh node |
| held constant | ctx 32768, mem-fraction 0.85, max-running-requests 256 |

Multi-node PP is explicitly **out of scope**; this round only establishes that
sharding across pipeline stages works at all and what it is worth.

### The staging order matters

`PreshardedModelLoader` gates on a `READY` sentinel. The overlapped stage
therefore copies `checksum.json` synchronously, stages `*.safetensor` in the
background, and creates `READY` **last**.

A loader that arrives before `READY` does not read half-staged bytes — it takes
a cache miss, which means a full HF load *and a re-dump into `/dev/shm`*. Safer
than `sharded_state`'s silent corruption, but for the 70B that is 141 GB of
tmpfs writes on top of the staged copy. The validity gate is mandatory.

## Jobs

```
./submit.sh apertus8b preflight                      # the gate — debug partition
./submit.sh <preset>  dump                           # one-off, offline
./submit.sh <preset>  default                        # note the node id
./submit.sh <preset>  presharded --exclude=<node>    # must be a different node
```

Preflight checks four things that can each ruin a measured run silently: the
nightly runs on GH200 at all; `--load-format presharded` with PP=4 reaches
ready; **servekit's profile regexes still match** (`profile.py:43-54` scrapes
literal SGLang log strings written against v0.5.10, and this image is months
ahead); and whether the image sets `BASH_ENV`.

If preflight fails, the round stops and gets written up — there is no fallback
to patching v0.5.10.

## What has to be true

1. **`checksum.json` shows 4 disjoint stages.** `scripts/inspect_dump.py` runs
   after every dump: `rank_to_reads` covers ranks 0..3, every stage holds
   private tensors, and no two stages share one outside a `-common` file. This
   is the round's real result and does not depend on any stopwatch.
2. Overlap gate **VALID** — `READY` written before weight loading began.
3. `errors=0`, `64/64` completed, throughput not materially below the default arm.
4. **`probe.py`'s 6 greedy completions byte-identical** between the two arms.
   Throughput alone proves nothing here: a model assembled from mismatched
   stages still emits fluent text at full speed.

Report `total`, `weight_loading` and `non-load = total − weight_loading` for
both arms; `non-load` flat across arms is the check that the technique moved
only the phase it targets.

## Deviations from this plan, as executed

- **`--attention-backend flashinfer` was added.** Not a tuning choice: the
  aarch64 nightly ships no FA3 while auto-selecting it, and Triton is broken at
  PP>1. Both arms use it, so the loader comparison is unaffected. See
  `scripts/models.sh` and `results/apertus-8b/results.md`.
- **TP=2 PP=2 was added**, via `TP_SIZE_OVERRIDE` / `PP_SIZE_OVERRIDE`, to
  exercise both rank fields at once.
- **All jobs run on `--partition=debug`**, including the measured ones — 1:30:00
  is well above the 1:00:00 they request, and it schedules far sooner.

## Known limits going in

- `n=1` per config. Node-to-node spread on Clariden reached 1.7x on non-loading
  phases last round, so read small differences through the load/non-load
  decomposition, not from `total`.
- PP's cold-start shape differs from TP=4 by construction: `pp_size > 1` forces
  `disable_overlap_schedule` (`server_args.py:2853`) and disables piecewise CUDA
  graph (`:1098`), so `piecewise_cuda_graph_capture` is expected to be absent.
- The loader is four days old and has no end-to-end PP test upstream; its unit
  tests use fake process groups with `pp=1`. Finding its PP bugs is part of the
  point.
