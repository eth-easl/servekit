# Phase 1 — `servekit launch`, stage-then-start, free on ready

**Question:** does the packaged wrapper reproduce the experiment's speedup with
no overlap, no engine hook, and no hand-run scripts — and does freeing the tmpfs
copy at ready actually give the RAM back?

**Answer: yes to both.** 4.64x on the 70B, 1.79x on Apertus-8B, RAM fully
returned for both the `sharded_state` and the default mmap loader. The
un-overlapped stage costs less than Phase 2 was budgeted to buy back.

One command, prepended to the command the user already had:

```bash
servekit launch -- python -m sglang.launch_server --model-path <ckpt> ...
```

## Results

One run per config, each on a fresh node. SGLang v0.5.10, TP=4, ctx 32768,
mem-fraction 0.85. Baselines are `experiments/clariden-loading-exp`'s, not
re-measured here — the model paths and every engine flag are sourced from that
experiment's `models.sh` so the two cannot drift apart.

| model | loader | job / node | stage | weight_load | total | vs default | tok/s |
|---|---|---|---|---|---|---|---|
| **Llama-3.1-70B**<br>141 GB, 30 shards | *default (capstor, mmap)* | — | — | 466.81 s | 586.33 s | — | 822.9 |
| | presharded + shm | 2922381 / nid006119 | 4.98 s @ 28.34 GB/s | **5.80 s** | **126.26 s** | **4.64x** | 800.1 |
| | stock + shm (mmap) | 2922382 / nid006044 | 4.29 s @ 32.89 GB/s | **7.79 s** | **128.40 s** | **4.57x** | 814.9 |
| **Apertus-8B**<br>16 GB, 4 shards | *default (capstor, mmap)* | — | — | 81.41 s | 172.68 s | — | 2825.7 |
| | presharded + shm | 2922371 / nid006073 | 0.98 s @ 16.45 GB/s | **0.98 s** | **96.35 s** | **1.79x** | 2767.1 |
| | stock + shm (mmap) | 2922383 / nid006622 | 1.34 s @ 12.03 GB/s | **1.20 s** | **98.10 s** | **1.76x** | 2780.9 |

Every run: 64/64 requests, errors=0, and a second bench a minute after the free
that did not drop (800.1 → 1064.3 tok/s on the 70B; the first bench starts the
instant the server is ready, with SGLang's lazy init still finishing, so the
later number is the higher one).

**Correctness is against the default run, not just self-consistent.** All four
runs reproduce the capstor/mmap baseline's 6 greedy probes byte-for-byte, as
does `clariden-loading-exp`'s overlapped run:

| vs `clariden-loading-exp` default (capstor, mmap) | Llama-3.1-70B | Apertus-8B |
|---|---|---|
| clariden preshard+shm+overlap | 6/6 identical | 6/6 identical |
| Phase 1 presharded + shm | 6/6 identical | 6/6 identical |
| Phase 1 stock + shm (mmap) | 6/6 identical | 6/6 identical |

The 70B phase table, with the stage now a visible phase rather than something
hidden in the total:

```
stage                            4.98  wall_clock
process_startup                 14.68  wall_clock
tp_worker_spawn                 15.18  wall_clock
torch_distributed_init           7.35  engine_reported
unknown                          1.27  wall_clock
weight_loading                   5.80  engine_reported
kv_cache_alloc                   0.57  wall_clock
cuda_graph_capture              19.52  engine_reported
piecewise_cuda_graph_capture    46.04  engine_reported
http_bind                        1.33  wall_clock
warmup_request(JIT)             10.68  wall_clock
total                          126.26  ready
```

## What this settles

- **The packaged pipeline loses nothing to the sbatch scripts it replaces.**
  126.26 s against the experiment's 127.06 s for preshard+shm+**overlapped** —
  i.e. Phase 1 without the overlap lands on the overlapped number. The vendored
  stager is not slower either: 28.34 GB/s here against the experiment's 17.02,
  which is node/OST variance in Phase 1's favour, not a change in the script
  (it is byte-identical, and a test asserts that).

- **Freeing at ready returns the RAM, for the mmap loader too.** This was the
  one caveat that could have invalidated the whole design — `unlink` only
  reclaims tmpfs pages once nothing maps them, so an engine still holding
  mappings at ready would have shown `df` recovering while `MemAvailable` did
  not. It does not happen: on the 70B mmap run `MemAvailable` bottomed at
  292.5 GB and was back to 432.0 GB a minute after ready — **+139.5 GB of the
  141.1 GB staged** — with `/dev/shm` down from a 141.4 GB peak to 0.3 GB.
  SGLang's mmap loader drops its mappings once the weights are on the GPU. The
  presharded run is the same (292.6 → 432.2 GB), as are both Apertus runs.

  The 1 s sampler cannot resolve *exactly* when within the free the pages come
  back — the "at ready" sample on the 70B mmap run caught the `rmtree` in
  progress at 67.5 of 141.4 GB — but the direction is unambiguous.

- **Presharding buys ~2 s once the bytes are in tmpfs**, not the bulk of the
  win. 5.80 vs 7.79 s of `weight_loading` on the 70B, 126.26 vs 128.40 s total.
  Nearly all of the 4.6x is `/dev/shm`, and the stock checkpoint with no loader
  flag at all gets 4.57x. This is the strongest argument for Phase 1's shape:
  the useful tool is the one that needs no prepared artifact, and Phase 3's
  `prepare` is worth ~2 s and a TP- and engine-version-locked artifact, which is
  a much weaker trade than the plan assumed.

- **Phase 2's overlap is worth less than budgeted.** The plan priced it at ~9 s
  from the experiment's 8.78 s stage; the stage measured 4.98 and 4.29 s here.
  It is an upper bound on what the overlap can save, and it is ~4% of the 70B's
  cold start.

## Timing verification

servekit times itself, so the totals were re-checked against clocks it does not
control. Nothing moved.

| check | result |
|---|---|
| servekit's `ready_at` vs SGLang's own "fired up and ready" log timestamp | **equal to the second**, all 4 runs |
| stage wall time (stager's `date +%s.%N`) vs an independent 1 Hz `/dev/shm` sampler | agrees within one sample interval — 4.98 s vs "staged by t+6.0", 4.29 vs t+5.1, 0.98 vs t+1.9, 1.34 vs t+1.9 |
| `sum(phases)` vs `total` | within 1.14 s (≤0.9%) on every run |
| SLURM step elapsed vs total + benches + the 60 s sleep | 176–226 s measured, consistent |

Two things this makes explicit:

- **`sum(phases)` slightly overshoots `total`** (127.40 vs 126.26 on the 70B).
  Not a wrapper bug: the `engine_reported` phases are a max over TP ranks, and
  ranks that finish at different times can make two adjacent phases overlap by
  a fraction of a second. Only `total` is a single wall-clock subtraction.
- **~8 s of job setup sits outside the measured window** — SLURM starts the srun
  7–9 s before servekit's `t0`, spent on container start and `pip install -e`.
  Real, but not attributable to the engine or the wrapper.

The comparison against the baselines is also **charged in the baseline's
favour**: `clariden-loading-exp`'s totals are engine-spawn → ready with no stage
at all, while Phase 1's total starts before the copy begins. The stage is
counted against Phase 1 and against nothing on the other side.

## What this does not settle

- **The cost of not overlapping is inside the noise.** The four runs land at
  +2.57, +0.82, +1.34 and −0.80 s against the overlapped reference — including a
  negative — on four different nodes. Cross-node variance here exceeds the
  ~5 s the overlap could save, so this data cannot measure Phase 2's benefit;
  it can only bound it by the stage wall time. Measuring it needs both arms on
  the same node.

- **One run per config.** Same limitation as `clariden-loading-exp`. The
  correctness checks (errors=0, throughput, greedy probes) are per-run; the
  timings are not repeated.

- **SGLang only, single node.** vLLM staging is deliberately refused by
  `launch` rather than guessed at (Phase 4), and multi-node TP is unmeasured.

- **A server that never reaches ready leaks the copy.** By design in Phase 1 —
  `rm -r` is the hatch. A server that never got ready is not serving, so nothing
  is waiting on that RAM. Not exercised on a real crash here, only in the unit
  tests.

- **138 GB of tmpfs while the copy is live.** On these `--exclusive` 288-core
  nodes with ~800 GB free that is comfortable; two large models staging
  concurrently on one node is not tested, and the stager's own `df` pre-flight
  is what refuses rather than filling it.

## Reproducing

```bash
cd experiments/servekit-fast-weight-load/phase-1-no-overlap
./submit.sh llama70b sharded --exclude=<node used before>
./submit.sh llama70b mmap    --exclude=<...>
```

One fresh node per run: the OS page cache survives across container runs, so a
reused node serves the weights from RAM and manufactures a win.
