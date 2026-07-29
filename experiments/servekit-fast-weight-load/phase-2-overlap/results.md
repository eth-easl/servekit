# Phase 2 — `--overlap`: is hiding the stage worth a barrier?

**Question:** overlapping the stage with engine startup can save at most the
stage's wall time. Phase 1 measured that at 4.3–5.0 s of the 70B's ~126 s, but
the arms ran on different nodes so the delta was inside noise. Is the win real,
and is it worth building the `sitecustomize` barrier that would make overlap
safe?

**Answer: the win is ~4–7 s and still not cleanly separated from node noise.**
Overlap reaches 126.70 s against sequential's 134.32 s — but the two ran on
different nodes, and the 7.62 s gap is *larger* than the 4.50 s stage that is
the theoretical ceiling on what overlap can save. So part of it is node
variance, not overlap.

## Results

SGLang v0.5.10, TP=4, Llama-3.1-70B presharded, `--load-format sharded_state`.
Baselines from `experiments/clariden-loading-exp`.

| arm | job / node | stage | weight_load | total | vs default | tok/s |
|---|---|---|---|---|---|---|
| *default (capstor, mmap)* | — | — | 466.81 s | 586.33 s | — | 822.9 |
| sequential | 2922933 / nid006073 | 4.50 s @ 31.4 GB/s | 6.04 s | 134.32 s | 4.37x | 813.5 |
| **overlap** | 2924555 / nid007661 | 6.99 s @ 20.2 GB/s, hidden | 6.22 s | **126.70 s** | **4.63x** | 801.0 |
| *clariden-loading-exp, overlapped* | 2916421 / nid007585 | 8.78 s @ 17.0 GB/s | 6.19 s | 127.06 s | 4.61x | 797.7 |

Both arms: 64/64, errors=0, and **6/6 greedy probes byte-identical to the
capstor/mmap default**. The overlap run served correct weights.

Overlap lands within 0.36 s of the sbatch experiment's overlapped number
(126.70 vs 127.06), which is the useful confirmation: the packaged `--overlap`
reproduces the hand-written pipeline.

## What this settles

- **`--overlap` works and is correct in the happy case.** Byte-identical
  outputs, and the stage is fully hidden — `weight_loading` is 6.22 s, the same
  as sequential's 6.04 s, so the loader never waited on the copy.
- **servekit's overlap needed the metadata split.** The first attempt staged
  everything concurrently, including `config.json`, which the engine reads
  seconds in. Fixed to copy the 10 non-`.safetensors` files synchronously first
  and overlap only the 28 shards, exactly as `preshard_shm_overlap.sbatch` does
  (`copied 10 metadata files up front`, `readers=1704`). Without it the engine
  can read a full-size zero-filled config.
- **The stage throughput varies more than the overlap win does.** 31.4, 20.2 and
  17.0 GB/s across three runs of the same 141 GB copy. The thing overlap hides
  is itself the noisiest term.

## What this does not settle

- **Whether the win is worth the barrier.** The 7.62 s measured gap exceeds the
  4.50 s ceiling that the sequential arm's own stage sets, so node variance is
  at least ~3 s of it. A same-node A/B is still what would answer this, and it
  is what I did not run.
- **The OOM on the first overlap attempt (job 2922932, nid007668) is
  unexplained.** A TP worker was OOM-killed during weight loading, 39 s after
  the stage had finished, with node `MemAvailable` at 532 GB and the step cgroup
  at ~207 GB of its 450 GB. It did not reproduce on nid007661. The re-run also
  carried the metadata fix, so it is not a clean re-test of the same binary.
- **n=1 per arm.**

## The 450 GB cgroup

`ReqMem=450G` on these jobs — the step is capped well below the node's ~800 GB,
and **tmpfs is charged to that cgroup**. The 141 GB staged copy is 31% of the
job's entire memory budget while it is live. Phase 1's write-up reported the
free-on-ready evidence in node-level `MemAvailable` and called it comfortable
headroom; measured against 450 GB that framing is wrong. The mechanism still
works — 141.4 GB of shm down to 0.3 GB, `MemAvailable` 306.6 → 445.3 GB — but
the ceiling is the cgroup, not the node.

## Reproducing

```bash
cd experiments/servekit-fast-weight-load/phase-2-overlap
./submit.sh llama70b seq
./submit.sh llama70b ovl --exclude=<node used above>
```
