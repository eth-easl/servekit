# servekit fast weight load — summary

Validation of the package built in `docs/packaging-fast-weight-load/PLAN.md`,
one sub-directory per phase. `experiments/lustre-loading-exp` and
`experiments/clariden-loading-exp` established the *technique*; these runs check
that the *packaged* version reproduces it.

| phase | what shipped | 70B result | see |
|---|---|---|---|
| **1** | `servekit launch`, stage-then-start, free on ready | 586.33 → **126.26 s**, **4.64x** | [phase-1-no-overlap](phase-1-no-overlap/results.md) |
| **2** | `--overlap` (opt-in, no barrier) | 586.33 → **126.70 s**, **4.63x** | [phase-2-overlap](phase-2-overlap/results.md) |

## Phase 1

`servekit launch -- <engine command>` reproduces the experiment's best
configuration with no overlap, no engine hook and no hand-run scripts, at
4.64x on Llama-3.1-70B and 1.79x on Apertus-8B (SGLang, TP=4, Clariden).

Two findings that change what the later phases are worth:

- **Presharding buys ~2 s once the bytes are in `/dev/shm`** (5.80 vs 7.79 s of
  weight loading on the 70B). The stock checkpoint with no loader flag gets
  4.57x on its own — so Phase 3's `prepare`, and the TP- and
  engine-version-locked artifact it produces, is a much weaker trade than the
  plan assumed.
- **Phase 2's overlap is bounded by a 4.29–4.98 s stage**, not the ~9 s the plan
  budgeted from `clariden-loading-exp` — ~4% of the 70B's cold start. The
  measured cost of *not* overlapping is inside cross-node noise here and needs a
  same-node comparison to resolve.

The design's one load-bearing risk is discharged: **freeing at ready really does
return the RAM**, for the default mmap loader as well as `sharded_state`
(+139.5 GB of the 141.1 GB staged, back within a minute). Caveat added in Phase
2: that was measured against node RAM, but the binding limit is a **450 GB step
cgroup** that tmpfs is charged to, so the 141 GB copy is 31% of the job's budget
while live — tighter than Phase 1 implied.

## Phase 2

`--overlap` reproduces the hand-written sbatch pipeline (126.70 s vs its
127.06 s) with byte-identical outputs, and the metadata files now have to be
copied synchronously ahead of the shards or the engine can read a full-size
zero-filled `config.json`.

**The win is still not separated from node noise.** Overlap's 7.62 s advantage
over the sequential arm exceeds the 4.50 s stage that bounds what it can save,
so ≥3 s of it is cross-node variance. Deciding whether to build the barrier
needs a same-node A/B, which has not been run.

Correctness is against the default run, not self-consistency: all four runs
reproduce the capstor/mmap baseline's 6 greedy probes byte-for-byte. Timings
were re-checked against clocks servekit does not control (SGLang's own log
timestamps, an independent `/dev/shm` sampler, SLURM step elapsed) and hold.
