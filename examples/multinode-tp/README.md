# Multi-node tensor parallel

Llama-3.1-70B at TP=8 across two 4-GPU Clariden nodes, weights loaded from
`/dev/shm` on each node.

```bash
sbatch prepare.sbatch   # once: the TP=8 presharded checkpoint (~10 min, 131 GB)
sbatch run.sbatch       # serve it, profile the cold start, bench, merge
```

## What is multi-node about it

Almost nothing. Every node runs the same command:

```bash
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --environment="$EDF" bash -c '
  unset BASH_ENV
  servekit launch --out logs/run.json -- \
    python -m sglang.launch_server --model-path "$SHARDED" \
      --load-format sharded_state --tensor-parallel-size 8 \
      --nnodes 2 --node-rank $SLURM_PROCID --dist-init-addr '"$HEAD"':20000 \
      --host 0.0.0.0 --port 8080
'
servekit report logs/ --out merged.json
```

servekit reads `--nnodes`/`--node-rank`/`--dist-init-addr` out of the command --
it never writes them -- and from them works out that this node holds ranks 4-7,
so it stages `model-rank-[4-7]-part-*.safetensors` and the metadata, and nothing
else. Two nodes then pull disjoint halves off Lustre at the same time rather
than the whole checkpoint twice.

There is no barrier, no control socket, and no shared rendezvous directory. Each
node frees its own copy when it reports itself up, which needs no cross-node
permission precisely because it staged only what its own ranks read. The head's
signal is the usual ready line; a worker runs no tokenizer process and binds no
API server, so its signal is the `Dummy health check server started` line SGLang
logs once all of its schedulers are constructed. `servekit report` merges the
per-node reports afterwards, maxing every phase across all eight ranks -- the
head's report alone only ever saw ranks 0-3.

## Two things you cannot drop

- **`unset BASH_ENV` inside the container.** The SGLang image sets
  `BASH_ENV=/etc/bash.bashrc`, so each of the stager's ~900 subshells runs
  `nvidia-smi` before its `dd`: 0.41 GB/s instead of 16.89 GB/s.
- **The `com.hooks.aws_ofi_nccl` annotation in the EDF.** It is what puts NCCL
  on Slingshot. Nothing else about the fabric needs setting -- no
  `NCCL_SOCKET_IFNAME`, no `NCCL_IB_HCA`.

## Measured (experiments/multinode-tp-exp, jobs 2964019 and 2964177)

| | baseline (HF checkpoint, mmap) | fast (presharded + /dev/shm) |
|---|---|---|
| stage | — | 3.10 s @ 22.8 GB/s (node 0), 2.74 s @ 25.8 GB/s (node 1) |
| weight_loading, max over 8 ranks | 553.02 s | 3.25 s |
| total to ready | 667.44 s | 120.30 s |
| total incl. stage | 667.44 s | **123.40 s (5.41x)** |
| throughput | 487.1 tok/s | 478.1 tok/s |

Each node staged 70.6 GB, not 141 GB, and used 66 GiB of its own tmpfs.

**Multi-node TP is for models that do not fit, not for capacity.** The same
bench gets 822.9 tok/s at TP=4 on a single node; splitting the tensor parallel
group across the interconnect costs ~40% of it, in both arms. Note also that the
mmap baseline gets *worse* with more nodes (553 s here against 467 s at TP=4 on
one node), because both nodes pull the same 141 GB off Lustre at once.

## Gotchas

- **debug-qos caps a job at 90 node-minutes**, so a 2-node job must ask for
  `--time=00:45:00` or less, and only one debug job runs per user at a time.
- **Use a fresh node pair when comparing against the baseline.** The OS page
  cache survives container runs; pass `--exclude=<the previous run's nodes>`.
- **`--wait=60` on the srun** so the worker does not hold the allocation to the
  time limit after the head exits. Its exit code is meaningless; the gates in
  `run.sbatch` decide whether the run worked.
- Benign teardown noise after a successful run: `TP4-7 Scheduler hit an
  exception`, gloo `Connection closed by peer`, and `Failed to destroy CXI
  Service ID` -- all of it after the bench has passed.
- Presharded checkpoints are locked to both the TP size **and** the engine.
  vLLM's `ShardedStateLoader` cannot read SGLang's shards. They are portable
  across CPU architectures.
