# vLLM — Llama-3.1-70B, TP=4, Clariden (GH200)

**Does preshard+shm+overlap reproduce on vLLM? Yes: 322.01 → 125.80 s, 2.56x**,
serving correctly (64/64, errors=0). Weight loading goes 199.90 → 7.50 s, a 27x
cut, and nothing else moves.

| | default (2918323) | preshard+shm+overlap (2918765) |
|---|---|---|
| node | nid007424 | nid006918 |
| process_startup | 20.28 | 20.52 |
| worker_spawn+dist_init | 43.86 | 44.08 |
| **weight_loading** | **199.90** | **7.50** |
| unknown | 2.14 | 2.05 |
| torch_compile | 37.19 | 35.45 |
| kv_cache_alloc | 2.72 | 2.30 |
| cuda_graph_capture | 10.00 | 8.00 |
| api_server_startup | 5.91 | 5.81 |
| **total** | **322.01** | **125.80** |
| non-load (total − weight_load) | 122.11 | 118.22 |
| stage (overlapped) | — | 16.89 s @ 8.62 GB/s, 47.80 s slack, VALID |
| throughput | 827.3 tok/s | 798.1 tok/s |
| ready_wait_s | 0.072 | 0.072 |

Every non-load phase agrees to well under a second between the two configs —
the check that the technique moved only the phase it targets. n=1 per config,
different nodes, one fresh node per run.

Next to SGLang on the same hardware and model (`../../sglang/llama-3.1-70b/`):

| | vLLM | SGLang |
|---|---|---|
| default total | 322.01 | 586.33 |
| preshard total | **125.80** | **127.06** |
| speedup | **2.56x** | 4.61x |
| weight_loading, default → preshard | 199.90 → 7.50 | 466.81 → 6.19 |
| non-load floor | 118–122 | 119–121 |

The two engines land within 1% of each other once weight loading is removed
(125.80 vs 127.06). The speedups differ only because the baselines differ, and
the baselines differ in `weight_loading`, whose spread on this storage is
430–939 s across the bristen reference runs — wider than the gap. **Do not read
322 vs 586 as an engine result.** The honest statement is that both engines have
a ~120 s floor here and the technique gets both of them to it.

## Two findings that cost most of the jobs

### 1. Presharded checkpoints are engine-specific

The experiment first pointed vLLM at the checkpoint SGLang wrote, on the
reasoning that `sharded_state` originated in vLLM and SGLang inherited it. vLLM
rejected it (job 2918412):

```
ValueError: Missing keys ('model.layers.0.self_attn.attn._k_scale',
  'model.layers.0.self_attn.attn._v_scale', 'model.layers.0.self_attn.attn._q_scale',
  'model.layers.0.self_attn.attn._prob_scale', ...) in loaded state!
```

Four attention scale params per layer, 320 for the 80-layer 70B, that vLLM's
model declares and SGLang's saver never writes. `ShardedStateLoader` treats every
one as mandatory.

The preflight gate had passed, and it was right to: the filename pattern matched
`ShardedStateLoader.DEFAULT_PATTERN` exactly, all four ranks had 7 parts, and the
keys were plain parameter names. The mismatch is in the key *set*, which is only
visible against an instantiated vLLM model. The gate now checks for the scale
params by name, which is the thing that actually discriminates.

So: presharded checkpoints are portable across **architectures** — the SGLang arm
reused a bristen x86-written checkpoint on GH200 with byte-identical output — and
locked to **TP size, engine, and engine version**. The packaging work needs one
artifact per engine, not one artifact.

vLLM's own checkpoint is at
`/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded-vllm`
(job 2918476: 28 shards, 132 GB, 48 scale params in rank-0 part-0).

### 2. `BASH_ENV` in the NGC image makes the stager pathological

The first run with the correct checkpoint came out at **468.54 s — worse than
doing nothing** (job 2918555). The stage took 348 s at 0.41 GB/s where SGLang's
hits 17 GB/s, and `process_startup` came out at 366.32 s against a normal 14.8 s.

It is neither the storage nor the checkpoint. Staged on the bare host, both
checkpoints read at full speed on the same node in the same minute (job 2918617):
SGLang's at 26.88 GB/s, vLLM's at 28.22 GB/s. Repeating the run inside the
container with no engine at all (job 2918657) reproduced the slowness, and the
node's process table said why:

```
3429 bash
1704 sed
1704 nvidia-smi      <- one per slice
1704 grep
   0 dd              <- none
```

`nvcr.io#nvidia/vllm:26.07-py3` sets `BASH_ENV=/etc/bash.bashrc`, so **every
non-interactive bash sources the NVIDIA banner, which shells out to
`nvidia-smi`**. The stager forks one `bash -c` per slice — 1704 of them at once —
so the node ran 1704 concurrent `nvidia-smi` processes at load average 1658 and
never got round to reading bytes. The same herd starved vLLM's own startup, which
is the whole of the 366 s `process_startup`; every other phase in that run is
normal (worker spawn 43.45, load 6.96, compile 33.81).

Fix, in `scripts/vllm/preshard_shm_overlap.sbatch`: run the stager under
`env -u BASH_ENV -u ENV`. The stager is a verbatim copy of the bristen one and
stays that way, so the fix lives at the call site. Stage 348 → **16.89 s**.

This is a portability lesson for the package, not a curiosity: the technique's
"spawn one worker per slice" shape is safe on the SGLang image and quietly
catastrophic on NVIDIA's. Anything shipped has to either not fork a shell per
slice, or neutralise the startup file.

**Note for the packaging work: the `dd`-per-slice design is probably the wrong
shape.** `env -u BASH_ENV` fixes this incident, but it treats a symptom of a
stager that forks 3408 short-lived processes (1704 `bash` + 1704 `dd`) to move
141 GB. That design is fragile in ways this experiment has now seen twice — it
inherits whatever the container does at shell startup, and it depends on process
spawn being cheap, which is not a property any image guarantees. It also leaves
the read rate at the mercy of the scheduler: 8.62 GB/s overlapped here against
28.22 standalone. A single process issuing async reads (io_uring / threads /
`fastsafetensors`-style) would do the same work with 1 process instead of 3408,
no shell involved, and a controllable queue depth. Worth measuring against the
current stager before any of this is packaged.

Worth noting the run that proved it landed on **nid006918 — the same node** as
the 468 s run, which kills the bad-node explanation outright. The stage reads
`O_DIRECT`, so nothing came from page cache either.

### Residual: the stage is still 2-3x off its standalone rate

16.89 s @ 8.62 GB/s against 28.22 GB/s measured standalone on the host. It is
overlapped with vLLM's startup by design, so some contention is expected, but the
gap is not attributed. It does not affect the result — the stage finished with
47.80 s of slack before the loader opened a file — and it is not worth chasing
while `torch_compile` + `worker_spawn` are 79 s of the remaining 126.

## What this does not settle

- n=1 per config, and the baseline's `weight_loading` is the noisiest quantity
  on this system. The 2.56x is a point estimate, not a bracket.
- vLLM 0.24.0.dev (NGC) against SGLang 0.5.10 — the engine versions are not
  matched, because upstream `vllm/vllm-openai:v0.25.0`'s arm64 build dies at CUDA
  graph capture on GH200 (`profile/llama-3.1-70b-clariden-vllm/results.md`). This
  compares each engine's working build on this hardware.
- The totals stop at different events: SGLang announces ready after its own
  warmup request, vLLM issues none. `ready_wait_s` is 0.072 s here, so vLLM
  really was serving at that point, but the vLLM-vs-SGLang totals are not the
  same measurement. The 2.56x is vLLM-vs-vLLM and unaffected.
- Only Llama-3.1-70B. The Apertus-8B case, where loading is a smaller share of
  cold start, is untested on vLLM.

## Raw artifacts

In this directory: `vllm-llama70b-{default,preshard,save,preflight}-<job>.out`,
`*-profile.json`, `*-stage.txt`, `*-timing.txt`. The 468 s run
(`...-2918555-...`) and the failed cross-engine run (`...-2918412-...`) are kept
deliberately — they are the evidence for both findings above. The host-side and
in-container stage probes are `../../stage-probe-2918617.out` and
`../../stage-probe-vllm-container-2918657.out`.

## Reproducing

```bash
cd experiments/clariden-loading-exp
./submit.sh vllm llama70b save                       # one-off, builds the checkpoint
./submit.sh vllm llama70b preflight                  # cheap gate
./submit.sh vllm llama70b default                    # note the node id
./submit.sh vllm llama70b preshard --exclude=<that node>
```
