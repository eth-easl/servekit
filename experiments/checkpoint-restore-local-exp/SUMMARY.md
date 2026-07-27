# checkpoint-restore-local-exp

Full process **checkpoint-to-disk and restore** for a warmed GPU/LLM process, on a
**local RTX 3090**, using NVIDIA `cuda-checkpoint` + CRIU. This is the continuation of
[`cuda-checkpoint-exp`](../cuda-checkpoint-exp/) — which proved the technique on CSCS
Bristen but found CRIU blocked there by a site-policy capability. Here that blocker is
absent, so we can run the round trip that matters for cold-start.

## Why a separate experiment

On Bristen the verdict was split: `cuda-checkpoint` (GPU eviction/restore of a *live*
process) worked, but **CRIU** — snapshot-to-disk-and-restore-*later*, the actual
cold-start win — could not run, because it needs `CAP_CHECKPOINT_RESTORE` in the
**initial** user namespace and no cluster user can get it. That's an outstanding ask to
CSCS. See [`cuda-checkpoint-exp/results/SUMMARY.md`](../cuda-checkpoint-exp/results/SUMMARY.md).

This machine is different in the one way that counts: the system `criu` binary already
carries `cap_checkpoint_restore=ep` as a **file capability** (the exact `setcap` remedy
the Bristen write-up asked for), and the bare host is seccomp-free. So the full round
trip works — and we can finally measure whether restore-from-snapshot beats a cold LLM
launch.

## Answer

**A warmed Apertus-8B SGLang server checkpoints to disk and restores — serving correct
inference — in ~18 s, vs ~59 s for a true-cold launch: a 3.3x cold-start reduction.**
Both numbers are page-cache-controlled, so the comparison is honest.

| gate | proves | result |
|---|---|---|
| `probe_env` | criu carries `cap_checkpoint_restore=ep`; host `Seccomp:0` | ✅ recorded |
| `criu-cpu` | CRIU can dump→disk→restore a plain process *(failed on Bristen)* | ✅ **PASS** |
| `criu-gpu` | `cuda-checkpoint` + CRIU full round trip through a **kill** | ✅ **PASS** |
| Gate 3a | Python+torch.cuda round trip (de-risk) | ✅ **PASS** |
| **Gate 3b** | **warmed Apertus-8B server: restore vs cold launch** | ✅ **PASS — 18.2 s vs 59.3 s (3.3x)** |

| path | time to serving |
|---|---:|
| cold launch (weights + JIT cold) | 59.3 s |
| **restore from 26 GB snapshot (cache cold)** | **18.2 s** |
| checkpoint (one-time) | 21.4 s → 26 GB snapshot |

**No correctness or performance change vs. a cold launch.** The restored server's
correctness-probe outputs are **byte-identical** to the cold-launch baseline's, across
all 6 probe prompts, and post-restore throughput is within noise: **278.3 tok/s
restored vs. 278.2 tok/s cold-launch** (64 requests, concurrency 16, 0 errors either
way). Checkpoint/restore only changes time-to-serving — see
[results/gate3b_server_checkpoint_restore/results.md](results/gate3b_server_checkpoint_restore/results.md#correctness--throughput-vs-cold-launch)
for the side-by-side.

Making a full SGLang server checkpointable unprivileged took five source-level fixes
(GPU VMAs on every tree pid, uvloop→epoll shim, `USE_LIBUV=0`, `--network-lock skip`,
`--tcp-close`) — all in `checkpoint_server.sh`. **Full write-up + the exact criu errors:
[results/gate3b_server_checkpoint_restore/results.md](results/gate3b_server_checkpoint_restore/results.md)**
· design + methodology: [PLAN.md](PLAN.md)

## Sub-experiments

| dir | question | verdict | detail |
|---|---|---|---|
| `gate0_probe` | Does this box carry the file capability Bristen lacked? | Yes — `criu` has `cap_checkpoint_restore=ep` as a file cap, host `Seccomp:0` | [results/gate0_probe/results.md](results/gate0_probe/results.md) |
| `gate1_criu_cpu` | Can CRIU dump/restore *any* process here at all? | ✅ PASS — this is the gate that failed on Bristen | [results/gate1_criu_cpu/results.md](results/gate1_criu_cpu/results.md) |
| `gate2_criu_gpu` | Does cuda-checkpoint + CRIU survive a real kill, on a CUDA process? | ✅ PASS — device memory survives kill + restore-from-disk | [results/gate2_criu_gpu/results.md](results/gate2_criu_gpu/results.md) |
| `gate3a_pytorch_cuda_roundtrip` | De-risk: does the C-counter result hold for a Python+torch.cuda process? | ✅ PASS (649 MB snapshot) | [results/gate3a_pytorch_cuda_roundtrip/results.md](results/gate3a_pytorch_cuda_roundtrip/results.md) |
| `baseline` | What does a true-cold Apertus-8B launch cost, phase by phase? | 59.3 s, both page-cache and JIT-cache confounds controlled | [results/baseline/results.md](results/baseline/results.md) |
| `gate3b_server_checkpoint_restore` | Does restoring a warmed full SGLang server beat that cold launch? | ✅ PASS — **18.2 s restore vs 59.3 s cold (3.3x)** | [results/gate3b_server_checkpoint_restore/results.md](results/gate3b_server_checkpoint_restore/results.md) |

## Environment (2026-07-23)

RTX 3090 24 GiB · driver 610.43.02 / CUDA UMD 13.3 · 125 GiB RAM · criu 4.2 (with file
caps) · host `Seccomp:0` · user `youssef` (uid 1005, no sudo). **Single GPU → TP1**,
unlike TP4 on Clariden/Bristen — flag any TP1-vs-TP4 differences explicitly.

## How to run

```bash
cd experiments/checkpoint-restore-local-exp

# one-time: venv + sglang 0.5.10 + servekit + download Apertus-8B (~16 GB) to ~/models
scripts/setup_env.sh all

# the cold-start reference (~59 s), both cache confounds controlled
scripts/baseline.sh --cold

# the payoff: launch -> warm -> checkpoint -> cold restore (~18 s), end to end
scripts/checkpoint_server.sh 3b
#   or split it:
scripts/checkpoint_server.sh checkpoint   # -> 26 GB snapshot in results/gate3b_server_checkpoint_restore/gate3b-run/img
scripts/restore_server.sh --cold          # restore it, timed, cache-cold
```

The early CUDA gates (foundation) need `bin/cuda-checkpoint` (prebuilt, from NVIDIA's
public repo) and `src/counter` (`nvcc src/counter.cu`), both gitignored:

```bash
curl -fsSL -o bin/cuda-checkpoint \
  https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/main/bin/x86_64_Linux/cuda-checkpoint
chmod +x bin/cuda-checkpoint && /usr/local/cuda/bin/nvcc -o src/counter src/counter.cu
scripts/gate_criu_cpu.sh && scripts/gate_criu_gpu.sh
```

## Files

| path | what |
|------|------|
| `scripts/setup_env.sh` | venv + `sglang[all]==0.5.10` + servekit + `hf download` |
| `scripts/baseline.sh` | `servekit profile` + `servekit bench` TP1; `--cold` controls **both** confounds |
| `scripts/cache_tools.py` | unprivileged page-cache evict + `mincore` verify |
| `scripts/checkpoint_server.sh` | `3a` (py+CUDA), `checkpoint`, `3b` (checkpoint+restore) |
| `scripts/restore_server.sh` | restore from snapshot `[--cold]`, timed + verified with `servekit bench` |
| `shim/uvloop/` | uvloop→epoll shim (kills io_uring so criu can dump); loaded via a `.pth` |
| `scripts/gate_criu_{cpu,gpu}.sh` | Gates 1–2 — the CRIU/cuda-checkpoint foundation |
| `src/counter.cu`, `src/python_cuda_counter.py` | device-memory counters (Gate 2 / 3a) |
| `bin/cuda-checkpoint` | prebuilt NVIDIA tool *(gitignored; fetch above)* |
| `results/<gate>/` | one dir per sub-experiment, each with its own `results.md` (see Sub-experiments above) |

## Key difference from Bristen, in one line

`criu` gets `cap_checkpoint_restore` from its **file capability** (effective in the
initial userns), so it is criu — not your `CapEff=0` shell — that satisfies the kernel
check. Invoke `criu … --unprivileged` (required for a non-root caller); do **not** wrap
it in `unshare -Urpf` (that only grants namespace-root, which the init-userns check
rejects — the Bristen dead end).

## This matches what the field is doing

Independently rediscovered the same path as: Fergus Finn's *Cloudburst* (70x for SGLang,
same io_uring/uvloop snag), **NVIDIA Dynamo Snapshot** (CRIU + cuda-checkpoint for
vLLM/SGLang/TRT-LLM on K8s, quiesce/resume hooks), and vLLM RFC #34303. Our
unprivileged bare-host recipe is the no-root equivalent of their approaches.

## Caveats / next

- **TP1 only.** TP4 (cluster) adds NCCL + a 4-rank process group; NCCL state
  checkpointing is untested here (single GPU) and is the main open risk for the cluster.
- **Snapshots are single-use** (see
  [results/gate3b_server_checkpoint_restore/results.md](results/gate3b_server_checkpoint_restore/results.md)) —
  a real deployment would need the `/dev/shm` link-remap set preserved alongside the
  image, or a dump that does not leave unlinked files behind, before one image could
  serve many cold nodes. That is the gap between this experiment and the actual use
  case.
- The **26 GB snapshot read** is the restore floor. Faster storage / O_DIRECT / parallel
  reads (as NVIDIA Dynamo does) would cut the 14 s criu-restore further.
- The uvloop shim forces plain asyncio (slightly slower loop than uvloop) — fine for
  cold-start/restore measurement; a production path would want the same on both sides.
- `--tcp-close` drops the torch TCPStore connection; safe at TP1 serving, revisit for TP>1.
