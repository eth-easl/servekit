# Result: local Apertus-8B checkpoint/restore vs cold start

**Verdict: it works, and it's ~3.3x faster than a cold launch.** On this box a warmed
Apertus-8B SGLang server (TP1) checkpoints to a 26 GB disk snapshot and restores —
serving correct inference — in **~18 s**, vs a **~59 s** true-cold launch. Both numbers
are page-cache-controlled, so the comparison is honest.

Environment: RTX 3090 24 GiB, driver 610.43.02, 125 GiB RAM, NVMe, kernel 7.0.0,
host `Seccomp:0`, user `youssef` (uid 1005, no sudo). criu 4.2 (file caps
`cap_checkpoint_restore,cap_sys_ptrace`). SGLang **v0.5.10**, torch 2.9.1+cu128,
native venv (not the container — see [PLAN.md](../PLAN.md)). Model
`swiss-ai/Apertus-8B-Instruct-2509`.

## Headline numbers

| path | time to serving | notes |
|---|---:|---|
| cold launch (weights + JIT cold) | **59.3 s** | `baseline.sh --cold`, reproducible ±0.3% |
| restore from snapshot (cache cold) | **18.2 s** | `restore_server.sh --cold` — **3.3x faster** |
| restore from snapshot (cache warm) | ~14 s | snapshot hot in page cache |
| checkpoint (one-time, to disk) | 21.4 s | cuda-checkpoint 10.5 s + criu dump 10.9 s → 26 GB |

Restore breakdown (cold): criu restore **14.2 s** (reads the 26 GB image from NVMe) +
cuda-checkpoint GPU reattach **3.7 s** + first request 0.26 s. The snapshot read
dominates — exactly the trade the page-cache method was built to measure.

**Why a restore wins:** the snapshot is taken *after* warmup, so restore skips
everything a cold launch pays — Python import + TP spawn (~14 s), weight load (~6 s),
and CUDA-graph capture + kernel compile (~35 s). See the baseline breakdown below.

## The true-cold baseline (what a restore replaces)

Two cache confounds, both controlled by `baseline.sh --cold` (see [PLAN.md](../PLAN.md)):
page cache (weights) and the JIT/compile cache (flashinfer/triton kernels — the cluster
neutralizes this with ephemeral `HOME=/root`). Phase breakdown, TP1, ctx 8192:

```
process_startup + imports   ~14.2 s
weight_loading                6.3 s   (cold read, 16 GB from NVMe)
cuda_graph_capture           17.0 s   (mostly first-touch flashinfer kernel COMPILE)
piecewise_cuda_graph_capture 17.9 s   (torch.compile eager, runs every launch)
warmup(JIT)                   1.7 s
────────────────────────────────────
total (ready)                59.3 s   |  throughput ~278 tok/s
```

Note `cuda_graph_capture` is dominated by kernel *compilation*, not graph capture:
warm-JIT it collapses to ~2 s. This is why controlling the compile cache matters.

## What it took to make a full SGLang server checkpointable (unprivileged, bare host)

Each was discovered by hitting criu's verbatim error, then fixing the source (not
working around criu). Encoded in `checkpoint_server.sh`.

| blocker | criu error | fix |
|---|---|---|
| GPU device VMAs | `handle_device_vma plugin failed` / `Can't handle non-regular mapping` | `cuda-checkpoint` **every** tree process holding `/dev/nvidia` VMAs (launcher + scheduler + detokenizer), not just the `nvidia-smi` compute-app |
| io_uring (uvloop) | `Unknown shit 600 (anon_inode:[io_uring])` | uvloop→epoll shim (`shim/uvloop`) loaded via a **.pth** so it reaches SGLang's env-scrubbed scheduler; a PYTHONPATH shim does not |
| io_uring (torch) | same | `USE_LIBUV=0` — torch's `TCPStore` uses bundled libuv→io_uring even at TP1; injected via the same .pth (scheduler env is scrubbed) |
| TCP connection lock | `Iptables configuration failed` | `--network-lock skip` (iptables lock needs CAP_NET_ADMIN) |
| TCP connection dump | `Can't turn TCP repair mode ON: Operation not permitted` | `--tcp-close` — drop the torch TCPStore TCP conn instead of preserving it (TCP_REPAIR needs CAP_NET_ADMIN); harmless at TP1 (no collectives during serving) |

Full criu dump args: `--shell-job --tcp-close --network-lock skip --file-locks
--link-remap --ext-unix-sk --unprivileged`. Restore: `--shell-job --tcp-close -d
--unprivileged`, then `cuda-checkpoint --action restore/unlock` on the saved GPU pids.

**Why not seccomp-disable io_uring?** A seccomp filter would flip the process to
`Seccomp:2`, and criu must then *suspend* it via `PTRACE_O_SUSPEND_SECCOMP`, which needs
`CAP_SYS_ADMIN` in the init userns — the exact Bristen wall. So we keep `Seccomp:0` and
remove io_uring at the source instead. (Community-confirmed: Cloudburst uses the same
uvloop-epoll route + `sysctl kernel.io_uring_disabled` where they have root; we don't.)

## Restoring: the snapshot is not a standalone artifact

The image alone is not enough — criu validates it against the *live filesystem* on the way
back in. Two ways that bites, both hit on 2026-07-24 trying to re-restore the Gate 3b image:

| symptom | criu error (verbatim) | why | fix |
|---|---|---|---|
| **A snapshot can only be restored once** | `Can't link dev/shm/link_remap.403 -> dev/shm/sem.m2PZEg: No such file or directory` | `--link-remap` preserves files that were *unlinked* at dump time (the POSIX semaphores SGLang's multiprocessing leaves in `/dev/shm`) by hard-linking them as `link_remap.*`. When a restored process exits it cleans up its `/dev/shm` entries — deleting the very inodes the image links against. The image is then permanently unrestorable. | re-run `checkpoint_server.sh checkpoint` (~22 s once the server is warm). No reboot needed to lose it — one restore+exit cycle is enough. |
| **Files open at dump time must still match** | `File .../results/ckpt_server.log has bad size 154990 (expect 30765)` | the checkpointed server holds that log as fd 1; criu records its size and refuses to restore onto a file that has changed. Any later run appending to the same log breaks the image. | `truncate -s <expected> <file>` (criu prints the expected size), or give each run its own log |

Consequence for benchmarking: **budget one checkpoint per restore measurement.** A/B'ing
restore latency across N runs needs N checkpoints, not one image restored N times.

## This matches what the field is doing

Independently rediscovered the same path as: Fergus Finn's *Cloudburst* (70x for SGLang,
same io_uring/uvloop snag), **NVIDIA Dynamo Snapshot** (CRIU + cuda-checkpoint for
vLLM/SGLang/TRT-LLM on K8s, quiesce/resume hooks), and vLLM RFC #34303. Our
unprivileged bare-host recipe is the no-root equivalent of their approaches.

## Gates

| gate | result |
|---|---|
| Gate 0 env probe / Gate 1 criu-cpu / Gate 2 criu-gpu (C counter) | ✅ PASS (see cuda-checkpoint-exp lineage) |
| Gate 3a — Python+torch.cuda round trip | ✅ PASS (649 MB snapshot, froze→+1) |
| Gate 3b — full SGLang server round trip | ✅ **PASS — 18.2 s restore vs 59.3 s cold (3.3x), correct inference** |

## Reproduce

```bash
scripts/setup_env.sh all                 # venv + sglang 0.5.10 + servekit + download
scripts/baseline.sh --cold               # true-cold reference (~59 s)
scripts/checkpoint_server.sh 3b          # launch → warm → checkpoint → cold restore
# or split: checkpoint_server.sh checkpoint ; restore_server.sh --cold
```

One checkpoint per restore — `restore_server.sh` consumes the image (see "Restoring").

## Caveats / next

- **TP1 only.** TP4 (cluster) adds NCCL + a 4-rank process group; NCCL state
  checkpointing is untested here (single GPU) and is the main open risk for the cluster.
- **Snapshots are single-use** (see "Restoring" above) — a real deployment would need the
  `/dev/shm` link-remap set preserved alongside the image, or a dump that does not leave
  unlinked files behind, before one image could serve many cold nodes. That is the gap
  between this experiment and the actual use case.
- The **26 GB snapshot read** is the restore floor. Faster storage / O_DIRECT / parallel
  reads (as NVIDIA Dynamo does) would cut the 14 s criu-restore further.
- The uvloop shim forces plain asyncio (slightly slower loop than uvloop) — fine for
  cold-start/restore measurement; a production path would want the same on both sides.
- `--tcp-close` drops the torch TCPStore connection; safe at TP1 serving, revisit for TP>1.
