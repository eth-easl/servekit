# Gate 3b — full SGLang server: checkpoint + restore vs. cold launch

**Status:** done — **PASS**, the experiment's headline result

## Goal

Everything before this gate (0–2, 3a) de-risked pieces of the technique in isolation.
This gate asks the question the whole experiment exists to answer: does checkpointing a
**warmed, full SGLang server** (Apertus-8B, TP1) to disk and restoring it actually beat
a true-cold launch — and by how much — once you account for the cost of reading a large
snapshot back from storage?

## Method

`scripts/checkpoint_server.sh 3b` launches Apertus-8B, warms it with one real request,
checkpoints it to disk (`scripts/checkpoint_server.sh checkpoint`), then immediately
calls `scripts/restore_server.sh --cold` to restore it with the snapshot evicted from
page cache first (honest cold-node timing) and verify it serves via `servekit bench`
(same correctness + throughput workload as the [baseline](../baseline/results.md)).

Snapshot dir: `gate3b-run/img` (gitignored, ~26 GB, regenerated per run).

## Result

Raw data: `gate3b.txt`, `gate3b_full.txt`, `gate3b_restore_cold.txt`,
`restore-*-bench.json`.

| path | time to serving | notes |
|---|---:|---|
| cold launch (weights + JIT cold) | **59.3 s** | see [`../baseline/results.md`](../baseline/results.md) |
| restore from snapshot (cache cold) | **18.2 s** | **3.3x faster** |
| restore from snapshot (cache warm) | ~14 s | snapshot hot in page cache |
| checkpoint (one-time, to disk) | 21.4 s | cuda-checkpoint 10.5 s + criu dump 10.9 s → 26 GB |

Restore breakdown (cold): criu restore **14.2 s** (reads the 26 GB image from NVMe) +
cuda-checkpoint GPU reattach **3.7 s** + first request 0.26 s. The snapshot read
dominates — exactly the trade the page-cache-cold method was built to measure honestly.

**Why a restore wins:** the snapshot is taken *after* warmup, so restore skips
everything a cold launch pays for — Python import + TP spawn (~14 s), weight load
(~6 s), and CUDA-graph capture + kernel compile (~35 s).

## Correctness & throughput vs. cold launch

Checkpoint/restore only has to change *when* the server is ready, not *what* it serves.
Both are checked directly against the [baseline](../baseline/results.md), which runs the
identical `servekit bench` workload (6 correctness prompts, then 64 requests / 16
concurrency / 512-in / 128-out for throughput):

| | cold launch (baseline) | restored server | delta |
|---|---|---|---|
| correctness (6 probe prompts) | see `../baseline/*-profile.json` | see `restore-*-bench.json` | **byte-identical output on all 6 prompts** |
| throughput | 278.2 tok/s | 278.3 tok/s | within noise (+0.04%) |
| errors | 0 / 64 | 0 / 64 | none |
| latency (mean / p50 / p99) | 6.90 / 7.36 / 7.37 s | 6.90 / 7.36 / 7.37 s | none |

E.g. the "The capital of France is" probe generates the exact same continuation
(`" Paris, which is also the country's largest city. not only is Paris the capital,
..."`) token-for-token in both the cold-launch and the restored server — expected,
since sampling is greedy (temperature 0) and the restored process is *the same warmed
process*, not a re-initialized one. **Verdict: checkpoint/restore is a pure
time-to-serving optimization — no measurable correctness or throughput change.**

## What it took to make a full SGLang server checkpointable (unprivileged, bare host)

Each blocker below was discovered by hitting criu's verbatim error, then fixing the
source (not working around criu). All encoded in `scripts/checkpoint_server.sh`.

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

**Why not seccomp-disable io_uring instead?** A seccomp filter would flip the process to
`Seccomp:2`, and criu must then *suspend* it via `PTRACE_O_SUSPEND_SECCOMP`, which needs
`CAP_SYS_ADMIN` in the init userns — the exact Bristen wall (see
[`../gate0_probe/results.md`](../gate0_probe/results.md)). So we keep `Seccomp:0` and
remove io_uring at the source instead. (Community-confirmed: Cloudburst uses the same
uvloop-epoll route + `sysctl kernel.io_uring_disabled` where they have root; we don't.)

## Restoring: the snapshot is not a standalone artifact

The image alone is not enough — criu validates it against the *live filesystem* on the
way back in. Two ways that bites, both hit on 2026-07-24 trying to re-restore an image:

| symptom | criu error (verbatim) | why | fix |
|---|---|---|---|
| **A snapshot can only be restored once** | `Can't link dev/shm/link_remap.403 -> dev/shm/sem.m2PZEg: No such file or directory` | `--link-remap` preserves files that were *unlinked* at dump time (the POSIX semaphores SGLang's multiprocessing leaves in `/dev/shm`) by hard-linking them as `link_remap.*`. When a restored process exits it cleans up its `/dev/shm` entries — deleting the very inodes the image links against. The image is then permanently unrestorable. | re-run `checkpoint_server.sh checkpoint` (~22 s once the server is warm). No reboot needed to lose it — one restore+exit cycle is enough. |
| **Files open at dump time must still match** | `File .../ckpt_server.log has bad size 154990 (expect 30765)` | the checkpointed server holds that log as fd 1; criu records its size and refuses to restore onto a file that has changed. Any later run appending to the same log breaks the image. | `truncate -s <expected> <file>` (criu prints the expected size), or give each run its own log |

**Consequence for benchmarking:** budget one checkpoint per restore measurement. A/B'ing
restore latency across N runs needs N checkpoints, not one image restored N times.

## Verdict

**GATE 3b: PASS — 18.2 s cold restore vs. 59.3 s cold launch, a 3.3x cold-start
reduction, serving correct inference.** This is the number the whole experiment set out
to measure, and it holds even after honestly paying the cost of reading a 26 GB snapshot
back from cold storage.

## Caveats

- **TP1 only.** TP4 (cluster) adds NCCL + a 4-rank process group; NCCL state
  checkpointing is untested here (single GPU) and is the main open risk for the cluster
  — see the top-level `SUMMARY.md`.
- **Snapshots are single-use** (see table above) — a real deployment would need the
  `/dev/shm` link-remap set preserved alongside the image, or a dump that does not leave
  unlinked files behind, before one image could serve many cold nodes.
- The **26 GB snapshot read** is the restore floor. Faster storage / O_DIRECT / parallel
  reads (as NVIDIA Dynamo does) would cut the 14 s criu-restore further.
- The uvloop shim forces plain asyncio (slightly slower loop than uvloop) — fine for
  cold-start/restore measurement; a production path would want the same on both sides.
- `--tcp-close` drops the torch TCPStore connection; safe at TP1 serving, revisit for TP>1.
