# Gate 2 — criu-gpu: full cuda-checkpoint + CRIU round trip through a kill

**Status:** done — **PASS**

## Goal

Gate 1 proved CRIU works for a plain process; this gate adds the piece that was
*impossible* on Bristen: a CUDA process, evicted from the GPU and dumped to disk through
a real **kill** (not a pause), then restored and reattached to the GPU with its device
state intact. This is the minimum viable proof that the checkpoint/restore technique
generalizes to GPU workloads before trying it on a full LLM server (Gate 3).

## Method

`scripts/gate_criu_gpu.sh` runs `src/counter` (a small CUDA program, `src/counter.cu`,
that increments a device-memory counter once a second and logs it). The round trip:

1. Start + warm the counter.
2. `cuda-checkpoint --action lock` then `--action checkpoint` — evicts GPU state to host
   RAM; the process's VRAM disappears from `nvidia-smi`.
3. `criu --unprivileged dump` — snapshots the now-CPU-only process to disk; **the
   process is killed** by this step.
4. Wait ~8 s with the process confirmed dead and holding no GPU memory.
5. `criu --unprivileged restore` — process comes back from the disk image.
6. `cuda-checkpoint --action restore` then `--action unlock` — reattaches to the GPU,
   device memory intact.
7. Confirm the counter resumes at `froze+1` and keeps advancing.

PASS requires the full round trip (including the real kill + restore-from-disk) to end
with the device counter resuming at exactly `froze+1`, sequence contiguous.

## Result

Raw output: `gate2_criu_gpu.txt`.

- GPU memory correctly vanished from `nvidia-smi` after `cuda-checkpoint checkpoint`.
- `criu dump` succeeded; process confirmed gone; snapshot on disk.
- `criu restore` succeeded from disk; `cuda-checkpoint restore + unlock` reattached the
  GPU; device memory present again.
- Tick sequence contiguous: resumed at exactly `froze+1`.

## Verdict

**GATE criu-gpu: PASS.** Device memory survives a full kill-and-restore-from-disk cycle,
unprivileged, on this box. This is the exact combination (`cuda-checkpoint` GPU eviction
+ CRIU disk snapshot) that was blocked on Bristen by the missing CRIU capability — here
it works end to end on a minimal CUDA target, clearing the way to try it on a real
Python/torch process (Gate 3a) and then a full SGLang server (Gate 3b).

## Caveats

Single-GPU, single-process, no multi-GPU/NCCL state — that risk is explicitly deferred,
see the top-level `SUMMARY.md` caveats.
