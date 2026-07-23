# cuda-checkpoint-exp

Feasibility probe: **can we checkpoint & restore a live CUDA process on Bristen?**
(NVIDIA `cuda-checkpoint` + CRIU), reproducing
<https://fergusfinn.com/blog/what-happens-when-you-checkpoint-a-cuda-process/>.

## Answer

**The GPU half works; the process half doesn't.**

- ✅ **`cuda-checkpoint` works** as an ordinary user: a live process's GPU state is
  evicted to host RAM (its 418 MiB vanishes from `nvidia-smi`, and it stays frozen
  across a window where ~8 ticks should have elapsed), then restored with **device
  memory intact** — a GPU-resident counter resumed at exactly `froze+1`, with the
  full tick sequence verified contiguous (no reset, gap, or jump).
- ❌ **CRIU cannot run at all.** It must `open()` `/proc/PID/map_files/*`, which the
  kernel gates behind CAP_SYS_ADMIN/CAP_CHECKPOINT_RESTORE **in the initial user
  namespace**. Unobtainable for us — in a container *or* on the bare host, even as
  namespace-root with full `CapEff`.

So: GPU eviction/restore on a **live** process, yes. Snapshot-to-disk-and-restore-
later (the real cold-start win), no — that needs CSCS to grant
`CAP_CHECKPOINT_RESTORE`. It's a site-policy blocker, not a driver or hardware one.

**Full write-up with evidence: [results/SUMMARY.md](results/SUMMARY.md)**
· rationale and design: [PLAN.md](PLAN.md)

## How to run

```bash
# from the repo root (~90s)
sbatch experiments/cuda-checkpoint-exp/scripts/ckpt_restore.sbatch
cat experiments/cuda-checkpoint-exp/results/cuda-cr-<jobid>.out
```

The job is three srun steps on one node:

1. **bundle** (inside the ubuntu/cuda container) — build criu 3.19, compile
   `counter.cu`. In the container purely because the **host has no `nvcc`, no criu
   build deps, and no root to install them**; the container is the only place we get
   apt + a toolchain. Artifacts are cached on `/iopsstor`, so this is a no-op on
   reruns. They run on the host because host glibc (2.38) is newer than jammy's (2.35).
2. **test on the bare host** — the **best case for criu**: `Seccomp:0`, and
   `unshare -Urpf` grants full caps. If criu fails *here*, it can't be blamed on the
   container. (It does fail here.)
3. **cuda-checkpoint inside the container** — as a normal user, seccomp active. This
   is where **real SGLang serving actually runs**, so it's the operationally relevant
   environment. criu gates are skipped there (seccomp rules them out regardless).

## Gates

| gate | what it proves |
|---|---|
| `cuda-ckpt` | cuda-checkpoint alone (no criu): GPU state evicted + restored in a live process |
| `map-files` | whether `open()` on `/proc/PID/map_files/*` is permitted — criu's hard prerequisite |
| `criu-cpu` | criu dump/restore of a plain process (no CUDA) — isolates privileges from GPU |
| `criu-gpu` | the full cuda-checkpoint + criu round trip (only runs if `criu-cpu` passes) |

The gates are independent, so CRIU's failure can't mask the GPU result.

## Files

| path | what |
|------|------|
| `src/counter.cu` | minimal CUDA program: a counter in **device** memory, ticking to a log |
| `bin/cuda-checkpoint` | prebuilt NVIDIA tool |
| `bin/criu` | criu 3.19, built by the bundle step and cached here |
| `hostlibs/` | `libnet.so.1` — the only lib the SLES host lacks |
| `scripts/bundle.sh` | step 1: build criu + compile counter (in container) |
| `scripts/run_host.sh` | step 2: the gates (bare host; re-execs into `unshare -Urpf`) |
| `scripts/ckpt_restore.sbatch` | SLURM wrapper — the only thing you run |
| `cuda-cr.toml` | enroot EDF (jammy-based `nvidia/cuda` devel image) |
