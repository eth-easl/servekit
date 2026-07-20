# Plan: Test CUDA checkpoint/restore (cuda-checkpoint + CRIU) on Bristen

## Context

The project goal is improving LLM inference **cold-start**. A promising technique is
GPU-process **checkpoint/restore**: warm up a serving process once, snapshot it (CUDA
context, device memory, threads), and later restore in seconds instead of
re-initializing from scratch. Before investing in that, we answer a prerequisite
feasibility question: **does GPU checkpoint/restore work at all on CSCS Bristen**,
given its NVIDIA driver and the unprivileged enroot/pyxis container sandbox?

This experiment reproduces the minimal example from
<https://fergusfinn.com/blog/what-happens-when-you-checkpoint-a-cuda-process/>: a
tiny CUDA program whose GPU state is checkpointed with NVIDIA's `cuda-checkpoint`,
the process dumped/restored with CRIU, then verified to keep running with its GPU
counter intact. Deliverable: a yes/no answer with evidence + a reproducible harness.
Standalone probe, isolated from the serving stack.

## How the flow works (and why it avoids the fragile bits)

`cuda-checkpoint --action lock` then `--action checkpoint` copies device memory to
host RAM and **detaches the process from the GPU** at the OS level. The process is
then an ordinary CPU process, so **plain `criu dump`/`restore` works — no CRIU CUDA
plugin needed** (the driver-570 "transparent"/plugin path is the risky dependency we
sidestep). On restore: `criu restore`, then `cuda-checkpoint --action restore` +
`--action unlock` reattaches to the GPU and resumes.

Requirements: driver **≥ 550**; ptrace access to `/proc/$PID/mem` (same-uid ptrace
already works in these containers); CRIU needs elevated caps (CAP_SYS_ADMIN /
CAP_CHECKPOINT_RESTORE, freeze, ptrace) — **whether the enroot user-namespace grants
enough is the open question this experiment settles.**

## Layout

```
cuda-checkpoint-exp/
  PLAN.md                     # this file
  README.md                   # how to run / how to read results
  bin/cuda-checkpoint         # prebuilt NVIDIA binary (also auto-fetched in-job)
  src/counter.cu              # minimal CUDA program (blog-style, UDP counter)
  scripts/run_ckpt_restore.sh # in-container: bootstrap + run->checkpoint->dump->restore->verify
  scripts/ckpt_restore.sbatch # SLURM job wrapping the in-container driver
  cuda-cr.toml                # enroot EDF (stock nvidia/cuda devel image + mounts)
  results/                    # *.out logs, criu logs, SUMMARY.md
```

**Setup is deliberately minimal:** compute nodes have outbound internet and the
stock `nvidia/cuda` devel image has `nvcc`+`apt`, so the job itself installs criu,
fetches `cuda-checkpoint`, and compiles `counter.cu` (all idempotent). No offline
image build / login-node prep step. One command: `sbatch scripts/ckpt_restore.sbatch`.

## Staged execution (go/no-go gates)

**Step 0 — Probe GPU node (blocking gate).** `nvidia-smi` on a `normal` GPU node:
record driver version (need ≥ 550), GPU model, `ptrace_scope`. Driver < 550 → stop.

**Step 1 — In-job bootstrap** (compute nodes have internet — verified). The job runs
on the stock `nvidia/cuda` devel image and, idempotently: `apt-get install -y criu
iproute2 netcat-openbsd`, fetches `cuda-checkpoint` from NVIDIA's repo, compiles
`counter.cu` with `nvcc`. No login-node prep / offline image build.

**Step 2 — CRIU sanity on a plain CPU process (privilege gate).** In-container on the
GPU node: dump+restore a trivial `sleep` loop. Isolates "can CRIU run in this userns
at all" from CUDA. Fail → capture exact `criu.log` error, report, stop.

**Step 3 — Full GPU checkpoint/restore.** In-container: start `counter`, warm it
(UDP pings → e.g. 101); `cuda-checkpoint lock`+`checkpoint`; `nvidia-smi` (no GPU mem
held); `criu dump`; `criu restore`; `cuda-checkpoint restore`+`unlock`; ping →
expect 102 → **PASS**. Log every step + `--get-state`.

**Step 4 — Report.** `results/SUMMARY.md`: driver version, PASS/FAIL per gate
(driver / CRIU-CPU / GPU-full), counter-continuity evidence, blocking errors verbatim.
Next step: same on a warmed SGLang/torch process (the real cold-start use case).

## Verification

The restored process replies to a UDP ping with the **incremented** counter value
continuing from before the checkpoint → CUDA context + device memory survived a full
dump→restore. Corroborated by clean `nvidia-smi` while checkpointed and a `--get-state`
transition. Every gate has an explicit pass condition and a captured error path, so a
"no" is still a documented, reproducible result.

## Risks (each still yields a useful answer)

1. Driver < 550 → `cuda-checkpoint` unsupported (Step 0).
2. Enroot userns lacks CRIU caps → fails on a plain process (Step 2 isolates it).
3. Compute nodes offline → pre-stage `.sqsh` + binary (Step 1).
4. CRIU chokes on the CUDA process despite passing Step 2 → captured with criu logs;
   motivates trying the driver-570 transparent/plugin path.
