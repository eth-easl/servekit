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

Making a full SGLang server checkpointable unprivileged took five source-level fixes
(GPU VMAs on every tree pid, uvloop→epoll shim, `USE_LIBUV=0`, `--network-lock skip`,
`--tcp-close`) — all in `checkpoint_server.sh`. **Full write-up + the exact criu errors:
[results/SUMMARY.md](results/SUMMARY.md)** · design + methodology: [PLAN.md](PLAN.md)

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
scripts/checkpoint_server.sh checkpoint   # -> 26 GB snapshot in results/gate3b-run/img
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
| `results/` | gate logs, profiles + `SUMMARY.md` |

## Key difference from Bristen, in one line

`criu` gets `cap_checkpoint_restore` from its **file capability** (effective in the
initial userns), so it is criu — not your `CapEff=0` shell — that satisfies the kernel
check. Invoke `criu … --unprivileged` (required for a non-root caller); do **not** wrap
it in `unshare -Urpf` (that only grants namespace-root, which the init-userns check
rejects — the Bristen dead end).
