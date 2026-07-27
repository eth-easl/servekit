# Plan: Full checkpoint/restore of a warmed LLM process — local machine

## Why this is a separate experiment from `cuda-checkpoint-exp`

`cuda-checkpoint-exp` answered the feasibility question on **CSCS Bristen** and got a
split verdict: `cuda-checkpoint` (GPU eviction/restore of a *live* process) works, but
**CRIU** — the piece that snapshots a process to disk and restores it *later*, which is
the actual cold-start win — is blocked. Not by driver or hardware, but by kernel
policy: CRIU's `open()` of `/proc/PID/map_files/*` needs `CAP_CHECKPOINT_RESTORE` in
the **initial** user namespace, and no cluster user can get that. It's an ask
outstanding with CSCS.

Rather than block on that grant, we continue on a **local machine** where we can
control the environment. Critically, this box is **not the same wall**: the system
`criu` binary already carries the exact file capability the Bristen SUMMARY asked CSCS
to set (`setcap cap_checkpoint_restore=ep`), and the bare host runs with `Seccomp: 0`.
So the full CRIU round trip that was impossible on Bristen is expected to work here —
which lets us measure the thing that actually matters: **how much cold-start time does
snapshot-restore buy for a warmed LLM process?**

## Local environment (recorded 2026-07-23)

| | |
|---|---|
| GPU | 1x NVIDIA GeForce RTX 3090, 24 GiB VRAM |
| Driver / CUDA UMD | 610.43.02 / 13.3 |
| RAM | 125 GiB total (~123 GiB free) |
| criu | v4.2, `/usr/sbin/criu` with file caps `cap_sys_ptrace,cap_checkpoint_restore=ep` |
| seccomp (bare host) | 0 (no filter) |
| ptrace_scope | 1 (criu bypasses via its `cap_sys_ptrace` file cap) |
| user | `youssef` uid=1005, **no sudo**, `CapEff=0` in the shell |

**Model:** Apertus-8B, which fits on the single 24 GiB GPU → all local runs are
**TP1**, unlike the TP4 setup on Clariden/Bristen. Any TP1-vs-TP4 difference in timing
or behaviour must be called out, not assumed to generalise. See the top-level project
`CLAUDE.md` and the [[local-test-machine]] note.

## The key insight about capabilities

The shell has `CapEff=0`; a naive `open()` of `map_files` from the shell returns EPERM
(verified) — *the same error seen on Bristen*. That is **not** the blocker here,
because `criu` does not inherit the shell's caps: its `cap_checkpoint_restore=ep` file
capability makes the cap effective **in `criu`'s own process, in the initial userns**,
which is exactly what the kernel check wants. This is the setup Bristen lacked. So the
right test is not "can my shell open map_files" but "can the criu binary complete a
dump/restore" — hence the gates below invoke `criu` directly, not `unshare -Urpf`
(which on Bristen only granted namespace-root and did *not* satisfy the init-userns
check).

## Staged gates (go/no-go)

**Gate 0 — Environment probe.** Record driver, criu version + `getcap`, seccomp,
ptrace_scope, caps. Confirms the file-cap precondition that differentiates this box
from Bristen. → `scripts/probe_env.sh`

**Gate 1 — `criu-cpu`: dump/restore a plain process.** A trivial tick-to-a-file loop,
no CUDA. Isolates "can CRIU run at all here" from anything GPU. Dump → process gone →
wait past several ticks → restore → counter resumes and advances. This is the gate that
**failed on Bristen**; passing it here is the whole premise. → `scripts/gate_criu_cpu.sh`

**Gate 2 — `criu-gpu`: full round trip on the CUDA counter.** Reuses `src/counter.cu`
(device-memory counter). `cuda-checkpoint lock`+`checkpoint` (evict GPU state to host)
→ `criu dump` (snapshot the now-CPU-only process to disk) → **kill** → `criu restore`
→ `cuda-checkpoint restore`+`unlock` → counter resumes at froze+1 with device memory
intact. Only runs if Gate 1 passes. → `scripts/gate_criu_gpu.sh`

**Gate 3 — the real target: warmed Apertus-8B (SGLang, TP1).** Split in two:

- **Gate 3a — Python+torch.cuda process.** De-risks the CPython/torch CRIU path (large
  RSS, many threads, the real CUDA libs) *before* the server. A `torch` device-memory
  counter, same round trip as Gate 2. → `src/python_cuda_counter.py`, `checkpoint_server.sh 3a`
- **Gate 3b — the full `launch_server` process tree.** Launch SGLang, warm it (weights
  loaded, CUDA graphs captured, first request served), `cuda-checkpoint` the GPU-holding
  scheduler PID, `criu dump` the **whole process tree** (HTTP + multiproc + open :8080
  socket) to disk, **kill**, evict the snapshot from page cache, `criu restore`,
  `cuda-checkpoint restore`, then confirm the restored server serves a *correct*
  `/generate`. Measure **restore-to-serving vs cold-launch-to-ready**, both cache-cold.
  → `checkpoint_server.sh 3b`

## Implementation

Three deliverable scripts, plus a baseline and the page-cache helper:

| script | role |
|---|---|
| `scripts/setup_env.sh` | venv + `pip install sglang[all]==0.5.10` + `servekit` + `hf download` (idempotent, phased) |
| `scripts/baseline.sh` | `servekit profile` + `servekit bench` wrapping `launch_server --tp-size 1` — the reference cold-start |
| `scripts/checkpoint_server.sh` | Gate 3a + 3b (the checkpoint/restore round trips) |
| `scripts/cache_tools.py` | unprivileged page-cache evict + `mincore` verify (see below) |

**Decisions locked:**

- **Checkpoint target = the full `launch_server` tree** (HTTP + multiproc + open socket),
  not the simpler in-process Engine. Production-faithful; higher CRIU risk (socket +
  tree), so `checkpoint_server.sh` surfaces criu's verbatim errors and iterates flags
  via `CRIU_DUMP_ARGS`.
- **Native venv, NOT a container.** The cluster serves from the `lmsysorg/sglang`
  container, but a container reintroduces the `Seccomp:2` filter that blocked CRIU on
  Bristen (`PTRACE_O_SUSPEND_SECCOMP` needs CAP_SYS_ADMIN). Bare-host is `Seccomp:0`, so
  the checkpoint half **must** run natively. Verified: driver 610 runs torch 2.9.1+cu128.
- **SGLang pin = v0.5.10** (matches `profile/apertus-8b-bristen/apertus-8b-sglang.toml`),
  torch 2.9.1/cu12.8. Model `swiss-ai/Apertus-8B-Instruct-2509`, `--trust-remote-code`.
- **24 GB-tuned launch knobs** vs cluster (32768/0.85/256): `--context-length 8192
  --mem-fraction-static 0.85 --max-running-requests 8`. cuda graphs stay **enabled**
  (graph capture is a cold-start phase we want to measure).
- **Disk: `~/models` + one snapshot at a time**, deleted between runs (80 GB free; abort
  guard if < 25 GB before a dump).

## Avoiding the page-cache confound (the load-bearing methodology)

**The trap.** We compare two routes to a ready engine: a **cold launch** (reads ~16 GB
safetensors, then captures CUDA graphs) vs a **restore** (reads a ~20 GB CRIU image,
then `cuda-checkpoint restore`). The OS page cache serves any recently read/written file
from RAM. Right after `criu dump` the snapshot is 100 % hot, so a back-to-back restore
reads from RAM and looks 5–10× too fast. Symmetrically, re-running a launch without
eviction reads warm weights. Uncontrolled, every number is biased.

**Why the usual fixes don't apply here.**

- `echo 3 > /proc/sys/vm/drop_caches` — root-only, and there is **no sudo** (`drop_caches`
  is mode `--w------- root`, confirmed). ✗
- Cache-bust by reading a >RAM (125 GB) file — only ~80 GB disk free. ✗
- O_DIRECT (the repo's Lustre method) — bypasses cache only on reads *we* issue, not
  SGLang's weight loader or criu's restore I/O (both buffered). ✗
- Fresh node per run (the repo's other method) — single local machine. ✗

**The fix that works unprivileged: surgical per-file eviction + verification** (`cache_tools.py`).

- `os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` drops a specific file's clean pages
  from cache — no privileges required. Before each *cold* run: `os.sync()` (so freshly
  written pages are clean/evictable), then fadvise-DONTNEED the target set — the
  **weights dir** before a cold launch, the **CRIU image dir** before a restore.
- **Verify with `mincore(2)`** (via ctypes): report resident-page % before/after and
  assert ≈ 0 %. fadvise is advisory, so verification is mandatory — every timed run is
  annotated with the measured resident % of its input set, so warm contamination can
  never slip in silently. (Validated: a 283 MB file went 100 % → 0.00 % resident.)

**Measurement protocol.** For each route run **cold** (evict inputs first) and **warm**
(immediately again); report both. Headline = **cold-launch vs cold-restore** (models a
real cold node); warm is a secondary data point. N repeats, evict between cold runs,
median + spread. Caveats stated not hidden: never evict mid-run (mmap'd weights);
GPU-side CUDA-graph/context state is *not* page cache — cold launch paying graph capture
while restore skips it is the effect under study, not a confound.

## Verification

Same discipline as `cuda-checkpoint-exp`: every gate has an explicit pass condition and
a captured error path. For the counter, correctness = the post-restore tick sequence is
contiguous from froze+1 (no reset/gap/jump) — a lost snapshot restarts at 1. For the
SGLang gate, the payoff is measured as wall-clock cold-launch vs restore-from-snapshot,
with the phase breakdown from `servekit profile` where applicable.

## Status (2026-07-23)

| step | state |
|---|---|
| Gate 0 — env probe | ✅ done — criu file caps confirmed, `Seccomp:0` |
| Gate 1 — criu-cpu (plain process) | ✅ **PASS** (the Bristen blocker; passes here) |
| Gate 2 — criu-gpu (C CUDA counter) | ✅ **PASS** (291 MB snapshot, froze→+1 contiguous) |
| env setup — venv + sglang 0.5.10 | ✅ done — `sglang 0.5.10`, `torch 2.9.1+cu128`, CUDA ok on 3090 |
| model download — Apertus-8B → `~/models` | ⏳ in progress (~16 GB) |
| Gate 3a — Python+torch.cuda | ✅ **PASS** (649 MB snapshot, froze→+1 contiguous) |
| baseline — `servekit profile` + `bench` TP1 | ✅ **done** — true-cold **59.3 s** (±0.3%) |
| Gate 3b — full server round trip | ✅ **PASS — cold restore 18.2 s vs 59.3 s = 3.3x** |

**Result:** a warmed Apertus-8B SGLang server (TP1) checkpoints to a **26 GB** snapshot
(21 s) and cold-restores to correct serving in **18.2 s**, vs **59.3 s** cold launch —
**3.3x**. Making the full server checkpointable unprivileged took five source-level
fixes (GPU VMAs on every tree pid; uvloop→epoll shim via `.pth`; `USE_LIBUV=0`;
`--network-lock skip`; `--tcp-close`) — all encoded in `checkpoint_server.sh`; the exact
criu errors and fixes are in
[results/gate3b_server_checkpoint_restore/results.md](results/gate3b_server_checkpoint_restore/results.md).
The snapshot-size
scaling held: 291 MB (C counter) → 649 MB (CPython+torch) → 26 GB (8B server), and that
26 GB read (~14 s) is the restore floor — exactly why page-cache control matters.

## Open questions this experiment feeds

- Does restore-from-disk actually beat a cold launch for an 8B model, once you account
  for reading the (large) snapshot back from storage? A 16 GiB weight set evicted to
  host and dumped to disk is a big image — the snapshot read may eat the savings.
- Does the restored process serve *correct* inference (not just "resumes"), including
  on the first real request (the JIT/lazy-init caveat flagged in `CLAUDE.md`)?
- How does any of this change under TP4 on the cluster, where there are 4 ranks and
  NCCL state to checkpoint — untestable here (single GPU).
