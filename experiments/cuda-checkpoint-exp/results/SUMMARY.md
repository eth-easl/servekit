# Result: CUDA checkpoint/restore on Bristen

**Verdict: the GPU half works, the process half does not.**

`cuda-checkpoint` fully works as an ordinary cluster user — we can evict a live
process's GPU state to host RAM and restore it, with device memory intact.
**CRIU cannot run at all**, because the kernel requires a capability in the
*initial* user namespace that no cluster user can obtain. So today we can do
*GPU eviction/restore on a live process*, but **not** snapshot-a-process-to-disk-
and-restore-it-later (which is the part that would actually buy cold-start time).

Evidence: jobs `75050`, `75058`, `75060`, `75064` (independent runs, identical conclusions).
Environment: Bristen, NVIDIA **A100-SXM4-80GB**, driver **590.48.01**, CUDA 13.1,
host SLES15-SP6 / glibc 2.38, kernel 6.4.0.

## Gates

| Gate | Result | Evidence |
|---|---|---|
| driver ≥ 550 | **PASS** | 590.48.01 — same version as the reference blog post |
| `cuda-ckpt` on the **bare host** | **PASS** | froze at 9, resumed at exactly 10 (job 75064) |
| `cuda-ckpt` **inside the pyxis container** | **PASS** | same result as normal user with `Seccomp=2` (job 75080) — *this is where real serving runs* |
| `map-files` (criu prerequisite) | **FAIL** | `open()` of `/proc/PID/map_files/*` → EPERM |
| `criu-cpu` (dump a plain process) | **FAIL** | `Can't open 22's mapfile link: Operation not permitted` |
| `criu-gpu` (full round trip) | **skipped** | blocked by `criu-cpu` |

**cuda-checkpoint works in the container**, as an ordinary user (`uid=1609`) with the
seccomp filter active (`Seccomp=2`) and no `--container-remap-root` — i.e. under the
exact conditions an SGLang serving job runs under (job 75080). The seccomp filter
blocks criu but not cuda-checkpoint: criu has to *suspend* the filter
(`PTRACE_O_SUSPEND_SECCOMP`, initial-userns only), whereas cuda-checkpoint only needs
same-uid ptrace. This matters — the bare-host result alone would not have told us
anything about the environment we actually deploy into.

## What PASSED — cuda-checkpoint (job 75064)

Run as the normal user (`uid=1609`), on the bare node, no container, no elevated
privileges. The counter ticks once per second and lives **in device memory**:

```
tick_ms=1000  freeze_window=8s  (=> ~8 ticks expected if NOT frozen)
counter pid=95735  tick=7
GPU procs while running:      95735, 418 MiB
   lock rc=0 / checkpoint rc=0      state: checkpointed
   last tick while checkpointed: tick=9
GPU procs while checkpointed:  (empty)      <-- 418 MiB fully released
   confirmed FROZEN: still tick=9 after 8s (~8 ticks should have elapsed)
   restore rc=0 / unlock rc=0       state: running
   first tick after restore : tick=10  <- the proof; must be tick=10 (= froze+1)
   counter now at           : tick=18  <- only how far it got during the ~7s sleep
GATE cuda-ckpt  : PASS  device memory intact: froze at 9, resumed at exactly 10
```

**Reading these numbers:** the counter froze at 9 and resumed at **10** — froze+1,
which is the whole claim. The trailing `tick=18` is *not* a jump from 9; it is simply
where the counter had got to seven seconds later, because the script deliberately
lets it run on to show it keeps ticking. That value carries no correctness
information (it would be 25 if we slept longer) and must not be read as the restore
result.

Three independent things are verified, deliberately — first/last values alone would
not be enough:

1. **The GPU really was released**: the process's 418 MiB disappears from
   `nvidia-smi --query-compute-apps` entirely while checkpointed.
2. **The process really was frozen**: it sat at `tick=9` across an 8s window in which
   ~8 ticks should have occurred (this is why the freeze window must scale with
   `TICK_MS` — "no ticks for 2s" would prove little at 1 tick/s).
3. **Device memory really survived, with no reset/gap/jump**: the first tick after
   restore is exactly `froze+1`, and the log is checked mechanically for the full
   sequence `tick=1..18` — each value present exactly once, in order:

```
  RESULT: contiguous tick=1..18, no reset/gap/jump (froze at 9)
  tick=9    <- before checkpoint
  tick=10   <- FIRST tick after restore (must be 10)
```

A lost or re-zeroed device counter would restart at 1 and the sequence check would
report BROKEN. This is the blog's first example, working on Bristen.

## What FAILED — CRIU, and exactly why

Not a missing package or a tunable — a kernel permission rule:

```
Error (criu/proc_parse.c:693): Can't open 22's mapfile link 55abde7ac000: Operation not permitted
Error (criu/cr-dump.c:1563): Collect mappings (pid: 22) failed with -1
```

CRIU must `open()` `/proc/PID/map_files/*` for every file-backed mapping. The kernel
gates that behind `checkpoint_restore_ns_capable(&init_user_ns)` — CAP_SYS_ADMIN or
CAP_CHECKPOINT_RESTORE **in the *initial* user namespace**. Being root inside our own
namespace does not count, by design.

The `map-files` gate isolates this precisely: on the same link, `readlink()` succeeds
(returns `/usr/bin/sleep`) while `open()` returns EPERM. Those are different kernel
paths; CRIU needs the one we don't have.

### `criu --unprivileged` does not help (job 75083)

`criu check --unprivileged` / `criu dump --unprivileged` selects criu's **non-root**
mode (Linux 5.9+). It is worth being precise about what that mode means: it removes
the need for **full root**, but it still requires **CAP_CHECKPOINT_RESTORE in the
initial userns**. It is not a "no capabilities" mode. Tested in every environment, it
changes nothing — the failure just moves to whichever check comes first:

| environment | uid / CapEff / Seccomp | `dump --unprivileged` fails at |
|---|---|---|
| container, normal user *(as real serving runs)* | 1609 / none / 2 | refuses immediately: needs CAP_SYS_ADMIN or CAP_CHECKPOINT_RESTORE |
| container, `--container-remap-root` | 0 / full / 2 | `suspending seccomp failed` (never reaches map_files) |
| bare host, `unshare -Urpf` | 0 / full / 0 | `Can't open map_files link` |

Three different walls, one shape: **every one needs a capability in the *initial*
user namespace**, and namespace-root satisfies none of them.

Usefully, criu names its own remedy verbatim:

```
CRIU needs to have the CAP_SYS_ADMIN or the CAP_CHECKPOINT_RESTORE capability:
setcap cap_checkpoint_restore+eip .../bin/criu
```

Two things ruled out along the way, each with evidence:

1. **`unshare -Urpf` is not a workaround.** It gives root, full `CapEff`, no seccomp
   and our own PID namespace — and CRIU *still* fails, because the map_files check is
   explicitly against `init_user_ns`. This is the load-bearing result: there is no
   unprivileged path to CRIU here.
2. **The pyxis/enroot container carries a *second, independent* blocker.** Be precise
   about this one: it is **not** why CRIU fails today (map_files above is — that
   blocks the bare host too, where `Seccomp: 0`). It matters for what happens *after*
   the capability is granted.

   Containers run `Seccomp: 2` (observed with and without `--container-remap-root`).
   The filter is registered by `enroot-nsenter` unconditionally: no seccomp flag among
   its options (`--target/--user/--mount/--remap-root/--envfile/--workdir`), no
   `ENROOT_*SECCOMP*` variable, nothing in `/etc/enroot/`, no pyxis `srun` option.
   There is no supported way to opt out. *(Established from strings/flags, not
   disassembly — a hidden conditional can't be fully excluded.)*

   That matters because `PTRACE_O_SUSPEND_SECCOMP` accepts **only CAP_SYS_ADMIN** in
   the initial userns — explicitly **not** CAP_CHECKPOINT_RESTORE. So granting
   `cap_checkpoint_restore` would make CRIU work **on the bare host but still not
   inside pyxis**, which is exactly where SGLang serving runs. Hence the tests run on
   the bare host: it isolates the fundamental blocker from this one.

Incidental (not the blocker, but they'd also need privileges): `RLIMIT_NOFILE` needs
CAP_SYS_RESOURCE, TCP-repair mode needs CAP_NET_ADMIN.

## What this means for cold-start

- **Available now:** a live process can release the GPU and take it back with its
  device memory intact. That enables GPU yielding / warm standby — a *resident*
  process parking its GPU state in host RAM and reclaiming it fast — but the process
  must stay alive, so it does not survive job boundaries.
- **Not available:** snapshot-to-disk + restore-later, i.e. the actual cold-start win
  (skip imports, weight load, CUDA graph capture). That needs CRIU.
- **To unblock — the concrete ask for CSCS.** criu states it itself, and it is smaller
  than "run privileged containers". Two things are needed, and the second is easy to
  miss:
  1. **`setcap cap_checkpoint_restore+eip <criu binary>`**, done by an admin, on a
     filesystem that supports the `security.capability` xattr (Lustre may not — the
     binary may need to live elsewhere). File caps set inside a userns would *not*
     work: they are namespaced, and these checks are against `init_user_ns`.
  2. **The target must not carry a seccomp filter**, or criu additionally needs
     CAP_SYS_ADMIN — `PTRACE_O_SUSPEND_SECCOMP` accepts *only* CAP_SYS_ADMIN, not
     CAP_CHECKPOINT_RESTORE. Since pyxis containers run with `Seccomp:2`, cap 1 alone
     would let us checkpoint processes on the **bare host** but still **not** an
     SGLang server inside pyxis. That needs a seccomp-free container path too.

  Worth asking: the driver is already new enough and the GPU half demonstrably works.
  This is purely a site-policy blocker, not a hardware or driver one.

## Reproduce

```bash
sbatch cuda-checkpoint-exp/scripts/ckpt_restore.sbatch   # ~90s
cat cuda-checkpoint-exp/results/cuda-cr-<jobid>.out
```

## Notes for whoever picks this up

- criu is **not packaged for Ubuntu 24.04** (removed from noble); the image is
  jammy-based for that reason.
- criu is built **without `libnftables-dev`** on purpose: its nftables kernel probe
  fails here and criu ≤3.16 makes that fatal. Built without it, the probe is compiled
  out. (Ubuntu's criu 3.16.1 dies on this; we build 3.19 from source.)
- Host glibc (2.38) is newer than jammy's (2.35), so jammy-built binaries run natively
  on the host; only `libnet.so.1` had to be bundled.
