# Gate 0 — environment probe

**Status:** done — this is the fact that motivates the whole experiment

## Goal

Before running anything, confirm this local box actually differs from CSCS Bristen in
the one way that matters. On Bristen, `cuda-checkpoint` worked but CRIU was blocked:
CRIU's `open()` of `/proc/PID/map_files/*` needs `CAP_CHECKPOINT_RESTORE` in the
**initial** user namespace, and no cluster user could obtain that. If this box has the
same wall, the rest of the gates below are moot.

## Method

`scripts/probe_env.sh` — a read-only, safe-to-rerun script — records user/caps,
GPU/driver, RAM, the `criu` binary's version and file capabilities (`getcap`), seccomp
state, and kernel knobs (`ptrace_scope`, `unprivileged_userns_clone`).

## Result

Raw output: `gate0_probe.txt`.

- `criu` (`/usr/sbin/criu`, v4.2) carries `cap_sys_ptrace,cap_checkpoint_restore=ep` as
  a **file capability** — this is the exact `setcap` remedy the Bristen write-up asked
  CSCS to apply, already present here.
- Host `Seccomp:0` (no filter) — the shell's own `CapEff` is `0000000000000000` (same
  `EPERM`-on-`map_files` symptom as Bristen would show from the shell), but that's
  irrelevant: a file capability on the `criu` binary makes the capability effective in
  **criu's own process**, in the initial userns, independent of the caller's shell caps.
- RTX 3090 24 GiB, driver 610.43.02, 125 GiB RAM, kernel 7.0.0-28-generic,
  `ptrace_scope=1`, `unprivileged_userns_clone=1`.

## Verdict

This box clears the precondition Bristen lacked. The right test from here on is not
"can my shell open `map_files`" (it can't, same as Bristen) but "can the `criu` binary
itself complete a dump/restore" — which is what Gates 1–2 test directly, invoking
`criu … --unprivileged` (not `unshare -Urpf`, which only grants namespace-root and does
**not** satisfy the init-userns check — the Bristen dead end).

## Caveats

None — this is a direct, read-only measurement.
