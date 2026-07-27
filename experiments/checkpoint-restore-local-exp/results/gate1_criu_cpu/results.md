# Gate 1 — criu-cpu: dump/restore a plain (non-CUDA) process

**Status:** done — **PASS** (this is the gate that failed on Bristen)

## Goal

Isolate "can CRIU run at all in this environment" from anything GPU-related, using the
simplest possible target. This is the exact gate that failed on Bristen (CRIU's
`map_files` open needs `CAP_CHECKPOINT_RESTORE` in the initial userns, unobtainable
there) — passing it here is the whole premise of doing this experiment locally instead
of waiting on a CSCS capability grant.

## Method

`scripts/gate_criu_cpu.sh` runs a trivial loop that appends an incrementing tick to a
file twice a second. It dumps the process with `criu --unprivileged dump`, confirms the
process is gone and the tick is frozen across a window where several more ticks would
otherwise have landed, then restores with `criu --unprivileged restore` and confirms the
counter resumes.

PASS requires: process disappears on dump, tick count stays frozen while dumped, and
after restore the **first** new tick is exactly `froze+1` (proving the in-memory counter
state survived, not just that some process is alive again), with the whole tick sequence
contiguous from start to finish.

## Result

Raw output: `gate1_criu_cpu.txt`.

- Dump: process vanished, tick frozen correctly.
- 3 s wait: tick file unchanged (confirms truly dumped/frozen, not just paused).
- Restore: first post-restore tick was exactly `froze+1`; full sequence contiguous.

## Verdict

**GATE criu-cpu: PASS.** CRIU can dump-to-disk and restore-from-disk here, unprivileged,
for a plain process — the step that was structurally impossible on Bristen. This
de-risks the CRIU side before adding CUDA state (Gate 2) or a real LLM server (Gate 3b).

## Caveats

None — deterministic pass/fail check on a synthetic counter, no GPU or model involved.
