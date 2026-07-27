# Probe — GDS on bristen

**Status:** done — **not available**

## Goal

InstantTensor's headline numbers (35–45 GB/s) come from GPUDirect Storage
(GDS). If GDS were live on bristen, the InstantTensor evaluation
([`../phase5_instanttensor/results.md`](../phase5_instanttensor/results.md))
would be a different conversation — check before spending more time on it.

## Method

`scripts/probes/gds_probe.sbatch` (job 74760, nid002324) checks three
independent signals: (A) whether the `nvidia_fs` kernel module is loaded and
whether `/dev/infiniband` exists; (B) asks `cuFile` itself via
`cuFileDriverGetProperties()`; (C) asks InstantTensor's own backend
auto-selection what it picks for a real shard on capstor.

## Result

1. **`nvidia_fs` is NOT in `/proc/modules`.** Not a container artifact —
   containers share the host kernel, so `/proc/modules` is the *node's*
   module list. No `/dev/infiniband` either.
2. **`cuFileDriverGetProperties()` reports `COMPATIBILITY MODE: False`**
   at face value — but see Caveats, this specific reading was later
   distrusted.
3. **InstantTensor selects `Backend.URING`** for a real shard on capstor,
   not `Backend.CUFILE` — the library's own auto-selection answers the
   question directly: it uses io_uring + direct I/O through the CPU, not
   DMA.

## Verdict

**GDS is not available on bristen.** Signal (1) is decisive on its own: the
nvidia-fs kernel driver isn't loaded, so true GDS DMA is impossible
regardless of what's bind-mounted into the container. Signal (3) confirms
it independently and is the one actually trusted (see Caveats) — the
library itself falls back to `Backend.URING`.

→ On this system, InstantTensor must win on pipelining + direct I/O alone,
against the fastsafetensors+patch baseline (38.2 s). It does not — see
[`../phase5_instanttensor/results.md`](../phase5_instanttensor/results.md).
Expectations were set before that run, not after.

## Caveats

- **Do not trust a hand-rolled `CUfileDrvProps` ctypes struct.** `cufile.h`
  nests `size_t` fields; an all-`c_uint` layout is misaligned and prints
  garbage — this probe's own early version confidently reported
  "COMPATIBILITY MODE: False" from junk bytes read that way. Signal (3),
  asking the library which backend it actually picked, is the trustworthy
  one; signal (2) above is kept for the record but not relied on.
- The probe's own segfault (`safe_open(..., load_now=False)` on teardown)
  is a known sharp edge in this InstantTensor version, not evidence of
  anything else — the PR path this repo evaluates uses `load_now=True` and
  doesn't hit it.
