# Phase 4 — Strategy B: `/dev/shm` staging, and 4b: the stager is CPU-bound

**Status:** done

## Goal

Test whether staging the model to node-local `/dev/shm` before serving beats
reading straight from Lustre, decoupling reads from capstor contention
entirely. mmap is again an explicit test axis, not assumed to win or lose —
tmpfs has no page-fault-from-disk cost, so the mmap-vs-Lustre finding
(Phase 1.3) might invert here.

## Method

Stage from a striped copy (Phase 2's `c8_s16M`) with a 60-worker parallel
copy script (`scripts/phase4_shm/stage_to_shm.sh` /
`stage_to_shm_sliced.sh`), then serve with `--model-path
/dev/shm/llama70b`, crossed with mmap / fastsafetensors / nommap.

Phase 4b isolates a second variable discovered mid-Phase-4: the sliced
stager measured 19.4–20.2 GB/s standalone but only 11.5–12.2 GB/s inside the
actual e2e jobs, on the *same node* minutes apart — ruling out node/Lustre
drift. `scripts/phase4_shm/stage_isolate_container.sbatch` (job 75713,
nid002280) varies CPU count and container presence independently, one job,
one node, four legs.

## Result

### Phase 4 — staging + loader on tmpfs

| variant | stage | weight_loading | server total | **e2e incl. staging** |
|---|---|---|---|---|
| **shm + mmap** | 21.2 s @ 6.65 GB/s | **19.6 s** | 177.6 s | **198.8 s** |
| shm + fastsafetensors | 21.8 s @ 6.46 | 25.1 | 195.8 | 217.6 |
| shm + nommap | 23.8 s @ 5.93 | 103.6 | 270.4 | 294.2 |

### Phase 4b — CPU count vs container, isolated (job 75713, nid002280)

| leg | CPUs | container | GB/s | stage |
|---|---|---|---|---|
| `batch_bare` | 128 | no | **20.54** | 6.87 s |
| `srun_bare_64` | 64 | no | 12.07 | 11.69 s |
| `srun_ctr_64` | 64 | yes | 11.59 | 12.18 s |
| `srun_ctr_128` | 128 | yes | **18.35** | 7.69 s |

`/dev/shm` came back byte-identical across all four legs (same tmpfs mount,
`size=369160116k`; container only adds `nosuid,noexec`) — a differing tmpfs
mount is ruled out.

### Repeatability check — sliced stager, shm + mmap, 3 more fresh nodes

The single shm+mmap point above (198.8 s) is corroborated by 3 later
fresh-node runs using the improved sliced stager (`stage_to_shm_sliced.sh`,
64 CPUs, direct reads unless noted):

| job | node | stage GB/s | stage_wall_s | weight_loading | server total | **e2e incl. staging** |
|---|---|---|---|---|---|---|
| 75168 | nid002297 | 11.66 | 12.10 | 19.58 | 187.43 | ≈199.5 |
| 75196 | nid002281 | 11.80 | 11.96 | 19.09 | 193.08 | ≈205.0 |
| 75335 | nid002280 | 11.66 | 12.10 | 18.61 | 185.23 | ≈197.3 |
| 75193 (buffered read) | nid002288 | 7.50 | 18.82 | 19.69 | 189.48 | ≈208.3 |

Direct-read staging is consistently ~11.7–11.8 GB/s; the one buffered-read
variant (75193) is markedly slower to stage (7.50 GB/s) with no compensating
benefit downstream — O_DIRECT is the right choice for this stager
regardless of the CPU-count question Phase 4b investigates below.
**shm + mmap e2e lands in a ~197–208 s band across 4 independent
measurements**, consistent with (not just a lucky single sample of) the
original 198.8 s figure.

## Verdict

**Two inversions worth remembering:**

1. **mmap is the BEST loader on tmpfs (19.6 s) and the WORST on Lustre
   (939 s, Phase 1.3).** The flag you must never use on Lustre is the one
   you want once the bytes are already in RAM — demand-paging is free when
   there's no disk behind the page.
2. **The staging script beat the engine's own loader at reading Lustre**
   (6.65 vs fastsafetensors-upstream's 1.78 GB/s from Phase 1.3/3) — not
   because tmpfs is magic, but because `cp` ran 60 workers where the loader
   ran 4. `/dev/shm` was never beating the storage; it was beating *the
   loader*. This is exactly what Phase 3 later fixed directly.

**Phase 4b: it is CPU count, not the container.** At fixed 64 CPUs the
container costs only 12.07 → 11.59 GB/s (4%). At fixed container, 64 → 128
CPUs is 11.59 → 18.35 GB/s (1.58×). The stage is CPU-bound: O_DIRECT reads
DMA into the user buffer, but all 141 GB are then memcpy'd by the CPU into
tmpfs pages (~40 GB/s of memory traffic at these rates). Mechanism: SLURM's
`TaskPlugin=task/affinity` restricts via `sched_setaffinity`, not the cgroup
cpuset — every leg printed `cpuset: 0-127` while `nproc` read 128/64/64/128,
and the affinity mask is inherited by all 1744 forked `dd`s. Reading
`cpuset.cpus.effective` alone would have shown "no restriction" and missed
this entirely.

**Overall for Phase 4**: shm+mmap (198.8 s incl. staging) edges out the
Phase-3-patched fastsafetensors-on-Lustre result (208.2 s) but costs 141 GB
of RAM and a stager. Its real value is for *warm restarts* (near-free once
staged) or once the stage is overlapped with startup — see
[`../phase7_overlap_stage/results.md`](../phase7_overlap_stage/results.md),
which builds directly on the Phase 4b CPU-count fix.

**Recommendation (not applied — recipe, not a code change):** raising the
job-level `--cpus-per-task` from 64 to 128 takes the stage ~12.2 s → ~7.7 s,
~4.5 s of cold start for free, if `/dev/shm` staging is adopted. Must be the
job-level SBATCH directive, not just the inner `srun` value — `srun_ctr_128`
only ran because `--exclusive` had 128 CPUs available; SLURM warns "Job
step's --cpus-per-task value exceeds that of job" otherwise. Scaling is
sublinear (2× CPUs → 1.58×), so 128 is roughly where this stops paying, and
giving SGLang 128 CPUs may shift other cold-start phases — the e2e total is
not guaranteed to drop by the full 4.5 s.

## Caveats

- The Phase-4b isolation script that would have measured "CPU count fixed,
  container varied" first (an earlier draft) would have concluded "bare ≈
  container ≈ 12 GB/s, container is innocent" and stopped there — the real
  variable (CPU count) was never moved in that draft. Worth remembering as a
  methodology trap: vary the thing you haven't yet excluded, not just the
  thing you suspect.
