# Lustre contention over 24h — experiment plan

## Question

How does **capstor Lustre read bandwidth vary over a full day/night cycle?**

The sibling `lustre-loading-exp` work established that capstor is
contention-dominated and time-varying — the *same* O_DIRECT read of the 70B
model measured **6.7–8.6 GB/s within ~20 min**, and swings 2–6× have been seen.
Its methodology rule #1 is blunt: *"a bandwidth number from a different job at a
different time is worthless."* But that work only ever sampled bandwidth
opportunistically, once per loader job. **We have never characterized the
diurnal pattern itself.** This experiment does exactly that.

**Hypothesis:** aggregate read bandwidth rises and falls across the day — lower
during business/training-heavy hours, higher overnight — as cluster-wide load on
the shared OSTs changes.

## Method

- **Probe:** read the whole 132 GB Llama-3.1-70B model (30 safetensors shards)
  from capstor, **O_DIRECT**, with a **32-worker pool**, and record aggregate
  **GiB/s**. The probe (`scripts/lib/dd_read_sweep.sh`) is **vendored** into this
  experiment so it is fully self-contained; it originates from the loading
  experiment.
- **Per slot:** run the probe **5×** back-to-back; keep all 5 (min/max shows the
  within-slot warm/cold spread — see caching note below).
- **Cadence:** one job every **30 min for 24 h = 48 slots**.
- **Node policy:** **fresh node per slot** (whatever SLURM hands out; no
  `--exclude`, no dependency). O_DIRECT bypasses the client page cache, so every
  slot's first read is genuinely cold; node-to-node hardware variance is accepted
  noise. This is the deliberate *opposite* of the loading experiment's chains,
  which forced a different cold node and serialized to *avoid* contention — here
  contention is the signal.
- **Scheduling:** all 48 jobs **pre-submitted at once**, each with an absolute
  `--begin` timestamp 30 min apart. Fire-and-forget; no daemon, survives
  disconnect; no single 24h job (each is `--time=00:20:00`).
- **Storage-only:** no GPU, no container, no server — a pure read measurement on
  the host `/capstor` mount.

### Caching note
O_DIRECT bypasses the **client** page cache but not the capstor **OSS-side**
cache. Within a slot, repeats 2–5 may run warmer than repeat 1. That is why we
keep all 5 repeats and plot min..max rather than averaging in-job — the spread is
informative, not noise to hide.

## Files

```
lustre-contention-exp/
  plan.md                        # this file
  scripts/
    contention_probe.sbatch      # one slot: 5× dd @ conc=32, writes GiB/s CSV
    submit_24h.sh                # pre-submits 48 --begin jobs, 30 min apart
    plot_contention.py           # bars(median) + min/max whiskers, y=GiB/s
    lib/dd_read_sweep.sh         # vendored O_DIRECT dd probe (self-contained)
  results/                       # contention-<jobid>-<node>.csv, .out logs, PNG
```

CSV columns: `iso_time,run_index,concurrency,total_bytes,wall_s,agg_GiBps`
(the shared lib emits decimal GB/s; the probe recomputes **binary GiB/s** =
`total_bytes / wall_s / 2^30` so the lib stays untouched).

## Run

From the repo root:

```bash
# 1. sanity-check one slot immediately (does not wait for the schedule)
sbatch lustre-contention-exp/scripts/contention_probe.sbatch

# 2. launch the 24h campaign (48 jobs, staggered --begin)
bash lustre-contention-exp/scripts/submit_24h.sh
squeue --me --sort=S -o '%.10i %.12j %.9T %.20S %R'

# 3. after the window, plot (needs matplotlib; a local venv is set up for it)
lustre-contention-exp/.venv/bin/python lustre-contention-exp/scripts/plot_contention.py
# -> results/contention_timeseries.png + a text summary table
```

The login node only has Python 3.6 with no matplotlib, so plotting uses a
dedicated `lustre-contention-exp/.venv` (Python 3.11 + matplotlib, `.gitignore`d).
Recreate it if missing:

```bash
servekit/.venv/bin/python -m venv lustre-contention-exp/.venv
lustre-contention-exp/.venv/bin/python -m pip install -q matplotlib
```

The probe and scheduler are pure bash + coreutils and need no venv.

Env overrides (both scripts pass `--export=ALL`): `MODEL`, `REPEATS`, `CONC`,
`BS`, `NSLOTS`, `INTERVAL_MIN`. E.g. a 2-slot 1-min smoke test:
`NSLOTS=2 INTERVAL_MIN=1 bash lustre-contention-exp/scripts/submit_24h.sh`.

## Known risk

A `MaxSubmitJobs` / pending-job QOS cap could reject some of the 48. If that
bites, submit in waves — rerun `submit_24h.sh` later with the remaining slots
pushed to later `--begin` times. Not built until it actually happens.
