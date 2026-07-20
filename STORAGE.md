# CSCS storage for model staging — investigation

Question: **can we stage model weights on `/iopsstor`?** Covers what's mounted,
what's writable, what's actually enforced, and what eats staged data.

Investigated 2026-07-16 on `bristen-ln001` as `yboughizane`. All numbers below
are from live probes on that node unless linked to CSCS docs.

---

## TL;DR

**Yes — `/iopsstor/scratch/cscs/$USER`, and there is no quota on it at all.**
`/iopsstor/store` exists but is not ours and never will be.

The binding constraint is **not** capacity, it's the **14-day atime reaper** —
which has *already silently deleted* our staged Qwen weights (see below).

| target | writable | quota | survives? |
|---|---|---|---|
| **`/iopsstor/scratch/cscs/$USER`** | **yes** | **none enforced** | 14-day atime reaper |
| `/iopsstor/store/cscs/*` | **no** — no allocation | n/a | n/a |
| `/capstor/scratch/cscs/$USER` | yes | 150 TiB soft, 1M inodes | 30-day atime reaper |
| `/capstor/store/cscs/swissai/infra02` | yes | 100 TiB soft / 150 TiB hard | **backed up, no cleanup** |

Pattern: **golden copy on `capstor/store/…/infra02`, working copy staged to
`iopsstor/scratch`, staged copy always treated as a rebuildable cache.**

---

## `/iopsstor/store` is not an option

It is mounted, but:

- Same Lustre backend as iopsstor/scratch — identical NIDs
  (`172.28.2.66–69@tcp`), same 3.0P at 75%. Two filesets, one filesystem.
- Contains only `HEPScore23-Benchmark/` and `swissai/a01…a11/`. We are in none
  of those gids.
- `touch` → Permission denied at every reachable level.
- The CSCS `quota` tool does not list it for us.
- **CSCS docs do not document an `/iopsstor/store` product.** Store exists only
  on capstor.

## There is no per-user quota on iopsstor scratch

This corrects a wrong first reading of the docs — the "150 TB / 1M inodes"
figure is **capstor scratch only**. Docs state *no* iopsstor quota; the only
iopsstor policy they publish is the 14-day deletion.

Project id `6980` (`lfs project -d /iopsstor/scratch/cscs/yboughizane`):

```
$ lfs quota -p 6980 /iopsstor/scratch/cscs
     kbytes   quota   limit   grace   files   quota   limit   grace
  107587072       0       0       -   90390       0       0       -
pid 6980 is using default block quota setting
```

...and the filesystem default it falls back to is itself unlimited:

```
$ lfs quota -P /iopsstor/scratch/cscs
   bquota  blimit  bgrace   iquota  ilimit  igrace
        0       0  604800        0       0  604800
```

In Lustre, `0` = no limit. The CSCS `quota` tool agrees — `-` in the limit
column for iopsstor, real numbers everywhere else:

```
| /iopsstor/scratch/cscs/yboughizane  | LUSTRE | 102.6G |    - |    - |        - |   90386 |    - |         - |
| /capstor/store/cscs/swissai/infra02 | LUSTRE |  30.2T | 30.2 |    - |   100.0T | 2479766 | 82.7 |   3000000 |
| /capstor/scratch/cscs/yboughizane   | LUSTRE |  22.7G |  0.0 |    - |   150.0T |      72 |  0.0 |   1000000 |
```

What *is* enforced, for contrast:

| path | soft | hard | inodes soft/hard |
|---|---|---|---|
| `iopsstor/scratch/…/$USER` | none | none | none |
| `capstor/scratch/…/$USER` | 150 TiB | none | 1M / none |
| `capstor/store/…/infra02` | 100 TiB | 150 TiB | 3M / 4M |

So capstor scratch's 150 TB is **soft**, with a two-week grace
(`bgrace=1209600`), not a wall. Note iopsstor's default grace is **7 days**
(`604800`) — half capstor's — if CSCS ever does set a soft quota there.

The missing quota almost certainly cuts both ways: nothing stops one user
filling shared NVMe, which is plausibly *why* the reaper is aggressive. **The
14 days is the enforcement mechanism, standing in for a limit.** Don't read
"no quota" as "no policy".

## The 14-day reaper — and proof it fires

> Files on `/iopsstor/scratch/cscs/$USER` that have not been **accessed** in 14
> days are automatically deleted. (capstor scratch: 30 days)

**It already ate our weights.** `models/Qwen3-32B`, `models/Qwen3-30B-A3B`,
`models/Qwen2.5-Math-7B` are empty shells — created in March, total 52K, 3
files, all that survives is a `.cache/huggingface/.gitignore` stub per model.
The safetensors are gone. Nobody noticed. This is not a theoretical policy.

### atime does update on read — but only on a *real* read

Worth stating precisely, because the first probe was misleading:

- Backdated atime to 2020, read the file, atime **did not move**. That read was
  served from **page cache** and never touched an OST.
- Same test with `dd iflag=direct` → **atime updated immediately.**

So: **a genuine cold read refreshes atime and protects the file.** A model
served entirely out of page cache, or idle for two weeks, does not get
protected. Refresh staged data with a **periodic O_DIRECT read of each shard**,
not `touch -a` — the reaper is atime-based and a real read is honest about it.

⚠️ **Unverified:** whether deliberately refreshing atime to dodge the reaper is
sanctioned by CSCS. The Confluence KB page on the scratch policy returned empty.
Worth confirming before automating it.

### Second risk: the occupancy sweep

iopsstor is at **75% full** (2.2P used, 761.8T free of 2.9P). Documented
thresholds: **60%** users asked to remove data, **80%** automatic removal.
We sit between them — staged data is exposed to a capacity sweep *regardless of
atime*.

## Group changed: infra01 → infra02

`lustre-loading-exp/NOTES.md` says *"I'm in group **infra01**"*. **No longer
true.** `id` now reports `infra02` only (uid 1609, gid 65604).

- `/capstor/store/cscs/swissai/infra01/hf_models` is now **read-only** for us —
  access comes via `other::r-x`, not group membership. Confirmed: `touch` →
  Permission denied.
- Every model path in the repo points into `infra01/hf_models` (Llama-3.1-70B,
  Apertus-8B, the `cold-start-experiments/llama70b_c*_s*` layouts). **Reads
  still work**; writes do not.
- Writable project space is now `/capstor/store/cscs/swissai/infra02`.

**Tightest real constraint anywhere:** infra02 inodes at **82.7% (2.48M / 3M
soft, 4M hard)** — not bytes (30.2 / 100 TiB).

## Why staging is worth it at all

Every model path in the repo currently loads from `/capstor/store/…` — the
**HDD-backed** store. CSCS docs explicitly say to avoid Store for jobs (fewer
metadata servers). iopsstor is NVMe: 240 × 30 TB SSD, 7.2 PB raw RAID 10, 20
OSSs / 2 MDSs, **782 GB/s read**, 8.6M read IOPS. Staging to iopsstor is
precisely the fix for reading weights off Store.

Caveat on OST count: **capstor has ≥150 OSTs, iopsstor scratch only ~20.** Per
`lustre-loading-exp`, aggregate loader throughput is dominated by reader
concurrency across OSTs, so iopsstor is **not automatically faster** for a
30-shard parallel read — it needs measuring, not assuming.

Not benchmarked here: the one number taken (`109 MB/s`, capstor/store,
single-stream O_DIRECT, login node) is **not representative** — login nodes are
throttled and `lustre-loading-exp` measures 6.7–8.6 GB/s on compute nodes at
N=30. **A fair iopsstor-vs-capstor comparison must be run on a compute node at
realistic concurrency.**

Shards are `stripe_count: 1` on capstor; the iopsstor default is also 1. Given
~20 OSTs, a wider stripe is worth testing if we stage — though note
`lustre-loading-exp` already found striping bought nothing on capstor.

---

## Open questions

- Is atime-refresh-to-dodge-the-reaper sanctioned? (KB page was empty.)
- Does iopsstor actually beat capstor for a 30-shard concurrent load, given
  ~20 OSTs vs ≥150? Compute node, realistic concurrency.
- Does a wider stripe help on iopsstor's smaller OST count?
- infra02 inode headroom (82.7%) — does a staged golden copy fit the budget?

## Reproducing these checks

```bash
quota -d                                    # CSCS view, all filesystems
lfs project -d /iopsstor/scratch/cscs/$USER # project id
lfs quota -p <id> /iopsstor/scratch/cscs    # enforced quota
lfs quota -P /iopsstor/scratch/cscs         # filesystem default (0 = unlimited)
lfs df -h /iopsstor/scratch/cscs            # occupancy vs 60/80% thresholds
find <dir> -type f -atime +14 | wc -l       # what the reaper is about to take
```

## Sources

- [File Systems — CSCS Docs](https://docs.cscs.ch/storage/filesystems/)
- [Storage — CSCS Docs](https://docs.cscs.ch/alps/storage/)
- [Machine Learning Platform — CSCS Docs](https://docs.cscs.ch/platforms/mlp/)
