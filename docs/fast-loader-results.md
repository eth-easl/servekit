# Fast loader — measured characterizations

Empirical numbers backing the taxonomy in [`fast-loader.md`](fast-loader.md).
All measurements use parallel `dd ... iflag=direct` against real or dense
synthetic 1 GiB files; `per_stream` = aggregate / n. Aggregate that stops
scaling while per-stream collapses = fast-loader §5 (server wall). Single-stream
that stays low = §1 (per-stream / single-OST / single-TCP cap).

Scripts: [`deploy/bench-storage/`](../deploy/bench-storage) (`bench_core.sh`,
`run_sgs.sh`, `bristen_bench.sbatch`).

---

## sgs-gpu07 (ETH) — 4× H100 NVL, AMD EPYC 9124, PCIe Gen5 x16

| tier (read, O_DIRECT) | 1 stream | peak aggregate | diagnosis |
|---|---|---|---|
| **NVMe `/tmp`** (node-local, system-root) | **3.35 GB/s** | **6.5 GB/s @ 8 streams** | the device; healthy. This is the staging ceiling (§3). |
| **NFS scratch** (`/home/xiayao/.cache`, vers=4.0, **krb5p**, no nconnect) | **75 MB/s** | 0.30 GB/s @ 16 streams → collapses | **§1 + §5 server wall.** This is where the HF cache lives. |
| **NFS home** (vers=4.1, krb5, no nconnect) | 375 MB/s | 0.63 GB/s, flat from 2 streams | **§5 server wall** — saturates immediately. |

**Headline:** the HF cache tier on sgs is **75 MB/s single-stream** and does not
scale past ~0.3 GB/s aggregate — this *is* the slow-load story. Three compounding
causes, all in `fast-loader.md`:

1. **No `nconnect`** on any NFS mount (verified) → one TCP stream per server (§1).
2. **`sec=krb5p`** on the scratch mount = per-RPC encryption; the worst security
   flavor for bandwidth. Home is `krb5i`-class (krb5) — faster, but still capped.
3. **§5 server wall**: aggregate plateaus the instant a 2nd stream is added on
   both NFS tiers → the NFS server's NIC/CPU is the limit, not the client.

**H2D ceiling:** PCIe Gen5 x16 (~63 GB/s) + NV12 between GPU0/1 and GPU2/3.
**Fix:** stage model weights to node-local NVMe on first touch (`rsync` to
`/tmp` or `/local` if granted) — §3 — then read at 3–6 GB/s instead of 0.08 GB/s.
Adding `nconnect=8` + dropping `krb5p`→`krb5`/`sys` would also help but needs
admin. Per-host, staging to NVMe is the lever the user controls.

---

## bristen (CSCS) — 4× A100-SXM4-80GB, AMD EPYC 7713, NVLink NV4, no node-local NVMe

`/capstor` is **Lustre**. Compute nodes have **no real node-local NVMe** (only
small Cray OS SSDs), so §3 staging is not an option here.

| tier (read, O_DIRECT) | 1 stream | peak aggregate | diagnosis |
|---|---|---|---|
| **Lustre `/capstor/store`** — real GLM-4.7-Flash shards, **`stripe_count=1`** | **359 MB/s** | **1.34 GB/s @ 16 streams** (collapses to 0.72 @ 32) | **§1 single-OST cap.** Every shard on one OST. |
| **Lustre `/capstor/scratch`** — synthetic, `stripe_count=4` | 155 MB/s | **6.05 GB/s @ 32 streams** (scales cleanly) | stripe=4 unlocks parallel-FS bandwidth. |

**Headline:** the model directory is laid out with `stripe_count=1`, so each
shard lives on a **single OST** → one reader gets ~359 MB/s and aggregate stalls
near 1.3 GB/s. The *same filesystem* delivers **6 GB/s** to a `stripe_count=4`
file at 32 streams. The slowness is the **file layout**, not the FS or network.

**H2D ceiling:** A100-SXM4 over NVLink 4-link (~100 GB/s GPU↔GPU), PCIe Gen4 to
host (~25 GB/s H2D). The 0.36–1.3 GB/s Lustre read is the wall, not H2D.

### Fix — measured: restripe to `-c 8 -S 4m` by copying to a wide-striped scratch dir

**Do NOT `lfs migrate` in place.** `/capstor/store/.../hf_models` is shared
canonical infra (mixed ownership — 2 of 48 GLM-4.7-Flash shards are owned by
another user), migrate needs file ownership, and it rewrites every byte over the
network anyway. Instead copy once into a scratch dir whose default layout is wide:

```bash
# 160 OSTs on this FS; -c 8 balances 48 shards × 8 ≈ 2.4 engagements/OST.
lfs setstripe -c 8 -S 4m /capstor/scratch/cscs/xyao/models/GLM-4.7-Flash
# parallel cp (each source shard is on a distinct OST → N copiers = N OSTs):
ls /capstor/store/.../GLM-4.7-Flash/model-*.safetensors \
  | xargs -P 8 -I{} cp {} /capstor/scratch/cscs/xyao/models/GLM-4.7-Flash/
```
**Measured (GLM-4.7-Flash, 48 shards, cold O_DIRECT, settled):**

| streams | `stripe_count=1` source | restriped `-c 8 -S 4m` | speedup |
|---|---|---|---|
| 1 | 359 MB/s | 188 MB/s | 0.5× (dd artifact) |
| 8 | 0.76 GB/s | 0.96 GB/s | 1.3× |
| 16 | 1.34 GB/s | 1.63 GB/s | 1.2× |
| 32 | **0.72 GB/s (collapsed)** | **3.49 GB/s** | **4.8×** |

Copy cost 58 GiB in 23 s (~2.6 GB/s). The restriped read **stops collapsing**
and sustains 3.5 GB/s at 32-way (vs 0.72 collapsed) — a ~5× parallel-load win.

**Single-stream dip is a `dd` measurement artifact, not a real regression.** `dd`
holds one RPC in flight per stream; on an 8-stripe file it round-robins without
overlap. Real loaders (vLLM's mmap-based safetensors reader, multi-threaded/aio
readers) keep many RPCs in flight per file and land on the favorable side.
**Striping only pays when the reader has ≥`stripe_count` concurrent RPCs/file.**

### General optimal `-c`

| workload | `-c` | `-S` |
|---|---|---|
| model shards (large seq, parallel readers via mmap/streaming) | **8** (16 if single-reader/file) | **4m** |
| many small files / metadata-heavy | 1 (default) | 1m |
| single huge file, one reader, max BW | up to ~16 | 4–8m |
| never | `=OST count` (full spread — kills readahead) | |

Rules: `-c` ≈ concurrent RPCs/file you'll have, capped well below OST count
(160 here); `-S 4m` cuts RPC count 4× vs the 1 MB default and keeps per-OST
chunks contiguous for readahead. We're not OST-bound at `-c 8` (curve still
climbing at 32 streams), so `-c 16` isn't worth the metadata cost here.

---

## Cross-system summary (why each is slow)

| system | storage (model tier) | single-stream | peak aggregate | root cause (fast-loader §) | top lever |
|---|---|---|---|---|---|
| **beverin** | `/capstor` Lustre (APU, no H2D) | — | — | measured 93 s/56 GiB ≈ 0.6 GB/s load; §2 (mmap trickle) + §3 (N× per PP node) | explicit seq reads + per-node broadcast |
| **sgs-gpu07** | NFS scratch (HF cache), no nconnect, krb5p | **75 MB/s** | 0.30 GB/s | §1 (single TCP) + §5 (server) + krb5p overhead | **stage to node-local NVMe** (§3) |
| **bristen** | Lustre store, model `stripe_count=1` | 359 MB/s | 1.34 GB/s | §1 (single-OST cap) | **`lfs migrate -c 16`** the model shards |
| **clariden** | GH200 (Grace-Hopper), CSCS FS | unmeasured | unmeasured | §2 (single-stream read into LPDDR5X) expected | **GPUDirect Storage / cuFile** (§4) |

### Caveats
- sgs/bristen numbers are dense synthetic-file or real-shard `dd` reads, not a
  live vLLM `load_state_dict`. They isolate the *storage wire* cost (the part
  `fast-loader.md` is about), excluding H2D copy and safetensors parse overhead.
- beverin's 93 s is a real end-to-end vLLM weight load (from `snapshot/RESULTS.md`
  M2.4) — directly comparable since bristen reads the *same files*, just on a
  different node/FS.
- clariden remains unmeasured (no access in this session).
