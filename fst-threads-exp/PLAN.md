# fastsafetensors reader threads — does the *second* concurrency axis move the loader?

> ## ⚠️ FALSIFIED — read `NOTES.md` first
>
> The hypothesis below (**"`max_threads` moves weight_loading"**) is **wrong**,
> measured: 16 → 32 made weight_loading **worse**, 76.75 → 104.01 s, with the
> in-job storage probe flat. `max_threads` is not a read-concurrency knob at
> SGLang's call site at all — fastsafetensors submits **one read per
> `max_copy_block_size` chunk** (default **16 GiB**), so a 5 GB shard is **one
> submit, one thread**, whatever `max_threads` says. It only allocates
> `bbuf_size_kb × max_threads` of pinned host memory per rank, which is pure cost.
>
> The plan's own "counter-hypothesis worth taking seriously" and its
> `bbuf_size_kb` risk note were both closer to the truth than the headline.
>
> **Live axis 2 is `max_copy_block_size`** (`scripts/submit_mcbs_sweep.sh`).
> Kept unedited below as the record of what was predicted and why.

## Question

**Weight loading is not storage-bound. Does raising fastsafetensors'
per-file thread count close the gap?**

`lustre-contention-exp/DD_VS_FASTSAFETENSORS.md` established, same node, same
minutes:

| reader | GB/s |
|---|---|
| `dd`, 30 files x **1** reader each (the old probe) | 0.72 |
| `dd`, 30 files x **32** readers each | **18.9** |
| fastsafetensors, whole model, TP=4 | **1.5–1.9** |

capstor hands over the whole 141 GB model in **~7 s** when asked properly. The
loader takes **69–93 s**. Storage has ~10–25× more headroom than the loader
uses, so the bottleneck is **the loader's own request concurrency**, not capstor.

There are **two** concurrency axes, and they compose:

| axis | knob | upstream default | tuned by |
|---|---|---|---|
| files in flight | `SGLANG_FST_FILES_PER_RANK` | 1 (→ 4 files at TP=4) | phase 3 ✅ (1→8 bought 2.3×) |
| **threads within a file** | **`max_threads`** | **16** | **never — this experiment** |

Phase 3 only ever tuned the first. The sliced `dd` sweep shows the second is
worth 26× to raw reads. **Hypothesis: `max_threads` moves weight_loading, and
composes multiplicatively with `files_per_rank`.**

Counter-hypothesis worth taking seriously: `lustre-loading-exp` measured that of
`fpr8`'s 38.2 s, only ~13–18 s is reading — the rest is non-overlapped H2D +
NCCL broadcast. **If so, no read-side knob can go below ~20 s** and `max_threads`
will do nothing once `fpr` is already high. A null result here is therefore a
*real* result: it would pin the floor and redirect the whole cold-start effort
at graph capture (~106 s of the ~208 s total).

## Method

- **Patch:** `scripts/fst_threads.patch` makes both axes env-tunable in
  `weight_utils.py::fastsafetensors_weights_iterator`. Generated with `git diff`
  against the pinned SHA, not hand-written. **With no env vars set it is
  byte-for-byte upstream** (verified: `files_per_rank=1` → `chunk = pg.size()`;
  empty `fst_kwargs` → `SafeTensorsFileLoader(pg, device)`; the round-robin
  `rank_file_map` reduces to upstream's `enumerate()`, short final chunk
  included). Applied in-container by the shared
  `lustre-loading-exp/scripts/lib/patch_sglang_in_container.sh`, which asserts
  the clone matches the image byte-for-byte before swapping.
  **It subsumes phase 3's `fst_files_per_rank.patch`** — both touch the same
  lines of the same function, and that harness cannot compose two patches over
  one region. Do not pass both.
- **Measure:** `servekit profile` → `weight_loading` (engine-reported) + total.
- **Grid** (1D through each axis first, then the interesting corner):

  | tag | fpr | max_threads | why |
  |---|---|---|---|
  | `ctl_first` | 1 | unset (16) | upstream. Brackets the sweep. |
  | `mt32` / `mt64` / `mt128` | 1 | 32 / 64 / 128 | **axis 2 alone** |
  | `fpr8` | 8 | unset (16) | phase 3's winner, reproduced |
  | `fpr8_mt64` | 8 | 64 | **do they compose?** |
  | `ctl_last` | 1 | unset (16) | brackets drift |

- **Per job:** an in-job **sliced** dd probe (T=8) — *not* the old probe, which
  reports the worst OST's latency as bandwidth and would mis-normalize every
  number here.
- **Node policy:** one job at a time, `--wait`, accumulating `--exclude`.
  `--exclusive` grants sole use of a node, **not a different node**; a
  `--dependency` chain hands the same node straight back. (Cost phase 3 a whole
  sweep.)

## Known risks

- **OST 8 is sick** (~480 ms/RPC, `lustre-contention-exp/NOTES.md`) and hosts
  `model-00018`. It is a *latency* fault, which is exactly what deeper queues
  hide — so `max_threads` may look artificially good right now. The in-job dd
  probe records the OST-8 shard's rate so this is visible rather than silent.
  Re-check any winner once OST 8 is fixed.
- `max_threads` is threads **per rank**, and 4 ranks share 32 CPUs. At
  `mt128` that is 512 threads on 32 cores; a regression there may be CPU
  contention, not storage. `bbuf_size_kb` (16 MB default) may also cap the win —
  16 MB across 128 threads is 128 KB each, and a 128 KB request costs the same
  ~480 ms as a 16 MB one on a sick OST. `SGLANG_FST_BBUF_SIZE_KB` is wired for
  exactly this; sweep it if `max_threads` alone plateaus early.

## Run

```bash
bash fst-threads-exp/scripts/submit_sweep.sh          # serial, fresh node per point
```
