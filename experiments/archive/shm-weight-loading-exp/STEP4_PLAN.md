# Step 4 — hugepage-backed staging + in-place H2D registration

## Context

`weight_loading` is 20.3 s of a 197.7 s cold start. An nsys trace of the real
load (job 76112, `results/nsys-76112-nid002292-stats.txt`) established what the
py-spy work could not:

- 141.1 GB moves in **3,956** H2D transfers (median 4.19 MB) — so the strided
  `RowParallelLinear` narrow does **not** fragment into ~14 KB DMAs. Torch
  materializes a contiguous host buffer first; the striding cost is a
  **host-side gather**, not fragmented DMA.
- Total GPU-side H2D time is 23.69 s summed over 4 ranks ≈ **5.9 s per rank**,
  against a ~20.3 s phase. **The DMA engine is idle ~71% of the window**, and
  while active it runs at ~23.8 GB/s aggregate — already the `dma_only` ceiling
  from `h2d_microbench.py`.

So nothing is slow; the pipeline is serial. The floor is ~5.9 s and the win is
in keeping the wire busy and removing host-side work — **not** in adding host
work, which is what the current `h2d_pinned_staging.patch` does (it bounces
every byte through a pinned buffer with a deliberately *blocking* copy).

Step 4 attacks the two host-side costs that staging-through-a-bounce cannot:
the ~4 KB page-fault load of touching 141 GB of tmpfs, and the bounce copy
itself. Both dissolve if the source is hugepage-backed and registered in place.

`/dev/shm` cannot get there: `shmem_enabled` is `[never]` (root-only global
knob), so tmpfs is 4 KB pages regardless of `madvise`. But two routes are open
on this host: anonymous THP is `[always]`, and the hugetlb pool has
`nr_overcommit_hugepages = 128426` (~250 GB of 2 MB pages on demand, no root
reservation needed) with hugetlbfs already mounted at 2M and 1G page sizes.

**Constraint that shapes everything:** hugetlbfs supports `mmap` but **not
`write()`**, so `dd` cannot stage into it. The tuned
`lustre-loading-exp/scripts/phase4_shm/stage_to_shm_sliced.sh` (64 slices,
O_DIRECT, 11.6 GB/s) stays untouched as the A/B control; a new mmap-based
stager is written alongside it.

## Phase 1.5 (2026-07-24) — merged staging+loader design, validated on real data

Reference: https://fergusfinn.com/blog/fast-sglang-starts/ describes the same
hugepage technique working in production. The difference from our dead Phase
1 (below): that design **never opens a filesystem path with safetensors**.
The consumer receives a raw fd over a Unix socket (`SCM_RIGHTS`), `mmap`s it,
and reconstructs tensors directly from the shared buffer -- bypassing
`safe_open` and its file-size validation entirely. The daemon/socket
machinery in that post is about surviving Kubernetes pod restarts, not
fundamental to the hugepage technique itself.

Applied to this problem: stage into an **anonymous `memfd_create(MFD_HUGETLB)`
buffer** (no mount, no file path, no `ftruncate`-exact-size constraint to
violate) and **patch the loader** to build tensor views straight from that
buffer instead of calling `safetensors.safe_open(path)`. The loader patch was
already required by Phase 2 ("in-place registration in the loader"), so this
isn't new invasiveness -- it just moves the staging step behind the same
patch boundary.

New wrinkle this surfaces: TP ranks in this SGLang are spawned via
`multiprocessing.spawn`, not forked (see project memory / CLAUDE.md deferred
idea on redundant imports) -- a spawned child does not inherit an anonymous
mapping the way a forked child would. `/dev/shm` sidesteps this today because
tmpfs has a path every rank can open independently; anonymous/memfd hugepage
memory has none, so sharing it across the 4 ranks needs an explicit handoff.

**Built and validated end-to-end (jobs 76127-76130), independent of SGLang:**
- `scripts/hugepage_safetensors.py` -- parses the safetensors header (8-byte
  LE length + JSON) and builds `torch.frombuffer` tensor views directly from
  a buffer, no `safe_open` call anywhere. Verified by hand against a real
  file first (login-node check, no cluster needed): header_len + data section
  end == exact file size for `model-00030-of-00030.safetensors`.
- `scripts/hugepage_stager.py` -- `memfd_create(MFD_HUGETLB)`, one combined
  buffer with each file at a 2 MB-aligned offset (tracked ourselves; the
  kernel-reported size is irrelevant since nothing calls `safe_open` on it),
  staged via O_DIRECT + `os.preadv` straight into the mmap'd memoryview (no
  bounce buffer at all, unlike the sliced `dd` stager).
- `scripts/hugepage_fd_broker.py` -- Unix-socket + `SCM_RIGHTS` handoff using
  stdlib `multiprocessing.reduction.sendfds`/`recvfds`, exactly the
  blog's mechanism, scoped to one node instead of a k8s DaemonSet.
- `scripts/test_hugepage_shared_pipeline.py` -- stages 2 real shards (7.07 GB,
  1.11-1.19 s), spawns 4 **separately-spawned** (`multiprocessing.spawn`)
  worker processes, each fetches the fd over the broker, reconstructs all 26
  tensors per file from the shared buffer, and checksums every one against a
  plain read of the source file.

**Result: PASS.** All 4 workers, all 26 tensors/file, byte-identical to the
source (job 76130). `cudaHostRegister` on the real 4.966 GB tensor span:
**9.46-17.77 GB/s** across the 4 concurrently-registering workers/GPUs --
lower than the isolated single-process 35.71 GB/s from the Phase 0 gate
(expected: 4-way contention), but still far above the 4 KB tmpfs baseline of
2.59-4.56 GB/s. The fd-sharing + safe_open-bypass mechanism is proven on real
model data, not just a synthetic buffer.

**Known-bad number from this run, do not use:** the H2D copy throughput
(0.01-0.17 GB/s) is a harness artifact, not a real measurement. `t =
torch.frombuffer(...)` has no way to tell PyTorch's own allocator that the
backing memory is `cudaHostRegister`-pinned (PyTorch tracks "pinned" via its
own bookkeeping, not by querying the driver), so `.copy_(t, non_blocking=True)`
silently takes a slow path. Getting a real number needs a direct
`cudaMemcpyAsync` call via `ctypes`/`libcudart` (which Phase 2 was already
going to need for the `cudaMemcpy2DAsync` strided-copy piece) -- not
`Tensor.copy_`.

**Not yet done:** none of this is wired into SGLang. The remaining work is (1)
find where SGLang actually spawns TP-rank subprocesses and get the staging
process's memfd fd to each rank (this test used our own launcher +
`multiprocessing.spawn`, which is representative but not the same code path
SGLang uses -- needs checking where in the engine that spawn happens), (2)
patch `safetensors_weights_iterator` (`weight_utils.py:703`, the
`safe_open`-based branch at line 724) to use `hugepage_safetensors` when
enabled, gated the same way as `SGLANG_H2D_PINNED_STAGING`, (3) replace
`h2d_copy_`'s pinned-buffer bounce with per-shard `cudaHostRegister` +
pipelined `cudaMemcpyAsync` against the now hugepage-backed source, measured
with the direct-`ctypes` approach above instead of `Tensor.copy_`.

## Phase 0/1 result (2026-07-24) — Phase 1 is dead, see below

**0a PASS.** `mmap(MAP_SHARED|MAP_ANONYMOUS|MAP_HUGETLB)` works with no mount,
no privileges (job 76117): confirmed real 2 MB pages via
`/proc/self/smaps` (`KernelPageSize: 2048 kB`) and `/proc/meminfo`
(`HugePages_Surp` 0 -> 3815 for an 8 GB request).

**0b borderline, treated as PASS.** `cudaHostRegister` measured **35.71 GB/s**
on the hugepage mapping (job 76117) — under the plan's round "≥40 GB/s" bar,
but Phase 2 pipelines registration behind DMA rather than doing it as one
blocking step, so the real gate is register-time-per-shard <
transfer-time-per-shard, not the isolated throughput number. Per-rank: ~35.3 GB
to register at 35.71 GB/s = ~0.99 s, against a measured ~5.9 s DMA window per
rank — comfortably hidden. (Still an 8-14x gain over the 4 KB tmpfs baseline
of 2.59-4.56 GB/s either way.)

Free `nsys cuda_api_sum` on job 76112's trace: `cudaMemcpyAsync` totals 24.05 s
across 3,980 calls, almost exactly matching the previously-measured 23.69 s of
GPU-side H2D time. Nearly all host-side time is already *inside* the
`cudaMemcpyAsync` call (the driver's own blocking bounce copy), not in a
separate fault/iterator gap outside it — so Phase 1's fault-elimination story
may have less room to matter than hoped; Phase 2's bounce-elimination is the
bigger lever.

**Phase 1 is dead — confirmed, not just the theorized risk.** `mount -t
hugetlbfs` needs CAP_SYS_ADMIN and fails in this container (job 76118), so
routes 1-2 (anonymous, no filesystem path) can't be used for a `--model-path`
directory. The only filesystem-backed hugetlbfs route is a pre-existing,
world-writable libhugetlbfs pool mount at
`/var/lib/hugetlbfs/global/pagesize-2097152` (job 76119). But on that mount,
`ftruncate` only accepts **exact multiples of 2 MB** (job 76124: 800, 4096,
2097153, 5242880 all `EINVAL`; only 2097152 itself succeeds) — there is no way
to get an exact-byte-size file. Padding a real shard up to the next 2 MB
boundary and pointing `safetensors.safe_open` at it (job 76125, real
`model-00030-of-00030.safetensors`, 2101346432 -> 2103443456 bytes) fails
outright: `SafetensorError: Error while deserializing header: incomplete
metadata, file not fully covered`. The known risk in Phase 1 below is real and
is not survivable on this container's hugetlbfs.

**Next: skip to Phase 2, staging source unchanged (`/dev/shm`).** Per the
Phase 1 fallback already specified below: keep `/dev/shm` staging as today and
do in-place registration against the tmpfs source directly (loses the fault
win Phase 1 was chasing, keeps the bounce-elimination win Phase 2 targets).
Registration throughput against 4 KB tmpfs pages is the already-measured
2.59-4.56 GB/s baseline, not the 35.71 GB/s hugepage number — Phase 2's
per-shard-registration-hidden-behind-pipelining math needs to be redone
against that baseline before assuming it still pays for itself.

## Phase 0 — Gate (build nothing until this passes)

One small job. Two independent questions; **both** must pass.

**0a. Is hugetlbfs actually usable in the container?** Try in order, stop at
first success:
1. `mmap(MAP_SHARED|MAP_ANONYMOUS|MAP_HUGETLB)` — no mount, no privileges.
2. `memfd_create(name, MFD_HUGETLB)` + `ftruncate` + `mmap`.
3. `mount -t hugetlbfs none /mnt/huge` (needs CAP_SYS_ADMIN — may fail).
4. Write access to the existing `/dev/hugepages` (root-owned 0755 — likely no).

Verify pages are *really* huge: `KernelPageSize: 2048 kB` for the mapping in
`/proc/self/smaps`, and `HugePages_Surp` rising in `/proc/meminfo`. Routes 1–2
give no filesystem path, which matters for Phase 1 — record which route works.

**0b. Is `cudaHostRegister` dramatically faster on 2 MB pages?** This is the
number the whole design rests on. Baseline is already measured: **2.59–4.56
GB/s on 4 KB tmpfs**, i.e. 31–54 s for 141 GB — the reason NOTES declared
whole-model pinning dead. Re-run the same measurement against a hugepage
mapping.

Extend the existing `bench_register()` in
`shm-weight-loading-exp/scripts/h2d_microbench.py:202` rather than writing new
code — it already does the timing and the break-even arithmetic.

**Pass criteria:** hugepages allocatable at ≥8 GB, and registration ≥40 GB/s
(→ 141 GB in <4 s, amortizable across a shard's ~989 tensors). If registration
stays single-digit GB/s, in-place registration is dead even with hugepages —
**stop here** and fall back to fixing the existing patch's blocking copy
(2–3 pinned buffers + `non_blocking=True` + events), which is worth ~20.3 → 14.5 s
on its own.

**Free, run alongside:** `nsys stats --report cuda_api_sum` on the existing
`results/nsys-76112-nid002292.nsys-rep`. Splits time *inside* `cudaMemcpy*`
(driver bounce) from time *outside* (faults + iterator), telling us how much of
the 14.4 s host-side gap Step 4 can even address.

## Phase 1 — mmap-based hugetlbfs stager

New: `shm-weight-loading-exp/scripts/stage_to_hugetlbfs.py`. Mirrors the proven
design of `stage_to_shm_sliced.sh` — that script's tuning is the reason the
stage hits 11.6 GB/s, so keep its structure and only change the write path:

- pre-create each dest file at full size (`ftruncate`) before any writer starts
- cut each file into `SLICES=64` contiguous ranges (Lustre needs the queue
  depth; one reader per OST is what made the naive version slow)
- workers `pread()` from the Lustre source with **O_DIRECT** into disjoint
  offsets of the **mmap'd** hugetlbfs file — O_DIRECT needs an aligned
  destination, and hugepage mappings are 2 MB aligned, so 4 KB-aligned slice
  offsets satisfy it
- slices tile exactly, no gaps/overlap; keep the caller's checksum gate

Then point `--model-path` at the hugetlbfs directory. `safe_open(st_file,
framework="pt", device="cpu")`
(`sglang/python/sglang/srt/model_loader/weight_utils.py:840`) mmaps zero-copy,
so the tensors the H2D copies read from inherit the hugepage backing with **no
loader change at all** — this phase alone should cut the fault component.

**Known risk — file-size alignment.** hugetlbfs rounds allocations to 2 MB. If
`ftruncate` forces a padded size and safetensors validates file length against
its header, `safe_open` will reject the shard. Test this on a *single small
shard* before staging 141 GB. If it rejects, fall back to Phase 2's
register-the-source approach with `/dev/shm` left in place (loses the fault win,
keeps the bounce-elimination win).

## Phase 2 — in-place registration in the loader

Replace the bounce in `h2d_copy_` (currently
`shm-weight-loading-exp/scripts/h2d_pinned_staging.patch`) with:

- **Register per shard, not per tensor.** On first touch of a shard's mapping,
  `cudaHostRegister` the whole region once and cache it; every one of that
  shard's tensors then DMAs directly from it. Registration cost amortizes over
  ~989 tensors instead of being paid per copy — this is what makes it beat the
  bounce even if registration isn't free.
- **Pipeline, don't block.** `cudaMemcpyAsync` + `torch.cuda.Event`, registering
  the next shard while the current shard's transfers are in flight. nsys says
  the wire is idle 71% of the time; overlap is the entire point, and the current
  patch explicitly forgoes it.
- **Strided slices: try `cudaMemcpy2DAsync`.** A pitched copy does the strided
  H2D on the DMA engine and removes the host gather completely. Torch doesn't
  expose it; reach it via `ctypes` against `libcudart`. This is the piece that
  targets the RowParallel path directly.
- Keep the existing fallbacks (not CUDA dest, not CPU source, already pinned,
  too small → plain `copy_`) so behavior is unchanged wherever staging can't pay.

Apply with the existing clone-verify-patch harness,
`lustre-loading-exp/scripts/lib/patch_sglang_in_container.sh`, which byte-checks
every touched file against the installed copy before patching.

Keep `SGLANG_H2D_PINNED_STAGING`-style env gating so the arm is switchable
without reverting, and set `OMP_NUM_THREADS=16` (cores/TP) in any arm that still
does a host gather — without it, 4 ranks × 64 OMP threads on 64 cores erases the
effect (measured 3.67× on gather).

## Verification

New sbatch modeled on `nsys_weight_load.sbatch` (which already encodes the two
non-obvious nsys requirements: `--sample=none --cpuctxsw=none` or it hangs
forever on Slurm, and `--duration=70` or the 113 s of graph capture makes the
report unfinishable).

Arms, **≥3 replicates each, one fresh node per run** (page cache survives
container runs, and the existing 4-arm study was n=1 — a ~10 s effect inside a
~198 s start cannot be resolved without repeats):

| arm | staging | loader |
|---|---|---|
| `ctl` | `/dev/shm` (dd, unchanged) | upstream |
| `huge` | hugetlbfs (new stager) | upstream |
| `huge_reg` | hugetlbfs | in-place registration |

Success = `weight_loading` drops from ~20.3 s toward the ~5.9 s DMA floor, with
`huge` isolating the fault win and `huge_reg` the bounce-elimination win.

Confirm with, not instead of, wall clock:
- `nsys stats --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum` — DMA time
  should approach the phase wall time as idle gaps close
- transfer count should stay ~3,956 (a spike means the 2D path fragmented)
- server reaches "fired up and ready to roll" and benches with 0 errors

## Out of scope

CUDA graph capture (113.7 s, 57.5% of cold start) and imports (38.6 s, 19.5%)
are larger levers but explicitly excluded here. Step 4's realistic ceiling is
~14 s off a ~198 s start (~7%).
