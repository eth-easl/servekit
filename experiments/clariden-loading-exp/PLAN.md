# Plan: does the weight-loading speedup reproduce on Clariden (GH200)?

## Context

`experiments/lustre-loading-exp` measured cold-start weight loading for
Llama-3.1-70B (TP=4) on **bristen** (x86, A100/sm_80) and landed a ~4.4x win on
total cold start:

| config | weight_loading | stage | total cold start |
|---|---|---|---|
| SGLang default (mmap, straight from capstor Lustre) | 634 s mean, n=4 (430–939) | — | 812 s mean |
| TP-presharded + sliced `/dev/shm` stage + overlapped | 9.8 s | 14.2 s (hidden) | **183.4 s** |

Every one of those numbers is bristen-only.
`docs/packaging-fast-weight-load/PLAN.md` says so explicitly: *"Clariden/GH200 is
unmeasured. Different memory topology and /dev/shm ..."*. `CLAUDE.md` names
**both** Bristen and Clariden as targets, so before this gets packaged we need to
know the speedup is a property of the technique and not of one machine.

This experiment answers exactly one question: **does the win reproduce on
Clariden?** Two configs, one run each, one fresh node per run.

## Why Clariden could plausibly behave differently

- **aarch64** (Grace) instead of x86 — different container build, different
  memcpy path, different Python/import cost.
- **288 cores across 4 NUMA sockets** vs bristen's 128 across 2. The sliced
  stager is CPU-bound (bristen: 64→128 cores took the stage 12.2 s → 7.7 s), so
  more cores should help. But 4 NUMA domains means `/dev/shm` pages land on
  whichever socket the `dd` that wrote them ran on, and the TP rank that later
  reads them may sit on a different socket. That is a real GH200-specific risk,
  not a formality — it is the single most likely way preshard+shm+overlap
  underperforms here.
- **870 GB RAM** → `/dev/shm` ≈ 435 GB; the 141 GB checkpoint fits easily.
  (Measured after the fact: `/dev/shm` is **334 GB**, not 435. Still ample.)
- **GH200 / sm_90a**, 4x96 GB — no memory pressure at TP=4.
  (Measured after the fact: **4x120 GB**.)
- Same capstor Lustre, reached over a different network path.

Verified while planning: the model and the presharded checkpoint are both
readable from Clariden, and `lmsysorg/sglang:v0.5.10` is a multi-arch manifest
with an arm64 variant — so the **engine version is held constant** and hardware
is the only changed variable.

## Design

| | |
|---|---|
| Configs | 2 only. **default** = mmap from capstor, no flags. **preshard+shm+overlap** = presharded + sliced shm stage + overlapped. |
| Model / TP | Llama-3.1-70B-Instruct, TP=4, ctx 32768, mem-fraction 0.85 — identical to bristen |
| Presharded ckpt | **reused** from the bristen run (`/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded`). safetensors are arch-independent and the `sharded_state` layout is a function of TP size + engine version, both unchanged. Assumption, not proof — the errors=0 / throughput check is what actually validates it. |
| Image | `lmsysorg/sglang:v0.5.10`, same tag as bristen |
| CPUs | whole node, 288 (bristen used 64). We are asking whether the speedup reproduces *on Clariden hardware*, so the runs use Clariden hardware; `nproc` is recorded in every run. |
| Reps | **n=1 per config**, each on a node no other run in this experiment has touched |
| Raw dd probe | none — kept minimal |

Intermediate configs (nommap, fastsafetensors, shm-without-presharding) are
deliberately omitted. They attribute *where* the win comes from, which bristen
already answered; this experiment only asks whether the endpoints hold.

## Files

Self-contained. Nothing under `lustre-loading-exp/` is read at runtime or
modified.

```
submit.sh                            the only entry point
scripts/shared/  models.sh           model paths + the constants both engines share
                 stage_to_shm_sliced.sh
                 save_sharded_state_fixed.py
scripts/sglang/  sglang-clariden.toml  preflight/baseline_mmap/
                 preshard_shm_overlap/save_sharded_ckpt .sbatch
scripts/vllm/    vllm-clariden.toml    preflight/baseline_mmap/
                 preshard_shm_overlap .sbatch
results/<engine>/<model>/            .out, -profile.json, -stage.txt,
                                     -timing.txt, results.md
```

- The `.toml`s are the EDFs. `HOME=/root` in both, so HF/Triton/pip caches are
  ephemeral and every srun is a cold start.
- `preflight.sbatch` — the cheap gate (debug partition, 15 min), one per engine.
- `stage_to_shm_sliced.sh` — verbatim copy of
  `lustre-loading-exp/scripts/phase4_shm/stage_to_shm_sliced.sh`. A copy, not a
  fork; do not diverge it. Shared by both engines: it copies bytes and knows
  nothing about either.
- `baseline_mmap.sbatch` — the **default** config.
- `preshard_shm_overlap.sbatch` — the **preshard+shm+overlap** config.
- `scripts/vllm/` has no `save_sharded_ckpt.sbatch` on purpose — see below.

Measurement is `servekit profile` (`../../servekit`, pure-Python, no deps), which
parses each engine's own timestamped log lines into per-phase durations and adds
a 64-request throughput + correctness block. Bench parameters match bristen
exactly so throughput is comparable across machines.

## Execution order

```
cd experiments/clariden-loading-exp
./submit.sh <engine> llama70b preflight                    # must pass first
./submit.sh <engine> llama70b default                      # note the node id
./submit.sh <engine> llama70b preshard --exclude=<that node>
```

The `--exclude` is not optional. Two rules learned the hard way in
`lustre-loading-exp/SUMMARY.md`: the OS page cache survives across container runs
on a node, and `--exclusive` grants sole use of a node but **not a different
one** — a naive back-to-back submission often reuses the same node, which would
let preshard+shm+overlap read the 141 GB from page cache and manufacture a fake
win.

## What counts as a valid result

- Preflight passes: aarch64, sglang 0.5.10, `sgl_kernel` imports, capability
  `(9,0)`, 4 GPUs, `/dev/shm` ≥ 160 GB free, sharded checkpoint readable.
- The two configs ran on **different** node ids.
- preshard+shm+overlap prints `VALID` with positive slack. `INVALID` → discard
  and re-run; do not interpret it.
- **Both configs serve correctly**: `errors=0`, `64/64` completed, throughput in
  the same ballpark. On bristen every loader hit 401–402 tok/s — loader choice
  must not move throughput. A fast preshard+shm+overlap that serves wrong is a
  failure, and this is the real check that `sharded_state` loaded the right
  bytes.

## Known limits, to be stated in the write-up

n=1 per config. Bristen's mmap runs varied 430–939 s across 4 repeats, so a
single Clariden baseline pins the speedup only to roughly a factor-of-two band. Enough
to answer "does it reproduce"; not enough to quote a precise Clariden speedup.

If preshard+shm+overlap does **not** reproduce the win, that is a genuine
finding and it blocks the packaging work in
`docs/packaging-fast-weight-load/PLAN.md`.

## Fallback

If the preflight fails on the arm64 image, the known-good GH200 alternative is
`/capstor/store/cscs/swissai/infra01/reasoning/imgs/projects/sgl_dev_90a/image.sqsh`
— but it is a much older SGLang, so results would no longer be version-comparable
to bristen and the comparison would need re-framing.

## Out of scope

No servekit changes, no engine patches, no new loader variants, no re-striping.
Two jobs plus a preflight, and a written answer.

---

# Extension: does it reproduce on vLLM?

The answer above is SGLang-only. `CLAUDE.md` names **both** vLLM and SGLang as
targets and the whole point of the package is to be plug-and-play, so a
technique that only pays off on one engine is not the deliverable. Same
question, same shape, one engine changed:

**Does preshard+shm+overlap reproduce on vLLM, on the same node type, same
model, same TP?**

## What is held constant, and what is not

Constant: model, TP=4, 32768 context, 0.85 memory fraction, 256 concurrent
requests, the stager, the bench, one fresh node per run — all read from
`scripts/shared/models.sh` by both engines, so they cannot drift.

**Not** constant: the engine *version*. SGLang runs 0.5.10 here and on bristen,
but vLLM has to run `nvcr.io#nvidia/vllm:26.07-py3` (vLLM 0.24.0.dev). Upstream
`vllm/vllm-openai:v0.25.0`'s arm64 build loads the model and then dies at CUDA
graph capture on a bf16 `cublasGemmEx`, twice, at the identical 32/51 capture —
`profile/llama-3.1-70b-clariden-vllm/results.md`. So this compares each engine's
*working build on GH200*, which is what a deployment gets, not two versions
matched by construction.

## The checkpoint is reused, not rebuilt

The preshard config points vLLM at the **shards SGLang wrote**
(`llama70b-tp4-sharded`), rather than adding a vLLM `save_sharded_state` job.
`sharded_state` originated in vLLM and SGLang inherited it, so the format should
be common — but "should" is the assumption under test, and it is worth testing:
if it holds, one presharded checkpoint serves both engines and the packaging
work has one artifact to produce instead of two.

It is gated in two places rather than assumed:

- `scripts/vllm/preflight.sbatch` checks the filenames against
  `ShardedStateLoader.DEFAULT_PATTERN` from the vLLM build itself, counts parts
  per rank, and reads the safetensors header of rank 0 to confirm the keys are
  plain parameter names with no wrapper prefix. Five minutes, on the debug
  partition.
- the run's own `errors=0` / `64/64` / correct output is what actually proves
  the right bytes loaded. Addressable is not the same as correct.

A hard load failure is a clean answer ("rebuild per engine"). Silent corruption
is the dangerous outcome, which is why correctness is a gate and not a footnote.

## Additional validity conditions

Everything under "What counts as a valid result" applies, plus:

- vLLM's preflight passes: aarch64, capability `(9,0)`, 4 GPUs, compiled ops
  registered, a real bf16 matmul executes, and the `sharded_state` gate passes.
- **The totals are not measured to the same boundary as SGLang's.** SGLang
  announces ready only after its own warmup request; vLLM never issues one, so
  its total stops earlier. Every vLLM total must be quoted with bench's
  `ready_wait_s` beside it. The vLLM-vs-vLLM speedup this experiment is actually
  after is unaffected — both configs stop at the same boundary.
- vLLM's `weight_loading` is its engine-reported `Model loading took`, max over
  the 4 ranks, exactly as for SGLang.

## Expected result, stated in advance

Weight loading was ~66% of vLLM's cold start (233.24 of 353.34 s, n=1) against
~80% for SGLang. Removing essentially all of it predicts roughly **2.9x**, below
SGLang's measured 4.61x, with a floor near 120 s that is dominated by worker
spawn, compile and capture. A result far from that is more interesting than one
near it: well below means the technique is engine-specific, and well above means
the n=1 baseline was not representative.
