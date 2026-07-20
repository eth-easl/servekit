# Notes — release the model, put it back

Model: Llama-3.1-70B-Instruct (30 shards, 141.1 GB), TP=4, `--load-format fastsafetensors`,
upstream loader (no phase3 patch). Bristen, 4 GPUs/node, ~515 GB RAM/node.
Runs: `ctl_nosaver` job 75093 on nid002320, `saver_cycle` job 75115 on nid002312
(distinct nodes, serialized, both after the contention campaign ended at 18:21).

---

## TL;DR

**The mechanism half-works, and the half that fails does so silently.**

| Step | Result |
|---|---|
| `release_memory_occupation(["weights"])` | ✅ **0.31 s** — frees ~43.7 GiB/rank, really freed (not swapped to host) |
| `resume_memory_occupation(["weights"])` | ✅ **0.20 s** — allocation returns to within 0.1 % of pre-release |
| `update_weights_from_disk` | ⚠️ **38.4 s**, reports `success: true` — **but the model is destroyed** |
| Serving after the cycle | ❌ 401.5 tok/s, 0 errors, HTTP 200 — **and pure garbage output** |
| `--enable-memory-saver` cold-start cost | ✅ **free** (88.5 s vs 92.8 s weight load — saver was *faster*, i.e. within noise) |

**Release and resume are essentially instant and behave exactly as documented.** The
GPU drops from 62.0 GB to 17.2 GB per rank in 0.31 s and comes back in 0.20 s. For
the checkpoint idea, that half is a green light: you *can* get a 70B server into a
model-free state, and back, in half a second.

**The reload is where it dies.** After `release → resume → update_weights_from_disk`,
the server answers every request at full speed with zero errors and the endpoint
returns `"Succeeded to update model weights."` — and emits:

```
prompt : "The capital of France is"
before : " a city of love, art, fashion, and cuisine. Paris is a must-visit ..."
after  : " QuestionQuestionQuestionQuestionQuestionQuestionQuestion ..."

prompt : "List the first 10 prime numbers."
before : " 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 ..."
after  : " assistantassistantassistantassistantassistantassistant"
```

6/6 correctness prompts destroyed. **Every cheap health signal says the cycle
succeeded.** HTTP 200, `success: true`, throughput identical to cold start (401.5 vs
400.6 tok/s), zero errors, GPU memory restored to within 0.1 %. Only reading the
generated text reveals it. Anything built on this that checks liveness instead of
output will report success forever.

**Do not adopt yet.** The prerequisite is *not* met.

**Probably a known upstream bug, corroborated by one report.**
[sglang#15246](https://github.com/sgl-project/sglang/issues/15246) reports our exact
failure — release → resume → update weights → gibberish — via the *upstream-tested*
`update_weights_from_tensor` API, closed "inactive" with no root cause. That points
at the **release/resume cycle** rather than our reload call. But it is one report, on
a different model family and an older version — corroboration, not proof. We used the
flow the docs prescribe ([RL docs](https://docs.sglang.io/docs/advanced_features/sglang_for_rl):
`from_disk` is "simplest and best for ... checkpointing"); it does not work. Mechanism
unidentified by anyone, us included.

**Not the KV cache, and not the reload call** — settled by the Apertus-8B matrix
below. On 8B, **every** tag combination (`weights` / `+kv_cache` / `+cuda_graph`) and
the no-release control all return **6/6 byte-identical** output. So:
`update_weights_from_disk` is fine on its own, and `weights`-only is not an
unsupported combination. **The trigger is model-specific** — the 70B differs by
weights-region size (44.8 vs 6.3 GB/rank) and by fastsafetensors batching (30 shards
→ 8 batches with a partial last batch, vs 4 shards → 1 clean batch). Next
discriminator: Llama-Guard-4-12B (5 shards → partial batch, 23 GB, ~3 min).

---

## What we verified, with numbers

### Release genuinely frees, and frees the whole region (not just the weights)

Per-rank GPU (`nvidia-smi --query-compute-apps`), rank 0:

| snapshot | GPU used | VmRSS |
|---|---|---|
| S1 post_bench | 61 998 MiB | 4 883 132 kB |
| S2 post_release | 17 242 MiB | 4 883 132 kB |
| S3 post_resume | 62 062 MiB | 4 883 132 kB |
| S4 post_reload | 61 888 MiB | 4 885 652 kB |

- **Freed: ~43.7 GiB/rank** (~44.7 GiB/GPU device-wide: 63 276 → 18 520 MiB).
- **VmRSS is flat to the kilobyte across release.** `enable_weights_cpu_backup=False`
  held: the weights were *freed*, not stashed in host RAM. `VmPin=0`, `VmLck=16 kB`
  throughout. This was the check that could have falsified the whole premise; it passed.
- **Resume restores to 62 062 MiB vs 61 998 MiB pre-release** — 0.1 %.

**The freed region is ~9 GiB larger than the weights themselves.** Weights per rank
are 141.1 GB / 4 = 32.9 GiB, but ~43.7 GiB is freed. That is not a mystery: the
engine's own log says `Load weight end ... mem usage=44.84 GB` — the memory-saver
`weights` region wraps the whole of `loader.load_model()`, so the fastsafetensors
loader's GPU scratch stays in the region via the caching allocator and is released
with it. Engine-reported and externally-observed numbers agree, which is why we
trust both.

### `--enable-memory-saver` is free at cold start

| | weight_loading | total | throughput |
|---|---|---|---|
| `ctl_nosaver` (75093) | 92.81 s | 261.4 s | 401.4 tok/s |
| `saver_cycle` (75115) | 88.51 s | 259.9 s | 400.6 tok/s |

The saver run was **4.3 s faster**, i.e. the VMM allocator hook costs nothing
measurable — the difference is well inside the 21 % node/time spread this repo has
measured for identical configs. Worth having measured rather than assumed: it means
a future adopter pays nothing to keep the option open.

### The reload was page-cache served — the 38.4 s is not a cold restore

`Cached` was **104 GB** at S1 (≈74 % of the 141 GB model; the node has 515 GB RAM)
and *did not increase* across the reload — it went 104 146 052 kB at S1/S2/S3 →
103 338 536 kB at S4. The reload read what was already in RAM.

Arithmetic: cold load 141 GB / 88.5 s = **1.59 GB/s**; reload 141 GB / 38.4 s =
**3.68 GB/s** — while the storage underneath was doing **0.65 GB/s** (below). A cold
restore on a fresh node would be far slower than 38.4 s. Treat 38.4 s as a
warm-cache floor.

---

## ⚠️ Correction to this experiment's own plan: the "storage healthy" gate was unsound

> **SUPERSEDED (Jul 17) — the gate was unsound, but not for the reason below.**
> This section concludes "capstor really was running at ~0.65 GB/s that evening"
> because two probes on different nodes agreed. They are not independent: **both
> are `dd`**. Agreement between two `dd` probes shows `dd` is reproducible, not
> that it is correct. The independent instrument is this experiment's own loader,
> and it read 1.52/1.59 GB/s on the same nodes minutes apart — 2.3× *more* than
> `dd`, with 4 shards in flight vs 30.
>
> Per-shard timing (`lustre-contention-exp`, jobs 75141/75142) found the cause:
> **one sick OST (8)**, whose read time *is* the pool's wall clock. Excluding it,
> capstor was at 3.0–8.6 GB/s — healthy all along. `dd`'s aggregate is a max over
> workers, i.e. a measurement of the worst OST.
>
> The 0.65 GB/s number below is real as a `dd` result and fictional as a capstor
> result. See `lustre-contention-exp/NOTES.md`.

PLAN.md said to pass only if the dd probe lands in the **6–9 GB/s band**. Both jobs
came in at **0.70 and 0.62 GB/s** — ~10× low. That looked like a hardware fault.

It was not. The `lustre-contention-exp` campaign was independently sampling capstor
all afternoon **on different nodes**, and measured:

```
17:01  0.484 GiB/s      17:32  0.589 GiB/s      18:02  0.674 GiB/s
17:09  0.703 GiB/s      17:46  0.700 GiB/s      18:15  0.683 GiB/s
```

Our 18:35 and 18:46 probes land exactly in that band. **capstor really was running at
~0.65 GB/s that evening, ~10× below the 6.7–8.6 GB/s this repo measured on other
days.** Two independent experiments, different nodes, agreeing.

The gate was wrong because it was **a remembered number from a different day** —
precisely the mistake `lustre-loading-exp`'s methodology rule #1 exists to prevent
("a bandwidth number from a different job at a different time is worthless"). The
in-job dd probe did its job; the *threshold* was the bug. Do not re-add a fixed band:
compare against the in-job probe, or against the contention campaign's curve for that
hour.

Side note worth keeping: the engine's cold load achieved **1.59 GB/s aggregate while
raw O_DIRECT dd managed 0.65 GB/s** — buffered `pread` + readahead beat O_DIRECT by
2.4× on degraded storage. The dd probe is a *lower* bound on what the loader sees,
not an upper one.

---

## Why the reload breaks — what the source rules OUT

Not resolved, but substantially narrowed by reading the pinned engine. Facts first:
every call returned 200, the loader visibly ran all 8 batches per rank,
`success: true`, no exception, no rollback message, and the
`torch_memory_saver` noop gate passed — **the saver was real**.

**Eliminated, each with evidence:**

| Suspect | Why it's dead |
|---|---|
| `process_weights_after_loading` rebinding params | Neither `UnquantizedLinearMethod` (`layers/linear.py`) nor `UnquantizedEmbeddingMethod` (`layers/vocab_parallel_embedding.py`) overrides it. **No-op for an unquantized Llama.** |
| Stale KV / radix cache | **Confirmed in the log, not just the code**: `Cache flushed successfully!` on all 4 TP ranks at 18:42:33, i.e. after the update and before the re-bench (`UpdateWeightFromDiskReqInput.flush_cache` defaults **True**, `io_struct.py:1245`). The re-bench computed fresh KV against the reloaded weights. Note the flush came from the *update*, **not** from the release: `release_memory_occupation` only calls `flush_cache()` inside the `kv_cache`-tag branch, which we never took. |
| CUDA graphs needing recapture *in general* | `recapture_cuda_graph` exists (`io_struct.py:1240`, default False) — but `update_weights_from_tensor`, which upstream tests and asserts correct output on, **also never recaptures**. |
| The reload using a different/blind iterator | `fastsafetensors_weights_iterator` (`weight_utils.py:729`) is **the same function cold start uses**, same batching (30 files / pg.size()=4 → 8 batches — confirmed in the log both times). |
| `load_weights` differing between the two paths | `update_weights_from_tensor` with `load_format=None` calls `self.model.load_weights(...)` (`model_runner.py:1675`) — **the identical call** `update_weights_from_disk` makes via `load_weights_and_postprocess` (`loader.py:697-698`). Both in-place. |

So the upstream-tested path and ours **converge on the same in-place `load_weights`**,
with a no-op postprocess and no recapture in either. The loader cannot be blamed on
static reading alone.

**What upstream actually tests** (`test/registered/rl/test_release_memory_occupation.py`):
every test uses `update_weights_from_tensor`, never `update_weights_from_disk`; and
every test releases **all three tags**, never `weights` alone. The
`..._with_weights_cpu_backup` variant confirms that *without* backup you must refill —
which is what we tried to do. **Our exact combination is untested upstream.**

**The two candidates left, which static reading cannot separate:**

1. **The memory-saver cycle corrupts something.** ← **upstream evidence says this one**
2. **`update_weights_from_disk` is broken on its own**, with the memory saver a red
   herring entirely.

---

## 🔴 This is a known, unfixed upstream bug — we are not holding it wrong

Searched the SGLang and torch_memory_saver trackers. **Exactly one genuine match**
— be precise about this, most of the superficially-similar reports are a different
thing:

| Report | Repro | Relevance |
|---|---|---|
| [sglang#15246](https://github.com/sgl-project/sglang/issues/15246) | release → resume → **`update_weights_from_tensor`** → gibberish. Qwen3-30B-A3B, `enable_memory_saver=True`, v0.5.4 | ✅ **The real match.** They *did* refill, via the upstream-tested API, and still got gibberish. **Closed "inactive"** — no root cause, no fix |
| [torch_memory_saver#71](https://github.com/fzyzcjy/torch_memory_saver/issues/71) | `pause`/`resume` corrupts weights, output "10–20× shorter" and wrong. tms 0.0.9 | ⚠️ Same library, but `enable_cpu_backup=**True**` — we run `False`, so a *different* code path. **OPEN**, no fix. Rules out CUDA graphs by ablation |
| [sglang#7939](https://github.com/sgl-project/sglang/issues/7939) | release → resume → generate → gibberish | ❌ **Not our bug** — no weight refill at all, so garbage is the *documented* outcome |
| [sglang#6367](https://github.com/sgl-project/sglang/issues/6367) | "do I have to update weights every time?" | ❌ Same: no refill. Expected behaviour |
| [sglang#19442](https://github.com/sgl-project/sglang/issues/19442) | "after `resume_memory_occupation`, the GPU occupation is indeed a random tensor rather than the actual weights" | ℹ️ Not a bug report — it's the **explanation** of why #7939/#6367 are expected |

**#15246 is the one that counts: our exact failure, via the upstream-*tested* API
(`update_weights_from_tensor`).** If refilling by tensor also produces gibberish, the
fault is not in `update_weights_from_disk` — it is in the release/resume cycle
itself. Candidate 1. But this is **a single report on a different model family and an
older version**, so treat it as corroboration, not proof.

**Two of my earlier hypotheses are now dead on upstream evidence too:**
- **CUDA graphs are not the mechanism.** torch_memory_saver exists precisely to
  preserve virtual addresses so graph replay stays valid; tms#71 rules graphs out by
  ablation; sglang#7939 reports `--disable-cuda-graph` doesn't help.
- **We used the prescribed flow.** [sglang#19442](https://github.com/sgl-project/sglang/issues/19442)
  and the [RL docs](https://docs.sglang.io/docs/advanced_features/sglang_for_rl) both
  describe release → resume → reload-the-weights as *the* intended sequence for LLMs
  (release frees; resume hands back an empty region; you must refill). `from_disk` is
  even called out as "simplest and best for elastic rollout scaling and
  checkpointing" — exactly our use case. The docs carry **no warning** that this is
  broken.

**Our run is a new data point, not a duplicate:** v0.5.10 (vs 0.5.4), dense Llama-70B
(vs MoE Qwen3), `update_weights_from_disk` (vs `from_tensor`), TP=4, with external
`nvidia-smi`/`VmRSS` evidence that release/resume themselves account for memory
correctly. Worth filing upstream — the existing issues died of stale-bot, not of
diagnosis.

**Honest caveats:** no report is a byte-for-byte match to our config, and tms#71 is
`enable_cpu_backup=True` while we run `False`. The common thread is the cycle, but
nobody has isolated the mechanism — including us.

---

## ✅ Apertus-8B tag matrix — the bug does NOT reproduce, and KV is not the trigger

Jobs 75135/75136/75139/75140. Apertus-8B-Instruct-2509, **TP=4 (same as the 70B — the
distributed path is a suspect, so it must not vary)**, same image, same EDF shape, same
`fastsafetensors`, memory saver ON for every point including the control. ~3 min/point.

| point | release tags | GPU freed/rank | reload | verdict | vs cold start |
|---|---|---|---|---|---|
| `norelease` | *(no release/resume)* | — | 1.17 s | ✅ coherent | **6/6 identical** |
| `w` | `weights` | 6 376 MiB | 1.15 s | ✅ coherent | **6/6 identical** |
| `w_kv` | `weights,kv_cache` | 60 388 MiB | 1.12 s | ✅ coherent | **6/6 identical** |
| `all` | `weights,kv_cache,cuda_graph` | 60 398 MiB | 1.17 s | ✅ coherent | **6/6 identical** |

All four answer *"Paris, which is also the country's largest city..."* — byte-identical
to their own cold-start bench. The differing freed-memory column proves the tags really
took effect (`kv_cache` releases the ~54 GB pool on top of the weights).

**Three things this settles:**

1. **The KV cache is not the trigger.** Releasing it, or not, changes nothing. Combined
   with the flush evidence above, the KV line of enquiry is closed.
2. **`update_weights_from_disk` works fine on its own** (`norelease` ✅) — the control we
   were missing. Candidate 2 is dead.
3. **`weights`-only is not an "unsupported combination".** It is exactly what failed on
   the 70B, and here it is perfect. The tag set is not the fault.

**So the trigger is model-specific**, and Apertus-8B is a *weak* reproducer: it differs
from the 70B in at least three ways at once, so it isolates nothing. It only proves the
bug is not universal.

| | Apertus-8B | Llama-70B |
|---|---|---|
| weights region / rank | **6.3 GB** | **44.8 GB** |
| shards | **4** → exactly 1 fastsafetensors batch, every rank gets a file | **30** → **8 batches**, and the last holds only 2 files so **ranks 2–3 get none** |
| architecture | Apertus | Llama |

The partial/multi-batch asymmetry is suggestive — `fastsafetensors_weights_iterator`
builds `rank_file_map = {i: [f] for i, f in enumerate(f_list)}`, so a short final batch
leaves ranks idle, and the 8B **never once takes that path**. But cold start survives it
on the 70B, so it is a lead, not a conclusion. Region size is an equally live suspect.

### Next discriminator: **Llama-Guard-4-12B** (already on disk)

5 shards, 23 GB → **2 batches with a partial last batch (1 file, ranks 1–3 idle)** at
~8B cost. It is the cheapest thing that separates *batching* from *model size*:
reproduce → the fastsafetensors batch path is implicated and we have a ~3-minute
upstream reproducer; clean → size/region-size is the axis, and the next test is the 70B
reloading with `load_format=safetensors` (a plausible **workaround**: cold-start fast on
fastsafetensors, reload on the safe loader).

## ⚠️ Methodology bug that nearly faked a result: `--export` eats commas

The first submission of this matrix ran `w_kv` and `all` as **silent duplicates of `w`**.
`sbatch --export=ALL,...,RELEASE_TAGS=weights,kv_cache,...` splits on commas, so
`RELEASE_TAGS` truncated to `weights` and the extra tags became phantom variable names.
Every job "succeeded", every point looked coherent, and the matrix would have read as
"tags make no difference" — **which is the right answer for the wrong reason**.

Caught only because the summarizer prints the tag set the driver actually received, and
it read `weights` on all four rows. Fixes: `export` the vars and use a plain
`--export=ALL`; plus a guard in `release_reload.sbatch` that aborts when a `*_kv*`/`*all*`
job receives a single-tag `RELEASE_TAGS`. Bad runs quarantined in
`results/failed_export_truncation/`.

**Print the parameter the job actually used, never the one you think you passed.** This
is the second time in this experiment that a green run meant nothing (the first being the
70B's `success: true` on a destroyed model).

## ⚠️ The control this experiment should have run and didn't — now run (see above)

**`update_weights_from_disk` on a healthy server, with no release/resume.** One
sequence, two HTTP calls removed. It discriminates the two candidates outright:

- output survives → the memory-saver cycle is implicated (candidate 1)
- output is garbage → `update_weights_from_disk` never worked here and the entire
  memory-saver framing is a red herring (candidate 2)

Releasing *and* reloading in one shot conflated two variables. The first run should
have priced each independently — the same lesson `lustre-loading-exp` records as
"one variable at a time".

## Next steps, cheapest first

1. **Move to Apertus-8B** (`profile/apertus-8b-bristen/`) — the whole cycle in ~2 min
   instead of ~20, so the matrix below costs minutes, not a node-hour. This should
   have been step 0.
2. **The missing control**: `update_weights_from_disk` with no release/resume.
3. **`recapture_cuda_graph: true`** on the update call — one JSON field, already
   wired through `tp_worker.py:99` to `init_device_graphs()`. If this fixes it,
   candidate 1 is confirmed and the fix is free.
4. **Release all three tags** (`weights,kv_cache,cuda_graph`) — the only combination
   upstream actually tests.
5. **Reload with `load_format=safetensors`** — the driver already takes
   `--load-format` independently of the server's, so it's a one-flag change.
6. Only once it works: re-run on a **fresh node** for an honest cold reload number,
   since 38.4 s is warm-cache.

## Method notes worth keeping

- **The correctness check is the only thing that caught this.** Liveness, HTTP status,
  the endpoint's own `success` flag, throughput, and error count *all* reported a
  healthy server. Any future work here must diff generated text.
- **Reusing `servekit.bench`'s own prompts and seeded workload for the re-bench** made
  the before/after comparison exact by construction — the "before" was already in the
  profile JSON.
- **The `torch_memory_saver` noop gate was worth building** even though it passed:
  without it, a silent Noop adapter would have produced a green run that proved
  nothing, and we would have believed the release numbers.
- Two jobs, distinct nodes, serialized after the contention campaign: no
  cross-experiment interference in either direction.
