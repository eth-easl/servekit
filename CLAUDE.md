# Project goal

Improve the cold-start performance of LLM inference. Concretely, build a
package that reduces the time from "start the process" to "serving traffic"
for LLM inference engines (vLLM, SGLang), in a **plug-and-play** fashion —
usable with minimal integration effort on top of an existing serving setup.

As the target application, this package should improve cold-start latency
for models deployed on the SwissAI serving platform, running on CSCS
**Bristen** and **Clariden**.

## Steps

1. **Profile an LLM start.** Break down where cold-start time actually goes
   (weight loading from storage, CUDA/driver init, kernel compilation /
   autotuning, memory allocation, distributed/NCCL setup, warmup requests,
   etc.) so later optimization work targets the real bottlenecks rather than
   assumptions.

   Delivered as `servekit/`, a minimal package with a `servekit profile --
   <command...>` CLI: wraps any launch command (e.g. `python -m
   sglang.launch_server ...`), parses SGLang's own timestamped log lines
   (`Init torch distributed`, `Load weight`, `Capture cuda graph`, the
   "fired up and ready to roll" line) with no changes to the engine, and
   emits a per-phase duration table + JSON report the moment the server
   reports ready — without killing the still-running server process.

   Scope is "process launch -> first request served" (not SLURM queue
   wait). **Follow-up needed:** this doesn't yet account for JIT/lazy-init
   effects that only trigger on the first real inference call (as opposed
   to warmup) — revisit once that's measured.

## Deferred ideas (not investigated yet)

- **Process/import startup** (`process_startup` + `tp_worker_spawn`, ~21% of
  post-load-fix cold start, ~38-40s): profile TP worker startup with py-spy
  to confirm/quantify the redundant-import hypothesis — each TP rank
  re-importing the full torch/transformers stack via `multiprocessing.spawn`
  rather than fork — before deciding on a fix.
