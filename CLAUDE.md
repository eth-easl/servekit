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
   <command...>` CLI: wraps any launch command (`python -m
   sglang.launch_server ...` or `vllm serve ...`), parses that engine's own
   timestamped log lines with no changes to the engine, and emits a per-phase
   duration table + JSON report the moment the server reports ready — without
   killing the still-running server process. The framework is detected from
   the launch command (no flag); an unrecognized command is rejected rather
   than guessed at. Note the two engines' ready signals are not the same
   event: SGLang's comes after its own warmup request, vLLM's ("Application
   startup complete") does not, so vLLM totals must be read together with
   bench's `ready_wait_s`.

   A second subcommand, `servekit bench --url ...`, loads a live server
   (correctness + throughput) over the OpenAI protocol both engines serve
   (`GET /v1/models` + `POST /v1/completions`), and is independent of
   `profile`: it needs only a URL, no launch command and no log to parse. Run
   alongside `profile` (`--into <report.json>`) it merges into the same JSON;
   run alone it benchmarks a server servekit never launched — which is what
   makes a CRIU-restored server measurable, since a restored process emits no
   startup log at all.

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
