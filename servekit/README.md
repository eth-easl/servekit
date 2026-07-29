# servekit

Plug-and-play optimization for LLM inference engines (SGLang, vLLM).
First target: eliminating cold-start latency. Currently implemented: `launch`,
`profile`, `bench`.

## Setup

```bash
pip install -e .
```

## Usage

### `servekit launch`

```bash
servekit launch -- python -m sglang.launch_server --model-path <model> ...
```

Prepend `servekit launch --` to an engine command and the model is copied into
`/dev/shm` first, the engine is started against the copy, and the copy is freed
the moment the server reports ready. `launch` behaves like the command it wraps
— same stdout, SIGTERM/SIGINT forwarded, child's exit code — so removing the
wrapper is how you get the baseline back.

The free is not tidying up: the weights are on the GPU by then, and the point is
to hand the RAM back to the job that is now serving, so the node can use it for
its own work (KV cache offloading and the like). For all but the first couple of
minutes of a job that may serve for hours, the node looks as if servekit was
never involved.

- `--out PATH` — JSON report path (default: `servekit-launch-<timestamp>.json`)
- `--timeout SECONDS` — max wait for ready signal (default: 1800)
- `--shm-root PATH` — where copies go (default: `/dev/shm/servekit`)
- `--slices N` — concurrent read slices per file (default: 64)
- `--overlap` — **unsafe, opt-in**; see below

The report is the `profile` report plus a leading `stage` phase, so the copy is
visible in the phase table rather than hidden in the total. With `--overlap` the
stage is concurrent with those phases, so it is not listed as one.

### `--overlap` (unsafe)

Starts the engine at the same time as the stage, saving at most the stage wall
time (4.3–5.0 s of the 70B's ~126 s). The stager truncates every destination to
full size before writing, so an engine that opens a file too early reads zeros
with no error from anything — and the final content is correct, so nothing is
left to find afterwards. There is no barrier making the engine wait; that is
Phase 2 of `docs/packaging-fast-weight-load/PLAN.md` and it is not built. Use it
to measure whether those seconds are worth the barrier, not in production.

config.json, the tokenizer and every other non-`.safetensors` file are copied
synchronously before the engine starts, so only the shards are overlapped — the
engine reads those small files within seconds and cannot be racing them.

Nothing else about the command is rewritten: `--load-format` and every other
flag pass through untouched. To load a TP-presharded checkpoint, produce it with
`experiments/clariden-loading-exp/scripts/shared/save_sharded_state_fixed.py`
and pass `--load-format sharded_state` yourself.

Known limits, all deliberate for now:

- **SGLang only.** vLLM staging is unmeasured, so `launch` refuses it rather
  than guessing at its argv; use `servekit profile` there.
- **Not overlapped by default.** The stage finishes before the engine starts,
  which costs its wall time (4.3–5.0 s for a 141 GB model on Clariden) but means
  a partially-written file can never be read. `--overlap` opts out of that
  guarantee; see above.
- **Freeing is `unlink`**, and tmpfs pages return only once nothing maps them.
  With `sharded_state` the loader reads into GPU memory and closes the files, so
  the RAM comes back at once; with the default mmap loader the engine may still
  hold mappings at ready time.
- **A server that never reaches ready leaves the copy behind** — no automatic
  recovery, and `rm -r /dev/shm/servekit/<name>` is the hatch. A server that
  never got ready is not serving, so there is no workload waiting on that RAM;
  guessing that the engine has let go of the files would risk pulling them out
  from under it for no gain.

### `servekit profile`

```bash
servekit profile -- python -m sglang.launch_server --model-path <model> ...
servekit profile -- vllm serve <model> --tensor-parallel-size 4 ...
```

`launch` without the staging: measures the cold start and changes nothing.

- `--out PATH` — JSON report path (default: `servekit-profile-<timestamp>.json`)
- `--timeout SECONDS` — max wait for ready signal (default: 1800)

Parses the engine's own log output for phase timings (no engine changes
needed) and prints a per-phase duration table once the server is ready. The
report is written the moment the server reports ready, not when it exits. The
server process keeps running — servekit only stops measuring. Send it SIGTERM
to shut the server down.

The engine is identified from the launch command, so there is no framework flag. 

### `servekit bench`

```bash
servekit bench --url http://127.0.0.1:8080 --out bench.json --wait-ready 300
```

- `--url URL` — server base URL
- `--into PATH` — merge results into an existing profile report (see below)
- `--out PATH` — write a standalone bench report
- `--wait-ready SECONDS` — poll until the server serves a request (0: fail immediately)
- `--requests` / `--concurrency` / `--input-len` / `--output-len` / `--seed`
- `--no-correctness` — skip the correctness probe

Runs a correctness check (greedy completions on fixed prompts) and a fixed
concurrent throughput workload against `POST /v1/completions`. 

### Both together

The two are separate commands run side by side, joined by the report file:

```bash
servekit profile --out run.json -- python -m sglang.launch_server ... &

servekit bench --url http://127.0.0.1:8080 --into run.json --wait-ready 1800 --requests 64
```

`run.json` ends up with both halves — the phase table under `phases` and the
benchmark under `benchmark` — in one file.

With `--into`, bench first waits for that file to appear before generating any
load. This ordering matters: an engine starts accepting traffic *before* it
announces readiness (SGLang binds Uvicorn and runs its warmup request seconds
before logging "fired up and ready to roll"), so a bench that only probed over
HTTP could start loading a server the profiler is still measuring. The report
file appearing is the signal that measurement is finished.

## Tests

```bash
PYTHONPATH=src python -m pytest tests -q
```
