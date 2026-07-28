# servekit

Plug-and-play optimization for LLM inference engines (SGLang, vLLM).
First target: eliminating cold-start latency. Currently implemented: `profile`, `bench`.

## Setup

```bash
pip install -e .
```

## Usage

### `servekit profile`

```bash
servekit profile -- python -m sglang.launch_server --model-path <model> ...
servekit profile -- vllm serve <model> --tensor-parallel-size 4 ...
```

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
