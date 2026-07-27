# servekit

Plug-and-play optimization for LLM inference engines (SGLang, vLLM).
First target: eliminating cold-start latency. Currently implemented: `profile`, `bench`.

## `servekit profile`

```bash
servekit profile -- python -m sglang.launch_server --model-path <model> ...
```

- `--out PATH` — JSON report path (default: `servekit-profile-<timestamp>.json`)
- `--timeout SECONDS` — max wait for ready signal (default: 1800)

Parses the engine's own log output for phase timings (no engine changes
needed) and prints a per-phase duration table once the server is ready. The
report is written the moment the server reports ready, not when it exits. The
server process keeps running — servekit only stops measuring. Send it SIGTERM
to shut the server down.

## `servekit bench`

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
concurrent throughput workload against `POST /generate`. It needs nothing but a
URL — no launch command, no log to parse — so it works against a server servekit
never started, such as one resurrected by `criu restore`. `--wait-ready` polls
`/generate` until the server answers and records how long that took as
`ready_wait_s`.

## Both together

The two are separate commands run side by side, joined by the report file:

```bash
servekit profile --out run.json -- python -m sglang.launch_server ... &
PROF=$!
trap 'kill $PROF 2>/dev/null' EXIT
servekit bench --url http://127.0.0.1:8080 --into run.json --wait-ready 1800 --requests 64
kill $PROF; wait $PROF
```

`run.json` ends up with both halves — the phase table under `phases` and the
benchmark under `benchmark` — in one file.

With `--into`, bench first waits for that file to appear before generating any
load. This ordering matters: an engine starts accepting traffic *before* it
announces readiness (SGLang binds Uvicorn and runs its warmup request seconds
before logging "fired up and ready to roll"), so a bench that only probed over
HTTP could start loading a server the profiler is still measuring. The report
file appearing is the signal that measurement is finished.
