# servekit

Plug-and-play optimization for LLM inference engines (SGLang, vLLM).
First target: eliminating cold-start latency. Currently implemented: `profile`.

## Usage

```bash
servekit profile -- python -m sglang.launch_server --model-path <model> ...
```

- `--out PATH` — JSON report path (default: `servekit-profile-<timestamp>.json`)
- `--timeout SECONDS` — max wait for ready signal (default: 1800)

Parses the engine's own log output for phase timings (no engine changes
needed) and prints a per-phase duration table once the server is ready. The
server process keeps running — servekit only stops measuring.
