# Servekit 🧊 → 🔥

Servekit is a kit of optimizations for serving large language models (LLMs) with minimal cold-start latency 🥶. It wraps sglang calls and implements some optimizations.

Currently, it supports fast weight loading for **lustre** storages like CSCS's `capstor` and `iopsstor` ⚡.


## Setup

Install directly from GitHub:

```bash
pip install git+https://github.com/eth-easl/servekit.git
```

Or, for local development (editable install, from this directory):

```bash
pip install -e .
```

## Usage

### `servekit launch`

```bash
servekit launch -- python -m sglang.launch_server --model-path <model> ...
```

Prepend `servekit launch --` to an engine command to enable servekit's optimizations. Currently, we offer:
* Fast weight loading: loads the weights in a multiprocessing fashion adapted to network storage like Lustre drives (e.g. CSCS `capstor` and `iopsstor`) to RAM (`/dev/shm`) and then to GPU memory.


### `servekit profile`

```bash
servekit profile -- python -m sglang.launch_server --model-path <model> ...
servekit profile -- vllm serve <model> --tensor-parallel-size 4 ...
```

Parses the engine's own log output for phase timings (no engine changes
needed) and prints a per-phase duration table once the server is ready. 


### `servekit bench`

```bash
servekit bench --url http://127.0.0.1:8080 --out bench.json --wait-ready 300
```

Runs a correctness check (greedy completions on fixed prompts) and a fixed
concurrent throughput workload against `POST /v1/completions`. 

### `servekit verify`

Checks that a served model produces the same numbers as a trusted reference —
per-token logprobs, mean NLL, and greedy continuations over a fixed prompt set.

```bash
# record a reference from a server you trust (e.g. plain sglang, no servekit)
servekit verify --url http://127.0.0.1:8080 --record gold.json --wait-ready 300

# check a later server (e.g. one started with `servekit launch`) against it
servekit verify --url http://127.0.0.1:8080 --reference gold.json --wait-ready 300
```

Exits 0 if every prompt is within tolerance (`--token-tol`, `--nll-tol`), 1
otherwise. `--out` writes the per-prompt result as JSON.

### Roadmap 👷‍♂️🚧

- [x] Support Multi-Node fast weight loading
- [ ] Support Pipeline parallelism
- [ ] Support vllm fast weight loading

## Tests

```bash
PYTHONPATH=src python -m pytest tests -q
```

The e2e suite needs a live SLURM cluster and runs only when asked for:

```bash
PYTHONPATH=src python -m pytest tests/e2e -m e2e -q
```
