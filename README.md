# Servekit

Servekit is a kit of optimizations for serving large language models (LLMs) with minimal cold-start latency. It wraps sglang calls and implements some optimizations. 
Currently, it support fast weight loading. 


## Setup

```bash
pip install -e .
```

## Usage

### `servekit launch`

```bash
servekit launch -- python -m sglang.launch_server --model-path <model> ...
```

Prepend `servekit launch --` to an engine command to enable servekit's optimizations. Curenntly, we offer:
* Fast weight loading: loads the weights in a multiprcessing fashion adapted to networks storage like Lustre drives (e.g CSCS `captstor` and `iopstore`) to RAM (`/dev/shm`) and then to GPU memory.


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

### Roadmap
[ ] Support Pipleine parallelism
[ ] Support Multi-Node fast weight loading
[ ] Support vllm fast weight loading

## Tests

```bash
PYTHONPATH=src python -m pytest tests -q
```
