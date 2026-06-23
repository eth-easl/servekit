# GLM-4.7-Flash on Beverin

This deploys the local `zai-org/GLM-4.7-Flash` snapshot with the ROCm vLLM image through the
CSCS Slurm Container Engine.

The service uses one full `mi300` node (4 GPUs). GLM-4.7-Flash is a 4.7B parameter model,
but the deployment runs with tensor parallelism across the node
(`--tensor-parallel-size 4 --pipeline-parallel-size 1`) and enables the GLM tool/reasoning
parsers (`--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice`).
Speculative decoding was tested but disabled because the `mtp` method triggered an assertion
failure in the ROCm MLA attention path during CUDA graph capture.

The default Slurm time limit is `12:00:00`.

Each vLLM instance is scraped by a per-rank `vmagent` that pushes to the SwissAI metrics
remote-write endpoint (`https://prometheus-dev.swissai.svc.cscs.ch/api/v1/write` by default).
The agent starts only after vLLM is serving so the Go scraper does not get stuck on an
unready target.

GLM-style checkpoints require `--trust-remote-code`, which is enabled in the serve script.

The serve script also launches an [OpenTela](https://github.com/swiss-ai/OpenTela) peer
that manages vLLM as its subprocess. It connects to the bootstrap peer
`/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9`,
registers the service as `llm` on port `8000`, and exposes the model as
`zai-org/GLM-4.7-Flash-rocm`.

## Remote layout

Files are copied to:

```text
/capstor/scratch/cscs/xyao/glm-47-flash-vllm
```

The EDF file lives in that directory and is found by exporting `EDF_PATH` before calling
`srun --environment=glm-47-flash-rocm`.

The model is loaded from:

```text
/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
```

vLLM, XDG, FlashInfer, Triton, and container `$HOME` writes are redirected to the deploy
directory to avoid `$HOME` quota failures on Beverin.

The service defaults are:

```text
MAX_MODEL_LEN=131072
MAX_NUM_SEQS=256
GPU_MEMORY_UTILIZATION=0.50
TP_SIZE=4
PP_SIZE=1
TIME_LIMIT=12:00:00
```

`--kv-cache-memory-bytes` was removed because it caused OOM on unified-memory MI300
nodes; vLLM now manages KV cache within `--gpu-memory-utilization 0.50`.

These can be overridden with environment variables at submission time.

## Run

The service script allocates multiple nodes (default 5) and runs one independent
vLLM + OpenTela instance (a replica) on each node, each at TP=4 across the
node's 4 GPUs. `rcc` resolves host and remote dir from `.rcc/config.toml`, so
there is no need to `cd` or hardcode paths:

```bash
rcc --profile glm-47-flash push
rcc --profile glm-47-flash run bash -lc "sbatch deploy/glm-47-flash-beverin/probe_vllm.sbatch"
rcc --profile glm-47-flash run bash -lc "sbatch deploy/glm-47-flash-beverin/serve_glm_47_flash.sbatch"
# N nodes -> N replicas (each TP=4), e.g. 10 nodes:
rcc --profile glm-47-flash run bash -lc "sbatch --nodes=10 deploy/glm-47-flash-beverin/serve_glm_47_flash.sbatch"
```

> The node count is set with `sbatch --nodes=N`. The script hardcodes
> `#SBATCH --nodes=5` and then derives the replica count from the actual
> allocation, so an environment variable like `NUM_NODES=3 sbatch ...` has
> **no effect** — Slurm honors the `#SBATCH` directive, not the env var.

## Image pinning

The image is pinned by digest in `glm-47-flash-rocm.toml`, not `:latest`. A
floating `:latest` can move under you (vLLM ROCm images are rebuilt frequently)
and silently change which kernels get JIT-compiled; pinning makes deployments
reproducible. To re-resolve the digest when intentionally upgrading:

```bash
REPO=vllm/vllm-openai-rocm
TOKEN=$(curl -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${REPO}:pull" \
  | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
curl -fsS -H "Authorization: Bearer ${TOKEN}" \
  -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' -D - -o /dev/null \
  "https://registry-1.docker.io/v2/${REPO}/manifests/latest" | grep -i docker-content-digest
```

> Note on cold start: GLM-4.7-Flash startup on MI300 is dominated by Triton JIT
> compilation during piecewise CUDA-graph capture (~15-30 min per replica). The
> `TRITON_CACHE_DIR` on shared `/capstor` does **not** reliably speed up
> subsequent multi-node launches — the compile cache written by many concurrent
> processes (5 nodes × 4 TP workers) is not safely shared and is effectively
> rebuilt each run. aiter (`VLLM_ROCM_USE_AITER=1`) uses pre-baked default fused_moe
> configs (`AITER_ONLINE_TUNE` is off), so it is not the bottleneck. There is no
> working cache-warmup shortcut currently; cold start is inherent to this image.

Check status with the rcc-backed helper:

```bash
./deploy/glm-47-flash-beverin/status_glm_47_flash.sh
```

The vLLM output is captured in the OpenTela log because vLLM runs as an OpenTela
subprocess. With `N` nodes there are `N` per-rank logs:

```text
/capstor/scratch/cscs/xyao/glm-47-flash-vllm/logs/opentela-<JOBID>-<RANK>.log
```

The job writes all endpoints and service metadata to:

```text
/capstor/scratch/cscs/xyao/glm-47-flash-vllm/last_service.env
```

From a workstation, tunnel the API after the job starts:

```bash
ssh -L 8000:<HEAD_NODE>:8000 beverin
curl http://127.0.0.1:8000/v1/models
```
