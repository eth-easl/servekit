# Kimi-K2.7 on Beverin

This deploys the local `moonshotai/Kimi-K2.7-Code` snapshot with the ROCm vLLM image through the
CSCS Slurm Container Engine.

The service uses two `mi300` nodes. Beverin exposes four MI300 GPUs per node,
so the launch maps vLLM to `--tensor-parallel-size 4 --pipeline-parallel-size 2`.
The Kimi-specific parser flags and ROCm AITER environment come from the vLLM
Kimi-K2.5 ROCm recipe; Kimi-K2.7-Code uses the same vLLM model architecture.

The serve script also launches an [OpenTela](https://github.com/swiss-ai/OpenTela) peer
on the head node that manages the multi-node vLLM job as its subprocess. It connects to
the bootstrap peer
`/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9`,
registers the service as `llm` on port `8000`, and exposes the model as
`moonshotai/Kimi-K2.7-Code`.

## Remote layout

Files are copied to:

```text
/capstor/scratch/cscs/xyao/kimi-k25-vllm
```

The EDF file lives in that directory and is found by exporting `EDF_PATH` before
calling `srun --environment=kimi-k25-rocm`.

The model is loaded from:

```text
/capstor/store/cscs/swissai/infra01/hf_models/models/moonshotai/Kimi-K2.7-Code
```

vLLM, XDG, FlashInfer, Triton, and container `$HOME` writes are redirected to
the deploy directory to avoid `$HOME` quota failures on Beverin.

The service defaults are intentionally conservative after the initial full
context run exceeded runtime memory during KV allocation:

```text
MAX_MODEL_LEN=131072
KV_CACHE_MEMORY_BYTES=20G
MAX_NUM_SEQS=1
GPU_MEMORY_UTILIZATION=0.80
```

These can be overridden with environment variables at submission time if you
want to trade stability for more context or concurrency.

## Run

`rcc` resolves host and remote dir from `.rcc/config.toml`, so there is no need
to `cd` or hardcode paths:

```bash
rcc --profile beverin push
rcc --profile beverin run bash -lc "sbatch deploy/kimi-k25-beverin/probe_vllm.sbatch"
rcc --profile beverin run bash -lc "sbatch deploy/kimi-k25-beverin/serve_kimi_k25.sbatch"
```

Unlike the GLM-4.7-Flash deployment, the model is **sharded across nodes**
(TP=4 within each node, PP=2 across them), so this is a single multi-node
vLLM instance, not per-node replicas. The script hardcodes `#SBATCH --nodes=2`
with `PP_SIZE=2`; to change the node count, pass `sbatch --nodes=N` **and** set
`PP_SIZE=N` to match — here `PP_SIZE` does not auto-follow the node count
(unlike the GLM-5.2 deployment):

```bash
# 4-node single instance: TP=4, PP=4
rcc --profile beverin run bash -lc "PP_SIZE=4 sbatch --nodes=4 deploy/kimi-k25-beverin/serve_kimi_k25.sbatch"
```

Check status with the rcc-backed helper:

```bash
./deploy/kimi-k25-beverin/status_kimi_k25.sh
```

The vLLM output is captured in the OpenTela log because vLLM runs as an OpenTela subprocess:

```text
/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/opentela-<JOBID>.log
```

The job writes its current endpoint metadata to:

```text
/capstor/scratch/cscs/xyao/kimi-k25-vllm/last_service.env
```

From a workstation, tunnel the API after the job starts:

```bash
ssh -L 8000:<HEAD_NODE>:8000 beverin
curl http://127.0.0.1:8000/v1/models
```
