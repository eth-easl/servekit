# GLM-5.2-FP8 on Beverin

This deploys the local `zai-org/GLM-5.2-FP8` snapshot with the ROCm vLLM image
through the CSCS Slurm Container Engine.

The model is a 704G FP8 checkpoint. A single `mi300` node has 4 GPUs, so the
service uses one 4-node vLLM instance with 16 GPUs total (TP=4 within each node,
PP=4 across them, ~44 GB/GPU of weights):

```text
TP_SIZE=4
PP_SIZE=4              # defaults to the Slurm node count (TP stays intra-node)
MAX_MODEL_LEN=262144  # 256K (256*1024); model is natively 1M-capable
MAX_NUM_SEQS=8        # fills the 4-stage pipeline; ~6 GB/GPU KV per 256K seq
GPU_MEMORY_UTILIZATION=0.90
KV_CACHE_MEMORY_BYTES unset
KV_CACHE_DTYPE=auto   # set to fp8 to halve KV cache (only for many concurrent long streams)
ENFORCE_EAGER=0
```

Tensor parallelism stays within each node and pipeline parallelism spans the
nodes (`PP_SIZE` defaults to the Slurm node count), matching the existing
multi-node Kimi launcher pattern. The 4-stage pipeline only pays off with
several requests in flight, so `MAX_NUM_SEQS=8` keeps the stages busy —
autoregressive decode of a single sequence cannot fill a pipeline. To scale
back to 2 nodes, submit with `sbatch --nodes=2 ...`; `PP_SIZE` follows
automatically (each 256K seq then costs ~12 GB/GPU at PP=2, so lower
`MAX_NUM_SEQS` accordingly).

## Context length

The deployment defaults to a 256K window (`MAX_MODEL_LEN=262144`). GLM-5.2 is a
`GlmMoeDsaForCausalLM`: it uses MLA (`kv_lora_rank=512`) plus DeepSeek sparse
attention (`index_topk=2048`), so the KV cache is a single compressed latent per
token rather than full per-head K/V. With PP=4 each GPU holds ~19.5 of the 78
layers, so a 256K sequence costs only ~6 GB/GPU of KV in BF16 on top of the
~44 GB/GPU of weights. At `MAX_NUM_SEQS=8` the worst case (8 full-length 256K
streams) is ~47 GB/GPU of KV — ~99 GB/GPU total, within the 128 GB MI300A
budget at `GPU_MEMORY_UTILIZATION=0.90`.

The model is natively 1M-capable (`max_position_embeddings=1048576`,
`rope_type=default`), so no RoPE scaling is required. At 4 nodes a single 1M
stream fits in BF16 (~24 GB/GPU KV + 44 GB weights ≈ 68 GB/GPU); set
`KV_CACHE_DTYPE=fp8` only if you want several concurrent 1M streams:

```bash
rcc --profile glm-52-fp8 run bash -lc \
  "MAX_MODEL_LEN=1048576 MAX_NUM_SEQS=1 sbatch deploy/glm-52-fp8-beverin/serve_glm_52_fp8.sbatch"
```

The GLM-specific vLLM flags follow the upstream GLM-5 recipe:

```text
--tool-call-parser glm47
--reasoning-parser glm45
--enable-auto-tool-choice
--chat-template-content-format string
```

MTP speculative decoding is wired behind `ENABLE_MTP=1`:

```text
--speculative-config.method mtp
--speculative-config.num_speculative_tokens 3
```

It defaults off for the first ROCm bring-up because the GLM-4.7 deployment hit
an ROCm MLA CUDA-graph assertion with MTP enabled. Enable it at submission time
after a baseline service is healthy.

If post-load setup runs out of memory, retry with eager execution and a lower
utilization (at PP=4, 8×256K needs ~47 GB/GPU of KV, so keep any fixed
`KV_CACHE_MEMORY_BYTES` budget above your `MAX_NUM_SEQS` × per-seq cost):

```bash
rcc --profile glm-52-fp8 run bash -lc "ENFORCE_EAGER=1 GPU_MEMORY_UTILIZATION=0.85 sbatch --time=04:00:00 deploy/glm-52-fp8-beverin/serve_glm_52_fp8.sbatch"
```

## Remote layout

Files are copied to:

```text
/capstor/scratch/cscs/xyao/glm-52-fp8-vllm
```

The model is loaded from:

```text
/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-5.2-FP8
```

vLLM, XDG, FlashInfer, Triton, and container `$HOME` writes are redirected to
the deploy directory to avoid `$HOME` quota failures on Beverin.

## Run

```bash
rcc --profile glm-52-fp8 push
rcc --profile glm-52-fp8 run bash -lc "sbatch deploy/glm-52-fp8-beverin/probe_vllm.sbatch"
rcc --profile glm-52-fp8 run bash -lc "sbatch deploy/glm-52-fp8-beverin/serve_glm_52_fp8.sbatch"
```

To retry with MTP:

```bash
rcc --profile glm-52-fp8 run bash -lc "ENABLE_MTP=1 sbatch deploy/glm-52-fp8-beverin/serve_glm_52_fp8.sbatch"
```

The job writes endpoint metadata to:

```text
/capstor/scratch/cscs/xyao/glm-52-fp8-vllm/last_service.env
```

From a workstation, tunnel the API after the job starts:

```bash
ssh -L 8000:<HEAD_NODE>:8000 beverin
curl http://127.0.0.1:8000/v1/models
```
