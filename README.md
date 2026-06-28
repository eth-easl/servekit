# serving-stack

Serving stack for LLM inference deployments (vLLM, SGLang) on CSCS clusters, managed with
[`rcc`](https://github.com/ResearchComputer/remote-cluster-controller)
(remote-cluster-controller).

> **How to start a model:** see [`docs/start_model.md`](docs/start_model.md) —
> the runbook across all current deployments (GLM-4.7-Flash, GLM-5.2-FP8,
> Kimi-K2.7-Code) on bristen / clariden / beverin, including the cache-aware
> SGLang router and DP=2 variants.

## Setup

Install `rcc` locally:

```bash
pip install remote-cluster-controller
# or: uv tool install remote-cluster-controller
```

The repository already contains `.rcc/config.toml` with a default `beverin`
profile plus GLM profiles. Each profile points at the canonical
remote working directory used by the corresponding deployment scripts.

## Sync code to the cluster

```bash
# push the working tree to the default profile (beverin)
rcc push

# push to a specific profile
rcc --profile beverin push
rcc --profile glm-47-flash push
rcc --profile glm-52-fp8 push
```

`rcc` respects `.gitignore` and the extra excludes in `.rcc/rccignore` (logs,
secrets, model weights, caches, etc.).

## Run commands remotely

```bash
# open a shell in the remote repo directory
rcc run bash -l

# submit the probe job
rcc run bash -lc "sbatch deploy/kimi-k25-beverin/probe_vllm.sbatch"

# submit the service job
rcc run bash -lc "sbatch deploy/kimi-k25-beverin/serve_kimi_k25.sbatch"

# submit the GLM-4.7-Flash service job (after switching to the glm-47-flash profile)
rcc --profile glm-47-flash run bash -lc "sbatch deploy/glm-47-flash-beverin/serve_glm_47_flash.sbatch"

# submit the GLM-5.2-FP8 service job
rcc --profile glm-52-fp8 run bash -lc "sbatch deploy/glm-52-fp8-beverin/serve_glm_52_fp8.sbatch"

# submit the GLM-4.7-Flash SGLang service job on bristen (5 nodes -> 5 replicas)
rcc --profile glm-47-flash-bristen push
rcc --profile glm-47-flash-bristen run bash -lc "sbatch deploy/glm-47-flash-bristen/serve_glm_47_flash_sglang.sbatch"

# submit the GLM-4.7-Flash cache-aware-router service on bristen (served as
# zai-org/GLM-4.7-Flash-xzyao; 5 nodes -> 5 backends + 1 router, behind OpenTela)
rcc --profile glm-47-flash-bristen-xzyao push
rcc --profile glm-47-flash-bristen-xzyao run bash -lc "sbatch deploy/glm-47-flash-bristen-xzyao/serve_glm_47_flash_sglang_router.sbatch"

# submit the GLM-4.7-Flash cache-aware-router service on bristen in DP=2 mode
# (two TP=2 backends per node -> all 4 A100s/node used; 5 nodes -> 10 backends
# + 1 router, behind OpenTela)
rcc --profile glm-47-flash-bristen-xzyao run bash -lc "sbatch deploy/glm-47-flash-bristen-xzyao/serve_glm_47_flash_sglang_router_dp2.sbatch"

# check the service endpoint written by the job
rcc run bash -lc "cat /capstor/scratch/cscs/xyao/kimi-k25-vllm/last_service.env"
```

## Local tunnel to the running API

After the service job starts and writes its endpoint metadata:

```bash
ssh -L 8000:<HEAD_NODE>:8000 beverin
curl http://127.0.0.1:8000/v1/models
```

## Layout

```text
.
├── .rcc/
│   ├── config.toml   # rcc profiles (host + remote_dir)
│   └── rccignore     # extra rsync excludes beyond .gitignore
├── deploy/
│   ├── kimi-k25-beverin/
│   │   ├── README.md
│   │   ├── probe_vllm.sbatch
│   │   └── serve_kimi_k25.sbatch
│   ├── glm-47-flash-beverin/
│       ├── README.md
│       ├── probe_vllm.sbatch
│       └── serve_glm_47_flash.sbatch
│   ├── glm-52-fp8-beverin/
│   │   ├── README.md
│   │   ├── probe_vllm.sbatch
│   │   └── serve_glm_52_fp8.sbatch
│   ├── glm-47-flash-bristen/
│   │   ├── README.md
│   │   ├── glm-47-flash-sglang.toml
│   │   ├── probe_sglang.sbatch
│   │   ├── probe_router.sbatch
│   │   ├── serve_glm_47_flash_sglang.sbatch
│   │   ├── serve_glm_47_flash_sglang_router.sbatch
│   │   └── status_glm_47_flash_sglang.sh
│   └── glm-47-flash-bristen-xzyao/
│       ├── README.md
│       ├── glm-47-flash-sglang-xzyao.toml
│       ├── probe_sglang.sbatch
│       ├── probe_router.sbatch
│       ├── serve_glm_47_flash_sglang_router.sbatch
│       ├── serve_glm_47_flash_sglang_router_dp2.sbatch   # TP=2 x DP=2/node -> all 4 A100s
│       ├── serve_glm_47_flash_sglang.sbatch
│       └── status_glm_47_flash_sglang.sh
└── README.md
```

Both service scripts launch an [OpenTela](https://github.com/swiss-ai/OpenTela)
peer that manages vLLM as its subprocess and registers the `llm` service on
port `8000` with the bootstrap peer
`/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9`.

## Profiles

| Profile      | Host    | Remote directory                              |
|--------------|---------|-----------------------------------------------|
| beverin      | beverin | `/capstor/scratch/cscs/xyao/kimi-k25-vllm`    |
| glm-47-flash | beverin | `/capstor/scratch/cscs/xyao/glm-47-flash-vllm` |
| glm-52-fp8   | beverin | `/capstor/scratch/cscs/xyao/glm-52-fp8-vllm`  |
| glm-47-flash-bristen | bristen | `/capstor/scratch/cscs/xyao/glm-47-flash-sglang` |
| glm-47-flash-bristen-xzyao | bristen | `/capstor/scratch/cscs/xyao/glm-47-flash-sglang-xzyao` |
