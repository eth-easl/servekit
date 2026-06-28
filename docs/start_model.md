# Starting models on the CSCS clusters

A runbook for bringing up the LLM serving stack (vLLM / SGLang) on the CSCS GPU
clusters behind OpenTela. Each model × cluster × engine combo lives as a
self-contained directory under `deploy/<model>-<cluster>/`, and is driven
through `rcc` profiles defined in `.rcc/config.toml`.

This doc is the index across all of them. For the deep "why" behind any step,
read the per-deployment `README.md` and the agent skill at
`.agents/skills/deploy/`.

> **Automating this runbook:** `tools/snapper/` provides a `snapper` CLI that
> runs these steps as commands (`snapper up/down/status/logs/verify`). It reads
> a per-deploy `deploy/<model>-<cluster>/snapper.toml`. See
> `tools/snapper/README.md`.

## The shape of every deployment

Every model is served with the **same runtime shape**, regardless of cluster
or engine:

```
N replicas = (1 OpenTela peer + 1 engine instance) per node
```

- Each node runs **one engine instance** at some tensor-parallel size **across
  that node's own GPUs only** (single-node TP — no cross-node `torchrun`).
- Each node runs an **OpenTela** peer that manages the engine as its
  subprocess (`otela start --subprocess …`), connects to the shared bootstrap
  peer, and registers the `llm` service on the engine's port.
- The engine is the **OpenTela subprocess**, so the engine log lives *inside*
  the OpenTela log. With N nodes there are N per-rank logs.

The cache-aware-router variants (SGLang, bristen only) change the topology to
**N backends + 1 router**, with the router registered to OpenTela instead of
the engine — see [Cache-aware router (SGLang)](#cache-aware-router-sglang).

## Clusters

All three use Slurm + the CSCS Container Engine (Enroot + Pyxis) with **EDF**
(Environment Definition File) images loaded via
`srun --environment=<EDF_NAME>`.

| Cluster | GPU / node | Partition | Account | Vendor | Default engine |
|---------|-----------|-----------|---------|--------|----------------|
| **bristen** | 4 × A100-80GB (sm_80, NVLink) | `normal` | `a-infra02` | NVIDIA (CUDA) | SGLang |
| **clariden** | 4 × GH200-120GB (sm_90) | `normal` | `a-infra02` | NVIDIA (CUDA) | SGLang |
| **beverin** | 4 × MI300X-192GB | `mi300` | `root` | AMD (ROCm) | vLLM |

Shared filesystems (all clusters):

- `/capstor/store/cscs/swissai/infra01/hf_models/models/` — staged model snapshots
- `/capstor/scratch/cscs/xyao/` — per-user scratch (deploy dirs, caches, opentela)
- `/capstor/store/cscs/swissai/infra01/ocf-share/` — shared binaries (`vmagent-amd64`)

`/users/xyao` (home) is quota-limited — every EDF redirects `$HOME`, HF/XDG/
Triton/pip caches into the deploy dir on `/capstor`.

## Deployment catalog

| Model | Cluster | Engine | Served name | Deploy dir | Profile | TP | Default nodes |
|-------|---------|--------|-------------|------------|---------|----|---------------|
| GLM-4.7-Flash | beverin | vLLM (ROCm) | `zai-org/GLM-4.7-Flash-rocm` | `deploy/glm-47-flash-beverin/` | `glm-47-flash` | 4 | 5 |
| GLM-4.7-Flash | bristen | SGLang (CUDA) | `zai-org/GLM-4.7-Flash` | `deploy/glm-47-flash-bristen/` | `glm-47-flash-bristen` | 4 | 5 |
| GLM-4.7-Flash | bristen | SGLang + **router** | `zai-org/GLM-4.7-Flash-xzyao` | `deploy/glm-47-flash-bristen-xzyao/` | `glm-47-flash-bristen-xzyao` | 4 | 5 |
| Kimi-K2.7-Code | beverin | vLLM (ROCm) | `moonshotai/Kimi-K2.7-Code` | `deploy/kimi-k25-beverin/` | `beverin` | 4 | 2 |
| GLM-5.2-FP8 | beverin | vLLM (ROCm) | `zai-org/GLM-5.2-FP8-rocm` | `deploy/glm-52-fp8-beverin/` | `glm-52-fp8` | 4 | 4 |

The **bristen-vllm** and **bench-storage** dirs under `deploy/` are
experimental (probe-only / no serve script) and are not in the catalog.

## Standard workflow (every deployment)

```bash
PROFILE=<profile>          # from the table above
DIR=<deploy dir>           # e.g. deploy/glm-47-flash-bristen

# 1. push this tree to the cluster's scratch dir
rcc --profile $PROFILE push

# 2. probe (1 node) — validates image, GPU, engine version, parsers, model config.
#    CHEAP and catches the expensive-to-debug problems early. Do NOT skip.
rcc --profile $PROFILE run bash -lc "sbatch $DIR/probe_<engine>.sbatch"
#   ... wait for the probe job, read its log ...

# 3. serve (N nodes) — each node becomes one replica
rcc --profile $PROFILE run bash -lc "sbatch $DIR/serve_<model>.sbatch"

# 4. verify readiness + inference (see "Verify" below)
```

Node count is set with `sbatch --nodes=N` on the CLI (overrides the
`#SBATCH --nodes` directive). An env var like `NUM_NODES=3 sbatch ...` does
**nothing** — Slurm honors the directive, not the env var.

### Per-model runbooks

**GLM-4.7-Flash on beverin (vLLM / ROCm):**
```bash
rcc --profile glm-47-flash push
rcc --profile glm-47-flash run bash -lc "sbatch deploy/glm-47-flash-beverin/probe_vllm.sbatch"
rcc --profile glm-47-flash run bash -lc "sbatch deploy/glm-47-flash-beverin/serve_glm_47_flash.sbatch"
```
Defaults: TP=4, `MAX_MODEL_LEN=131072`, `GPU_MEMORY_UTILIZATION=0.50` (low —
ROCm/MI300X needs headroom for AITER scratch), reasoning parser `glm45`,
tool parser `glm47`. Engine port 8000.

**GLM-4.7-Flash on bristen (SGLang / CUDA):**
```bash
rcc --profile glm-47-flash-bristen push
rcc --profile glm-47-flash-bristen run bash -lc "sbatch deploy/glm-47-flash-bristen/probe_sglang.sbatch"
rcc --profile glm-47-flash-bristen run bash -lc "sbatch deploy/glm-47-flash-bristen/serve_glm_47_flash_sglang.sbatch"
```
Defaults: TP=4, `MAX_MODEL_LEN=131072`, `MEM_FRACTION_STATIC=0.85`,
`MAX_RUNNING_REQUESTS=256`, parsers `glm45`/`glm47`, `--enable-metrics` (needed
for vmagent — SGLang exposes `/metrics` only with this flag). Engine port 8080.

**Kimi-K2.7-Code on beverin (vLLM / ROCm):**
```bash
rcc --profile beverin push
rcc --profile beverin run bash -lc "sbatch deploy/kimi-k25-beverin/probe_vllm.sbatch"
rcc --profile beverin run bash -lc "sbatch deploy/kimi-k25-beverin/serve_kimi_k25.sbatch"
```
Defaults: TP=4, 2 nodes, `MAX_MODEL_LEN=131072`, `GPU_MEMORY_UTILIZATION=0.80`,
parsers `kimi_k2`/`kimi_k2`. Engine port 8000.

**GLM-5.2-FP8 on beverin (vLLM / ROCm):**
```bash
rcc --profile glm-52-fp8 push
rcc --profile glm-52-fp8 run bash -lc "sbatch deploy/glm-52-fp8-beverin/probe_vllm.sbatch"
rcc --profile glm-52-fp8 run bash -lc "sbatch deploy/glm-52-fp8-beverin/serve_glm_52_fp8.sbatch"
```
Defaults: TP=4, 4 nodes, `MAX_MODEL_LEN=262144` (long context),
`GPU_MEMORY_UTILIZATION=0.90`, parsers `glm45`/`glm47`. Engine port 8000.

## Cache-aware router (SGLang)

**Bristen SGLang only.** The base SGLang deploy registers each replica to
OpenTela as its own `llm` peer, so the gateway balances **cache-blind** —
shared-prefix traffic scatters across replicas and each replica's RadixCache
under-hits. Putting the [SGLang Router](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway)
in front with a `cache_aware` policy steers prefix-affine requests to the
**same** backend, lifting cross-replica prefix-cache hit rate. Reach for it
when the workload has shared prefixes (system prompts, few-shot, multi-turn)
and you run ≥2 replicas.

Topology change:

```
external client → OpenTela gateway → [rank 0] sglang_router (cache_aware)
                                           ├─ http://node0:8080  (SGLang TP=N)
                                           ├─ http://node1:8080  (SGLang TP=N)
                                           └─ …                   (one per node)
```

- Every node runs a **plain SGLang backend** — *not* an OpenTela subprocess.
  Backends are private, reachable only inside the allocation by hostname.
- **Rank 0** additionally runs the router (CPU-only) and the **single**
  OpenTela peer that supervises the router as its subprocess and registers
  it as the public `llm` service.
- The router (`sglang-router`) is **not** in the `lmsysorg/sglang` image — it
  pip-installs from PyPI at startup (~30 MB, ~1 s). Pin it
  (`sglang-router==0.3.2`). Always run `probe_router.sbatch` before the first
  router serve.

**Worked deploy:** `deploy/glm-47-flash-bristen-xzyao/`
(served as `zai-org/GLM-4.7-Flash-xzyao`).

```bash
rcc --profile glm-47-flash-bristen-xzyao push
rcc --profile glm-47-flash-bristen-xzyao run bash -lc "sbatch deploy/glm-47-flash-bristen-xzyao/probe_router.sbatch"
rcc --profile glm-47-flash-bristen-xzyao run bash -lc "sbatch deploy/glm-47-flash-bristen-xzyao/serve_glm_47_flash_sglang_router.sbatch"
```

The public endpoint is the **router** on the head node
(`ENDPOINT=<head>:8090` in `last_service.env`), not the per-node backends.

### DP=2 variant (use all GPUs at TP=2)

The base router script runs **one** backend per node, so at TP=2 it uses only
2 of the node's 4 A100s (the other 2 sit idle). `serve_glm_47_flash_sglang_router_dp2.sbatch`
runs **two TP=2 backends per node** (data-parallel across the node's GPUs) so
all 4 A100s are used:

```
per node:  backend A  CUDA_VISIBLE_DEVICES=0,1  port 8080  TP=2  (GPUs 0,1)
           backend B  CUDA_VISIBLE_DEVICES=2,3  port 8081  TP=2  (GPUs 2,3)
```

N nodes × DP=2 → **2N backends** (10 on the default 5-node job), all fronted
by the single cache-aware router, with OpenTela still a single `llm` peer on
rank 0. GPU partitioning is per-process (`CUDA_VISIBLE_DEVICES`), so the two
backends on a node never touch each other's GPUs.

```bash
rcc --profile glm-47-flash-bristen-xzyao run bash -lc \
  "sbatch deploy/glm-47-flash-bristen-xzyao/serve_glm_47_flash_sglang_router_dp2.sbatch"
```

Defaults: `TP_SIZE=2`, `DP_PER_NODE=2` (TP×DP = 4 GPUs/node, guarded by a
startup check). Per-backend logs are `sglang-backend-<JOBID>-<RANK>-<d>.log`
(`d` ∈ {0,1}).

**DP=2 gotcha — pip race.** The DP=2 backends share one container's
site-packages, so if each backend ran `pip install transformers==…` they'd
race and corrupt the package (seen: "invalid distribution ~ransformers" →
`ValueError tensorflow_text` at import). The DP=2 script therefore does the
transformers upgrade **once per node in the launcher, before any backend
spawns**; the backends just `exec sglang`. Keep that pattern for any future
multi-backend-per-container variant.

### Running multiple deployments concurrently

Two router deployments serving the **same** model name are fine — the gateway
treats them as two providers of the same `llm` service and aggregates them
(e.g. two 5-node DP=2 router jobs = one logical model backed by 20 A100s).
The only hard requirement is that the OpenTela peers keep **distinct libp2p
ports** (a process-level constraint), so the second job bumps them:

```bash
# second concurrent deployment of the SAME model, isolated ports
rcc --profile glm-47-flash-bristen-xzyao run bash -lc '
  cd /capstor/scratch/cscs/xyao/glm-47-flash-sglang-xzyao
  export OPENTELA_TCP_PORT=45905     # +1000 vs the canonical 44905
  export OPENTELA_UDP_PORT=61820     # +1000 vs the canonical 60820
  sbatch deploy/glm-47-flash-bristen-xzyao/serve_glm_47_flash_sglang_router_dp2.sbatch
'
```

**Do not** mix the *base* (cache-blind, each replica a separate `llm` peer)
and *router* (cache-aware) topologies under the same `llm` name at once —
they'd both register `llm` and the gateway would mix cache-blind and
cache-aware paths. Cancel one first, or isolate with a distinct served-model
name **and** bumped ports.

## Verify a running service

Wait until each rank prints `fired up and ready to roll` (SGLang) /
`APIV1-server ready` (vLLM) and `vmagent_started`, then:

```bash
PROFILE=<profile>
# endpoints written by the job into the deploy dir
cat /capstor/scratch/cscs/xyao/<deploy-scratch-dir>/last_service.env

EP=$(rcc --profile $PROFILE run bash -lc "grep '^ENDPOINT' last_service.env | head -1 | cut -d= -f2")
curl http://$EP/v1/models
curl http://$EP/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<served-model-name>","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```

OpenTela registration shows up on the gateway dashboard at
<https://serving.swissai.svc.cscs.ch/>. The router deploy's rank-0 OpenTela
log confirms registration: look for `register_workers` /
`register_tokenizer '<served name>'` and the gateway proxying
`GET /v1/_service/llm/...` requests.

To **prove cache-awareness** of the router: fire several identical-prefix
requests and compare per-backend `sglang:num_requests_total` before/after — a
`cache_aware` router concentrates them on ONE worker (verified: 10/10 to one
backend); `round_robin` would split them evenly.

## Common operations

```bash
# status (rcc-backed helper per deploy)
./deploy/<model>-<cluster>/status_<model>*.sh

# node count on the CLI (overrides #SBATCH --nodes)
rcc --profile <p> run bash -lc "sbatch --nodes=10 deploy/.../serve_*.sbatch"

# override TP / memory / parsers at submit time
rcc --profile <p> run bash -lc "TP_SIZE=2 sbatch deploy/.../serve_*.sbatch"

# cancel
rcc --profile <p> run bash -lc "scancel <JOBID>"

# local tunnel to the API
ssh -L 8000:<HEAD_NODE>:8000 beverin        # or :8090 for the bristen router
curl http://127.0.0.1:8000/v1/models
```

Readiness markers and per-rank log locations:

| Engine | Per-rank log | Ready when you see |
|--------|--------------|--------------------|
| SGLang | `logs/opentela-<JOB>-<RANK>.log` | `The server is fired up and ready to roll!` + `sglang_ready rank=<R>` |
| vLLM | `logs/opentela-<JOB>-<RANK>.log` | `APIV1-server started/ready` + `vllm_ready rank=<R>` |
| SGLang router | `logs/opentela-<JOB>-0.log` | `sglang_router_version=…` → `router_ready` |

Filter log noise with `grep -v go-ds-crdt` (OpenTela/libp2p gossip timeouts
are non-fatal). First launch per image is slow (cold Lustre weight read +
FlashInfer/Triton JIT + CUDA-graph capture) — 10–20 min is normal, restarts
are much faster once the shared `/capstor` caches are warm.

## Key rules

1. **Pin the container image by digest** in the EDF, never a floating tag —
   floating tags silently break the shared Triton/FlashInfer JIT caches.
2. **OpenTela manages the engine as a subprocess** (`otela start --subprocess …`),
   not as a peer-launched sidecar.
3. **Per-rank OpenTela libp2p ports are offset by `SLURM_PROCID`** and peer
   joins are staggered (10 s × rank) — avoids NAT port collision and
   thundering-herd on the single bootstrap.
4. **Always probe before serve.** The probe is cheap and catches
   config/version problems on 1 node instead of N.
5. **Set `HF_HUB_OFFLINE=1`** in the EDF (models are pre-staged), but compute
   nodes *do* have PyPI access for in-container `pip` upgrades.
6. **GLM-4.7-Flash needs a transformers upgrade.** Its `model_type`
   (`glm4_moe_lite`) is newer than the image's pinned `transformers`; the
   serve scripts upgrade to `transformers==5.12.1` at startup (`--trust-remote-code`
   doesn't help — the checkpoint ships no custom modeling code).
7. **SGLang needs `--enable-metrics`** for vmagent (vLLM has it on by default).

The full rationale for every rule is in `.agents/skills/deploy/references/gotchas.md`.
