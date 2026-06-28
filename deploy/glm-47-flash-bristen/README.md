# GLM-4.7-Flash on Bristen (SGLang, CUDA)

This deploys the local `zai-org/GLM-4.7-Flash` snapshot with the **official
SGLang CUDA image** (`lmsysorg/sglang:v0.5.9`) through the CSCS Container Engine
(Enroot + Pyxis EDF).

Bristen is the NVIDIA partition: `4 × A100-SXM4-80GB` per node on NVLink, Slurm
`normal` partition, account `a-infra02`. This is the CUDA/NVIDIA counterpart to
the beverin ROCm/vLLM deploy — same operational shape (N replicas, one
SGLang + OpenTela instance per node), different stack.

## Architecture

The service script allocates multiple nodes (default **5**) and runs **one
independent SGLang + OpenTela instance (a replica) on each node**:

- Each node runs SGLang at **TP=4** across the node's 4 A100s (single-node
  tensor parallel — no cross-node distributed init, no `torchrun`).
- Each node runs its own **OpenTela** peer that manages SGLang as its subprocess
  (`otela start --subprocess ...`), connects to the bootstrap peer, and
  registers the `llm` service on port `8080`.
- 5 nodes ⇒ 5 SGLang servers ⇒ 5 `llm` peers in OpenTela.

SGLang flags: `--trust-remote-code` (GLM remote code), `--tensor-parallel-size 4`,
`--context-length` (default 131072), `--mem-fraction-static` (default 0.85),
`--max-running-requests` (default 256), and `--enable-metrics` so the per-rank
vmagent can scrape `/metrics`. Prefix caching (RadixCache) is on by default in
SGLang. The reasoning parser `glm45` and tool-call parser `glm47` are on by
default — the same pair the beverin vLLM deploy uses. Override via
`SGLANG_REASONING_PARSER` / `SGLANG_TOOL_CALL_PARSER` (run the probe to list the
parser names available in this image).

## Container image

The image is **pinned by digest** in `glm-47-flash-sglang.toml`
(`lmsysorg/sglang@sha256:e216b7dc...`, resolved 2026-06-26 from the `v0.5.9`
tag — CUDA 12.9.1, NCCL 2.27.3, FlashInfer 0.6.3, A100/sm_80 compatible). A
floating `:latest` would silently move and invalidate the Triton/FlashInfer JIT
caches kept on shared `/capstor`. To re-resolve the digest when upgrading:

```bash
REPO=lmsysorg/sglang
TOKEN=$(curl -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${REPO}:pull" \
  | sed -n 's/.*"token":"\([^"]*\)".*/\1/p')
curl -fsS -H "Authorization: Bearer ${TOKEN}" \
  -H 'Accept: application/vnd.docker.distribution.manifest.list.v2+json' -D - -o /dev/null \
  "https://registry-1.docker.io/v2/${REPO}/manifests/v0.5.9" | grep -i docker-content-digest
```

The CSCS Container Engine auto-pulls and caches the image at
`${SCRATCH}/.edf_imagestore` (i.e. `/capstor/scratch/cscs/xyao/.edf_imagestore`).
The first launch per architecture pulls (~6 GB); later launches reuse the
cached squashfs. See the CSCS docs:
<https://docs.cscs.ch/software/container-engine/run/>.

## Model compatibility (transformers upgrade)

GLM-4.7-Flash uses `model_type = glm4_moe_lite`, which is newer than the
`transformers==4.57.1` pinned in the official SGLang 0.5.9 image — stock
4.57.1 (and even 5.0.0rc0) reject the config with *"Transformers does not
recognize this architecture"*, and the checkpoint ships no custom modeling
code for `--trust-remote-code` to fall back on. Native support lands later in
the transformers 5.x line (verified on 5.12.1).

The serve script therefore upgrades `transformers` in the container at startup
(bristen compute nodes have PyPI access), pinned to `TRANSFORMERS_VERSION`
(default `5.12.1`), with a persistent pip cache under the deploy dir so repeat
launches stay fast. SGLang 0.5.9 still imports and serves correctly on
transformers 5.12.1 (the image's pinned 4.57.1 is a lower bound, not a hard
requirement). Override the version with `TRANSFORMERS_VERSION=...` at
submission time.

## Remote layout

Files are synced to:

```text
/capstor/scratch/cscs/xyao/glm-47-flash-sglang
```

The EDF (`glm-47-flash-sglang.toml`) lives in that tree and is found by
exporting `EDF_PATH` before calling `srun --environment=glm-47-flash-sglang`.

The model is loaded from:

```text
/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
```

SGLang, XDG, Triton, and container `$HOME` writes are redirected to the deploy
directory (same rationale as beverin: avoid `$HOME`/`/users` quota failures).

The service defaults are:

```text
MAX_MODEL_LEN=131072
MAX_RUNNING_REQUESTS=256
MEM_FRACTION_STATIC=0.85
TP_SIZE=4
SGLANG_REASONING_PARSER=glm45
SGLANG_TOOL_CALL_PARSER=glm47
DISABLE_CUDA_GRAPH=0
TIME_LIMIT=12:00:00
```

These can be overridden with environment variables at submission time.

## Run

`rcc` resolves host and remote dir from `.rcc/config.toml` (the
`glm-47-flash-bristen` profile), so there is no need to `cd` or hardcode paths:

```bash
# probe: confirm the image, GPU type, SGLang version, and available parsers
rcc --profile glm-47-flash-bristen push
rcc --profile glm-47-flash-bristen run bash -lc "sbatch deploy/glm-47-flash-bristen/probe_sglang.sbatch"

# serve: default #SBATCH --nodes=5 -> 5 replicas (each TP=4)
rcc --profile glm-47-flash-bristen run bash -lc "sbatch deploy/glm-47-flash-bristen/serve_glm_47_flash_sglang.sbatch"

# N nodes -> N replicas, e.g. 10 nodes:
rcc --profile glm-47-flash-bristen run bash -lc "sbatch --nodes=10 deploy/glm-47-flash-bristen/serve_glm_47_flash_sglang.sbatch"
```

> Node count is set with `sbatch --nodes=N`. The script hardcodes
> `#SBATCH --nodes=5` and derives the replica count from the actual allocation,
> so an env var like `NUM_NODES=3 sbatch ...` has **no effect** — Slurm honors
> the `#SBATCH` directive, not the env var. (Same caveat as the beverin deploy.)

Check status with the rcc-backed helper:

```bash
./deploy/glm-47-flash-bristen/status_glm_47_flash_sglang.sh
```

The SGLang output is captured in the OpenTela log because SGLang runs as an
OpenTela subprocess. With `N` nodes there are `N` per-rank logs:

```text
/capstor/scratch/cscs/xyao/glm-47-flash-sglang/logs/opentela-<JOBID>-<RANK>.log
```

The job writes all endpoints and service metadata to:

```text
/capstor/scratch/cscs/xyao/glm-47-flash-sglang/last_service.env
```

## Cache-aware router variant

`serve_glm_47_flash_sglang_router.sbatch` is an alternative to the base serve
script that puts the **[SGLang Router](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway)**
in front of the replicas with a **`cache_aware`** routing policy. The base
script registers each SGLang replica to the OpenTela gateway as its own `llm`
peer, so the gateway balances **cache-blind** — shared-prefix traffic scatters
across replicas and each replica's RadixCache under-hits. The router variant
steers prefix-affine requests to the **same** backend, lifting cross-replica
prefix-cache hit rate.

**Topology change:**

```
external client → OpenTela gateway → [rank 0] sglang_router (cache_aware)
                                          ├─ http://node0:8080  (SGLang TP=4)
                                          ├─ http://node1:8080  (SGLang TP=4)
                                          └─ …                   (one per node)
```

- Every node runs a **plain SGLang backend** (TP=4, port `8080`) — *not* an
  OpenTela subprocess. Backends are private, reachable only inside the
  allocation by hostname.
- **Rank 0** additionally runs the router (CPU-only, co-located with its own
  backend on port `8090`) and the **single** OpenTela peer, which supervises the
  router as its subprocess and registers it as the public `llm` service. So
  OpenTela exposes one endpoint = the router; the router owns request→backend.
- The router is installed at startup (`pip install sglang-router==0.3.2`, pinned)
  because it ships **outside** the `sglang:v0.5.9` image — confirmed by
  `probe_router.sbatch`, which also dumps the version's real `launch_router`
  flag surface. The wheel is ~30 MB and installs in ~1 s from the persistent
  pip cache (same mechanism as the transformers upgrade).

**Trade-off:** the router is a single front-door (SPOF) and OpenTela no longer
sees individual replicas. In exchange you get cache-aware routing plus
health-checked backends, retries, and circuit breaking from one Rust hop.

**Run** (probe first to confirm the router installs + flags for the pinned image):

```bash
rcc --profile glm-47-flash-bristen push
rcc --profile glm-47-flash-bristen run bash -lc "sbatch deploy/glm-47-flash-bristen/probe_router.sbatch"

# serve with the cache-aware router (default #SBATCH --nodes=5):
rcc --profile glm-47-flash-bristen run bash -lc "sbatch deploy/glm-47-flash-bristen/serve_glm_47_flash_sglang_router.sbatch"

# short 2-node smoke test before committing a full 12h job:
rcc --profile glm-47-flash-bristen run bash -lc "sbatch --nodes=2 --time=00:40:00 deploy/glm-47-flash-bristen/serve_glm_47_flash_sglang_router.sbatch"
```

**Public endpoint** is the **router** on the head node (`ENDPOINT=<head>:8090`
in `last_service.env`), not the per-node backends (listed as `BACKEND_i=` for
debugging). Cache-aware tuning is overridable at submit time:
`ROUTER_PORT`, `ROUTER_POLICY` (default `cache_aware`; also `random`,
`round_robin`, `power_of_two`, `manual`), `CACHE_THRESHOLD` (router default
0.3), `BALANCE_ABS_THRESHOLD` (64), `BALANCE_REL_THRESHOLD` (1.5),
`ROUTER_VERSION`, `ROUTER_PROM_PORT` (router's own Prometheus, also scraped by
rank 0's vmagent).

Readiness markers in the rank-0 OpenTela log
(`logs/opentela-<JOBID>-0.log`): `sglang_router_version=…` then
`router_ready`. Per-backend logs are `logs/sglang-backend-<JOBID>-<RANK>.log`.

## OpenTela

Each rank launches an [OpenTela](https://github.com/swiss-ai/OpenTela) peer that
manages SGLang as its subprocess. It connects to the bootstrap peer
`/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9`,
registers the service as `llm` on port `8080`, and exposes the model as
`zai-org/GLM-4.7-Flash`. Per-rank libp2p tcp/udp ports are offset by
`SLURM_PROCID` so the peers do not collide on a shared NAT egress, and peer
joins are staggered (10 s × rank) to avoid thundering the single bootstrap.

## Cold start

GLM-4.7-Flash on A100 is a small model (~4.7 B), so weight load is fast, but the
**first launch per image pays a one-time FlashInfer/Triton JIT compile** (CUDA
graph capture + kernel autotune). The shared `TRITON_CACHE_DIR` on `/capstor`
warms across runs *for a pinned image* — another reason the EDF pins by digest.

## Local tunnel to the API

After the service job starts and writes its endpoint metadata:

```bash
ssh -L 8080:<HEAD_NODE>:8080 bristen
curl http://127.0.0.1:8080/v1/models
```
