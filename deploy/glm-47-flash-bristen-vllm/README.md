# GLM-4.7-Flash — vLLM-CUDA on bristen (A100/sm_80)

This deploy proves G1: vLLM with CUDA backend serves `glm4_moe_lite` (GLM-4.7-Flash, 28B,
TP=4) on the bristen cluster (4x A100-80GB nodes) before any snapshot/cold-start measurement.

## Files

| File | Purpose |
|------|---------|
| `glm-47-flash-vllm-cuda.toml` | pyxis EDF — pinned vLLM-CUDA image + cache dirs |
| `probe_vllm_cuda.sbatch` | G1 probe: upgrades transformers, launches TP=4, checks `/health`, fires a completion |

## Image pin

```
vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
```

Resolved 2026-06-26 from Docker Hub `vllm/vllm-openai:latest`.

Re-resolve with (run from bristen login node):
```bash
REPO=vllm/vllm-openai; TAG=latest
TOKEN=$(curl -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${REPO}:pull" | sed -n "s/.*\"token\":\"\([^\"]*\)\".*/\1/p")
curl -fsS -H "Authorization: Bearer ${TOKEN}" \
  -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" -D - -o /dev/null \
  "https://registry-1.docker.io/v2/${REPO}/manifests/${TAG}" | grep -i docker-content-digest
```

## transformers 5.12.1 overlay

`glm4_moe_lite` (the GLM-4.7-Flash architecture) requires transformers >= 5.12.1.
The bundled `vllm/vllm-openai:latest` may ship an older version. The probe (and any serve
script) runs `pip install transformers==5.12.1` inside the container before launching vLLM.

Risk: if upgrading transformers breaks vLLM's internal imports, the server will exit with
an import error. In that case re-resolve a newer `vllm/vllm-openai` tag that already bundles
transformers >= 5.12.1 and update the digest in the EDF.

## Model path

```
/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
```

## Cache directory

`/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda` — **distinct from the beverin ROCm cache**
(`/capstor/scratch/cscs/xyao/glm-47-flash-vllm`). CUDA Triton / vLLM / FlashInfer compile
caches are architecture-specific (sm_80 vs. gfx942) and must not collide.

## Running

```bash
# From your local machine in the serving-stack repo root:
rcc --profile glm-47-flash-bristen-vllm push

# Submit the probe:
ssh bristen 'cd /capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda && sbatch deploy/glm-47-flash-bristen-vllm/probe_vllm_cuda.sbatch'

# Poll:
ssh bristen 'squeue -u xyao'
ssh bristen 'tail -n 60 /capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/logs/glm-47-vllm-cuda-probe-<JOBID>.out'
```

G1 pass criteria: log shows `transformers post: 5.12.1`, `PROBE READY at Ns`,
a `PROBE completion:` line, and `PROBE_CORRECT=1`.
