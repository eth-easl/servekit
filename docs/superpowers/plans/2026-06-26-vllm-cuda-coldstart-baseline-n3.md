# N3 — vLLM-CUDA Cold-Start Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up **vLLM-CUDA serving GLM-4.7-Flash at TP=4** on bristen and establish a clean, fully-decomposed **baseline cold start** — `READY` time, per-phase breakdown, and the characterized eliminable **capture-phase** cost — that N4 (skip-capture) and N5 (snapshot/restore) are measured against. N3 only *measures* the capture cost.

**Architecture:** Mirror the proven beverin (ROCm) vLLM cold-start measurement on CUDA. The measurement machinery is vendor-neutral (`vllm serve` + `/health` poll + vLLM-log grep + a `torch.cuda.CUDAGraph` monkeypatch + a vLLM-API skip-capture probe), so it ports near-verbatim — N3 swaps only the container/image/env/paths and adds the `glm4_moe_lite` transformers overlay. A **probe-first** task confirms the image actually serves the model before the measurement stack is built. Baseline runs **clean** (no snapshot interposer).

**Tech Stack:** vLLM-CUDA (`vllm/vllm-openai` image, sm_80 / CUDA 12.x), PyTorch CUDA, `transformers==5.12.1` (the `glm4_moe_lite` fix), enroot/pyxis on CSCS **bristen** (4× A100-80GB/node, SLURM `-A a-infra02`, partition `normal`), `rcc --profile glm-47-flash-bristen-vllm push` for code sync.

## Global Constraints

- **Model:** `GLM-4.7-Flash` at `/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash` (shared `/capstor`, ~56 GiB / 48 shards — the AMD-measured figure; the SGLang README's "~4.7B" is a misnomer).
- **`glm4_moe_lite` needs transformers ≥ 5.12.1.** Stock vLLM images may bundle an older transformers; `--trust-remote-code` does NOT help (the checkpoint ships no custom modeling code). The serve/probe does `pip install transformers==5.12.1` at container startup (`PIP_BREAK_SYSTEM_PACKAGES=1`, bristen compute nodes have PyPI access, persistent pip cache on `/capstor`). If the bundled transformers already serves it, the overlay is a fast no-op.
- **Baseline config (headline defaults, env-overridable):** TP=4, `--gpu-memory-utilization 0.90`, `--max-num-seqs 256`, `--max-model-len 131072`, default cudagraph capture sizes. Single node / single replica.
- **Image pinned by digest** (the `:latest` JIT-cache-invalidation lesson). The exact digest is resolved empirically in Task 1 and recorded in the EDF.
- **Distinct CUDA deploy/cache dir:** `/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda` — NOT beverin's `glm-47-flash-vllm` (Triton/vLLM/FlashInfer caches are arch-specific; sharing the beverin ROCm dir would collide and break the warm-cache A/B).
- **Warm-cache A/B:** the snapshot win targets warm-cache *restart* cold start, so a warm-up run populates `TRITON_CACHE_DIR`/`VLLM_CACHE_ROOT` on `/capstor` before the measured run. `PYTHONHASHSEED=0` pins inductor fusion order. First-run (cold-cache) compile is reported separately, never as the baseline.
- **Baseline runs clean** — no `LD_PRELOAD`, no `redirect_cuda`/`record_cuda`. The numbers must be uncontaminated by any interposer.
- **Run on hardware, not the login node.** Every run is a bristen `srun`/`sbatch` job. Account `-A a-infra02`, partition `normal`.
- **Do NOT modify** the SGLang `deploy/glm-47-flash-bristen/`, the beverin deploy, or any `snapshot/csrc/*`. N3 is additive: new files under `deploy/glm-47-flash-bristen-vllm/` and `snapshot/recipe/`, the rcc profile, and a RESULTS section.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `deploy/glm-47-flash-bristen-vllm/glm-47-flash-vllm-cuda.toml` | enroot EDF: vLLM-CUDA image (pinned digest), mounts, CUDA cache env | Create |
| `deploy/glm-47-flash-bristen-vllm/probe_vllm_cuda.sbatch` | G1 probe: transformers overlay + single-node `vllm serve` TP=4 → `/health` + correct completion | Create |
| `deploy/glm-47-flash-bristen-vllm/README.md` | image pin, transformers overlay, cache layout, run commands | Create |
| `snapshot/recipe/_vllm_coldstart_cuda.sh` | cold-start driver: serve (graph\|eager) → `/health` READY → correctness probe; honors `VLLM_CG_INSTRUMENT`/`VLLM_CG_SKIP_CAPTURE`/`CAPTURE_SIZES`; pins caches | Create (port of `_vllm_coldstart.sh`) |
| `snapshot/recipe/_vllm_measure_cuda.sh` | per-phase log-grep breakdown (one run; `EXTRA_ARGS=--enforce-eager` for the eager variant) | Create (port of `_vllm_measure.sh`) |
| `snapshot/recipe/vllm_coldstart_cuda.sbatch` | sbatch wrapper: transformers overlay, warm-up run, then graph + eager + instrumented runs | Create |
| `snapshot/recipe/cginst/sitecustomize.py`, `snapshot/recipe/cginst_skip/{cg_skip.py,sitecustomize.py}` | per-graph capture timing + skip-capture `measure` mode (vendor-neutral; hook `torch.cuda.CUDAGraph` + vLLM API) | Provide verbatim (already in the repo working tree; see Task 3) |
| `.rcc/config.toml` | add `[profiles.glm-47-flash-bristen-vllm]` | Modify |
| `snapshot/RESULTS.md` | append an N3 section | Modify |

---

## Task 1: vLLM-CUDA deploy EDF + probe (G1 — does it serve?)

The single biggest risk is whether vLLM-CUDA serves `glm4_moe_lite` at all. This task resolves+pins the image, adds the EDF + rcc profile, and proves serving with a probe **before** any measurement work.

**Files:**
- Create: `deploy/glm-47-flash-bristen-vllm/glm-47-flash-vllm-cuda.toml`
- Create: `deploy/glm-47-flash-bristen-vllm/probe_vllm_cuda.sbatch`
- Create: `deploy/glm-47-flash-bristen-vllm/README.md`
- Modify: `.rcc/config.toml`

**Interfaces:**
- Produces: a pinned EDF `glm-47-flash-vllm-cuda`; a probe job that prints `PROBE READY at Ns` + `PROBE completion: <text>` confirming a correct answer; the `glm-47-flash-bristen-vllm` rcc profile (remote `/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda`).

- [ ] **Step 1: Resolve the vLLM-CUDA image digest**

From the bristen login node (has internet), resolve a concrete digest for a candidate tag (mirrors the SGLang README's re-resolve snippet). Start with `latest`; if the probe (Step 4) shows no `glm4_moe_lite` support, re-resolve a newer/specific tag and repeat.
```bash
ssh bristen 'REPO=vllm/vllm-openai; TAG=latest
  TOKEN=$(curl -fsSL "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${REPO}:pull" | sed -n "s/.*\"token\":\"\([^\"]*\)\".*/\1/p")
  curl -fsS -H "Authorization: Bearer ${TOKEN}" \
    -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" -D - -o /dev/null \
    "https://registry-1.docker.io/v2/${REPO}/manifests/${TAG}" | grep -i docker-content-digest'
```
Record the printed `sha256:...` for the EDF `image =` line.

- [ ] **Step 2: Add the rcc profile**

In `.rcc/config.toml`, append:
```toml
[profiles.glm-47-flash-bristen-vllm]
host = "bristen"
remote_dir = "/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda"
```

- [ ] **Step 3: Write the EDF**

Create `deploy/glm-47-flash-bristen-vllm/glm-47-flash-vllm-cuda.toml` (CUDA adaptation of the beverin vLLM EDF: drop all ROCm/AITER/HSA env, keep the cache-dir + offline + no-user-site shape; `<DIGEST>` from Step 1):
```toml
# vLLM-CUDA EDF for the GLM-4.7-Flash cold-start baseline on bristen (A100/sm_80).
# Pinned by digest so the JIT compile caches (TRITON_CACHE_DIR / VLLM_CACHE_ROOT /
# VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR on shared /capstor) stay warm across launches.
# A floating :latest silently moves and invalidates the whole cache. Resolve the
# digest with the snippet in README.md. transformers is upgraded to 5.12.1 at
# container startup (the glm4_moe_lite fix) by the probe/serve scripts.
image = "vllm/vllm-openai@<DIGEST>"

mounts = [
  "/capstor:/capstor",
  "/users/xyao:/users/xyao",
  "/iopsstor:/iopsstor",
]

workdir = "/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda"

[env]
HOME = "/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/home"
HF_HOME = "/capstor/store/cscs/swissai/infra02/xyao/cache/hf"
XDG_CACHE_HOME = "/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/cache/xdg"
XDG_CONFIG_HOME = "/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/cache/config"
VLLM_CACHE_ROOT = "/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/cache/vllm"
VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR = "/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/cache/flashinfer"
TRITON_CACHE_DIR = "/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/cache/triton"
PIP_CACHE_DIR = "/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/cache/pip"
VLLM_NO_USAGE_STATS = "1"
NCCL_DEBUG = "WARN"
PIP_BREAK_SYSTEM_PACKAGES = "1"
PYTHONHASHSEED = "0"
PYTHONNOUSERSITE = "1"
HF_HUB_OFFLINE = "1"
TRANSFORMERS_OFFLINE = "1"
TOKENIZERS_PARALLELISM = "false"

[annotations]
com.pyxis.entrypoint_log = "true"
```

- [ ] **Step 4: Write the probe sbatch**

Create `deploy/glm-47-flash-bristen-vllm/probe_vllm_cuda.sbatch`:
```bash
#!/bin/bash
#SBATCH --job-name=glm-47-vllm-cuda-probe
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:40:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/logs/%x-%j.out

set -euo pipefail
DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda}"
export DEPLOY_DIR
export EDF_PATH="${DEPLOY_DIR}/deploy/glm-47-flash-bristen-vllm:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/home" "${DEPLOY_DIR}/cache/triton" \
  "${DEPLOY_DIR}/cache/vllm" "${DEPLOY_DIR}/cache/flashinfer" "${DEPLOY_DIR}/cache/pip"

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"
PORT="${PORT:-8000}"

srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -euo pipefail
  echo "=== transformers overlay (glm4_moe_lite) ==="
  python -c "import transformers; print(\"pre:\", transformers.__version__)"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'" || { echo "TRANSFORMERS_OVERLAY_FAILED"; exit 1; }
  python -c "import transformers; print(\"post:\", transformers.__version__)"
  echo "=== launch vllm serve TP=4 ==="
  t0=$(date +%s)
  vllm serve "'"${MODEL}"'" --host 127.0.0.1 --port '"${PORT}"' --served-model-name probe \
    --tensor-parallel-size 4 --trust-remote-code \
    --gpu-memory-utilization 0.90 --max-model-len 131072 --max-num-seqs 256 \
    > "${DEPLOY_DIR}/logs/probe-vllm.log" 2>&1 &
  PID=$!
  while :; do
    el=$(( $(date +%s) - t0 ))
    if kill -0 $PID 2>/dev/null && curl -fsS "http://127.0.0.1:'"${PORT}"'/health" >/dev/null 2>&1; then
      echo "PROBE READY at ${el}s"; break; fi
    if ! kill -0 $PID 2>/dev/null; then echo "PROBE SERVER_EXITED at ${el}s"; tail -40 "${DEPLOY_DIR}/logs/probe-vllm.log"; exit 1; fi
    if [ $el -gt 1200 ]; then echo "PROBE DEADLINE"; exit 1; fi
    sleep 5
  done
  RESP=$(curl -sS "http://127.0.0.1:'"${PORT}"'/v1/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"probe\",\"prompt\":\"The capital of France is\",\"max_tokens\":3,\"temperature\":0}")
  echo "PROBE completion: ${RESP}"
  echo "${RESP}" | grep -qi "paris" && echo "PROBE_CORRECT=1" || echo "PROBE_CORRECT=0"
  kill -9 $PID 2>/dev/null; pkill -9 -f "vllm serve" 2>/dev/null || true
'
```

- [ ] **Step 5: Write the README** documenting: the image pin + the resolve snippet, the transformers 5.12.1 overlay rationale, the model path, the distinct CUDA cache dir (and why not the beverin one), and the run commands (`rcc --profile glm-47-flash-bristen-vllm push` then `sbatch …probe_vllm_cuda.sbatch`).

- [ ] **Step 6: Run the probe (G1)**

```bash
rcc --profile glm-47-flash-bristen-vllm push
ssh bristen 'cd /capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda && sbatch deploy/glm-47-flash-bristen-vllm/probe_vllm_cuda.sbatch'
ssh bristen 'tail -n 60 /capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/logs/glm-47-vllm-cuda-probe-*.out'
```
Expected: `transformers post: 5.12.1`, `PROBE READY at Ns`, a `PROBE completion:` line, and `PROBE_CORRECT=1` (the completion contains "Paris"). **If the server exits with an architecture/`glm4_moe_lite` error**, the bundled vLLM is too old: re-resolve a newer `vllm/vllm-openai` tag in Step 1, update the EDF digest, and re-run. This is the milestone's gating risk — resolve it here before proceeding.

- [ ] **Step 7: Commit**
```bash
git add deploy/glm-47-flash-bristen-vllm/ .rcc/config.toml
git commit -m "snapshot(cuda): N3 Task 1 — vLLM-CUDA GLM-4.7-Flash deploy + probe (serves TP=4 on A100)"
```

---

## Task 2: Cold-start driver + READY + per-phase breakdown + warm-cache A/B (G2, G3)

Port the vendor-neutral cold-start driver and phase-decomposition script, and run the warm-cache baseline: a clean `READY` (graph mode) reproducible across 2 runs, plus the log-grep phase breakdown and the `--enforce-eager` variant.

**Files:**
- Create: `snapshot/recipe/_vllm_coldstart_cuda.sh`
- Create: `snapshot/recipe/_vllm_measure_cuda.sh`
- Create: `snapshot/recipe/vllm_coldstart_cuda.sbatch`

**Interfaces:**
- Consumes: the `glm-47-flash-vllm-cuda` EDF + rcc profile (Task 1).
- Produces: `READY at Ns` (graph) + `COLD_START_SECONDS` per variant; a `=== sub-phase breakdown ===` section (Model loading / init engine / Maximum concurrency / Application startup) + the `Capturing CUDA graphs … [MM:SS` bars.

- [ ] **Step 1: Write `_vllm_coldstart_cuda.sh`**

Create `snapshot/recipe/_vllm_coldstart_cuda.sh` (port of `_vllm_coldstart.sh`: CUDA deploy dir, TP=4/gmu=0.90 defaults, the "Paris" correctness probe, CUDA cache dirs; the instrumentation hooks are unchanged — they are vendor-neutral):
```bash
#!/bin/bash
# Measure vLLM-CUDA cold-start time to /health, with or without CUDA graph
# capture. Runs WITHOUT any LD_PRELOAD so the timing reflects real vLLM startup.
set -uo pipefail
DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda}"
cd "$DEPLOY_DIR"
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
TP="${TP:-4}"
GMU="${GMU:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
PORT="${PORT:-8821}"
MODE="${MODE:-graph}"   # graph | eager

LOG="${DEPLOY_DIR}/logs/vllm_coldstart_${MODE}_${SLURM_JOB_ID:-local}.log"
rm -f "$LOG"
DEADLINE="${DEADLINE:-1200}"

CAPTURE_ARGS=()
if [ -n "${CAPTURE_SIZES:-}" ]; then
  # shellcheck disable=SC2206
  CAPTURE_ARGS=(--cudagraph-capture-sizes $CAPTURE_SIZES)
fi
ARGS=(--host 127.0.0.1 --port "$PORT" --served-model-name cs \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --trust-remote-code \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" "${CAPTURE_ARGS[@]}")
if [ "$MODE" = "eager" ]; then ARGS+=(--enforce-eager); fi

echo "[coldstart] MODE=$MODE TP=$TP gmu=$GMU start=$(date +%T)"

# Vendor-neutral instrumentation (hooks torch.cuda.CUDAGraph): per-graph timing.
if [ -n "${VLLM_CG_INSTRUMENT:-}" ]; then
  export PYTHONPATH="${DEPLOY_DIR}/snapshot/recipe/cginst:${PYTHONPATH:-}"
  rm -f "$VLLM_CG_INSTRUMENT"
  echo "[coldstart] CG instrumentation active -> $VLLM_CG_INSTRUMENT"
fi
# Skip-capture probe (measure mode quantifies the capture-phase forward cost).
if [ -n "${VLLM_CG_SKIP_CAPTURE:-}" ]; then
  export PYTHONPATH="${DEPLOY_DIR}/snapshot/recipe/cginst_skip:${PYTHONPATH:-}"
  echo "[coldstart] CG skip-capture active mode=$VLLM_CG_SKIP_CAPTURE"
fi
# Pin the CUDA caches for a fair warm-cache A/B.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DEPLOY_DIR}/cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DEPLOY_DIR}/cache/vllm}"
export PYTHONHASHSEED=0

t0=$(date +%s)
vllm serve "$MODEL" "${ARGS[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  if kill -0 "$SERVER_PID" 2>/dev/null && \
     curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "[coldstart] MODE=$MODE READY at ${elapsed}s"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[coldstart] MODE=$MODE SERVER_EXITED at ${elapsed}s"; tail -40 "$LOG"; break; fi
  if [ "$elapsed" -gt "$DEADLINE" ]; then echo "[coldstart] MODE=$MODE DEADLINE at ${elapsed}s"; break; fi
  sleep 2
done

RESP=$(curl -sS "http://127.0.0.1:${PORT}/v1/completions" -H 'Content-Type: application/json' \
  -d '{"model":"cs","prompt":"The capital of France is","max_tokens":3,"temperature":0}' 2>/dev/null || true)
echo "[coldstart] MODE=$MODE completion: ${RESP}"
echo "${RESP}" | grep -qi "paris" && echo "[coldstart] MODE=$MODE inference=ok" || echo "[coldstart] MODE=$MODE inference=failed"

sleep 2
kill -9 "$SERVER_PID" 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f 'VLLM_RPC' 2>/dev/null
sleep 3
echo "[coldstart] MODE=$MODE done=$(date +%T)"
```

- [ ] **Step 2: Write `_vllm_measure_cuda.sh`**

Create `snapshot/recipe/_vllm_measure_cuda.sh` (port of `_vllm_measure.sh`: CUDA dir, TP=4/gmu=0.90 defaults; the grep patterns are unchanged — the same log lines appear on CUDA):
```bash
#!/bin/bash
# Per-phase decomposition of a warm-cache vLLM-CUDA start (no interposer).
# RUN_TAG + EXTRA_ARGS select the variant (EXTRA_ARGS="--enforce-eager" disables capture).
set -uo pipefail
DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda}"
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash
DIR="${DEPLOY_DIR}/measure-init/${RUN_TAG:-graphs}"
rm -rf "$DIR"; mkdir -p "$DIR"
export VLLM_LOGGING_LEVEL=INFO
export PYTHONHASHSEED=0
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${DEPLOY_DIR}/cache/triton}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${DEPLOY_DIR}/cache/vllm}"
TP="${TP:-4}"; GMU="${GMU:-0.90}"; MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
PORT="${PORT:-8801}"; DEADLINE="${RUN_SECS:-1200}"; EXTRA_ARGS="${EXTRA_ARGS:-}"
FULL="$DIR/vllm.log"
echo "[measure] tag=${RUN_TAG:-graphs} extra='${EXTRA_ARGS}' tp=$TP gmu=$GMU start=$(date +%T)"
t0=$(date +%s)
# shellcheck disable=SC2086
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name m \
  --tensor-parallel-size "$TP" --pipeline-parallel-size 1 --trust-remote-code \
  --gpu-memory-utilization "$GMU" --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" $EXTRA_ARGS > "$FULL" 2>&1 &
SERVER_PID=$!
ready=""
while :; do
  elapsed=$(( $(date +%s) - t0 ))
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "[measure] server exited before ready (elapsed=${elapsed}s)"; break; fi
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then ready="$elapsed"; echo "[measure] READY: COLD_START_SECONDS=${elapsed}"; break; fi
  if [ "$elapsed" -gt "$DEADLINE" ]; then echo "[measure] DEADLINE ${DEADLINE}s without ready"; break; fi
  sleep 2
done
kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f 'from multiprocessing' 2>/dev/null; sleep 3
echo "=== sub-phase breakdown (${RUN_TAG:-graphs}) ==="
grep -aE "Model loading took|init engine .* took|Maximum concurrency|Application startup complete" "$FULL" | tail -6
echo "--- capture wall-clock (sum of PIECEWISE+FULL bars) ---"
grep -aoE "Capturing CUDA graphs \([^)]*\): 100%[^[]*\[[0-9]{2}:[0-9]{2}" "$FULL" | tail -6
echo "[measure] RESULT tag=${RUN_TAG:-graphs} ready=${ready:-NONE}"
```

- [ ] **Step 3: Write the wrapper sbatch**

Create `snapshot/recipe/vllm_coldstart_cuda.sbatch` (transformers overlay once, warm-up run to populate caches, then the measured graph + eager runs):
```bash
#!/bin/bash
#SBATCH --job-name=vllm-coldstart-cuda
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/logs/%x-%j.out

set -euo pipefail
DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda}"
export DEPLOY_DIR
export EDF_PATH="${DEPLOY_DIR}/deploy/glm-47-flash-bristen-vllm:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.12.1}"

srun --environment=glm-47-flash-vllm-cuda bash -lc '
  set -euo pipefail
  cd "${DEPLOY_DIR}"
  pip install --quiet "transformers=='"${TRANSFORMERS_VERSION}"'"
  python -c "import transformers; print(\"transformers:\", transformers.__version__)"
  echo "=== warm-up run (populate Triton/vLLM caches; not measured) ==="
  MODE=graph DEADLINE=1800 bash snapshot/recipe/_vllm_coldstart_cuda.sh || true
  echo "=== measured: graph run #1 (warm cache) ==="
  RUN_TAG=graphs1 bash snapshot/recipe/_vllm_measure_cuda.sh
  echo "=== measured: graph run #2 (reproducibility) ==="
  RUN_TAG=graphs2 bash snapshot/recipe/_vllm_measure_cuda.sh
  echo "=== measured: enforce-eager (no capture) ==="
  RUN_TAG=eager EXTRA_ARGS=--enforce-eager bash snapshot/recipe/_vllm_measure_cuda.sh
'
```

- [ ] **Step 4: Run (G2, G3)**

```bash
rcc --profile glm-47-flash-bristen-vllm push
ssh bristen 'cd /capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda && sbatch snapshot/recipe/vllm_coldstart_cuda.sbatch'
ssh bristen 'tail -n 80 /capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/logs/vllm-coldstart-cuda-*.out'
```
Expected: two graph runs with close `COLD_START_SECONDS` (reproducible, G2); each with a `=== sub-phase breakdown ===` (Model loading / init engine / Maximum concurrency / Application startup) and `Capturing CUDA graphs … [MM:SS` bars; the eager run's `COLD_START_SECONDS` notably lower (no capture). The phases should roughly sum to `READY`, with **weight-load** and **capture** both quantified (G3). Capture cost ≈ graph `READY` − eager `READY`.

- [ ] **Step 5: Commit**
```bash
git add snapshot/recipe/_vllm_coldstart_cuda.sh snapshot/recipe/_vllm_measure_cuda.sh snapshot/recipe/vllm_coldstart_cuda.sbatch
git commit -m "snapshot(cuda): N3 Task 2 — cold-start driver + phase breakdown + warm-cache A/B"
```

---

## Task 3: Capture-phase instrumentation cross-check (G4)

Quantify the eliminable capture cost three independent ways and confirm they agree: per-graph timing (`VLLM_CG_INSTRUMENT`), total skip-capture cost (`VLLM_CG_SKIP_CAPTURE=measure`), and the `--enforce-eager` delta (Task 2).

**Files:**
- Provide (verbatim, vendor-neutral — already in the repo working tree): `snapshot/recipe/cginst/sitecustomize.py`, `snapshot/recipe/cginst_skip/cg_skip.py`, `snapshot/recipe/cginst_skip/sitecustomize.py`. These hook `torch.cuda.CUDAGraph.{capture_begin,capture_end,replay}` and `vllm.compilation.cuda_graph.CUDAGraphWrapper` / `GPUModelRunner` — the same classes on CUDA — so they port unchanged. The execution workspace must contain them (they are not CUDA-specific; do not rewrite them).
- Modify: `snapshot/recipe/vllm_coldstart_cuda.sbatch` (add the instrumented runs)

**Interfaces:**
- Consumes: the driver from Task 2 (already honors `VLLM_CG_INSTRUMENT` and `VLLM_CG_SKIP_CAPTURE` via `PYTHONPATH`).
- Produces: a per-graph CSV (`capture_wall_ms` summed) and a `[cg_skip] CAPTURE_PHASE_S=...` line; both ≈ the Task 2 graph−eager delta.

- [ ] **Step 1: Confirm the instrumentation is present and activates on CUDA**

Verify the three files exist in the workspace and that `cginst_skip/sitecustomize.py` dispatches `cg_skip` on `VLLM_CG_SKIP_CAPTURE` and `cginst/sitecustomize.py` gates on `VLLM_CG_INSTRUMENT`. (No edits — they are vendor-neutral. If absent from the workspace, they live at `snapshot/recipe/cginst/` and `snapshot/recipe/cginst_skip/` in the repo; ensure the push includes them.)

- [ ] **Step 2: Add the instrumented runs to the wrapper sbatch**

In `snapshot/recipe/vllm_coldstart_cuda.sbatch`, inside the `srun` heredoc after the eager run, add:
```bash
  echo "=== instrumented: per-graph capture timing (VLLM_CG_INSTRUMENT) ==="
  VLLM_CG_INSTRUMENT="${DEPLOY_DIR}/measure-init/cg_per_graph.csv" \
    MODE=graph bash snapshot/recipe/_vllm_coldstart_cuda.sh
  echo "--- per-graph capture_wall_ms sum ---"
  awk -F, "NR>1 && \$6!=\"\" {s+=\$6} END{printf \"PER_GRAPH_CAPTURE_MS_SUM=%.0f\n\", s}" \
    "${DEPLOY_DIR}/measure-init/cg_per_graph.csv" || true
  echo "=== instrumented: total capture-phase cost (VLLM_CG_SKIP_CAPTURE=measure) ==="
  VLLM_CG_SKIP_CAPTURE=measure MODE=graph bash snapshot/recipe/_vllm_coldstart_cuda.sh \
    2>&1 | grep -E "CAPTURE_PHASE_S|READY" || true
```

- [ ] **Step 3: Run (G4)**

```bash
rcc --profile glm-47-flash-bristen-vllm push
ssh bristen 'cd /capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda && sbatch snapshot/recipe/vllm_coldstart_cuda.sbatch'
ssh bristen 'tail -n 100 /capstor/scratch/cscs/xyao/glm-47-flash-vllm-cuda/logs/vllm-coldstart-cuda-*.out'
```
Expected: `PER_GRAPH_CAPTURE_MS_SUM=...`, `[cg_skip] CAPTURE_PHASE_S=...`, and the Task-2 (graph − eager) delta should **cross-agree** within a reasonable margin (the three are different views of the same capture-phase forward cost). This confirms the eliminable fraction is characterized and the vendor-neutral instrumentation activates correctly on CUDA. If `cg_skip measure` throws (vLLM API drift on this image version), capture the traceback and report it — the per-graph + eager-delta still bound the cost; note the discrepancy honestly.

- [ ] **Step 4: Commit**
```bash
git add snapshot/recipe/vllm_coldstart_cuda.sbatch snapshot/recipe/cginst snapshot/recipe/cginst_skip
git commit -m "snapshot(cuda): N3 Task 3 — capture-phase instrumentation cross-check on CUDA"
```

---

## Task 4: RESULTS N3 section + regression check (G5)

Document the baseline honestly and confirm N3 is additive.

**Files:**
- Modify: `snapshot/RESULTS.md`

- [ ] **Step 1: Regression check** — confirm N3 touched no source/SGLang-deploy/beverin paths:
```bash
git diff --stat "$(git merge-base main HEAD)" -- \
  snapshot/csrc deploy/glm-47-flash-bristen deploy/glm-47-flash-beverin
```
Expected: **no output**. (N3 adds only `deploy/glm-47-flash-bristen-vllm/`, `snapshot/recipe/` CUDA recipes + the vendor-neutral instrumentation, the rcc profile, and the RESULTS section.)

- [ ] **Step 2: Append the N3 RESULTS section** documenting: the bristen environment (resolved image digest, CUDA version, sm_80, transformers 5.12.1), the config (TP=4, gmu 0.90, max-num-seqs 256, max-model-len 131072), the warm-cache `READY` (graph) reproducible across the 2 runs, the per-phase decomposition (with **weight-load I/O called out as non-eliminable** and **capture as the eliminable target**), the three cross-agreeing capture-cost measures (`PER_GRAPH_CAPTURE_MS_SUM` ≈ `CAPTURE_PHASE_S` ≈ graph−eager delta), the first-run (cold-cache) compile reported separately, and a note that N4 (skip-capture) is next. Match the AMD RESULTS style and decompose the eliminable fraction truthfully.

- [ ] **Step 3: Commit**
```bash
git add snapshot/RESULTS.md
git commit -m "snapshot(cuda): N3 complete — vLLM-CUDA baseline cold start characterized on A100"
```

---

## Self-Review

**Spec coverage (against the N3 design):**
- Deploy (EDF + transformers overlay + serve config, distinct CUDA cache dir) → Task 1 ✓
- Probe-first de-risking (G1: serves GLM TP=4 + correct completion) → Task 1 ✓
- READY timer + log-grep phase breakdown + warm-cache A/B (G2, G3) → Task 2 ✓
- Capture instrumentation: per-graph + measure-mode + enforce-eager cross-check (G4) → Task 3 ✓
- RESULTS N3 honest eliminable-fraction accounting (G5) + additive regression check → Task 4 ✓
- Production config (TP=4, gmu 0.90, max-num-seqs 256, max-model-len 131072) → Global Constraints + Tasks 1–2 ✓
- Baseline runs clean (no interposer) → Global Constraints + drivers (no LD_PRELOAD) ✓

**Placeholder scan:** the only intentional placeholder is `<DIGEST>` (the vLLM-CUDA image digest), resolved empirically in Task 1 Step 1 and pinned — flagged at first use, exactly like N1's image resolution. No TBD/TODO. The verbatim instrumentation (`cginst`/`cginst_skip`) is intentionally referenced-not-reproduced because it ports unchanged (vendor-neutral); Task 3 Step 1 makes its presence an explicit check.

**Type/interface consistency:** the driver's env contract (`MODE`, `TP`, `GMU`, `MAX_NUM_SEQS`, `MAX_MODEL_LEN`, `CAPTURE_SIZES`, `VLLM_CG_INSTRUMENT`, `VLLM_CG_SKIP_CAPTURE`, `DEPLOY_DIR`) is consistent across `_vllm_coldstart_cuda.sh`, `_vllm_measure_cuda.sh`, and the wrapper sbatch. `DEPLOY_DIR` and the EDF name (`glm-47-flash-vllm-cuda`) and rcc profile (`glm-47-flash-bristen-vllm`) match across all tasks. The grep patterns and CSV column index (`$6` = `capture_wall_ms`) match the provided `cginst/sitecustomize.py`.

**Scope:** N3 only — deploy + baseline measurement. N4 (skip-capture *win*), N5 (snapshot/restore + serving overhead), and the `redirect_cuda`/`record_cuda` interposers are separate milestones (the design's scope-OUT list).

---

## Execution Handoff

Two notes specific to N3 before execution:
1. **Instrumentation availability:** Task 3 reuses the vendor-neutral `snapshot/recipe/cginst/` and `cginst_skip/` (currently *untracked* in `main`). The execution workspace must contain them — either execute in the main checkout (where they exist) or copy them into the worktree before Task 3. Task 4's commit adds them to the branch.
2. Every gate is a bristen cluster job (`rcc --profile glm-47-flash-bristen-vllm push` → `sbatch -A a-infra02`), and the cold-start runs are multi-minute (model load + capture), so each task's verification is a longer `sbatch` round-trip than N1/N2.
