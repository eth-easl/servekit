#!/bin/bash
#SBATCH --job-name=glm-47-flash-sglang-router-probe
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-47-flash-sglang/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-sglang}"
SCRIPT_DIR="${DEPLOY_DIR}/deploy/glm-47-flash-bristen"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/cache/pip"

echo "router probe started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"
echo "EDF_PATH=${EDF_PATH}"

# Does the SGLang router/gateway ship in the pinned image, and if not can we
# pip-install it on a compute node? Dump the ACTUAL launch_router flag surface
# for THIS version so we wire cache_aware routing against real flags, not the
# latest-docs flag list (image is v0.5.9; docs describe the newer gateway).
srun --environment=glm-47-flash-sglang \
  bash -lc 'set -uo pipefail
    export PIP_CACHE_DIR="'"${DEPLOY_DIR}"'/cache/pip"

    echo "=== inside container: $(hostname) ==="
    python3 -c "import sglang, importlib.metadata as m; print(\"sglang=\", m.version(\"sglang\"))"

    echo "=== base image: is sglang_router importable? ==="
    if python3 -c "import sglang_router" 2>/dev/null; then
      echo "ROUTER_IN_BASE_IMAGE=yes"
    else
      echo "ROUTER_IN_BASE_IMAGE=no"
    fi

    echo "=== pip packages matching router/gateway/sgl ==="
    python3 -m pip list 2>/dev/null | grep -iE "router|gateway|^sgl" || echo "(none)"

    # If absent from the base image, try installing from PyPI (compute nodes
    # have network access; this is the same mechanism the serve script uses for
    # the transformers upgrade). Best-effort; never fail the probe.
    if ! python3 -c "import sglang_router" 2>/dev/null; then
      echo "=== attempting: pip install sglang-router ==="
      python3 -m pip install --no-input --break-system-packages sglang-router >&2 || \
        echo "PIP_INSTALL_sglang-router=FAILED"
    fi

    echo "=== after install attempt: importable? + version ==="
    if python3 -c "import sglang_router" 2>/dev/null; then
      echo "ROUTER_IMPORTABLE=yes"
      python3 -c "import importlib.metadata as m
for n in (\"sglang-router\",\"sglang_router\"):
    try:
        print(\"sglang_router_version=\", m.version(n)); break
    except Exception:
        pass" || true
    else
      echo "ROUTER_IMPORTABLE=no"
    fi

    echo "=== python -m sglang_router.launch_router --help ==="
    python3 -m sglang_router.launch_router --help 2>&1 | head -200 || echo "launch_router --help FAILED"

    echo "=== python -m sglang_router.launch_server --help (co-launch DP mode) ==="
    python3 -m sglang_router.launch_server --help 2>&1 | head -120 || echo "launch_server --help FAILED"

    echo "=== console scripts (sgl-router / sglang-router) ? ==="
    for c in sgl-router sglang-router sgl-model-gateway; do
      echo "which $c: $(command -v "$c" || echo MISSING)"
    done
  '

echo "router probe finished: $(date --iso-8601=seconds)"
