#!/bin/bash
#SBATCH --job-name=glm-47-flash-sglang-probe
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/xyao/glm-47-flash-sglang/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-47-flash-sglang}"
SCRIPT_DIR="${DEPLOY_DIR}/deploy/glm-47-flash-bristen"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs"

echo "probe started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"
echo "EDF_PATH=${EDF_PATH}"

srun --environment=glm-47-flash-sglang \
  bash -lc 'set -euo pipefail
    echo "inside container: $(hostname)"
    python3 - <<'"'"'PY'"'"'
import importlib.metadata as m
import importlib, os, shutil, sys
print("python", sys.version.split()[0], "exe=", shutil.which("python3"))
for p in ("sglang", "torch", "transformers", "numpy", "flashinfer"):
    try:
        print(f"{p}={m.version(p)}")
    except Exception:
        print(f"{p}=NOT_INSTALLED")
import torch
print("torch.cuda.is_available", torch.cuda.is_available(),
      "cuda", torch.version.cuda, "ndev", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  gpu", i, torch.cuda.get_device_name(i))

# Surface the parser names actually registered in THIS image so the serve
# defaults (SGLANG_REASONING_PARSER / SGLANG_TOOL_CALL_PARSER) can be set
# without guessing. Best-effort: never crashes the probe.
try:
    from sglang.srt.parser.reasoning_parser import ReasoningParser
    print("reasoning_parsers:", sorted(ReasoningParser.DetectorMap.keys()))
except Exception as e:
    print("reasoning_parsers lookup failed:", e)
try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
    print("tool_call_parsers:", sorted(FunctionCallParser.ToolCallParserEnum.keys()))
except Exception as e:
    print("tool_call_parsers lookup failed:", e)
PY
    echo "--- nvidia-smi ---"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head
    echo "--- model config head ---"
    head -3 /capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash/config.json 2>/dev/null
  '

echo "probe finished: $(date --iso-8601=seconds)"
