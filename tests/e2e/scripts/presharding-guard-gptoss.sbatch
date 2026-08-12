#!/bin/bash
# `servekit prepare` must refuse gpt-oss on the mxfp4 + triton_kernel path.
#
#   sbatch presharding-guard-gptoss.sbatch
#
# That combination is the one from sgl-project/sglang#34448: the postprocess
# hook parks the swizzled expert weights on the quant-method object, so the dump
# comes out ~69% short and the server answers fluently from zeros. The guard has
# to fire before the engine loads anything, so this job needs a GPU only to get
# the sglang container, not to finish a dump.

#SBATCH --job-name=servekit-guard-gptoss
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/openai/gpt-oss-20b}"
OUT="${OUT:-${SCRATCH:-/tmp}/servekit-guard-e2e-${SLURM_JOB_ID}}"
TP="${TP:-1}"

EDF="${SLURM_SUBMIT_DIR}/../../../examples/fast-weight-load/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../.."
CAPTURE="${SLURM_SUBMIT_DIR}/logs/guard-${SLURM_JOB_ID}.txt"
RESULT="${SLURM_SUBMIT_DIR}/logs/guard-${SLURM_JOB_ID}.json"

INNER_CMD=$(cat <<EOF
set -uo pipefail
ulimit -c 0
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

servekit prepare --model "${MODEL}" --out "${OUT}" --tp "${TP}" \
  -- \
  --moe-runner-backend triton_kernel \
  --mem-fraction-static 0.85 \
  > "${CAPTURE}" 2>&1
echo \$? > "${CAPTURE}.rc"
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"

cat "${CAPTURE}"

# Read back by the test: whether prepare refused, and whether it left anything
# behind that a later `servekit launch` could pick up and serve.
python3 - "$CAPTURE" "$OUT" "$RESULT" <<'PY'
import json, sys, glob, os
capture, out, result = sys.argv[1:4]
with open(capture + ".rc") as f:
    rc = int(f.read().strip())
json.dump({
    "rc": rc,
    "output": open(capture).read(),
    "shards": len(glob.glob(os.path.join(out, "*.safetensors"))),
    "manifest": os.path.isfile(os.path.join(out, "servekit.json")),
}, open(result, "w"))
PY

rm -rf "${OUT}"
echo "wrote ${RESULT}"
