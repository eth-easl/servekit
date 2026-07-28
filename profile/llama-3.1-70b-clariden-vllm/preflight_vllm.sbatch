#!/bin/bash
# Preflight - the cheap gate before the 4-GPU run.
#
# The unverified assumption is that the arm64 half of the vllm/vllm-openai
# manifest is usable on GH200 (sm_90a). If it is not, this dies in ~5 minutes
# instead of burning a 1-hour 4-GPU allocation. Fails loudly on a wrong arch, a
# wrong compute capability, or a compiled extension that will not import --
# it never downgrades any of those to a warning.
#
#   cd <repo root> && sbatch profile/llama-3.1-70b-clariden-vllm/preflight_vllm.sbatch
#
#SBATCH --job-name=vllm-clariden-preflight
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-clariden-vllm/llama-3.1-70b-vllm.toml"
MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct

echo "job=${SLURM_JOB_ID} node=$(hostname)"

INNER_CMD=$(cat <<EOF
set -euo pipefail

echo "--- node ---"
echo "arch=\$(uname -m)  nproc=\$(nproc)"
df -h /dev/shm
# No \`| head\`: SIGPIPE would trip the \`||\` and misreport numactl as missing.
numactl -H 2>/dev/null || echo "(numactl not present)"

echo "--- checkpoint ---"
ls -l "${MODEL}/config.json"
echo "shards=\$(ls "${MODEL}"/*.safetensors | wc -l)"

echo "--- engine ---"
# python3, not python: the vllm image has no \`python\` alias, unlike sglang's.
which python3 pip3 vllm || true
python3 - <<'PY'
import platform, sys
import torch, vllm

print("python  =", sys.version.split()[0])
print("machine =", platform.machine())
print("vllm    =", vllm.__version__)
print("torch   =", torch.__version__)
# NOT `vllm._C`: 0.25 ships the compiled ops as _C_stable_libtorch.abi3.so,
# reached via _custom_ops, which registers them under torch.ops._C.
import vllm._custom_ops  # noqa: F401
print("custom ops:", hasattr(torch.ops, "_C"))
print("gpus    =", torch.cuda.device_count())
print("capab   =", torch.cuda.get_device_capability())
print("name    =", torch.cuda.get_device_name(0))
# Importable is not enough: a real kernel must run on this arch.
x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", bool((x @ x).isfinite().all()))

ok = hasattr(torch.ops, "_C")
if not ok:
    print("FAIL: compiled ops did not register")
if platform.machine() != "aarch64":
    print("FAIL: not aarch64"); ok = False
if torch.cuda.get_device_capability() != (9, 0):
    print("FAIL: not sm_90 (Hopper)"); ok = False
if torch.cuda.device_count() != 4:
    print("FAIL: expected 4 GPUs"); ok = False
if not vllm.__version__.startswith("0.25"):
    print("FAIL: unexpected vllm version"); ok = False
print("PREFLIGHT", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
PY

echo "--- servekit ---"
python3 -m pip install -e "${SLURM_SUBMIT_DIR}/servekit" --no-build-isolation --no-deps -q
python3 -c "from servekit.profile import detect_framework; print('framework =', detect_framework(['vllm','serve','${MODEL}']).name)"
which servekit
EOF
)

srun \
  --nodes=1 \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"
