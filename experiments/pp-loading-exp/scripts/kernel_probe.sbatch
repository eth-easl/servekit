#!/bin/bash
# Disambiguate the preflight failure. Job 2922510 died on
#   ImportError: cannot import name 'flash_ops' from 'sgl_kernel'
# in lmsysorg/sglang:nightly-dev-20260729-16a52bff (torch 2.11.0+cu130).
#
# PP=4 itself was fine up to that point -- all four stages initialised and
# loaded weights -- so the question is narrow: is the aarch64 sgl_kernel wheel
# simply missing FA3, and is that true of the cu12 build of the SAME commit?
#
# Imports only. No model, no GPU work beyond a device query, so it costs
# minutes and settles whether this round needs a different image variant, a
# different attention backend, or is blocked.
#
#   IMAGE_EDF=scripts/sglang-nightly-cu12.toml sbatch scripts/kernel_probe.sbatch
#
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/${IMAGE_EDF:-scripts/sglang-nightly-cu12.toml}"
echo "job=${SLURM_JOB_ID} node=$(hostname) edf=${EDF}"
grep '^image' "${EDF}"

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 \
  --environment="${EDF}" \
  python - <<'PY'
import importlib
import torch
import sgl_kernel

print("torch      =", torch.__version__)
print("device     =", torch.cuda.get_device_name(0))
print("sgl_kernel =", sgl_kernel.__file__)
print("has flash_ops:", hasattr(sgl_kernel, "flash_ops"))

# Which attention backends can actually be constructed in this build. The
# engine picked fa3 by default; if another one imports, the round survives with
# an explicit --attention-backend rather than a different image.
for name, mod in [
    ("fa3",        "sglang.srt.layers.attention.flashattention_backend"),
    ("flashinfer", "sglang.srt.layers.attention.flashinfer_backend"),
    ("triton",     "sglang.srt.layers.attention.triton_backend"),
    ("torch_native", "sglang.srt.layers.attention.torch_native_backend"),
]:
    try:
        importlib.import_module(mod)
        print(f"  {name:13s} import OK")
    except Exception as e:
        print(f"  {name:13s} FAIL: {type(e).__name__}: {e}")
PY
echo "finished=$(date --iso-8601=seconds)"
