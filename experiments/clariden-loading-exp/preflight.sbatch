#!/bin/bash
# Preflight - the cheap gate. Run this BEFORE either run.
#
# The one unverified assumption in this experiment is that the arm64 half of the
# lmsysorg/sglang:v0.5.10 manifest is actually usable on GH200 (sm_90a). If it
# is not, this dies in ~5 minutes instead of burning a 1-hour 4-GPU allocation.
# It also records the node facts the results write-up needs (nproc, /dev/shm,
# NUMA), which differ from bristen and are part of the answer.
#
#   cd experiments/clariden-loading-exp && sbatch preflight.sbatch
#
#SBATCH --job-name=clariden-preflight
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

SHARDED_SRC="${SHARDED_SRC:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded}"
MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
EDF="${SLURM_SUBMIT_DIR}/sglang-clariden.toml"

echo "job=${SLURM_JOB_ID} node=$(hostname)"

INNER_CMD=$(cat <<EOF
set -euo pipefail

echo "--- node ---"
echo "arch=\$(uname -m)  nproc=\$(nproc)"
df -h /dev/shm
# No `| head`: SIGPIPE would trip the `||` and misreport numactl as missing.
numactl -H 2>/dev/null || echo "(numactl not present)"

echo "--- checkpoints ---"
ls -l "${MODEL}/config.json"
ls -l "${SHARDED_SRC}/model-rank-0-part-0.safetensors"
echo "sharded shards=\$(ls "${SHARDED_SRC}"/*.safetensors | wc -l)"

echo "--- engine ---"
python - <<'PY'
import sys, torch, sglang
print("python  =", sys.version.split()[0])
print("machine =", __import__("platform").machine())
print("sglang  =", sglang.__version__)
print("torch   =", torch.__version__)
import sgl_kernel                      # the compiled ext -- most likely to be wrong-arch
print("sgl_kernel ok:", sgl_kernel.__file__)
print("gpus    =", torch.cuda.device_count())
print("capab   =", torch.cuda.get_device_capability())
print("name    =", torch.cuda.get_device_name(0))

ok = True
if __import__("platform").machine() != "aarch64":
    print("FAIL: not aarch64"); ok = False
if not sglang.__version__.startswith("0.5.10"):
    print("FAIL: sglang is %s, expected 0.5.10*" % sglang.__version__); ok = False
if torch.cuda.get_device_capability() != (9, 0):
    print("FAIL: capability %s, expected (9, 0)" % (torch.cuda.get_device_capability(),)); ok = False
if torch.cuda.device_count() != 4:
    print("FAIL: %d GPUs, expected 4" % torch.cuda.device_count()); ok = False
print("PREFLIGHT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY

# /dev/shm must hold the 141 GB checkpoint plus headroom.
shm_free=\$(df --output=avail -B1 /dev/shm | tail -1 | tr -d ' ')
echo "shm_free_GB=\$(( shm_free / 1000000000 ))"
if (( shm_free < 160000000000 )); then
  echo "FAIL: /dev/shm has < 160 GB free"; exit 1
fi
echo "ALL CHECKS PASSED"
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"
