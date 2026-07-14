#!/bin/bash
# Is GPUDirect Storage actually usable on bristen over capstor (Lustre)?
#
# The phase5 preflight INFERRED "no GDS" from absent /dev/nvidia-fs* nodes. That
# is a hint, not a measurement:
#   - the device nodes may just not be bind-mounted into the container even if
#     the host driver is loaded (fixable by exposing them);
#   - Lustre is NOT disqualifying (DDN EXAScaler supports GDS).
# So ASK cuFile directly instead of reading tea leaves:
#   1. cuFileDriverOpen() + cuFileDriverGetProperties() -> does it report
#      COMPATIBILITY MODE (POSIX + CPU bounce buffer) or true DMA?
#   2. which backend does InstantTensor actually select for a file on capstor?
#
# This decides how to read the whole phase-5 result: with GDS, InstantTensor's
# 35-45 GB/s headline is in play; without it, it must win on pipelining alone.
#
# Needs a GPU (cuFile needs a CUDA context). ~3 min.
#
# Submit:
#   sbatch lustre-loading-exp/scripts/probes/gds_probe.sbatch
#
#SBATCH --job-name=gds-probe
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=lustre-loading-exp/results/probes/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
MODEL_DIR=/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct

echo "job=${SLURM_JOB_ID} node=$(hostname)"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
pip install 'instanttensor>=0.1.9' -q

echo "=============== A. host/container GDS surface ==============="
echo "--- /dev/nvidia* ---"; ls /dev/nvidia* 2>/dev/null || echo "  none"
echo "--- nvidia-fs in /proc/modules (is the kernel driver loaded at all?) ---"
grep -i nvidia_fs /proc/modules 2>/dev/null || echo "  nvidia_fs NOT in /proc/modules"
echo "--- MOFED / rdma ---"; ls /dev/infiniband 2>/dev/null || echo "  no /dev/infiniband"

echo
echo "=============== B. ASK cuFile ITSELF (authoritative) ==============="
python - <<'PY'
import ctypes, torch
torch.zeros(1, device="cuda")   # force a CUDA context first

lib = ctypes.CDLL("libcufile.so")

class CUfileDrvProps(ctypes.Structure):
    # cufile.h: nvfs{major,minor,...}, fflags, max_direct_io_size, max_device_cache_size, ...
    _fields_ = [("nvfs_major", ctypes.c_uint), ("nvfs_minor", ctypes.c_uint),
                ("nvfs_dstatusflags", ctypes.c_uint), ("nvfs_dcontrolflags", ctypes.c_uint),
                ("fflags", ctypes.c_uint),
                ("max_direct_io_size", ctypes.c_uint),
                ("max_device_cache_size", ctypes.c_uint),
                ("per_buffer_cache_size", ctypes.c_uint),
                ("max_device_pinned_mem_size", ctypes.c_uint),
                ("max_batch_io_size", ctypes.c_uint),
                ("max_batch_io_timeout_msecs", ctypes.c_uint)]

rc = lib.cuFileDriverOpen()
print("cuFileDriverOpen() ->", rc, "(0 = success)")
if rc == 0:
    p = CUfileDrvProps()
    rc2 = lib.cuFileDriverGetProperties(ctypes.byref(p))
    print("cuFileDriverGetProperties() ->", rc2)
    if rc2 == 0:
        # fflags bit 0 == "compatibility mode" per cufile.h (CU_FILE_ALLOW_COMPAT_MODE)
        compat = bool(p.fflags & 0x1)
        print("  nvfs driver version : %d.%d" % (p.nvfs_major, p.nvfs_minor))
        print("  fflags              : 0x%x" % p.fflags)
        print("  max_direct_io_size  :", p.max_direct_io_size)
        print()
        print("  >>> COMPATIBILITY MODE:", compat)
        print("  >>> (True  = POSIX read + CPU bounce buffer, NOT true GDS DMA)")
        print("  >>> (False = real GPUDirect Storage DMA path)")
else:
    print("  >>> cuFile driver could NOT be opened -> no GDS on this node.")
PY

echo
echo "=============== C. which backend does InstantTensor pick? ==============="
python - <<'PY'
import glob, torch, instanttensor
print("Backend       :", [b for b in dir(instanttensor.Backend) if not b.startswith("_")])
print("BackendPolicy :", [b for b in dir(instanttensor.BackendPolicy) if not b.startswith("_")])

f = sorted(glob.glob("${MODEL_DIR}/*.safetensors"))[:1]
print("\nopening one real shard on capstor:", f[0].split("/")[-1])
torch.zeros(1, device="cuda")
try:
    with instanttensor.safe_open(f, framework="pt", device=0, load_now=False) as h:
        for attr in ("backend", "_backend", "selected_backend"):
            if hasattr(h, attr):
                print("  selected backend ->", getattr(h, attr))
                break
        else:
            print("  (handle exposes no backend attr; attrs:", [a for a in dir(h) if not a.startswith("__")][:20], ")")
except Exception as e:
    print("  safe_open failed:", type(e).__name__, e)
PY
echo
echo "[gds-probe] done"
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo "finished=$(date --iso-8601=seconds)"
