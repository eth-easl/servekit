#!/bin/bash
#SBATCH --job-name=snap-build7
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --partition=mi300
#SBATCH --account=root
set -euo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
export EDF_PATH="${PWD}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"
srun --environment=glm-47-flash-rocm bash -c '
  cd /capstor/scratch/cscs/xyao/kimi-k25-vllm/snapshot/build7
  cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
  cmake --build . -j${SLURM_CPUS_PER_TASK:-8} --target snapshot_record snapshot_redirect 2>&1 | tail -5
  echo "=== build7 done; .so mtime ==="
  ls -la libsnapshot_record.so* libsnapshot_redirect.so* 2>/dev/null | head
'
