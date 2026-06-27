#!/bin/bash
# _build_cuda_so.sh — build the snapshot record+redirect .so for CUDA.
# Invoked inside whichever enroot container is active (snapshot-cuda devel OR the
# vllm image). Building INSIDE the target image is required for cross-container
# LD_PRELOAD: the .so must require a glibc <= the serving image's and link the
# serving image's libcudart (the snapshot-cuda devel image is ubuntu24/glibc2.39
# + CUDA 12.6; the vllm image is ubuntu22/glibc2.35 + CUDA 13 — a .so built in
# one will not load in the other, caught by the ldd gate).
#
# Args: $1 = build dir name under snapshot/ (default: build-vllm).
# Outputs: snapshot/<build>/libsnapshot_{record,redirect}_cuda.so
set -euo pipefail
SNAP_CUDA_DIR="${SNAP_CUDA_DIR:-/capstor/scratch/cscs/xyao/snapshot-cuda}"
BUILD_DIR="${1:-build-vllm}"
cd "${SNAP_CUDA_DIR}"

# Standalone cmake tarball (the vllm runtime image has no cmake; the devel image
# does, but this works in both). No-op if a system cmake is present.
CMAKE_BIN=""
for c in "${SNAP_CUDA_DIR}/cmake/cmake-3.30.8-linux-x86_64/bin/cmake" /usr/bin/cmake; do
  if [ -x "$c" ]; then CMAKE_BIN="$c"; break; fi
done
[ -n "$CMAKE_BIN" ] || { echo "no cmake found"; exit 1; }
export PATH="$(dirname "$CMAKE_BIN"):${PATH}"

# libnvrtc: runtime images ship only the versioned libnvrtc.so.13 (no unversioned
# symlink), so find_library(nvrtc) misses it. Resolve it explicitly if absent.
NVRTC_HINT=""
if [ ! -e /usr/local/cuda/lib64/libnvrtc.so ] && \
   [ -e /usr/local/cuda/lib64/libnvrtc.so.13 ]; then
  NVRTC_HINT="-DSNAPSHOT_NVRTC_LIBRARY=/usr/local/cuda/lib64/libnvrtc.so.13"
fi

cmake -S snapshot -B "snapshot/${BUILD_DIR}" \
  -DSNAPSHOT_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=80 ${NVRTC_HINT}
cmake --build "snapshot/${BUILD_DIR}" -j"${SLURM_CPUS_PER_TASK:-8}" \
  --target snapshot_record_cuda snapshot_redirect_cuda
ls -l "snapshot/${BUILD_DIR}/libsnapshot_record_cuda.so" \
      "snapshot/${BUILD_DIR}/libsnapshot_redirect_cuda.so"
