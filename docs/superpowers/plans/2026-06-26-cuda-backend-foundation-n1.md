# N1 — CUDA Backend Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `CudaBackend` (CUDA driver-API VMM + graph capture/rebuild + nvrtc synthetic module) so the standalone `snapshot` CLI gates pass on an A100, mirroring the proven HIP backend — with the HIP path left byte-for-byte untouched.

**Architecture:** Fill the three stubbed CUDA backend translation units (`cuda_vmm.cpp`, `cuda_graph.cpp`, `cuda_backend.cpp`) behind the existing vendor-neutral `GpuBackend` interface, using the CUDA **driver API** (`cu*`, link `libcuda`) plus `nvrtc`. The vendor-neutral CLI, IR, serialization, relocation, and allocator (`csrc/core/*`, `csrc/cli/*`) are reused unchanged and act as the test oracle: the same gates that pass on HIP must pass on CUDA. The build selects this backend with `-DSNAPSHOT_BACKEND=CUDA`.

**Tech Stack:** C++17, CUDA Driver API (CUDA ≥12.4), nvrtc, CMake ≥3.20, enroot/pyxis containers on the CSCS **bristen** A100 cluster (SLURM `-A a-infra02`), `rcc --profile bristen push` for code sync.

## Global Constraints

- **Dual-vendor, build-time selection:** the backend is chosen at configure time via `SNAPSHOT_BACKEND` (`AUTO|HIP|CUDA|STUB`). This plan only adds/edits the CUDA branch and CUDA source files.
- **Do NOT modify (regression invariant):** `csrc/backends/hip/*`, `csrc/preload/*`, `csrc/core/*`, `include/snapshot/*`, `csrc/cli/*`. The only edits outside `csrc/backends/cuda/` are: the CUDA branch of `cmake/SnapshotBackend.cmake`, and new files under `snapshot/recipe/`. `cuda_backend.hpp` gets ONE additive declaration (`ensure_cuda_context`). All HIP/core/cli sources stay unchanged so the AMD path cannot regress.
- **Fixed base:** `snapshot::kDefaultRequestedBase` = `0x600000000000` (vendor-neutral constant, already defined). The CUDA VMM reserve must honor it when free.
- **Target arch:** A100 = `sm_80`. nvrtc compiles to a cubin for the device's actual compute capability.
- **Bit-identical:** the synthetic kernels use exact unsigned-integer arithmetic, identical source to `hip_kernels.cpp`, so captured-then-restored device memory is byte-identical regardless of base address.
- **Run on hardware, not the login node:** bristen login nodes have no GPU and no CUDA toolkit. Every build/gate runs inside the CUDA container on a compute node via `sbatch`/`srun`.
- **Account/partition:** bristen uses `-A a-infra02`, partition `normal`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `snapshot/csrc/backends/cuda/cuda_vmm.cpp` | CUDA context init, `arch`, VMM (granularity/reserve/release/create/map/unmap/set_access) | Replace stubs |
| `snapshot/csrc/backends/cuda/cuda_graph.cpp` | module load/get_function, stream, capture, introspect, rebuild, instantiate, launch, kernel-launch, memcpy | Replace stubs |
| `snapshot/csrc/backends/cuda/cuda_backend.cpp` | `make_cuda_backend`, `compile_synthetic_module` (nvrtc → cubin) | Replace stubs |
| `snapshot/csrc/backends/cuda/cuda_backend.hpp` | `CudaBackend` class decl | Add ONE line: `Status ensure_cuda_context();` decl |
| `snapshot/cmake/SnapshotBackend.cmake` | `snapshot_configure_cuda` | Add nvrtc include/lib |
| `snapshot/recipe/snapshot-cuda.toml` | enroot EDF for the CUDA build container | Create |
| `snapshot/recipe/build_snapshot_cuda.sbatch` | build + ctest + CLI gates on bristen | Create |

`ensure_cuda_context()` is defined once (non-anonymous) in `cuda_vmm.cpp` and called from all three CUDA TUs; the per-TU `cu_status` error helper is duplicated in each CUDA `.cpp` (matching how HIP duplicates `hip_status`).

---

## Task 1: CUDA build toolchain + cluster harness

Establish a working CUDA build container on bristen, wire nvrtc into CMake, and get the (still-stubbed) CUDA build to configure, compile, and pass the host-only tests. This locks in the toolchain before any backend code is written.

**Files:**
- Create: `snapshot/recipe/snapshot-cuda.toml`
- Create: `snapshot/recipe/build_snapshot_cuda.sbatch`
- Modify: `snapshot/cmake/SnapshotBackend.cmake` (the `snapshot_configure_cuda` function)

**Interfaces:**
- Consumes: existing `CMakeLists.txt` CUDA branch (builds `snapshot_core` + `snapshot` CLI + tests under `SNAPSHOT_BACKEND=CUDA`).
- Produces: a reproducible `sbatch snapshot/recipe/build_snapshot_cuda.sbatch` that configures + builds + runs `ctest -E _gpu` inside the CUDA container.

- [ ] **Step 1: Probe the container toolchain on bristen**

Sync the repo and run a one-off probe to confirm the chosen image has the full toolchain. First create the EDF (next step needs it); for the probe, run an inline srun against a candidate image.

Run (from local):
```bash
rcc --profile bristen push
ssh bristen 'cd /capstor/scratch/cscs/xyao/<REMOTE_DIR> && \
  srun -A a-infra02 -p normal --gpus-per-node=1 -t 00:10:00 \
    --environment="nvidia/cuda:12.6.3-devel-ubuntu24.04" \
    bash -lc "nvcc --version | tail -2; cmake --version | head -1; g++ --version | head -1; \
      ls /usr/local/cuda/include/cuda.h /usr/local/cuda/include/nvrtc.h; \
      echo CUDA_DRIVER:; ls /usr/lib/x86_64-linux-gnu/libcuda.so* 2>/dev/null || echo no-libcuda"'
```
Expected: prints CUDA toolkit version (≥12.4), `cmake version 3.x`, a gcc version, and confirms `cuda.h` + `nvrtc.h` exist. `libcuda.so` is injected by the CSCS container engine's NVIDIA hook at runtime (it may be absent from the listing but present when a GPU job runs).

If `cmake` is missing from the devel image: build a one-time enroot overlay that adds it (see `clusters/cscs/containers.md` overlay recipe — `enroot create → start --rw → apt-get install -y cmake → export` to `snapshot-cuda.sqsh`, ~1 min), and point the EDF `image` at that `.sqsh`. Record the resolved image (digest or `.sqsh` path) in `snapshot-cuda.toml`.

- [ ] **Step 2: Write the build EDF**

Create `snapshot/recipe/snapshot-cuda.toml` (mirror `deploy/glm-47-flash-bristen/glm-47-flash-sglang.toml`'s mount/env shape):
```toml
# CUDA build/test container for the snapshot CudaBackend on bristen (A100/sm_80).
# Any CUDA >= 12.4 devel image with cmake + g++ + nvrtc headers works. Pin a
# concrete digest (or a local .sqsh overlay that adds cmake) once Step 1 resolves
# the toolchain. libcuda.so is injected by the CSCS container engine at runtime.
image = "nvidia/cuda:12.6.3-devel-ubuntu24.04"

mounts = [
  "/capstor:/capstor",
  "/users/xyao:/users/xyao",
  "/iopsstor:/iopsstor",
]

workdir = "/capstor/scratch/cscs/xyao/<REMOTE_DIR>"

[env]
HOME = "/capstor/scratch/cscs/xyao/<REMOTE_DIR>/home"
```
Replace `<REMOTE_DIR>` with the bristen `rcc` profile's `remote_dir` (add a `[profiles.bristen-snapshot]` to `.rcc/config.toml` if needed, or reuse the existing bristen remote dir).

- [ ] **Step 3: Add nvrtc to `snapshot_configure_cuda`**

In `snapshot/cmake/SnapshotBackend.cmake`, the `snapshot_configure_cuda(target)` function currently finds `cuda.h` + `libcuda`. Add nvrtc so `cuda_backend.cpp` can `#include <nvrtc.h>` and link `libnvrtc`. Replace the function body's find/link section with:
```cmake
function(snapshot_configure_cuda target)
  find_path(SNAPSHOT_CUDA_INCLUDE_DIR
    NAMES cuda.h
    HINTS "$ENV{CUDA_HOME}/include" "/usr/local/cuda/include")
  find_library(SNAPSHOT_CUDA_DRIVER_LIBRARY
    NAMES cuda
    HINTS "$ENV{CUDA_HOME}/lib64" "/usr/local/cuda/lib64" "/usr/lib/x86_64-linux-gnu")
  find_library(SNAPSHOT_NVRTC_LIBRARY
    NAMES nvrtc
    HINTS "$ENV{CUDA_HOME}/lib64" "/usr/local/cuda/lib64" "/usr/lib/x86_64-linux-gnu")

  if(NOT SNAPSHOT_CUDA_INCLUDE_DIR OR NOT SNAPSHOT_CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "SNAPSHOT_BACKEND=CUDA requires cuda.h and libcuda")
  endif()
  if(NOT SNAPSHOT_NVRTC_LIBRARY)
    message(FATAL_ERROR "SNAPSHOT_BACKEND=CUDA requires libnvrtc for synthetic kernel compilation")
  endif()

  target_include_directories(${target} PRIVATE "${SNAPSHOT_CUDA_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE
    "${SNAPSHOT_CUDA_DRIVER_LIBRARY}" "${SNAPSHOT_NVRTC_LIBRARY}")
endfunction()
```

- [ ] **Step 4: Write the build sbatch (the failing gate)**

Create `snapshot/recipe/build_snapshot_cuda.sbatch` (mirror `build_snapshot.sbatch`, swap partition/account/EDF/backend):
```bash
#!/bin/bash
#SBATCH --job-name=snapshot-build-cuda
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/<REMOTE_DIR>/logs/%x-%j.out

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/<REMOTE_DIR>}"
SCRIPT_DIR="${DEPLOY_DIR}/snapshot/recipe"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/snapshot/build-cuda"

echo "snapshot CUDA build started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"

srun --environment=snapshot-cuda \
  bash -lc 'set -euo pipefail
    cmake -S snapshot -B snapshot/build-cuda -DSNAPSHOT_BACKEND=CUDA
    cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}"
    echo "--- host-only ctest (GPU tests excluded until backend is implemented) ---"
    ctest --test-dir snapshot/build-cuda -E "_gpu" --output-on-failure
  '

echo "snapshot CUDA build finished: $(date --iso-8601=seconds)"
```

- [ ] **Step 5: Run the build job — verify it fails as expected, then passes**

Run:
```bash
rcc --profile bristen push
ssh bristen 'cd /capstor/scratch/cscs/xyao/<REMOTE_DIR> && sbatch snapshot/recipe/build_snapshot_cuda.sbatch'
# then tail the log:
ssh bristen 'tail -n 40 /capstor/scratch/cscs/xyao/<REMOTE_DIR>/logs/snapshot-build-cuda-*.out'
```
Expected: the CUDA build **configures and compiles** (the stubbed `cuda_*.cpp` include only `cuda_backend.hpp`/`gpu_backend.hpp`, so they compile without `cuda.h`; CMake's `find_path` validated `cuda.h`/`nvrtc.h` exist). Host-only `ctest -E _gpu` reports **all passing** (`test_allocator`, `test_relocation`, `test_serialize_roundtrip`, `test_record_ir`, `test_amdgpu_msgpack`). The GPU tests are excluded here (they'd fail against the stub backend — implemented in Tasks 2–4).

- [ ] **Step 6: Commit**

```bash
git add snapshot/recipe/snapshot-cuda.toml snapshot/recipe/build_snapshot_cuda.sbatch snapshot/cmake/SnapshotBackend.cmake
git commit -m "snapshot(cuda): N1 build harness — EDF, build sbatch, nvrtc in cmake"
```

---

## Task 2: `cuda_vmm.cpp` — context init, arch, VMM

Implement CUDA driver-API context init plus the VMM methods, so `probe-base` reports a honored fixed base on the A100.

**Files:**
- Modify: `snapshot/csrc/backends/cuda/cuda_backend.hpp` (add `ensure_cuda_context` decl)
- Replace: `snapshot/csrc/backends/cuda/cuda_vmm.cpp`

**Interfaces:**
- Consumes: `kDefaultRequestedBase` (from `snapshot/allocator.hpp` via the CLI), `MemHandle`, `MemoryAccess`.
- Produces: `Status ensure_cuda_context();` (free function, namespace `snapshot`) — called by Tasks 3 entry points; `CudaBackend::{arch,get_allocation_granularity,reserve_address,release_address,create_physical,release_physical,map,unmap,set_access}`.

- [ ] **Step 1: Add the `ensure_cuda_context` declaration to the header**

In `snapshot/csrc/backends/cuda/cuda_backend.hpp`, after the `#include` and inside `namespace snapshot {`, before `class CudaBackend`, add:
```cpp
// Establishes the device-0 primary CUDA context once (driver API requires
// explicit cuInit + a current context). Defined in cuda_vmm.cpp; called by every
// CudaBackend entry point that issues driver calls.
Status ensure_cuda_context();
```

- [ ] **Step 2: Write `cuda_vmm.cpp` (the implementation)**

Replace the entire contents of `snapshot/csrc/backends/cuda/cuda_vmm.cpp` with:
```cpp
#include "cuda_backend.hpp"

#include <sstream>
#include <string>

#include <cuda.h>

namespace snapshot {
namespace {

Status cu_status(CUresult error, const char* call) {
  if (error == CUDA_SUCCESS) {
    return Status::Ok();
  }
  const char* msg = nullptr;
  cuGetErrorString(error, &msg);
  std::ostringstream message;
  message << call << " failed: " << (msg ? msg : "unknown CUDA error");
  return Status::backend(message.str());
}

CUmemGenericAllocationHandle to_physical(MemHandle handle) {
  return static_cast<CUmemGenericAllocationHandle>(handle.value);
}

MemHandle from_physical(CUmemGenericAllocationHandle handle) {
  MemHandle out;
  out.value = static_cast<std::uintptr_t>(handle);
  return out;
}

// The physical allocation and its access grant must target the device the
// caller is actually using (matches hip_vmm.cpp's current_device rationale).
int current_device() {
  CUdevice dev = 0;
  cuCtxGetDevice(&dev);
  return static_cast<int>(dev);
}

}  // namespace

Status ensure_cuda_context() {
  // Run-once: cuInit + retain device-0 primary context + make it current. The
  // primary context is shared with any CUDA runtime-API user in the process.
  static Status init = [] {
    CUresult rc = cuInit(0);
    if (rc != CUDA_SUCCESS) return cu_status(rc, "cuInit");
    CUdevice dev = 0;
    rc = cuDeviceGet(&dev, 0);
    if (rc != CUDA_SUCCESS) return cu_status(rc, "cuDeviceGet");
    CUcontext ctx{};
    rc = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (rc != CUDA_SUCCESS) return cu_status(rc, "cuDevicePrimaryCtxRetain");
    rc = cuCtxSetCurrent(ctx);
    if (rc != CUDA_SUCCESS) return cu_status(rc, "cuCtxSetCurrent");
    return Status::Ok();
  }();
  return init;
}

Status CudaBackend::arch(ArchInfo* out) {
  if (out == nullptr) {
    return Status::invalid_argument("arch output is null");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }
  CUdevice dev = 0;
  status = cu_status(cuCtxGetDevice(&dev), "cuCtxGetDevice");
  if (!status.ok()) {
    return status;
  }
  int major = 0, minor = 0;
  status = cu_status(
      cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                           dev),
      "cuDeviceGetAttribute(major)");
  if (!status.ok()) {
    return status;
  }
  status = cu_status(
      cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           dev),
      "cuDeviceGetAttribute(minor)");
  if (!status.ok()) {
    return status;
  }
  out->name = "sm_" + std::to_string(major) + std::to_string(minor);
  return Status::Ok();
}

Status CudaBackend::get_allocation_granularity(std::uint64_t* out) {
  if (out == nullptr) {
    return Status::invalid_argument("granularity output is null");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = current_device();
  std::size_t granularity = 0;
  status = cu_status(
      cuMemGetAllocationGranularity(&granularity, &prop,
                                    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
      "cuMemGetAllocationGranularity");
  if (!status.ok()) {
    return status;
  }
  *out = granularity;
  return Status::Ok();
}

Status CudaBackend::reserve_address(std::uint64_t size, std::uint64_t alignment,
                                    std::uint64_t requested_base,
                                    std::uint64_t* out_base) {
  if (out_base == nullptr) {
    return Status::invalid_argument("reserve output is null");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }
  CUdeviceptr ptr = 0;
  status = cu_status(
      cuMemAddressReserve(&ptr, size, alignment,
                          static_cast<CUdeviceptr>(requested_base), 0),
      "cuMemAddressReserve");
  if (!status.ok()) {
    return status;
  }
  *out_base = static_cast<std::uint64_t>(ptr);
  return Status::Ok();
}

Status CudaBackend::release_address(std::uint64_t base, std::uint64_t size) {
  return cu_status(cuMemAddressFree(static_cast<CUdeviceptr>(base), size),
                   "cuMemAddressFree");
}

Status CudaBackend::create_physical(std::uint64_t size, MemHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("physical output is null");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = current_device();
  CUmemGenericAllocationHandle handle{};
  status = cu_status(cuMemCreate(&handle, size, &prop, 0), "cuMemCreate");
  if (!status.ok()) {
    return status;
  }
  *out = from_physical(handle);
  return Status::Ok();
}

Status CudaBackend::release_physical(MemHandle handle) {
  return cu_status(cuMemRelease(to_physical(handle)), "cuMemRelease");
}

Status CudaBackend::map(std::uint64_t va, std::uint64_t size,
                        std::uint64_t offset, MemHandle handle) {
  return cu_status(cuMemMap(static_cast<CUdeviceptr>(va), size, offset,
                            to_physical(handle), 0),
                   "cuMemMap");
}

Status CudaBackend::unmap(std::uint64_t va, std::uint64_t size) {
  return cu_status(cuMemUnmap(static_cast<CUdeviceptr>(va), size), "cuMemUnmap");
}

Status CudaBackend::set_access(std::uint64_t va, std::uint64_t size,
                               const MemoryAccess& access) {
  CUmemAccessDesc desc{};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = access.device_ordinal != 0 ? access.device_ordinal
                                                 : current_device();
  desc.flags = access.read && access.write
                   ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                   : (access.read ? CU_MEM_ACCESS_FLAGS_PROT_READ
                                  : CU_MEM_ACCESS_FLAGS_PROT_NONE);
  return cu_status(
      cuMemSetAccess(static_cast<CUdeviceptr>(va), size, &desc, 1),
      "cuMemSetAccess");
}

}  // namespace snapshot
```

- [ ] **Step 3: Add the `probe-base` gate to the build sbatch**

In `snapshot/recipe/build_snapshot_cuda.sbatch`, inside the `srun` heredoc, after the `ctest` line add:
```bash
    echo "--- probe-base (fixed-base determinism on A100) ---"
    snapshot/build-cuda/snapshot probe-base
```

- [ ] **Step 4: Run and verify**

Run:
```bash
rcc --profile bristen push
ssh bristen 'cd /capstor/scratch/cscs/xyao/<REMOTE_DIR> && sbatch snapshot/recipe/build_snapshot_cuda.sbatch'
ssh bristen 'tail -n 20 /capstor/scratch/cscs/xyao/<REMOTE_DIR>/logs/snapshot-build-cuda-*.out'
```
Expected output includes:
```
requested_base=0x600000000000
returned_base=0x600000000000
honored=1
```
(`honored=1` confirms `cuMemAddressReserve` accepts the fixed base on the A100. If `honored=0`, the VMM still works — relocation handles a shifted base — but note it; the Δ=0 fast path needs `honored=1`.)

- [ ] **Step 5: Commit**

```bash
git add snapshot/csrc/backends/cuda/cuda_vmm.cpp snapshot/csrc/backends/cuda/cuda_backend.hpp snapshot/recipe/build_snapshot_cuda.sbatch
git commit -m "snapshot(cuda): N1 Task 2 — CUDA context + arch + VMM (probe-base honored)"
```

---

## Task 3: nvrtc synthetic module + CUDA graph forward path

Implement `compile_synthetic_module` (nvrtc → cubin) and the forward-path graph methods (module/stream/capture/introspect/instantiate/launch/kernel-launch/memcpy), so `snapshot capture` produces a snapshot file. `rebuild_graph` stays stubbed until Task 4.

**Files:**
- Replace: `snapshot/csrc/backends/cuda/cuda_backend.cpp`
- Replace: `snapshot/csrc/backends/cuda/cuda_graph.cpp`

**Interfaces:**
- Consumes: `ensure_cuda_context()` (Task 2); `KernelLaunchParams` (`function`, `grid`, `block`, `shared_mem_bytes`, `param_blob`); `GraphIR`/`GraphNodeIR`.
- Produces: `make_cuda_backend()`, `compile_synthetic_module(image, entry_names)` emitting a cubin + `{"mul_bias","relu_offset","in_place"}`; `CudaBackend::{load_module,unload_module,get_function,stream_create,stream_destroy,begin_capture,end_capture,introspect_graph,instantiate,exec_set_kernel_node_params,launch,synchronize,launch_kernel,memcpy_h2d,memcpy_d2h}`. `rebuild_graph` returns `unsupported` (Task 4).

- [ ] **Step 1: Write `cuda_backend.cpp` (nvrtc compile)**

Replace the entire contents of `snapshot/csrc/backends/cuda/cuda_backend.cpp` with:
```cpp
#include "cuda_backend.hpp"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <nvrtc.h>

#include "snapshot/workload_kernels.hpp"

namespace snapshot {

std::unique_ptr<GpuBackend> make_cuda_backend() {
  return std::make_unique<CudaBackend>();
}

namespace {

// Identical exact-unsigned-integer arithmetic to hip_kernels.cpp, so a
// captured-then-restored pipeline reproduces byte-identical device memory.
// extern "C" => unmangled names, so cuModuleGetFunction resolves them by name.
constexpr const char* kSource = R"cuda(
extern "C" __global__ void mul_bias(unsigned int* a, unsigned int* b,
                                    unsigned int* c, int bias, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] * b[i] + (unsigned int)bias;
}

extern "C" __global__ void relu_offset(unsigned int* c, unsigned int* out,
                                       int offset, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = c[i] + (unsigned int)offset;
}

extern "C" __global__ void in_place(unsigned int* out, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = out[i] ^ 0x9e3779b9u;
}
)cuda";

Status nvrtc_fail(nvrtcResult result, const char* call) {
  std::ostringstream message;
  message << call << " failed: " << nvrtcGetErrorString(result);
  return Status::backend(message.str());
}

// "--gpu-architecture=sm_XY" from the current device's compute capability.
// Defaults to sm_80 (A100) when no context is current yet (nvrtc itself needs
// no context; this only picks the codegen target).
std::string arch_option() {
  int major = 8, minor = 0;
  CUdevice dev = 0;
  if (cuCtxGetDevice(&dev) == CUDA_SUCCESS) {
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                         dev);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                         dev);
  }
  return "--gpu-architecture=sm_" + std::to_string(major) +
         std::to_string(minor);
}

}  // namespace

Status compile_synthetic_module(std::vector<std::byte>* image,
                                std::vector<std::string>* entry_names) {
  if (image == nullptr || entry_names == nullptr) {
    return Status::invalid_argument("compile_synthetic_module output is null");
  }

  // Establish a context first so arch_option targets the real device.
  (void)ensure_cuda_context();

  nvrtcProgram program{};
  nvrtcResult result = nvrtcCreateProgram(&program, kSource,
                                          "snapshot_synthetic.cu", 0, nullptr,
                                          nullptr);
  if (result != NVRTC_SUCCESS) {
    return nvrtc_fail(result, "nvrtcCreateProgram");
  }

  const std::string arch = arch_option();
  const char* options[] = {arch.c_str()};
  result = nvrtcCompileProgram(program, 1, options);
  if (result != NVRTC_SUCCESS) {
    std::size_t log_size = 0;
    nvrtcGetProgramLogSize(program, &log_size);
    std::string log(log_size, '\0');
    if (log_size > 0) {
      nvrtcGetProgramLog(program, log.data());
    }
    nvrtcDestroyProgram(&program);
    std::ostringstream message;
    message << "nvrtcCompileProgram failed: " << log;
    return Status::backend(message.str());
  }

  std::size_t cubin_size = 0;
  result = nvrtcGetCUBINSize(program, &cubin_size);
  if (result != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    return nvrtc_fail(result, "nvrtcGetCUBINSize");
  }
  std::vector<char> cubin(cubin_size);
  result = nvrtcGetCUBIN(program, cubin.data());
  if (result != NVRTC_SUCCESS) {
    nvrtcDestroyProgram(&program);
    return nvrtc_fail(result, "nvrtcGetCUBIN");
  }
  nvrtcDestroyProgram(&program);

  image->assign(reinterpret_cast<const std::byte*>(cubin.data()),
                reinterpret_cast<const std::byte*>(cubin.data()) + cubin.size());
  *entry_names = {"mul_bias", "relu_offset", "in_place"};
  return Status::Ok();
}

}  // namespace snapshot
```

- [ ] **Step 2: Write `cuda_graph.cpp` (forward path, rebuild stubbed)**

Replace the entire contents of `snapshot/csrc/backends/cuda/cuda_graph.cpp` with:
```cpp
#include "cuda_backend.hpp"

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <vector>

#include <cuda.h>

namespace snapshot {
namespace {

Status cu_status(CUresult error, const char* call) {
  if (error == CUDA_SUCCESS) {
    return Status::Ok();
  }
  const char* msg = nullptr;
  cuGetErrorString(error, &msg);
  std::ostringstream message;
  message << call << " failed: " << (msg ? msg : "unknown CUDA error");
  return Status::backend(message.str());
}

template <typename T>
T as_handle(OpaqueHandle handle) {
  return reinterpret_cast<T>(handle.value);
}

template <typename Handle, typename T>
Handle from_handle(T value) {
  Handle out;
  out.value = reinterpret_cast<std::uintptr_t>(value);
  return out;
}

GraphNodeType from_cuda_node_type(CUgraphNodeType type) {
  switch (type) {
    case CU_GRAPH_NODE_TYPE_KERNEL:
      return GraphNodeType::kKernel;
    case CU_GRAPH_NODE_TYPE_MEMCPY:
      return GraphNodeType::kMemcpyD2D;
    case CU_GRAPH_NODE_TYPE_MEMSET:
      return GraphNodeType::kMemset;
    default:
      return GraphNodeType::kKernel;
  }
}

}  // namespace

Status CudaBackend::load_module(const std::byte* image, std::size_t n,
                                ModuleHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("module output is null");
  }
  if (image == nullptr || n == 0) {
    return Status::invalid_argument("module image is empty");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }
  CUmodule module{};
  status = cu_status(cuModuleLoadData(&module, image), "cuModuleLoadData");
  if (!status.ok()) {
    return status;
  }
  *out = from_handle<ModuleHandle>(module);
  return Status::Ok();
}

Status CudaBackend::unload_module(ModuleHandle module) {
  return cu_status(cuModuleUnload(as_handle<CUmodule>(module)),
                   "cuModuleUnload");
}

Status CudaBackend::get_function(ModuleHandle module, const std::string& name,
                                 FunctionHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("function output is null");
  }
  CUfunction function{};
  Status status = cu_status(
      cuModuleGetFunction(&function, as_handle<CUmodule>(module), name.c_str()),
      "cuModuleGetFunction");
  if (!status.ok()) {
    return status;
  }
  *out = from_handle<FunctionHandle>(function);
  return Status::Ok();
}

Status CudaBackend::stream_create(StreamHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("stream output is null");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }
  CUstream stream{};
  status = cu_status(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
                     "cuStreamCreate");
  if (!status.ok()) {
    return status;
  }
  *out = from_handle<StreamHandle>(stream);
  return Status::Ok();
}

Status CudaBackend::stream_destroy(StreamHandle stream) {
  return cu_status(cuStreamDestroy(as_handle<CUstream>(stream)),
                   "cuStreamDestroy");
}

Status CudaBackend::begin_capture(StreamHandle stream) {
  return cu_status(cuStreamBeginCapture(as_handle<CUstream>(stream),
                                        CU_STREAM_CAPTURE_MODE_GLOBAL),
                   "cuStreamBeginCapture");
}

Status CudaBackend::end_capture(StreamHandle stream, GraphHandle* out_graph) {
  if (out_graph == nullptr) {
    return Status::invalid_argument("graph output is null");
  }
  CUgraph graph{};
  Status status = cu_status(
      cuStreamEndCapture(as_handle<CUstream>(stream), &graph),
      "cuStreamEndCapture");
  if (!status.ok()) {
    return status;
  }
  *out_graph = from_handle<GraphHandle>(graph);
  return Status::Ok();
}

Status CudaBackend::introspect_graph(GraphHandle graph, GraphIR* out) {
  if (out == nullptr) {
    return Status::invalid_argument("graph IR output is null");
  }
  CUgraph cuda_graph = as_handle<CUgraph>(graph);

  std::size_t count = 0;
  Status status = cu_status(cuGraphGetNodes(cuda_graph, nullptr, &count),
                            "cuGraphGetNodes(count)");
  if (!status.ok()) {
    return status;
  }

  std::vector<CUgraphNode> nodes(count);
  if (count > 0) {
    status = cu_status(cuGraphGetNodes(cuda_graph, nodes.data(), &count),
                       "cuGraphGetNodes");
    if (!status.ok()) {
      return status;
    }
  }

  // Recovers structure (count + node types) for validation against the recorded
  // IR; a captured node's kernel identity/args are NOT recovered here (recorded
  // at issue time by the workload/interposer instead).
  out->nodes.clear();
  out->edges.clear();
  for (std::size_t i = 0; i < count; ++i) {
    CUgraphNodeType node_type{};
    status = cu_status(cuGraphNodeGetType(nodes[i], &node_type),
                       "cuGraphNodeGetType");
    if (!status.ok()) {
      return status;
    }
    GraphNodeIR node;
    node.id = static_cast<std::uint64_t>(i + 1);
    node.type = from_cuda_node_type(node_type);
    out->nodes.push_back(std::move(node));
  }
  return Status::Ok();
}

Status CudaBackend::rebuild_graph(const GraphIR& /*ir*/, GraphHandle* /*out*/) {
  return Status::unsupported("CUDA rebuild_graph is implemented in N1 Task 4");
}

Status CudaBackend::instantiate(GraphHandle graph, GraphExecHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("graph exec output is null");
  }
  CUgraphExec exec{};
  Status status = cu_status(
      cuGraphInstantiateWithFlags(&exec, as_handle<CUgraph>(graph), 0),
      "cuGraphInstantiateWithFlags");
  if (!status.ok()) {
    return status;
  }
  *out = from_handle<GraphExecHandle>(exec);
  return Status::Ok();
}

Status CudaBackend::exec_set_kernel_node_params(
    GraphExecHandle /*exec*/, std::uint64_t /*node_id*/,
    const KernelLaunchParams& /*params*/) {
  return Status::unsupported(
      "CUDA graph exec kernel parameter update needs node-handle mapping");
}

Status CudaBackend::launch(GraphExecHandle exec, StreamHandle stream) {
  return cu_status(cuGraphLaunch(as_handle<CUgraphExec>(exec),
                                 as_handle<CUstream>(stream)),
                   "cuGraphLaunch");
}

Status CudaBackend::synchronize(StreamHandle stream) {
  return cu_status(cuStreamSynchronize(as_handle<CUstream>(stream)),
                   "cuStreamSynchronize");
}

Status CudaBackend::launch_kernel(StreamHandle stream,
                                  const KernelLaunchParams& params) {
  // The synthetic workload records a FLAT kernarg buffer (param_blob) padded to
  // the kernel's exact kernarg-segment size, with ptr_offsets marking pointer
  // args for relocation. Launch it via the driver buffer (extra) format sized to
  // the blob. The buffer must be valid only for the duration of this call (the
  // driver copies kernargs at enqueue time), so a local copy is safe.
  std::vector<std::byte> blob(params.param_blob);
  std::size_t size = blob.size();
  void* config[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, blob.data(),
                    CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
  return cu_status(
      cuLaunchKernel(as_handle<CUfunction>(params.function), params.grid.x,
                     params.grid.y, params.grid.z, params.block.x,
                     params.block.y, params.block.z, params.shared_mem_bytes,
                     as_handle<CUstream>(stream), nullptr, config),
      "cuLaunchKernel");
}

Status CudaBackend::memcpy_h2d(std::uint64_t dst_device, const void* src_host,
                               std::uint64_t bytes, StreamHandle stream) {
  return cu_status(
      cuMemcpyHtoDAsync(static_cast<CUdeviceptr>(dst_device), src_host, bytes,
                        as_handle<CUstream>(stream)),
      "cuMemcpyHtoDAsync");
}

Status CudaBackend::memcpy_d2h(void* dst_host, std::uint64_t src_device,
                               std::uint64_t bytes, StreamHandle stream) {
  return cu_status(
      cuMemcpyDtoHAsync(dst_host, static_cast<CUdeviceptr>(src_device), bytes,
                        as_handle<CUstream>(stream)),
      "cuMemcpyDtoHAsync");
}

}  // namespace snapshot
```

- [ ] **Step 3: Add the `capture` gate to the build sbatch**

In `snapshot/recipe/build_snapshot_cuda.sbatch`, after the `probe-base` line in the heredoc, add:
```bash
    echo "--- capture (forward path: nvrtc -> module -> VMM -> capture -> serialize) ---"
    snapshot/build-cuda/snapshot capture snapshot/build-cuda/cap.snap
    ls -l snapshot/build-cuda/cap.snap
```

- [ ] **Step 4: Run and verify capture succeeds**

Run:
```bash
rcc --profile bristen push
ssh bristen 'cd /capstor/scratch/cscs/xyao/<REMOTE_DIR> && sbatch snapshot/recipe/build_snapshot_cuda.sbatch'
ssh bristen 'tail -n 30 /capstor/scratch/cscs/xyao/<REMOTE_DIR>/logs/snapshot-build-cuda-*.out'
```
Expected: the `capture` command exits 0 and `cap.snap` is a non-empty file (a few hundred KB — it embeds the cubin). No `unsupported`/`backend_error` lines. This proves nvrtc compile, module load, fixed-base VMM alloc, input seed, real stream capture, introspect (node-count match), instantiate, and graph launch all work on the A100.

- [ ] **Step 5: Commit**

```bash
git add snapshot/csrc/backends/cuda/cuda_backend.cpp snapshot/csrc/backends/cuda/cuda_graph.cpp snapshot/recipe/build_snapshot_cuda.sbatch
git commit -m "snapshot(cuda): N1 Task 3 — nvrtc synthetic module + CUDA graph forward path (capture works)"
```

---

## Task 4: `rebuild_graph` — the restore path (bit-identical gate)

Implement `rebuild_graph` (build a CUDA graph node-by-node with buffer-format kernargs, keeping the kernarg buffers alive until instantiate), so the full `verify`/`restore`/`bench` gates pass bit-identically.

**Files:**
- Modify: `snapshot/csrc/backends/cuda/cuda_graph.cpp` (add a side-data registry + implement `rebuild_graph`; free the registry in `instantiate`)

**Interfaces:**
- Consumes: `GraphIR` with each kernel node's resolved `function` handle and relocated `param_blob` (the CLI's restore path resolves functions + relocates pointers by Δ before calling `rebuild_graph`).
- Produces: `CudaBackend::rebuild_graph` returning a `GraphHandle` whose kernel nodes carry the recorded kernargs; `instantiate` drops the side data after the exec bakes its copy.

- [ ] **Step 1: Add the side-data registry above the methods**

In `snapshot/csrc/backends/cuda/cuda_graph.cpp`, add `#include <map>` and `#include <memory>` to the include block, then inside the anonymous `namespace {` (after `from_cuda_node_type`) add:
```cpp
// A rebuilt kernel node references its kernarg buffer through the driver `extra`
// launch config. That buffer, the `size` it points at, and the config array must
// outlive cuGraphAddKernelNode and stay valid until the graph is instantiated
// (after which the executable graph owns a baked copy). Park them in a registry
// keyed by the graph handle; drop it in instantiate(). unique_ptr gives each
// NodeParam a stable address as more nodes are added.
struct NodeParam {
  std::vector<std::byte> blob;
  std::size_t size = 0;
  void* config[5] = {};  // {BUFFER_POINTER, blob.data(), BUFFER_SIZE, &size, END}
};

struct GraphSideData {
  std::vector<std::unique_ptr<NodeParam>> nodes;
};

std::map<std::uintptr_t, GraphSideData>& graph_side_registry() {
  static std::map<std::uintptr_t, GraphSideData> registry;
  return registry;
}
```

- [ ] **Step 2: Replace the `rebuild_graph` stub with the implementation**

Replace the `CudaBackend::rebuild_graph` stub body with:
```cpp
Status CudaBackend::rebuild_graph(const GraphIR& ir, GraphHandle* out) {
  if (out == nullptr) {
    return Status::invalid_argument("graph output is null");
  }
  Status status = ensure_cuda_context();
  if (!status.ok()) {
    return status;
  }

  CUgraph graph{};
  status = cu_status(cuGraphCreate(&graph, 0), "cuGraphCreate");
  if (!status.ok()) {
    return status;
  }

  GraphSideData side;
  std::map<std::uint64_t, CUgraphNode> node_by_id;

  for (const GraphNodeIR& node : ir.nodes) {
    if (node.type != GraphNodeType::kKernel) {
      static_cast<void>(cuGraphDestroy(graph));
      return Status::unsupported(
          "rebuild_graph currently supports kernel nodes only");
    }
    if (!node.kernel.function.valid()) {
      static_cast<void>(cuGraphDestroy(graph));
      return Status::invalid_argument(
          "kernel node is missing a resolved function handle");
    }

    // Buffer-format kernargs: a stable copy of the (already-relocated) flat
    // kernarg blob, sized to itself, referenced via the extra config array.
    auto param = std::make_unique<NodeParam>();
    param->blob = node.kernel.param_blob;
    param->size = param->blob.size();
    param->config[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
    param->config[1] = param->blob.data();
    param->config[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
    param->config[3] = &param->size;
    param->config[4] = CU_LAUNCH_PARAM_END;

    CUDA_KERNEL_NODE_PARAMS kparams{};
    kparams.func = as_handle<CUfunction>(node.kernel.function);
    kparams.gridDimX = node.kernel.grid.x;
    kparams.gridDimY = node.kernel.grid.y;
    kparams.gridDimZ = node.kernel.grid.z;
    kparams.blockDimX = node.kernel.block.x;
    kparams.blockDimY = node.kernel.block.y;
    kparams.blockDimZ = node.kernel.block.z;
    kparams.sharedMemBytes = node.kernel.shared_mem_bytes;
    kparams.kernelParams = nullptr;
    kparams.extra = param->config;

    std::vector<CUgraphNode> deps;
    for (const GraphEdgeIR& edge : ir.edges) {
      if (edge.to == node.id) {
        auto it = node_by_id.find(edge.from);
        if (it != node_by_id.end()) {
          deps.push_back(it->second);
        }
      }
    }

    CUgraphNode graph_node{};
    status = cu_status(
        cuGraphAddKernelNode(&graph_node, graph, deps.data(), deps.size(),
                             &kparams),
        "cuGraphAddKernelNode");
    if (!status.ok()) {
      static_cast<void>(cuGraphDestroy(graph));
      return status;
    }
    node_by_id[node.id] = graph_node;
    side.nodes.push_back(std::move(param));
  }

  graph_side_registry()[reinterpret_cast<std::uintptr_t>(graph)] =
      std::move(side);
  *out = from_handle<GraphHandle>(graph);
  return Status::Ok();
}
```

- [ ] **Step 3: Free the side data in `instantiate`**

In `CudaBackend::instantiate`, after the successful `cuGraphInstantiateWithFlags` and before `*out = ...`, add:
```cpp
  // The executable graph owns baked copies of every node's params now, so the
  // rebuild-time kernarg buffers can be released.
  graph_side_registry().erase(
      reinterpret_cast<std::uintptr_t>(as_handle<CUgraph>(graph)));
```

- [ ] **Step 4: Add the verify/restore/bench gates to the build sbatch**

In `snapshot/recipe/build_snapshot_cuda.sbatch`, after the `capture` lines in the heredoc, add:
```bash
    echo "--- verify (single-process bit-identical, relocation exercised) ---"
    snapshot/build-cuda/snapshot verify snapshot/build-cuda/verify.snap
    echo "--- two-process: capture then restore in a fresh process ---"
    snapshot/build-cuda/snapshot capture snapshot/build-cuda/two.snap
    snapshot/build-cuda/snapshot restore snapshot/build-cuda/two.snap
    echo "--- bench (cold capture vs warm restore) ---"
    snapshot/build-cuda/snapshot bench snapshot/build-cuda/bench.snap --scaled
```

- [ ] **Step 5: Run and verify bit-identical**

Run:
```bash
rcc --profile bristen push
ssh bristen 'cd /capstor/scratch/cscs/xyao/<REMOTE_DIR> && sbatch snapshot/recipe/build_snapshot_cuda.sbatch'
ssh bristen 'tail -n 40 /capstor/scratch/cscs/xyao/<REMOTE_DIR>/logs/snapshot-build-cuda-*.out'
```
Expected `verify` output:
```
captured_base=0x600000000000
restored_base=0x...            (a DIFFERENT base — capture region kept mapped)
relocation_delta_nonzero=1
known_patches=6
capture_matches_reference=1
restore_matches_reference=1
restore_matches_capture=1
verify ok
```
Expected `restore` (two-process) output: `bit_identical_vs_reference=1`. Expected `bench`: prints `cold_capture_ms`, `warm_restore_ms`, `speedup=...`. All commands exit 0. `known_patches=6` confirms the 3+2+1 synthetic pointers were relocated by Δ and the rebuilt graph is bit-identical.

- [ ] **Step 6: Commit**

```bash
git add snapshot/csrc/backends/cuda/cuda_graph.cpp snapshot/recipe/build_snapshot_cuda.sbatch
git commit -m "snapshot(cuda): N1 Task 4 — rebuild_graph + restore path (verify bit-identical on A100)"
```

---

## Task 5: Full ctest green + RESULTS section + AMD regression check

Bring the GPU ctests online on CUDA, document the N1 result, and confirm the HIP path is untouched.

**Files:**
- Modify: `snapshot/recipe/build_snapshot_cuda.sbatch` (run full `ctest`, no exclusion)
- Modify: `snapshot/RESULTS.md` (add an "N1 — CUDA backend on A100" section)

**Interfaces:**
- Consumes: the working `CudaBackend` (Tasks 2–4).
- Produces: a green full `ctest` on the CUDA build; a documented result; a verified-clean HIP diff.

- [ ] **Step 1: Run the full ctest (including GPU tests) on CUDA**

Edit `snapshot/recipe/build_snapshot_cuda.sbatch`: change the host-only `ctest --test-dir snapshot/build-cuda -E "_gpu" ...` line to:
```bash
    echo "--- full ctest (host + GPU) on CUDA ---"
    ctest --test-dir snapshot/build-cuda --output-on-failure
```
Run the job and confirm all tests pass:
```bash
rcc --profile bristen push
ssh bristen 'cd /capstor/scratch/cscs/xyao/<REMOTE_DIR> && sbatch snapshot/recipe/build_snapshot_cuda.sbatch'
ssh bristen 'grep -E "tests passed|failed" /capstor/scratch/cscs/xyao/<REMOTE_DIR>/logs/snapshot-build-cuda-*.out | tail'
```
Expected: `100% tests passed` (now including `test_graph_capture_gpu` and `test_e2e_roundtrip_gpu`, which run the real CUDA backend via `make_backend()`). If a GPU test fails, debug it with `superpowers:systematic-debugging` before proceeding — the vendor-neutral test is the oracle.

- [ ] **Step 2: Verify the HIP / core paths are untouched (regression-by-construction)**

Run:
```bash
git diff --stat cd5da1b -- snapshot/csrc/backends/hip snapshot/csrc/preload snapshot/csrc/core snapshot/include snapshot/csrc/cli
```
Expected: **no output** (zero changes to HIP backend, preload interposers, core, headers, or CLI). The only changed/added files are under `snapshot/csrc/backends/cuda/`, the CUDA branch of `snapshot/cmake/SnapshotBackend.cmake`, `snapshot/recipe/snapshot-cuda.toml`, `snapshot/recipe/build_snapshot_cuda.sbatch`, and `snapshot/RESULTS.md`. This is the AMD regression invariant verified mechanically.

(Optional, recommended) Confirm the AMD build still passes by submitting the beverin build job — it should be unaffected:
```bash
ssh beverin 'cd /capstor/scratch/cscs/xyao/kimi-k25-vllm && sbatch snapshot/recipe/build_snapshot.sbatch'
# expect: ctest 5/5, probe-base honored=1, verify ok (unchanged)
```

- [ ] **Step 3: Add the N1 RESULTS section**

Append to `snapshot/RESULTS.md` a section documenting: the A100/bristen environment (CUDA version, sm_80), `probe-base` honored result, `verify` (Δ≠0, known_patches=6, bit-identical), two-process `restore` bit-identical, `bench` numbers, full `ctest` green, and the table of CUDA↔HIP API equivalences proven (VMM, graph capture/rebuild, nvrtc). Keep the format consistent with the existing AMD milestone sections. Note explicitly that the HIP path is unchanged (build-time `SNAPSHOT_BACKEND` selection) and that N2 (CUDA interposers) is the next milestone.

- [ ] **Step 4: Commit**

```bash
git add snapshot/recipe/build_snapshot_cuda.sbatch snapshot/RESULTS.md
git commit -m "snapshot(cuda): N1 complete — full ctest green on A100, RESULTS documented, HIP untouched"
```

---

## Self-Review

**Spec coverage (against the §4 Layer-1 design):**
- CudaBackend VMM (`cuMemAddressReserve` fixed base + one `cuMemSetAccess`) → Task 2 ✓
- arch → `sm_80` → Task 2 ✓
- module load/get_function, capture/introspect/rebuild/instantiate/launch, launch_kernel, memcpy → Tasks 3–4 ✓
- nvrtc synthetic module (same exact-uint kernels) → Task 3 ✓
- M1-analog gates (probe-base / verify / capture→restore / bench, ctest) → Tasks 2–5 ✓
- Dual-vendor build-time selection + AMD untouched → Global Constraints + Task 1 + Task 5 Step 2 ✓
- Build harness on bristen (EDF + sbatch, run on compute node) → Task 1 ✓

**Placeholder scan:** the only intentional placeholder is `<REMOTE_DIR>` (the bristen scratch path), which the implementer fills from the `rcc` bristen profile; flagged at first use. The image tag in `snapshot-cuda.toml` is resolved empirically in Task 1 Step 1. No "TBD/TODO/handle edge cases" — all code is complete.

**Type consistency:** `ensure_cuda_context()` is declared in `cuda_backend.hpp` (Task 2 Step 1), defined in `cuda_vmm.cpp` (Task 2 Step 2), and called in `cuda_graph.cpp` (Task 3) and `cuda_backend.cpp` (Task 3). `cu_status` is duplicated per-TU (matches HIP). `NodeParam`/`graph_side_registry` are defined and used only within `cuda_graph.cpp` (Task 4). Handle round-trips use the shared `as_handle`/`from_handle` helpers. The CUDA driver types (`CUdeviceptr`, `CUmemGenericAllocationHandle`, `CUDA_KERNEL_NODE_PARAMS`, `CUgraphNode`) are consistent across tasks.

**Scope:** N1 only (foundation). N2 (CUDA interposers), N3 (vLLM-CUDA deploy), N4 (skip-capture), N5 (snapshot/restore) are separate plans per the design's milestone table.

---

## Execution Handoff

Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks. Note: every gate runs on the bristen cluster, so each task's verification is an `sbatch` round-trip (~minutes); the reviewer checks the SLURM log before approving.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
