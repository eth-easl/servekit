cmake_minimum_required(VERSION 3.20)

project(snapshot LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SnapshotBackend)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# The core static lib is linked into shared interposer libraries, so build all
# objects position-independent.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(snapshot_core
  csrc/core/allocator.cpp
  csrc/core/capture.cpp
  csrc/core/graph_model.cpp
  csrc/core/hashing.cpp
  csrc/core/record.cpp
  csrc/core/relocation.cpp
  csrc/core/restore.cpp
  csrc/core/serialize.cpp
  csrc/backends/backend_factory.cpp
)

target_include_directories(snapshot_core
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_definitions(snapshot_core PUBLIC SNAPSHOT_BACKEND_${SNAPSHOT_BACKEND}=1)

if(SNAPSHOT_BACKEND STREQUAL "HIP")
  target_sources(snapshot_core PRIVATE
    csrc/backends/hip/hip_backend.cpp
    csrc/backends/hip/hip_graph.cpp
    csrc/backends/hip/hip_vmm.cpp
    csrc/backends/hip/hip_kernels.cpp
  )
  snapshot_configure_hip(snapshot_core)

  # LD_PRELOAD interposer: standalone shared object, not linked into the core.
  # Forwards to real HIP via dlsym(RTLD_NEXT); used to observe/measure graph
  # capture in any HIP application (including vLLM) without modifying it.
  add_library(snapshot_preload SHARED csrc/preload/snapshot_preload.cpp)
  snapshot_configure_hip(snapshot_preload)
  target_link_libraries(snapshot_preload PRIVATE ${CMAKE_DL_LIBS})

  # LD_PRELOAD redirector: serves hipMalloc from the deterministic VMM region so
  # an unmodified application's device addresses become reproducible.
  add_library(snapshot_redirect SHARED csrc/preload/snapshot_redirect.cpp)
  snapshot_configure_hip(snapshot_redirect)
  target_link_libraries(snapshot_redirect PRIVATE snapshot_core ${CMAKE_DL_LIBS})

  # LD_PRELOAD recorder: interposes module-load / get-function / launch /
  # capture / alloc, and serializes each captured graph's IR to a snapshot
  # file via snapshot_core. The recorder (not the workload) is the interceptor,
  # so the same mechanism that captures our synthetic workload also captures a
  # real vLLM cold start. Load order with redirect is LD_PRELOAD="redirect record"
  # so redirect serves hipMalloc and record observes identity + launches.
  add_library(snapshot_record SHARED csrc/preload/snapshot_record.cpp)
  snapshot_configure_hip(snapshot_record)
  target_link_libraries(snapshot_record PRIVATE snapshot_core ${CMAKE_DL_LIBS})

  # Standalone raw-hipMalloc GPU program to demonstrate transparent redirect.
  add_executable(redirect_smoke csrc/cli/redirect_smoke.cpp)
  snapshot_configure_hip(redirect_smoke)
  target_link_libraries(redirect_smoke PRIVATE snapshot_core)
elseif(SNAPSHOT_BACKEND STREQUAL "CUDA")
  target_sources(snapshot_core PRIVATE
    csrc/backends/cuda/cuda_backend.cpp
    csrc/backends/cuda/cuda_graph.cpp
    csrc/backends/cuda/cuda_vmm.cpp
  )
  snapshot_configure_cuda(snapshot_core)
endif()

add_executable(snapshot
  csrc/cli/main.cpp
  csrc/cli/workload.cpp
  csrc/cli/workload_kernels.cpp
)
target_link_libraries(snapshot PRIVATE snapshot_core)

# M3b diagnostic: load a captured code object via hipModuleLoadData to isolate
# whether a rebuild "no kernel image available" is the image vs. the wiring.
if(SNAPSHOT_BACKEND STREQUAL "HIP")
  add_executable(modload_test csrc/cli/modload_test.cpp)
  snapshot_configure_hip(modload_test)
endif()

enable_testing()

function(snapshot_add_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE snapshot_core)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

snapshot_add_test(test_allocator)
snapshot_add_test(test_relocation)
snapshot_add_test(test_serialize_roundtrip)
snapshot_add_test(test_record_ir)
snapshot_add_test(test_amdgpu_msgpack)
snapshot_add_test(test_graph_capture_gpu)
snapshot_add_test(test_e2e_roundtrip_gpu)
