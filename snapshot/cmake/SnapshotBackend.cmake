set(SNAPSHOT_BACKEND "AUTO" CACHE STRING "Snapshot GPU backend: AUTO, HIP, CUDA, or STUB")
set_property(CACHE SNAPSHOT_BACKEND PROPERTY STRINGS AUTO HIP CUDA STUB)

string(TOUPPER "${SNAPSHOT_BACKEND}" SNAPSHOT_BACKEND)

if(SNAPSHOT_BACKEND STREQUAL "AUTO")
  find_program(SNAPSHOT_HIPCC hipcc HINTS "$ENV{ROCM_PATH}/bin" "/opt/rocm/bin")
  find_program(SNAPSHOT_NVCC nvcc HINTS "$ENV{CUDA_HOME}/bin" "/usr/local/cuda/bin")

  if(SNAPSHOT_HIPCC)
    set(SNAPSHOT_BACKEND "HIP")
  elseif(SNAPSHOT_NVCC)
    set(SNAPSHOT_BACKEND "CUDA")
  else()
    set(SNAPSHOT_BACKEND "STUB")
  endif()
endif()

if(NOT SNAPSHOT_BACKEND MATCHES "^(HIP|CUDA|STUB)$")
  message(FATAL_ERROR "SNAPSHOT_BACKEND must be AUTO, HIP, CUDA, or STUB")
endif()

message(STATUS "Snapshot backend: ${SNAPSHOT_BACKEND}")

function(snapshot_configure_hip target)
  find_path(SNAPSHOT_HIP_INCLUDE_DIR
    NAMES hip/hip_runtime_api.h
    HINTS "$ENV{ROCM_PATH}/include" "/opt/rocm/include")
  find_library(SNAPSHOT_HIP_LIBRARY
    NAMES amdhip64
    HINTS "$ENV{ROCM_PATH}/lib" "$ENV{ROCM_PATH}/lib64" "/opt/rocm/lib" "/opt/rocm/lib64")
  find_library(SNAPSHOT_HIPRTC_LIBRARY
    NAMES hiprtc
    HINTS "$ENV{ROCM_PATH}/lib" "$ENV{ROCM_PATH}/lib64" "/opt/rocm/lib" "/opt/rocm/lib64")

  if(NOT SNAPSHOT_HIP_INCLUDE_DIR OR NOT SNAPSHOT_HIP_LIBRARY)
    message(FATAL_ERROR "SNAPSHOT_BACKEND=HIP requires ROCm HIP headers and libamdhip64")
  endif()
  if(NOT SNAPSHOT_HIPRTC_LIBRARY)
    message(FATAL_ERROR "SNAPSHOT_BACKEND=HIP requires libhiprtc for synthetic kernel compilation")
  endif()

  target_include_directories(${target} PRIVATE "${SNAPSHOT_HIP_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE "${SNAPSHOT_HIP_LIBRARY}" "${SNAPSHOT_HIPRTC_LIBRARY}")
  # The HIP backend uses only host-side HIP API (no __global__ kernels at M1),
  # so it is compiled by the plain CXX compiler rather than hipcc. The HIP
  # headers then require the platform to be selected explicitly.
  target_compile_definitions(${target} PRIVATE __HIP_PLATFORM_AMD__=1)
endfunction()

function(snapshot_configure_cuda target)
  find_path(SNAPSHOT_CUDA_INCLUDE_DIR
    NAMES cuda.h
    HINTS "$ENV{CUDA_HOME}/include" "/usr/local/cuda/include")
  find_library(SNAPSHOT_CUDA_DRIVER_LIBRARY
    NAMES cuda
    HINTS "$ENV{CUDA_HOME}/lib64" "/usr/local/cuda/lib64" "/usr/lib/x86_64-linux-gnu")

  if(NOT SNAPSHOT_CUDA_INCLUDE_DIR OR NOT SNAPSHOT_CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR "SNAPSHOT_BACKEND=CUDA requires cuda.h and libcuda")
  endif()

  target_include_directories(${target} PRIVATE "${SNAPSHOT_CUDA_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE "${SNAPSHOT_CUDA_DRIVER_LIBRARY}")
endfunction()
