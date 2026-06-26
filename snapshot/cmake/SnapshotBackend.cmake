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
  snapshot_detect_launch_ex()
  if(SNAPSHOT_HAS_LAUNCH_EX_CACHE)
    target_compile_definitions(${target} PRIVATE SNAPSHOT_HAS_LAUNCH_EX=1)
  endif()
endfunction()

# Detect whether the ROCm headers define hipLaunchConfig_t / HIP_LAUNCH_CONFIG
# (added in newer HIP; absent from some ROCm 6.3 builds). The recorder guards
# the hipLaunchKernelExC / hipDrvLaunchKernelEx interposers behind this define.
include(CheckCXXSourceCompiles)
function(snapshot_detect_launch_ex)
  if(DEFINED SNAPSHOT_HAS_LAUNCH_EX_CACHE)
    return()
  endif()
  set(SNAPSHOT_HAS_LAUNCH_EX_CACHE OFF CACHE INTERNAL "")
  if(SNAPSHOT_HIP_INCLUDE_DIR)
    set(CMAKE_REQUIRED_INCLUDES "${SNAPSHOT_HIP_INCLUDE_DIR}")
    set(CMAKE_REQUIRED_DEFINITIONS "-D__HIP_PLATFORM_AMD__=1")
    check_cxx_source_compiles("
      #include <hip/hip_runtime_api.h>
      hipLaunchConfig_t cfg;
      int main() { (void)cfg; return 0; }
    " SNAPSHOT_LAUNCH_EX_TEST)
    if(SNAPSHOT_LAUNCH_EX_TEST)
      set(SNAPSHOT_HAS_LAUNCH_EX_CACHE ON CACHE INTERNAL "")
      message(STATUS "Snapshot: hipLaunchConfig_t available (SNAPSHOT_HAS_LAUNCH_EX)")
    else()
      message(STATUS "Snapshot: hipLaunchConfig_t absent (guarding launch-ex interposers)")
    endif()
    set(CMAKE_REQUIRED_INCLUDES "")
    set(CMAKE_REQUIRED_DEFINITIONS "")
  endif()
endfunction()

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
