#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "snapshot/status.hpp"

namespace snapshot {

// Compiles the synthetic workload kernel module for the active backend
// (hiprtc on HIP, nvrtc on CUDA) and returns a loadable code object plus the
// exported entry-point names. The code object is what gets serialized into a
// snapshot and reloaded by a fresh process via GpuBackend::load_module.
//
// Returns a kUnsupported status on the stub backend, which has no real device.
//
// The synthetic module exports three kernels, all over unsigned 32-bit
// elements so the pipeline is bit-identical by construction:
//   mul_bias(uint* a, uint* b, uint* c, int bias, uint n)  -> c = a*b + bias
//   relu_offset(uint* c, uint* out, int offset, uint n)    -> out = c + offset
//   in_place(uint* out, uint n)                            -> out ^= 0x9e3779b9
Status compile_synthetic_module(std::vector<std::byte>* image,
                                std::vector<std::string>* entry_names);

}  // namespace snapshot
