#pragma once

#include <cstdint>

namespace snapshot {

struct OpaqueHandle {
  std::uintptr_t value = 0;

  bool valid() const { return value != 0; }
  explicit operator bool() const { return valid(); }
};

struct ModuleHandle : OpaqueHandle {};
struct FunctionHandle : OpaqueHandle {};
struct GraphHandle : OpaqueHandle {};
struct GraphExecHandle : OpaqueHandle {};
struct MemHandle : OpaqueHandle {};
struct StreamHandle : OpaqueHandle {};

}  // namespace snapshot
