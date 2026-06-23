#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "snapshot/allocator.hpp"
#include "snapshot/gpu_backend.hpp"
#include "snapshot/status.hpp"

namespace snapshot {

constexpr std::uint32_t kSnapshotVersion = 1;

enum class SectionTag : std::uint32_t {
  kAllocLog = 1,
  kModules = 2,
  kGraphNodes = 3,
  kGraphEdges = 4,
};

struct SnapshotData {
  Vendor vendor = Vendor::kStub;
  std::string arch;
  AllocatorState allocator;
  std::vector<ModuleImage> modules;
  GraphIR graph;
};

std::uint32_t crc32(const std::byte* data, std::size_t n);
std::uint64_t hash_bytes(const std::byte* data, std::size_t n);

Status write_snapshot_file(const std::string& path, const SnapshotData& data);
Status read_snapshot_file(const std::string& path, SnapshotData* out);

Status validate_snapshot_for_backend(const SnapshotData& data,
                                     GpuBackend& backend);

}  // namespace snapshot
