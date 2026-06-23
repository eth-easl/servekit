#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "snapshot/handles.hpp"
#include "snapshot/status.hpp"

namespace snapshot {

enum class Vendor : std::uint32_t {
  kStub = 0,
  kHip = 1,
  kCuda = 2,
};

struct ArchInfo {
  std::string name;
};

struct Dim3 {
  std::uint32_t x = 1;
  std::uint32_t y = 1;
  std::uint32_t z = 1;
};

struct KernelLaunchParams {
  FunctionHandle function;
  Dim3 grid;
  Dim3 block;
  std::uint32_t shared_mem_bytes = 0;
  std::vector<std::byte> param_blob;
  std::vector<std::uint32_t> ptr_offsets;
};

enum class GraphNodeType : std::uint32_t {
  kKernel = 1,
  kMemcpyH2D = 2,
  kMemcpyD2H = 3,
  kMemcpyD2D = 4,
  kMemset = 5,
};

struct GraphNodeIR {
  std::uint64_t id = 0;
  GraphNodeType type = GraphNodeType::kKernel;
  std::uint64_t module_hash = 0;
  std::string entry_name;
  KernelLaunchParams kernel;
  std::uint64_t dst = 0;
  std::uint64_t src = 0;
  std::uint64_t bytes = 0;
  std::uint32_t memset_value = 0;
};

struct GraphEdgeIR {
  std::uint64_t from = 0;
  std::uint64_t to = 0;
};

struct GraphIR {
  std::vector<GraphNodeIR> nodes;
  std::vector<GraphEdgeIR> edges;
};

struct ModuleImage {
  std::uint64_t hash = 0;
  std::vector<std::byte> image;
  std::vector<std::string> entry_names;
};

struct MemoryAccess {
  int device_ordinal = 0;
  bool read = true;
  bool write = true;
};

class GpuBackend {
 public:
  virtual ~GpuBackend() = default;

  virtual Vendor vendor() const = 0;
  virtual Status arch(ArchInfo* out) = 0;

  virtual Status get_allocation_granularity(std::uint64_t* out) = 0;
  virtual Status reserve_address(std::uint64_t size, std::uint64_t alignment,
                                 std::uint64_t requested_base,
                                 std::uint64_t* out_base) = 0;
  virtual Status release_address(std::uint64_t base, std::uint64_t size) = 0;
  virtual Status create_physical(std::uint64_t size, MemHandle* out) = 0;
  virtual Status release_physical(MemHandle handle) = 0;
  virtual Status map(std::uint64_t va, std::uint64_t size,
                     std::uint64_t offset, MemHandle handle) = 0;
  virtual Status unmap(std::uint64_t va, std::uint64_t size) = 0;
  virtual Status set_access(std::uint64_t va, std::uint64_t size,
                            const MemoryAccess& access) = 0;

  virtual Status load_module(const std::byte* image, std::size_t n,
                             ModuleHandle* out) = 0;
  virtual Status unload_module(ModuleHandle module) = 0;
  virtual Status get_function(ModuleHandle module, const std::string& name,
                              FunctionHandle* out) = 0;

  virtual Status stream_create(StreamHandle* out) = 0;
  virtual Status stream_destroy(StreamHandle stream) = 0;
  virtual Status begin_capture(StreamHandle stream) = 0;
  virtual Status end_capture(StreamHandle stream, GraphHandle* out_graph) = 0;
  virtual Status introspect_graph(GraphHandle graph, GraphIR* out) = 0;
  virtual Status rebuild_graph(const GraphIR& ir, GraphHandle* out) = 0;
  virtual Status instantiate(GraphHandle graph, GraphExecHandle* out) = 0;
  virtual Status exec_set_kernel_node_params(GraphExecHandle exec,
                                             std::uint64_t node_id,
                                             const KernelLaunchParams& params) = 0;
  virtual Status launch(GraphExecHandle exec, StreamHandle stream) = 0;
  virtual Status synchronize(StreamHandle stream) = 0;

  virtual Status launch_kernel(StreamHandle stream,
                               const KernelLaunchParams& params) = 0;
  virtual Status memcpy_h2d(std::uint64_t dst_device,
                            const void* src_host, std::uint64_t bytes,
                            StreamHandle stream) = 0;
  virtual Status memcpy_d2h(void* dst_host, std::uint64_t src_device,
                            std::uint64_t bytes, StreamHandle stream) = 0;
};

std::unique_ptr<GpuBackend> make_backend();
// Always returns the host-only stub backend regardless of the compiled-in GPU
// backend, so host-only logic tests stay deterministic and device-independent.
std::unique_ptr<GpuBackend> make_stub_backend();
const char* vendor_name(Vendor vendor);
Vendor vendor_from_name(const std::string& name);

}  // namespace snapshot
