#include "snapshot/capture.hpp"

namespace snapshot {

CaptureSession::CaptureSession(GpuBackend& backend, std::uint64_t region_size)
    : backend_(backend), region_size_(region_size) {}

Status CaptureSession::begin() {
  return allocator_.init(backend_, region_size_);
}

Status CaptureSession::finish(const std::vector<ModuleImage>& modules,
                              GraphHandle graph, SnapshotData* out) {
  if (out == nullptr) {
    return Status::invalid_argument("capture finish output is null");
  }

  GraphIR ir;
  Status status = backend_.introspect_graph(graph, &ir);
  if (!status.ok()) {
    return status;
  }

  ArchInfo arch;
  status = backend_.arch(&arch);
  if (!status.ok()) {
    return status;
  }

  out->vendor = backend_.vendor();
  out->arch = arch.name;
  out->allocator = allocator_.state();
  out->modules = modules;
  out->graph = ir;
  return Status::Ok();
}

}  // namespace snapshot
