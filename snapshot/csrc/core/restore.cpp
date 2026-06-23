#include "snapshot/restore.hpp"

#include <unordered_map>

#include "snapshot/record.hpp"

namespace snapshot {

RestoreSession::RestoreSession(GpuBackend& backend) : backend_(backend) {}

Status RestoreSession::restore(const SnapshotData& snapshot,
                               RestoreResult* out) {
  if (out == nullptr) {
    return Status::invalid_argument("restore output is null");
  }

  Status status = validate_snapshot_for_backend(snapshot, backend_);
  if (!status.ok()) {
    return status;
  }

  std::uint64_t restored_base = 0;
  status = allocator_.replay(backend_, snapshot.allocator,
                             snapshot.allocator.region_base, &restored_base);
  if (!status.ok()) {
    return status;
  }

  GraphIR relocated = snapshot.graph;

  // Populate precise pointer offsets from parsed AMDGPU kernel signatures so
  // relocation patches only real pointer args (not blind-scanned scalars).
  // Mirrors rebuild_check_snapshot. Best-effort: unmatched nodes fall back to
  // blind-scan inside relocate_graph_ir.
  std::unordered_map<std::string, KernelSig> sig_by_name;
  for (const ModuleImage& module : snapshot.modules) {
    if (module.image.empty()) continue;
    for (const KernelSig& ks : extract_amdgpu_kernels(module.image.data(),
                                                     module.image.size())) {
      sig_by_name[ks.name] = ks;
    }
  }
  for (GraphNodeIR& node : relocated.nodes) {
    if (node.type != GraphNodeType::kKernel || node.entry_name.empty()) continue;
    auto sit = sig_by_name.find(node.entry_name);
    if (sit == sig_by_name.end()) continue;
    auto offs = tag1_blob_ptr_offsets(node.kernel.param_blob, sit->second);
    if (!offs.empty()) {
      node.kernel.ptr_offsets = std::move(offs);
    }
  }

  RelocationStats stats;
  status = relocate_graph_ir(
      &relocated,
      Relocation{snapshot.allocator.region_base, restored_base,
                 snapshot.allocator.region_size},
      true, &stats);
  if (!status.ok()) {
    return status;
  }

  // Load all modules first. Like rebuild_check_snapshot, we do NOT assume a
  // node's function lives in the module its node.module_hash names: multi-
  // target fat binaries (host-registered kernels, Triton code objects) can have
  // a function's entry in a sibling ELF. So we collect distinct entry names and
  // resolve each against any module that contains it.
  std::vector<ModuleHandle> module_handles;
  module_handles.reserve(snapshot.modules.size());
  for (const ModuleImage& module : snapshot.modules) {
    ModuleHandle module_handle;
    status = backend_.load_module(module.image.data(), module.image.size(),
                                  &module_handle);
    if (!status.ok()) {
      return status;
    }
    module_handles.push_back(module_handle);
  }

  // Collect the distinct entry names the graph references.
  std::vector<std::string> referenced;
  for (const GraphNodeIR& node : relocated.nodes) {
    if (node.type != GraphNodeType::kKernel || node.entry_name.empty()) {
      continue;
    }
    bool seen = false;
    for (const std::string& n : referenced) {
      if (n == node.entry_name) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      referenced.push_back(node.entry_name);
    }
  }

  // Resolve each entry against any module that contains it.
  for (const std::string& entry : referenced) {
    FunctionHandle function;
    bool resolved = false;
    for (ModuleHandle module_handle : module_handles) {
      status = backend_.get_function(module_handle, entry, &function);
      if (status.ok()) {
        resolved = true;
        break;
      }
    }
    if (!resolved) {
      return Status::format("entry '" + entry +
                            "' not found in any loaded module");
    }
    for (GraphNodeIR& node : relocated.nodes) {
      if (node.type == GraphNodeType::kKernel && node.entry_name == entry) {
        node.kernel.function = function;
      }
    }
  }

  GraphHandle graph;
  status = backend_.rebuild_graph(relocated, &graph);
  if (!status.ok()) {
    return status;
  }

  GraphExecHandle exec;
  status = backend_.instantiate(graph, &exec);
  if (!status.ok()) {
    return status;
  }

  out->restored_base = restored_base;
  out->relocation_stats = stats;
  out->graph = graph;
  out->exec = exec;
  return Status::Ok();
}

}  // namespace snapshot
