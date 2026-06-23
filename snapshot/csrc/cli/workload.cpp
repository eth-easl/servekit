#include "workload.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <unordered_map>
#include <vector>

#include "snapshot/allocator.hpp"
#include "snapshot/gpu_backend.hpp"
#include "snapshot/record.hpp"
#include "snapshot/relocation.hpp"
#include "snapshot/restore.hpp"
#include "snapshot/workload_kernels.hpp"

namespace snapshot::cli {
namespace {

constexpr std::uint64_t kBufferBytes = 16ULL * 1024ULL * 1024ULL;
constexpr std::uint32_t kElems = static_cast<std::uint32_t>(kBufferBytes / 4);
constexpr std::uint64_t kRegionSize = 4ULL * kBufferBytes;
constexpr std::uint32_t kBlock = 256;
constexpr std::int32_t kBias = 17;
constexpr std::int32_t kOffset = 3;
constexpr std::uint32_t kXor = 0x9e3779b9u;
constexpr int kBenchWarmupChains = 64;  // models per-shape eager warmup

void append_u64(std::vector<std::byte>* out, std::uint64_t value) {
  for (int i = 0; i < 8; ++i) {
    out->push_back(static_cast<std::byte>((value >> (i * 8)) & 0xFFU));
  }
}

void append_u32(std::vector<std::byte>* out, std::uint32_t value) {
  for (int i = 0; i < 4; ++i) {
    out->push_back(static_cast<std::byte>((value >> (i * 8)) & 0xFFU));
  }
}

void append_i32(std::vector<std::byte>* out, std::int32_t value) {
  append_u32(out, static_cast<std::uint32_t>(value));
}

void pad_to(std::vector<std::byte>* out, std::size_t size) {
  while (out->size() < size) {
    out->push_back(std::byte{0});
  }
}

struct Buffers {
  std::uint64_t a = 0;
  std::uint64_t b = 0;
  std::uint64_t c = 0;
  std::uint64_t out = 0;
};

Buffers buffers_at(std::uint64_t base) {
  return Buffers{base + 0 * kBufferBytes, base + 1 * kBufferBytes,
                 base + 2 * kBufferBytes, base + 3 * kBufferBytes};
}

Dim3 grid_for(std::uint32_t n) {
  return Dim3{(n + kBlock - 1) / kBlock, 1, 1};
}

std::uint64_t synthetic_module_hash(const ModuleImage& module) {
  return module.hash;
}

// mul_bias(uint* a, uint* b, uint* c, int bias, uint n) -> kernarg 32 bytes.
GraphNodeIR make_mul_bias(std::uint64_t id, std::uint64_t hash,
                          const Buffers& bufs) {
  GraphNodeIR node;
  node.id = id;
  node.type = GraphNodeType::kKernel;
  node.module_hash = hash;
  node.entry_name = "mul_bias";
  node.kernel.grid = grid_for(kElems);
  node.kernel.block = Dim3{kBlock, 1, 1};
  node.kernel.ptr_offsets = {0, 8, 16};
  append_u64(&node.kernel.param_blob, bufs.a);
  append_u64(&node.kernel.param_blob, bufs.b);
  append_u64(&node.kernel.param_blob, bufs.c);
  append_i32(&node.kernel.param_blob, kBias);
  append_u32(&node.kernel.param_blob, kElems);
  pad_to(&node.kernel.param_blob, 32);
  return node;
}

// relu_offset(uint* c, uint* out, int offset, uint n) -> kernarg 24 bytes.
GraphNodeIR make_relu_offset(std::uint64_t id, std::uint64_t hash,
                             const Buffers& bufs) {
  GraphNodeIR node;
  node.id = id;
  node.type = GraphNodeType::kKernel;
  node.module_hash = hash;
  node.entry_name = "relu_offset";
  node.kernel.grid = grid_for(kElems);
  node.kernel.block = Dim3{kBlock, 1, 1};
  node.kernel.ptr_offsets = {0, 8};
  append_u64(&node.kernel.param_blob, bufs.c);
  append_u64(&node.kernel.param_blob, bufs.out);
  append_i32(&node.kernel.param_blob, kOffset);
  append_u32(&node.kernel.param_blob, kElems);
  pad_to(&node.kernel.param_blob, 24);
  return node;
}

// in_place(uint* out, uint n) -> kernarg 16 bytes.
GraphNodeIR make_in_place(std::uint64_t id, std::uint64_t hash,
                          const Buffers& bufs) {
  GraphNodeIR node;
  node.id = id;
  node.type = GraphNodeType::kKernel;
  node.module_hash = hash;
  node.entry_name = "in_place";
  node.kernel.grid = grid_for(kElems);
  node.kernel.block = Dim3{kBlock, 1, 1};
  node.kernel.ptr_offsets = {0};
  append_u64(&node.kernel.param_blob, bufs.out);
  append_u32(&node.kernel.param_blob, kElems);
  pad_to(&node.kernel.param_blob, 16);
  return node;
}

// Builds the serializable graph IR (module/entry/params, no live handles). A
// single chain is mul_bias -> relu_offset -> in_place; --scaled repeats it to
// stress graph size. verify only uses one chain so the host reference is exact.
GraphIR build_workload_ir(std::uint64_t hash, const Buffers& bufs, bool scaled) {
  GraphIR ir;
  const int chains = scaled ? 64 : 1;
  std::uint64_t id = 0;
  for (int chain = 0; chain < chains; ++chain) {
    ir.nodes.push_back(make_mul_bias(++id, hash, bufs));
    if (id > 1) {
      ir.edges.push_back(GraphEdgeIR{id - 1, id});
    }
    ir.nodes.push_back(make_relu_offset(++id, hash, bufs));
    ir.edges.push_back(GraphEdgeIR{id - 1, id});
    ir.nodes.push_back(make_in_place(++id, hash, bufs));
    ir.edges.push_back(GraphEdgeIR{id - 1, id});
  }
  return ir;
}

std::vector<std::uint32_t> host_reference() {
  std::vector<std::uint32_t> out(kElems);
  for (std::uint32_t i = 0; i < kElems; ++i) {
    const std::uint32_t a = i;
    const std::uint32_t b = 2u * i + 1u;
    const std::uint32_t c = a * b + static_cast<std::uint32_t>(kBias);
    std::uint32_t o = c + static_cast<std::uint32_t>(kOffset);
    o ^= kXor;
    out[i] = o;
  }
  return out;
}

// Resolves every kernel node's live FunctionHandle from a loaded module so the
// chain can be issued (during capture) or launched directly (during warmup).
Status resolve_functions(GpuBackend& backend, const ModuleImage& module,
                         GraphIR* ir) {
  ModuleHandle module_handle;
  Status status =
      backend.load_module(module.image.data(), module.image.size(),
                          &module_handle);
  if (!status.ok()) {
    return status;
  }
  for (const std::string& entry : module.entry_names) {
    FunctionHandle function;
    status = backend.get_function(module_handle, entry, &function);
    if (!status.ok()) {
      return status;
    }
    for (GraphNodeIR& node : ir->nodes) {
      if (node.type == GraphNodeType::kKernel && node.entry_name == entry) {
        node.kernel.function = function;
      }
    }
  }
  return Status::Ok();
}

Status seed_inputs(GpuBackend& backend, const Buffers& bufs,
                   StreamHandle stream) {
  std::vector<std::uint32_t> a(kElems);
  std::vector<std::uint32_t> b(kElems);
  for (std::uint32_t i = 0; i < kElems; ++i) {
    a[i] = i;
    b[i] = 2u * i + 1u;
  }
  Status status = backend.memcpy_h2d(bufs.a, a.data(), kBufferBytes, stream);
  if (!status.ok()) {
    return status;
  }
  status = backend.memcpy_h2d(bufs.b, b.data(), kBufferBytes, stream);
  if (!status.ok()) {
    return status;
  }
  return backend.synchronize(stream);
}

// Captures the workload into a serializable snapshot. The allocator region is
// left mapped (caller owns release) so a same-process restore is forced to a
// different base, exercising relocation. If gpu_out != nullptr, the captured
// graph is launched once and OUT is read back.
Status capture_snapshot(GpuBackend& backend, DeterministicAllocator* allocator,
                        bool scaled, int warmup_chains, SnapshotData* snapshot,
                        std::vector<std::uint32_t>* gpu_out) {
  if (backend.vendor() == Vendor::kStub) {
    return Status::unsupported("real capture requires the HIP or CUDA backend");
  }

  ModuleImage module;
  Status status = compile_synthetic_module(&module.image, &module.entry_names);
  if (!status.ok()) {
    return status;
  }
  module.hash = hash_bytes(module.image.data(), module.image.size());

  status = allocator->init(backend, kRegionSize, kDefaultRequestedBase);
  if (!status.ok()) {
    return status;
  }
  const std::uint64_t base = allocator->state().region_base;
  std::uint64_t va = 0;
  for (const char* tag : {"A", "B", "C", "OUT"}) {
    status = allocator->alloc(backend, kBufferBytes, tag, &va);
    if (!status.ok()) {
      return status;
    }
  }
  const Buffers bufs = buffers_at(base);

  GraphIR live_ir = build_workload_ir(module.hash, bufs, scaled);
  status = resolve_functions(backend, module, &live_ir);
  if (!status.ok()) {
    return status;
  }

  StreamHandle stream;
  status = backend.stream_create(&stream);
  if (!status.ok()) {
    return status;
  }
  status = seed_inputs(backend, bufs, stream);
  if (!status.ok()) {
    return status;
  }

  // Warmup models the eager per-shape work that the original process performs
  // and a restored process skips entirely.
  for (int w = 0; w < warmup_chains; ++w) {
    for (const GraphNodeIR& node : live_ir.nodes) {
      status = backend.launch_kernel(stream, node.kernel);
      if (!status.ok()) {
        return status;
      }
    }
  }
  if (warmup_chains > 0) {
    status = backend.synchronize(stream);
    if (!status.ok()) {
      return status;
    }
  }

  status = backend.begin_capture(stream);
  if (!status.ok()) {
    return status;
  }
  for (const GraphNodeIR& node : live_ir.nodes) {
    status = backend.launch_kernel(stream, node.kernel);
    if (!status.ok()) {
      return status;
    }
  }
  GraphHandle graph;
  status = backend.end_capture(stream, &graph);
  if (!status.ok()) {
    return status;
  }

  GraphIR introspected;
  status = backend.introspect_graph(graph, &introspected);
  if (!status.ok()) {
    return status;
  }
  if (introspected.nodes.size() != live_ir.nodes.size()) {
    return Status::mismatch("introspected graph node count does not match the "
                            "recorded chain");
  }

  GraphExecHandle exec;
  status = backend.instantiate(graph, &exec);
  if (!status.ok()) {
    return status;
  }
  status = backend.launch(exec, stream);
  if (!status.ok()) {
    return status;
  }
  status = backend.synchronize(stream);
  if (!status.ok()) {
    return status;
  }

  if (gpu_out != nullptr) {
    gpu_out->assign(kElems, 0);
    status = backend.memcpy_d2h(gpu_out->data(), bufs.out, kBufferBytes, stream);
    if (!status.ok()) {
      return status;
    }
    status = backend.synchronize(stream);
    if (!status.ok()) {
      return status;
    }
  }

  ArchInfo arch;
  status = backend.arch(&arch);
  if (!status.ok()) {
    return status;
  }

  snapshot->vendor = backend.vendor();
  snapshot->arch = arch.name;
  snapshot->allocator = allocator->state();
  snapshot->modules = {module};
  snapshot->graph = build_workload_ir(module.hash, bufs, scaled);

  backend.stream_destroy(stream);
  return Status::Ok();
}

// Restores from a parsed snapshot, re-seeds inputs, launches the rebuilt graph,
// and reads OUT back. The restore session's allocator is released afterwards.
Status restore_and_run(GpuBackend& backend, const SnapshotData& snapshot,
                       std::vector<std::uint32_t>* gpu_out,
                       std::uint64_t* restored_base, RelocationStats* stats) {
  RestoreSession session(backend);
  RestoreResult result;
  Status status = session.restore(snapshot, &result);
  if (!status.ok()) {
    return status;
  }
  *restored_base = result.restored_base;
  *stats = result.relocation_stats;

  const Buffers bufs = buffers_at(result.restored_base);
  StreamHandle stream;
  status = backend.stream_create(&stream);
  if (!status.ok()) {
    session.allocator().release(backend);
    return status;
  }
  status = seed_inputs(backend, bufs, stream);
  if (status.ok()) {
    status = backend.launch(result.exec, stream);
  }
  if (status.ok()) {
    status = backend.synchronize(stream);
  }
  if (status.ok() && gpu_out != nullptr) {
    gpu_out->assign(kElems, 0);
    status = backend.memcpy_d2h(gpu_out->data(), bufs.out, kBufferBytes, stream);
    if (status.ok()) {
      status = backend.synchronize(stream);
    }
  }

  backend.stream_destroy(stream);
  session.allocator().release(backend);
  return status;
}

bool buffers_equal(const std::vector<std::uint32_t>& a,
                   const std::vector<std::uint32_t>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(std::uint32_t)) == 0;
}

}  // namespace

Status capture_synthetic(const std::string& path, bool scaled) {
  auto backend = make_backend();
  DeterministicAllocator allocator;
  SnapshotData snapshot;
  Status status = capture_snapshot(*backend, &allocator, scaled,
                                   /*warmup_chains=*/0, &snapshot,
                                   /*gpu_out=*/nullptr);
  if (status.ok()) {
    status = write_snapshot_file(path, snapshot);
  }
  // The captured region can be released now; the file is self-contained.
  allocator.release(*backend);
  return status;
}

Status restore_synthetic(const std::string& path, std::ostream& out) {
  auto backend = make_backend();
  SnapshotData snapshot;
  Status status = read_snapshot_file(path, &snapshot);
  if (!status.ok()) {
    return status;
  }

  std::vector<std::uint32_t> got;
  std::uint64_t restored_base = 0;
  RelocationStats stats;
  status = restore_and_run(*backend, snapshot, &got, &restored_base, &stats);
  if (!status.ok()) {
    return status;
  }

  const auto expected = host_reference();
  const bool ok = buffers_equal(expected, got);
  out << "vendor=" << vendor_name(snapshot.vendor) << "\n";
  out << "arch=" << snapshot.arch << "\n";
  out << "captured_base=0x" << std::hex << snapshot.allocator.region_base
      << "\n";
  out << "restored_base=0x" << restored_base << std::dec << "\n";
  out << "nodes=" << snapshot.graph.nodes.size() << "\n";
  out << "known_patches=" << stats.known_patches << "\n";
  out << "blind_patches=" << stats.blind_patches << "\n";
  out << "bit_identical_vs_reference=" << (ok ? 1 : 0) << "\n";
  return ok ? Status::Ok()
            : Status::mismatch("restored output does not match host reference");
}

Status verify_synthetic(const std::string& path, std::ostream& out) {
  auto backend = make_backend();
  if (backend->vendor() == Vendor::kStub) {
    return Status::unsupported("verify requires the HIP or CUDA backend");
  }

  // Capture, keeping the region mapped so restore is forced to a different base.
  DeterministicAllocator capture_allocator;
  SnapshotData snapshot;
  std::vector<std::uint32_t> captured_out;
  Status status = capture_snapshot(*backend, &capture_allocator, /*scaled=*/false,
                                   /*warmup_chains=*/0, &snapshot, &captured_out);
  if (!status.ok()) {
    capture_allocator.release(*backend);
    return status;
  }

  // Persist + reload so the file format is exercised on the verify path too.
  status = write_snapshot_file(path, snapshot);
  if (status.ok()) {
    SnapshotData reloaded;
    status = read_snapshot_file(path, &reloaded);
    if (status.ok()) {
      snapshot = reloaded;
    }
  }
  if (!status.ok()) {
    capture_allocator.release(*backend);
    return status;
  }

  std::vector<std::uint32_t> restored_out;
  std::uint64_t restored_base = 0;
  RelocationStats stats;
  status = restore_and_run(*backend, snapshot, &restored_out, &restored_base,
                           &stats);
  capture_allocator.release(*backend);
  if (!status.ok()) {
    return status;
  }

  const auto reference = host_reference();
  const bool capture_ok = buffers_equal(reference, captured_out);
  const bool restore_ok = buffers_equal(reference, restored_out);
  const bool match = buffers_equal(captured_out, restored_out);
  const bool relocated = restored_base != snapshot.allocator.region_base;

  out << "captured_base=0x" << std::hex << snapshot.allocator.region_base
      << "\n";
  out << "restored_base=0x" << restored_base << std::dec << "\n";
  out << "relocation_delta_nonzero=" << (relocated ? 1 : 0) << "\n";
  out << "known_patches=" << stats.known_patches << "\n";
  out << "capture_matches_reference=" << (capture_ok ? 1 : 0) << "\n";
  out << "restore_matches_reference=" << (restore_ok ? 1 : 0) << "\n";
  out << "restore_matches_capture=" << (match ? 1 : 0) << "\n";

  if (!capture_ok || !restore_ok || !match) {
    return Status::mismatch("verify failed: outputs are not bit-identical");
  }
  if (!relocated) {
    out << "WARN: relocation delta was zero; relocation path not exercised\n";
  }
  out << "verify ok\n";
  return Status::Ok();
}

Status bench_synthetic(const std::string& path, int iters, bool scaled,
                       std::ostream& out) {
  if (iters <= 0) {
    return Status::invalid_argument("bench iterations must be positive");
  }
  auto backend = make_backend();
  if (backend->vendor() == Vendor::kStub) {
    return Status::unsupported("bench requires the HIP or CUDA backend");
  }

  using clock = std::chrono::steady_clock;

  // Cold path: capture including warmup (the work a fresh process avoids).
  DeterministicAllocator capture_allocator;
  SnapshotData snapshot;
  const auto capture_begin = clock::now();
  Status status = capture_snapshot(*backend, &capture_allocator, scaled,
                                   kBenchWarmupChains, &snapshot,
                                   /*gpu_out=*/nullptr);
  const auto capture_end = clock::now();
  if (!status.ok()) {
    capture_allocator.release(*backend);
    return status;
  }
  status = write_snapshot_file(path, snapshot);
  if (!status.ok()) {
    capture_allocator.release(*backend);
    return status;
  }

  // Warm path: restore from the snapshot, repeated. The capture region stays
  // mapped so each restore reserves a fresh base (no capture re-run).
  std::chrono::nanoseconds restore_total{0};
  for (int i = 0; i < iters; ++i) {
    std::vector<std::uint32_t> got;
    std::uint64_t restored_base = 0;
    RelocationStats stats;
    const auto restore_begin = clock::now();
    status = restore_and_run(*backend, snapshot, &got, &restored_base, &stats);
    const auto restore_end = clock::now();
    if (!status.ok()) {
      capture_allocator.release(*backend);
      return status;
    }
    restore_total += restore_end - restore_begin;
  }
  capture_allocator.release(*backend);

  const auto capture_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(capture_end -
                                                            capture_begin)
          .count();
  const auto restore_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(restore_total /
                                                            iters)
          .count();
  out << "bench: nodes=" << snapshot.graph.nodes.size()
      << " warmup_chains=" << kBenchWarmupChains << " iters=" << iters << "\n";
  out << "cold_capture_ms=" << capture_ms << "\n";
  out << "warm_restore_ms=" << restore_ms << "\n";
  out << "speedup=" << (restore_ms > 0
                            ? static_cast<double>(capture_ms) / restore_ms
                            : 0.0)
      << "\n";
  return Status::Ok();
}

// ---- M3a: recorded-snapshot inspection + rebuild ---------------------------

Status inspect_snapshot(const std::string& path, std::ostream& out) {
  SnapshotData snapshot;
  Status status = read_snapshot_file(path, &snapshot);
  if (!status.ok()) {
    return status;
  }
  SnapshotSummary sum;
  status = summarize_snapshot(snapshot, &sum);
  if (!status.ok()) {
    return status;
  }
  out << "file=" << path << "\n";
  out << "vendor=" << vendor_name(snapshot.vendor) << "\n";
  out << "arch=" << snapshot.arch << "\n";
  out << "captured_base=0x" << std::hex << sum.captured_base << std::dec
      << "\n";
  out << "region_size=" << sum.region_size << "\n";
  out << "alloc_events=" << sum.alloc_events << "\n";
  out << "modules=" << sum.module_count
      << " (image_bytes=" << sum.total_module_bytes;
  if (sum.modules_with_empty_image > 0) {
    out << ", EMPTY_IMAGE=" << sum.modules_with_empty_image;
  }
  out << ")\n";
  out << "nodes=" << sum.node_count << " (kernel=" << sum.kernel_nodes
      << ")\n";
  out << "nodes_with_identity=" << sum.nodes_with_identity << "\n";
  out << "nodes_without_identity=" << sum.nodes_without_identity << "\n";
  if (sum.nodes_with_empty_module_image > 0) {
    out << "nodes_with_empty_module_image="
        << sum.nodes_with_empty_module_image << "\n";
  }
  out << "total_param_bytes=" << sum.total_param_bytes << "\n";

  // The M3a.3 identity-recovery gate: every captured launch must map back to a
  // known (module, entry). Any unknown node means we missed a
  // hipModuleGetFunction call (or a module load) for that kernel.
  if (sum.nodes_without_identity > 0) {
    out << "IDENTITY_GATE=FAIL (" << sum.nodes_without_identity
        << " unknown node(s))\n";
    return Status::mismatch("identity recovery incomplete: " +
                            std::to_string(sum.nodes_without_identity) +
                            " node(s) have no recorded (module, entry)");
  }
  out << "IDENTITY_GATE=PASS\n";
  return Status::Ok();
}

Status rebuild_check_snapshot(const std::string& path, std::ostream& out) {
  auto backend = make_backend();
  if (backend->vendor() == Vendor::kStub) {
    return Status::unsupported(
        "rebuild-check requires the HIP or CUDA backend");
  }
  SnapshotData snapshot;
  Status status = read_snapshot_file(path, &snapshot);
  if (!status.ok()) {
    return status;
  }
  SnapshotSummary sum;
  status = summarize_snapshot(snapshot, &sum);
  if (!status.ok()) {
    return status;
  }
  if (sum.nodes_without_identity > 0) {
    return Status::mismatch("refusing to rebuild: " +
                            std::to_string(sum.nodes_without_identity) +
                            " node(s) have no recorded identity");
  }

  // Validate against the live backend, then load every recorded module and
  // resolve every entry the graph references.
  status = validate_snapshot_for_backend(snapshot, *backend);
  if (!status.ok()) {
    return status;
  }
  GraphIR ir = snapshot.graph;  // rebuild mutates function handles into the IR
  std::uint64_t loaded_modules = 0;
  std::uint64_t resolved_entries = 0;

  // Load all modules first, then resolve each distinct entry name by trying
  // every module until one provides the function. This is robust to
  // multi-target fat binaries (host-registered kernels), where a function's
  // entry lives in a sibling ELF of the module its node.module_hash names.
  std::vector<ModuleHandle> module_handles;
  module_handles.reserve(snapshot.modules.size());
  for (const ModuleImage& module : snapshot.modules) {
    if (module.image.empty()) {
      return Status::format("module hash " + std::to_string(module.hash) +
                            " has an empty image; cannot reload");
    }
    ModuleHandle module_handle;
    status = backend->load_module(module.image.data(), module.image.size(),
                                  &module_handle);
    if (!status.ok()) {
      return status;
    }
    module_handles.push_back(module_handle);
    ++loaded_modules;
  }

  // Collect the distinct entry names the graph actually references.
  std::vector<std::string> referenced;
  for (const GraphNodeIR& node : snapshot.graph.nodes) {
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
      status = backend->get_function(module_handle, entry, &function);
      if (status.ok()) {
        resolved = true;
        break;
      }
    }
    if (!resolved) {
      return Status::format("entry '" + entry +
                            "' not found in any loaded module");
    }
    ++resolved_entries;
    for (GraphNodeIR& node : ir.nodes) {
      if (node.type == GraphNodeType::kKernel && node.entry_name == entry) {
        node.kernel.function = function;
      }
    }
  }

  // Replay the deterministic allocator so the captured device pointers in
  // each node's param_blob resolve to MAPPED memory in this process (the
  // recorder captures raw device VAs; without replay they point at reserved-
  // but-unmapped VA and hipGraphAddKernelNode faults probing them — the node#3
  // cascade segfault in job 522840). Then relocate the IR by the (usually zero,
  // on the same node) delta between captured and restored base, with blind-scan
  // fallback to find pointer bytes inside the tagged kernarg blob.
  //
  // Arena mode (the default): the redirect serves hipMalloc from ONE big
  // hipMalloc-backed arena, so the recorder captures region_base/size (via the
  // env vars the redirect publishes) but ZERO individual alloc events. Rather
  // than map the whole 72 GiB arena (which fails hipMemCreate OOM on MI300A),
  // scan the param_blobs for the captured pointer VA span and map just that
  // subrange — the rebuild only needs the referenced VAs backed.
  AllocatorState alloc_state = snapshot.allocator;
  const std::uint64_t cbase = alloc_state.region_base;
  const std::uint64_t csize = alloc_state.region_size;
  if (alloc_state.events.empty() && cbase != 0 && csize != 0) {
    std::uint64_t span_min = 0, span_max = 0;
    bool any_ptr = false;
    for (const GraphNodeIR& node : ir.nodes) {
      const std::vector<std::byte>& blob = node.kernel.param_blob;
      if (blob.size() < sizeof(std::uint64_t)) continue;
      for (std::size_t off = 0;
           off + sizeof(std::uint64_t) <= blob.size(); ++off) {
        std::uint64_t v = 0;
        std::memcpy(&v, blob.data() + off, sizeof(v));
        if (v < cbase || v - cbase >= csize) continue;
        if (!any_ptr || v < span_min) span_min = v;
        if (!any_ptr || v > span_max) span_max = v;
        any_ptr = true;
        off += sizeof(std::uint64_t) - 1;  // skip pointer body
      }
    }
    if (any_ptr) {
      const std::uint64_t gran =
          alloc_state.granularity ? alloc_state.granularity : 4096;
      const std::uint64_t lo = span_min / gran * gran;
      std::uint64_t hi = span_max + 1;
      hi = (hi + gran - 1) / gran * gran;
      alloc_state.events.push_back(
          AllocEvent{lo - cbase, hi - lo, "ptrspan"});
    }
  }
  DeterministicAllocator allocator;
  std::uint64_t restored_base = 0;
  out << "rebuild-check: allocator region_base=0x" << std::hex
      << snapshot.allocator.region_base << " size=0x"
      << snapshot.allocator.region_size << std::dec << " events="
      << alloc_state.events.size() << "\n";
  for (const AllocEvent& ev : alloc_state.events) {
    out << "rebuild-check:   map off=0x" << std::hex << ev.offset << " size=0x"
        << ev.size << std::dec << " tag=" << ev.tag << "\n";
  }
  status = allocator.replay(*backend, alloc_state, alloc_state.region_base,
                            &restored_base);
  if (!status.ok()) {
    return status;
  }
  out << "rebuild-check: replay restored_base=0x" << std::hex << restored_base
      << " (captured=0x" << snapshot.allocator.region_base << ")"
      << std::dec << "\n";

  // Populate precise pointer offsets from parsed AMDGPU kernel signatures, so
  // relocation patches ONLY the args the metadata marks as pointers (replacing
  // the blind 8-byte scan that can false-positive on scalar values). Best-effort:
  // nodes without a matching signature keep empty ptr_offsets and fall back to
  // blind-scan inside relocate_graph_ir.
  std::unordered_map<std::string, KernelSig> sig_by_name;
  for (const ModuleImage& module : snapshot.modules) {
    if (module.image.empty()) continue;
    for (const KernelSig& ks : extract_amdgpu_kernels(module.image.data(),
                                                     module.image.size())) {
      sig_by_name[ks.name] = ks;
    }
  }
  std::uint64_t precise_nodes = 0;
  for (GraphNodeIR& node : ir.nodes) {
    if (node.type != GraphNodeType::kKernel || node.entry_name.empty()) continue;
    auto sit = sig_by_name.find(node.entry_name);
    if (sit == sig_by_name.end()) continue;
    auto offs = tag1_blob_ptr_offsets(node.kernel.param_blob, sit->second);
    if (!offs.empty()) {
      node.kernel.ptr_offsets = std::move(offs);
      ++precise_nodes;
    }
  }
  out << "rebuild-check: sigs=" << sig_by_name.size()
      << " precise_ptr_nodes=" << precise_nodes << "\n";

  RelocationStats reloc_stats;
  status = relocate_graph_ir(
      &ir,
      Relocation{snapshot.allocator.region_base, restored_base,
                 snapshot.allocator.region_size},
      /*blind_scan_fallback=*/true, &reloc_stats);
  if (!status.ok()) {
    return status;
  }
  out << "rebuild-check: relocated known=" << reloc_stats.known_patches
      << " blind=" << reloc_stats.blind_patches << "\n";

  GraphHandle graph;
  status = backend->rebuild_graph(ir, &graph);
  if (!status.ok()) {
    return status;
  }
  GraphExecHandle exec;
  status = backend->instantiate(graph, &exec);
  if (!status.ok()) {
    return status;
  }

  out << "rebuild-check: " << snapshot.graph.nodes.size() << " nodes, "
      << loaded_modules << " modules, " << resolved_entries
      << " entry handles, instantiate=ok\n";
  out << "REBUILD_GATE=PASS\n";
  return Status::Ok();
}

}  // namespace snapshot::cli
