#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "snapshot/allocator.hpp"
#include "snapshot/gpu_backend.hpp"
#include "snapshot/snapshot_format.hpp"
#include "snapshot/status.hpp"

namespace snapshot {

// ---- Host-testable helpers for the LD_PRELOAD recorder ---------------------
//
// The recorder interposer (csrc/preload/snapshot_record.cpp) observes HIP
// module-load / get-function / launch / capture events inside an unmodified
// application (our synthetic workload, or vLLM) and accumulates them. These
// helpers turn that accumulated, vendor-neutral record into a SnapshotData
// that the existing serialize/relocate/restore pipeline already understands.
//
// Keeping the assembly logic here (not in the HIP-only interposer) means the
// IR-building math is unit-testable on the stub backend with no device.

// A code object passed to hipModuleLoadData has no size argument; it is an
// ELF/HSACO image whose total length is recoverable from its header. Returns
// the image size in bytes, or 0 if the buffer is not a parseable ELF64 (the
// caller should then record an empty image and flag it). `max_cap` bounds the
// trusted size (a sane upper bound for a single code object).
std::uint64_t elf_code_object_size(const std::byte* image, std::size_t max_cap);

// Walk an AMDGPU ELF64 image's symbol tables (SHT_SYMTAB and SHT_DYNSYM) and
// return every symbol name, bounded by each section's strtab size so a missing
// terminator cannot run off the buffer. Used to build a name -> image map so a
// kernel named by hipKernelNameRef or __hipRegisterFunction can be linked back
// to the code object it lives in — without depending on the (unreliable)
// hipFunction_t handle matching the one hipModuleGetFunction returned. Returns
// an empty vector for a non-ELF/unparseable image.
std::vector<std::string> extract_elf_symbols(const std::byte* image,
                                             std::size_t size);

// One kernel's signature parsed from the AMDGPU code-object metadata note
// (MessagePack, code-object v4+). Used by the rebuild to get EXACT arg counts
// and per-arg sizes/offsets (the launch-time readability heuristic undercounts)
// and precise pointer offsets (replaces blind-scan relocation). `is_pointer`
// is true when value_kind is a device-address kind (global_buffer /
// dynamic_shared_pointer / image / sampler / queue / etc.).
struct KernelArgSig {
  std::uint32_t offset = 0;   // byte offset within the kernarg segment
  std::uint32_t size = 0;     // size in bytes
  bool is_pointer = false;    // value_kind is a device-address kind
};
struct KernelSig {
  std::string name;                       // the .name field
  std::uint32_t kernarg_segment_size = 0;  // total kernarg buffer size
  std::uint32_t kernarg_alignment = 1;
  std::vector<KernelArgSig> args;
};
// Parse the AMDGPU code-object metadata note (MessagePack) from an ELF image.
// Returns one entry per kernel in amdhsa.kernels. Empty for non-ELF, for the
// uncompressed YAML form (code-object v3; use extract_elf_symbols for names),
// or on any parse error. Pure parsing; no HIP dependency.
std::vector<KernelSig> extract_amdgpu_kernels(const std::byte* image,
                                              std::size_t size);

// Rewrite every amdhsa.kernels[].max_flat_workgroup_size to `target`, IN PLACE,
// within the AMDGPU metadata note of an ELF code object. `image` must be
// writable; all ELF/note offsets and byte sizes are unchanged (same-width
// integer rewrites only). Returns the number of values rewritten.
//
// Rationale: Triton's ROCm codegen emits an over-restrictive
// max_flat_workgroup_size (256) for num_warps=8 pointwise kernels that are in
// fact compiled for and run correctly at 512 threads. Eager launch ignores the
// field (baseline serving works), but hipGraphAddKernelNode validates
// block.x <= MAX_THREADS_PER_BLOCK (derived from this field) and rejects such
// nodes. Bumping it to the device max (1024 on MI300A) makes the code object
// self-consistent so graph-add succeeds. The field is a runtime hint only;
// compile-time resource allocation (COMPUTE_PGM_RSRC1/2) is unaffected, so the
// patch does not change execution. Only rewrites values whose current msgpack
// int encoding (uint16/uint32/uint64) can hold `target` in place; smaller
// encodings (fixint/uint8) are left untouched. Pure byte manipulation; no HIP
// dependency.
std::size_t patch_amdgpu_max_flat_workgroup_size(std::byte* image,
                                                 std::size_t size,
                                                 std::uint32_t target);

// Given a tag-1 (array-format) kernarg blob {u8 tag=1, u32 count, per arg:
// {u32 len, bytes[len]}} and the kernel's parsed signature, compute the
// blob-relative byte offsets where POINTER args live. arg i's value starts at
// 5 + sum(4 + prev.len); a pointer arg contributes its start offset. Only args
// present in BOTH the blob and the signature are considered. Empty on any
// mismatch (caller falls back to blind-scan). Pure data manipulation.
std::vector<std::uint32_t> tag1_blob_ptr_offsets(
    const std::vector<std::byte>& blob, const KernelSig& sig);

// Returns the byte offsets within a tag-1 kernarg blob that START each
// NON-pointer arg's value bytes. These ranges may contain embedded device
// pointers (e.g. a struct arg {ptr, stride, size}) that the signature does not
// advertise as pointers — they need blind-scan relocation to catch. Each
// returned offset is the START of the arg value; the caller scans [off, off+len)
// for embedded pointers by reading the per-arg length prefix the same way
// tag1_blob_ptr_offsets does. Returns {offset, length} pairs.
std::vector<std::pair<std::uint32_t, std::uint32_t>>
tag1_blob_nonptr_arg_ranges(const std::vector<std::byte>& blob,
                            const KernelSig& sig);

struct RecordedModule {
  std::uint64_t hash = 0;
  std::vector<std::byte> image;  // may be empty if the ELF size could not be parsed
};

struct RecordedFunction {
  std::uint64_t function_id = 0;  // opaque id the recorder assigned to hipFunction_t
  std::uint64_t module_hash = 0;  // 0 = the function's module was not recorded
  std::string entry_name;        // empty = not recorded
};

struct RecordedLaunch {
  std::uint64_t function_id = 0;
  Dim3 grid;
  Dim3 block;
  std::uint32_t shared_mem_bytes = 0;
  std::vector<std::byte> param_blob;  // raw kernarg; pointer offsets discovered
                                      // by blind-scan relocation at restore time
};

// Everything the recorder gathered for a single capture window, plus the shared
// module/function tables and the deterministic-memory region the pointers live
// in. The recorder fills one of these on hipStreamEndCapture.
struct RecordAssembly {
  Vendor vendor = Vendor::kHip;
  std::string arch;  // e.g. "gfx942:sramecc+:xnack-"; must match the restore backend

  std::uint64_t region_base = 0;   // arena/VMM base the captured pointers live in
  std::uint64_t region_size = 0;
  std::uint64_t granularity = 4096;
  std::vector<AllocEvent> alloc_events;

  std::vector<RecordedModule> modules;      // need not be deduped; hash groups them
  std::vector<RecordedFunction> functions;
  std::vector<RecordedLaunch> launches;     // the ordered kernel chain captured
};

// Builds a serializable SnapshotData from a RecordAssembly:
//  - dedupes modules by hash, collecting each module's entry names from the
//    functions that reference it;
//  - emits one kernel node per recorded launch (serial-chain edges), resolving
//    function_id -> (module_hash, entry_name);
//  - leaves kernel.ptr_offsets empty (the interposer cannot know which param
//    bytes are pointers); restore-time blind-scan relocation discovers them.
Status assemble_recorded_snapshot(const RecordAssembly& in, SnapshotData* out);

// A read-only summary of a parsed snapshot, for the `inspect` verb and for the
// M3a.3 identity-recovery gate (nodes_without_identity must be 0).
struct SnapshotSummary {
  std::uint64_t captured_base = 0;
  std::uint64_t region_size = 0;
  std::size_t module_count = 0;
  std::uint64_t total_module_bytes = 0;
  std::size_t modules_with_empty_image = 0;
  std::size_t alloc_events = 0;
  std::size_t node_count = 0;
  std::size_t kernel_nodes = 0;
  std::size_t nodes_with_identity = 0;    // module_hash != 0 and entry_name non-empty
  std::size_t nodes_without_identity = 0;
  std::size_t nodes_with_empty_module_image = 0;  // identity known, image missing
  std::uint64_t total_param_bytes = 0;
};

Status summarize_snapshot(const SnapshotData& in, SnapshotSummary* out);

}  // namespace snapshot
