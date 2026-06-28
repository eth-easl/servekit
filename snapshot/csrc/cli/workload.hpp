#pragma once

#include <iosfwd>
#include <string>

#include "snapshot/snapshot_format.hpp"

namespace snapshot::cli {

Status capture_synthetic(const std::string& path, bool scaled);
Status restore_synthetic(const std::string& path, std::ostream& out);
Status verify_synthetic(const std::string& path, std::ostream& out);
Status bench_synthetic(const std::string& path, int iters, bool scaled,
                       std::ostream& out);

// M3a: host-only snapshot reader. Parses a recorded snapshot and reports the
// module / node / identity summary. No GPU backend is needed, so this is the
// part of the M3a.3 identity-recovery gate that runs anywhere (and locally on
// the stub build for format testing).
Status inspect_snapshot(const std::string& path, std::ostream& out);

// M3a: GPU rebuild check. Loads the recorded module images, resolves every
// entry, rebuilds the graph from the recorded IR, and instantiates it. The
// gate is "no HIP error through instantiate"; it does NOT launch (launching a
// real vLLM graph in isolation reads unseeded internal buffers, and bit-identical
// for vLLM is M3b). Requires the HIP backend.
Status rebuild_check_snapshot(const std::string& path, std::ostream& out);

// GPU-free snapshot analysis: parses module ELF images for AMDGPU kernel
// signatures (MessagePack metadata), reports per-node arg counts / pointer
// offsets / signature matches, and validates the tagged kernarg blobs. Runs on
// the login node (no HIP). Used to validate the msgpack parser + exact-arg-
// count logic against real snapshots before spending a GPU allocation.
Status analyze_snapshot(const std::string& path, std::ostream& out);

// M3f: restore ALL snapshots in a directory in one process. Loads each unique
// module image once (deduplicated by hash), then for each .snap file replays
// the allocator, rebuilds the graph, and instantiates it. Reports per-graph and
// aggregate timing. This is the "warm restore" time that replaces vLLM's cold
// capture phase — the core cold-start win.
Status restore_all_snapshots(const std::string& dir, std::ostream& out);

}  // namespace snapshot::cli
