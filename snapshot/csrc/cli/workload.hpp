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

}  // namespace snapshot::cli
