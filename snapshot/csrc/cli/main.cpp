#include <cstdlib>
#include <iostream>
#include <string>

#include "snapshot/gpu_backend.hpp"
#include "workload.hpp"

namespace {

void usage(std::ostream& out) {
  out << "usage: snapshot <verb> <file> [options]\n"
      << "verbs:\n"
      << "  capture <file> [--scaled]           synthetic capture (workload is recorder)\n"
      << "  restore <file>                       restore + bit-identical vs reference\n"
      << "  verify <file>                        one-process bit-identical gate\n"
      << "  bench <file> [--scaled] [--iters N]  cold capture vs warm restore\n"
      << "  probe-base                           M1.0 fixed-base determinism\n"
      << "  inspect <file>                       M3a: parse a recorded snapshot, report identity\n"
      << "  rebuild-check <file>                 M3a: reload modules, rebuild, instantiate\n"
      << "  analyze <file>                       GPU-free: parse kernel signatures, report arg counts\n"
      << "  restore-all <dir>                   M3f: rebuild + instantiate ALL snapshots, time it\n"
      << "options:\n"
      << "  --scaled       use ~200 synthetic graph nodes\n"
      << "  --iters N      bench iterations (default: 10)\n";
}

int finish(const snapshot::Status& status) {
  if (status.ok()) {
    return 0;
  }
  std::cerr << snapshot::status_code_name(status.code()) << ": "
            << status.message() << "\n";
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);  // line-buffered so SLURM logs survive crashes
  if (argc < 2) {
    usage(std::cerr);
    return 2;
  }

  std::string verb = argv[1];
  if (verb == "probe-base") {
    auto backend = snapshot::make_backend();
    std::uint64_t granularity = 0;
    snapshot::Status status = backend->get_allocation_granularity(&granularity);
    if (!status.ok()) {
      return finish(status);
    }
    std::uint64_t base = 0;
    status = backend->reserve_address(64ULL * 1024ULL * 1024ULL, granularity,
                                      snapshot::kDefaultRequestedBase, &base);
    if (!status.ok()) {
      return finish(status);
    }
    std::cout << "requested_base=0x" << std::hex
              << snapshot::kDefaultRequestedBase << "\n";
    std::cout << "returned_base=0x" << base << "\n";
    std::cout << "honored=" << std::dec
              << (base == snapshot::kDefaultRequestedBase ? 1 : 0) << "\n";
    backend->release_address(base, 64ULL * 1024ULL * 1024ULL);
    return 0;
  }

  if (argc < 3) {
    usage(std::cerr);
    return 2;
  }

  std::string path = argv[2];
  bool scaled = false;
  int iters = 10;
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--scaled") {
      scaled = true;
    } else if (arg == "--iters" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else {
      usage(std::cerr);
      return 2;
    }
  }

  if (verb == "capture") {
    return finish(snapshot::cli::capture_synthetic(path, scaled));
  }
  if (verb == "restore") {
    return finish(snapshot::cli::restore_synthetic(path, std::cout));
  }
  if (verb == "verify") {
    return finish(snapshot::cli::verify_synthetic(path, std::cout));
  }
  if (verb == "bench") {
    return finish(snapshot::cli::bench_synthetic(path, iters, scaled,
                                                std::cout));
  }
  if (verb == "inspect") {
    return finish(snapshot::cli::inspect_snapshot(path, std::cout));
  }
  if (verb == "rebuild-check") {
    return finish(snapshot::cli::rebuild_check_snapshot(path, std::cout));
  }
  if (verb == "analyze") {
    return finish(snapshot::cli::analyze_snapshot(path, std::cout));
  }
  if (verb == "restore-all") {
    return finish(snapshot::cli::restore_all_snapshots(path, std::cout));
  }

  usage(std::cerr);
  return 2;
}
