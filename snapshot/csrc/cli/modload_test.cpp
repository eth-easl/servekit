// modload_test <file.co> — call hipModuleLoadData on a captured code object and
// report whether ROCm accepts it. Decisive for "no kernel image available":
// isolates whether a rebuild failure is the image itself vs. the rebuild wiring.
#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <file.co>\n", argv[0]);
    return 2;
  }
  // Force a device + context so hipModuleLoadData has a device to bind to.
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0) {
    std::fprintf(stderr, "no HIP device\n");
    return 3;
  }
  hipDevice_t dev{};
  if (hipDeviceGet(&dev, 0) != hipSuccess) {
    std::fprintf(stderr, "hipDeviceGet failed\n");
    return 3;
  }
  std::ifstream in(argv[1], std::ios::binary);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  std::vector<char> buf((std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());
  std::fprintf(stderr, "loading %s (%zu bytes)\n", argv[1], buf.size());
  hipModule_t mod{};
  hipError_t rc = hipModuleLoadData(&mod, buf.data());
  std::fprintf(stderr, "hipModuleLoadData rc=%d (%s)\n", static_cast<int>(rc),
               hipGetErrorString(rc));
  if (rc == hipSuccess) {
    hipModuleUnload(mod);
    return 0;
  }
  return 1;
}
