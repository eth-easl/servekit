// Probe v3: read libnccl's .nv_fatbin section from its on-disk ELF (sh_addr),
// scan ONLY that runtime range for FATBIN_MAGIC, load each, resolve the NCCL
// kernel name. Reliable + bounded (no false positives from other sections).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <elf.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>

static const uint32_t FATBIN_MAGIC = 0xBA55ED50;
static uintptr_t g_nccl_base = 0;
static char g_nccl_path[1024] = {};
static std::vector<CUmodule> g_mods;

static int find_base(struct dl_phdr_info* info, size_t, void*) {
  if (info->dlpi_name && strstr(info->dlpi_name, "libnccl")) {
    g_nccl_base = info->dlpi_addr;
    if (info->dlpi_name[0]) { strncpy(g_nccl_path, info->dlpi_name, sizeof(g_nccl_path)-1); g_nccl_path[sizeof(g_nccl_path)-1]=0; }
    return 1;
  }
  return 0;
}

// returns runtime [start,end) of .nv_fatbin by reading the on-disk ELF
static bool find_nvfatbin_range(const char* path, uintptr_t base, uintptr_t& start, size_t& size) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return false;
  Elf64_Ehdr eh; ssize_t r = read(fd, &eh, sizeof eh); if (r != (ssize_t)sizeof eh) { close(fd); return false; }
  std::vector<Elf64_Shdr> sh(eh.e_shnum);
  lseek(fd, eh.e_shoff, SEEK_SET);
  read(fd, sh.data(), sizeof(Elf64_Shdr) * eh.e_shnum);
  // shstrtab
  std::string strtab; {
    std::vector<char> buf(sh[eh.e_shstrndx].sh_size);
    lseek(fd, sh[eh.e_shstrndx].sh_offset, SEEK_SET);
    read(fd, buf.data(), buf.size());
    strtab.assign(buf.data(), buf.size());
  }
  for (auto& s : sh) {
    if (s.sh_type != SHT_PROGBITS) continue;
    const char* nm = strtab.c_str() + s.sh_name;
    if (strcmp(nm, ".nv_fatbin") == 0) {
      start = base + s.sh_addr; size = s.sh_size; close(fd); return true;
    }
  }
  close(fd);
  return false;
}

int main() {
  CUcontext ctx{}; CUdevice dev{};
  cuInit(0); cuDeviceGet(&dev, 0);
  cuDevicePrimaryCtxRetain(&ctx, dev); cuCtxSetCurrent(ctx);
  void* h = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
  if (!h) h = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
  fprintf(stderr, "[probe] dlopen libnccl=%p\n", h);
  dl_iterate_phdr(find_base, nullptr);
  fprintf(stderr, "[probe] libnccl base=0x%lx path=%s\n", g_nccl_base, g_nccl_path);

  const char* path = g_nccl_path[0] ? g_nccl_path : nullptr;

  uintptr_t start = 0; size_t size = 0;
  if (path && find_nvfatbin_range(path, g_nccl_base, start, size)) {
    fprintf(stderr, "[probe] .nv_fatbin runtime=[0x%lx, +0x%zx)\n", start, size);
    for (size_t o = 0; o + 4 <= size; o += 8) {
      if (*(uint32_t*)(start + o) == FATBIN_MAGIC) {
        CUmodule mod{};
        if (cuModuleLoadFatBinary(&mod, (void*)(start + o)) == CUDA_SUCCESS && mod) g_mods.push_back(mod);
      }
    }
  }
  fprintf(stderr, "[probe] loaded %zu fatbin modules\n", g_mods.size());

  const char* names[] = {
    "_Z40ncclDevKernel_AllReduce_Sum_bf16_RING_LL24ncclDevKernelArgsStorageILm4096EE",
    "_Z38ncclDevKernel_AllReduce_Sum_u8_RING_LL24ncclDevKernelArgsStorageILm4096EE",
    nullptr };
  for (int n = 0; names[n]; ++n) {
    int found = -1;
    for (size_t i = 0; i < g_mods.size(); ++i) {
      CUfunction f{};
      if (cuModuleGetFunction(&f, g_mods[i], names[n]) == CUDA_SUCCESS && f) { found = (int)i; break; }
    }
    fprintf(stderr, "[probe] resolve '%.30s...' -> module #%d\n", names[n], found);
  }
  return 0;
}
