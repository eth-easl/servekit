// Minimal CUDA program for testing cuda-checkpoint + CRIU checkpoint/restore.
//
// In the spirit of the example in
// https://fergusfinn.com/blog/what-happens-when-you-checkpoint-a-cuda-process/
//
// - Creates a CUDA context (cudaFree(0)) and a counter that lives in DEVICE memory.
// - Every ~200ms launches a kernel that increments that counter *on the GPU*,
//   copies it back, and appends "tick=<n>" to a log file.
//
// The blog's version is poked over a UDP socket. We deliberately drop the socket:
// checkpointing an open socket drags in CRIU's network-lock machinery, which is
// orthogonal to the question we're asking (does GPU state survive?). A timer +
// log file gives the same evidence with far fewer moving parts -- if the counter
// keeps climbing from where it stopped, the CUDA context and device memory
// survived the checkpoint/restore round trip.
//
// Build:  nvcc -o counter counter.cu
// Run:    ./counter <logfile> [tick_ms]
//
// tick_ms (default 1000) sets the pace. Slower ticks keep the process mostly idle
// rather than hammering the GPU while cuda-checkpoint's `lock` waits for CUDA calls
// to quiesce, and make the log readable. The caller must scale its "is it frozen?"
// observation window with this, or the freeze evidence gets weaker.

#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>

__global__ void inc(int *counter) { atomicAdd(counter, 1); }

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),      \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  const char *path = (argc > 1) ? argv[1] : "counter.log";
  int tick_ms = (argc > 2) ? atoi(argv[2]) : 1000;
  if (tick_ms <= 0) tick_ms = 1000;

  // Establish a CUDA context and a device-side counter initialized to 0.
  CUDA_CHECK(cudaFree(0));
  int *d_counter = nullptr;
  CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

  FILE *log = fopen(path, "a");
  if (!log) { perror("fopen"); return 1; }
  setvbuf(log, nullptr, _IOLBF, 0);
  fprintf(log, "start pid=%d tick_ms=%d\n", getpid(), tick_ms);

  for (;;) {
    inc<<<1, 1>>>(d_counter);
    CUDA_CHECK(cudaGetLastError());
    int h = -1;
    CUDA_CHECK(cudaMemcpy(&h, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    fprintf(log, "tick=%d\n", h);
    usleep(tick_ms * 1000);
  }
  return 0;
}
