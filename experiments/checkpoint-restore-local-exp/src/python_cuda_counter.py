#!/usr/bin/env python3
"""Gate 3a probe: a minimal *Python + torch.cuda* process that holds a CUDA context
and a device-memory counter, ticking to a log file.

The C `counter.cu` (Gate 2) proved cuda-checkpoint+criu works on a tiny CUDA program.
Before attempting the full SGLang server, this de-risks the piece in between: a real
CPython interpreter with the torch/CUDA runtime loaded -- many threads, a big RSS, and
the actual CUDA libraries the server uses. Same verification contract as counter.cu:
the counter lives in *device* memory and appends every tick, so a restore that resumes
at froze+1 with a contiguous sequence proves the CUDA context + device memory survived.

Run:  python_cuda_counter.py <logfile> [tick_ms]
"""
import os
import sys
import time

import torch


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "py_counter.log"
    tick_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    assert torch.cuda.is_available(), "CUDA not available"
    # Counter in DEVICE memory (like counter.cu's d_counter).
    counter = torch.zeros(1, dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()

    with open(path, "a", buffering=1) as log:
        log.write("start pid=%d tick_ms=%d dev=%s\n"
                  % (os.getpid(), tick_ms, torch.cuda.get_device_name(0)))
        while True:
            counter += 1                     # increment on the GPU
            h = int(counter.item())          # copy back to host
            log.write("tick=%d\n" % h)
            time.sleep(tick_ms / 1000.0)


if __name__ == "__main__":
    main()
