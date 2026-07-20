#!/usr/bin/env python3
"""Buffered vs O_DIRECT on ONE fd, long sequential stream, with a ramp curve.

Resolves the loose end in DD_VS_FASTSAFETENSORS.md. fastsafetensors reads
BUFFERED (`os.open(src, O_RDONLY)`, no O_DIRECT) and -- per its C++ -- with ONE
thread per file (submit_io chunks by max_copy_block_size, default 16 GiB, so a
5 GB shard is a single submit). Yet it read the sick OST-8 shard at 213 MB/s
where `dd bs=16M iflag=direct` got 24.9. If it is not thread fan-out, the
remaining candidate is Lustre client READAHEAD, which O_DIRECT bypasses.

An earlier buffered-vs-direct spot check showed only 31.0 vs 25.2 MB/s (1.2x),
which does NOT explain 8x. The suspicion: that test read only 128 MB, and
readahead is per-fd and ramps -- 8 reads is not enough to reach the window.
This probe is built to test exactly that:

  * ONE fd per stream (a fresh `dd` per chunk would reset readahead every time,
    which is precisely how the earlier check hid the effect)
  * long stretches (default 1.2 GB), so the window has room to open
  * per-segment throughput => the ramp is visible, not inferred

O_DIRECT needs alignment: the buffer is an anonymous mmap (page-aligned) and
offsets/lengths are multiples of bs.

Usage: readahead_probe.py <file> <offset_mib> <length_mib> <bs_mib> <direct|buffered>
"""
import mmap
import os
import sys
import time

O_DIRECT = 0o40000  # Linux; not in os.* on all builds


def stream(path, offset, length, bs, direct, seg=128 * 1024 * 1024):
    flags = os.O_RDONLY | (O_DIRECT if direct else 0)
    fd = os.open(path, flags)
    buf = mmap.mmap(-1, bs)  # anonymous mmap => page-aligned, safe for O_DIRECT
    marks = []
    try:
        os.lseek(fd, offset, os.SEEK_SET)
        done = 0
        t0 = time.time()
        seg_t0, seg_b = t0, 0
        while done < length:
            n = os.readv(fd, [buf])
            if n <= 0:
                break
            done += n
            seg_b += n
            if seg_b >= seg:
                now = time.time()
                marks.append((done, seg_b / (now - seg_t0) / 1e6))
                seg_t0, seg_b = now, 0
        dt = time.time() - t0
    finally:
        buf.close()
        os.close(fd)
    return done, dt, marks


def main():
    path, off_mib, len_mib, bs_mib, mode = sys.argv[1:6]
    offset = int(off_mib) * 1024 * 1024
    length = int(len_mib) * 1024 * 1024
    bs = int(bs_mib) * 1024 * 1024
    direct = mode == "direct"

    done, dt, marks = stream(path, offset, length, bs, direct)
    mbps = done / dt / 1e6
    print(
        f"  {mode:8s} off={off_mib:>5}MiB len={done/1048576:.0f}MiB bs={bs_mib}M"
        f" -> {dt:7.2f} s  {mbps:8.1f} MB/s"
    )
    # Ramp: if readahead is the mechanism, buffered starts slow and climbs while
    # the window opens; O_DIRECT should be flat (it has no window to open).
    if marks:
        curve = "  ".join(f"{m:.0f}" for _, m in marks[:12])
        print(f"           ramp (MB/s per 128MiB): {curve}")
        first, last = marks[0][1], marks[-1][1]
        print(f"           first={first:.0f} last={last:.0f} ramp_factor={last/first:.1f}x")


if __name__ == "__main__":
    main()
