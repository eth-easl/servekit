#!/usr/bin/env python3
"""External stager for the sharded_state hugepage integration.

Runs OUTSIDE sglang, before `python -m sglang.launch_server` starts: stages
every *.safetensors file in a pre-sharded checkpoint directory
(`--load-format sharded_state`, see save_sharded_state_fixed.py) into ONE
memfd(MFD_HUGETLB) buffer via hugepage_stager.stage(), then serves that fd to
every TP rank over a Unix-socket + SCM_RIGHTS broker (hugepage_fd_broker).

Does not need to know which rank owns which file -- each rank filters the
combined files_meta down to its own model-rank-{rank}-part-*.safetensors
files after fetching (see hugepage_client.py). No sglang import here.

Usage: hugepage_stage_daemon.py <checkpoint_dir> --sock <path> [--ready-file <path>]
"""
import argparse
import glob
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hugepage_fd_broker as broker
import hugepage_stager as stager

_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--sock", required=True)
    ap.add_argument("--ready-file", default=None)
    ap.add_argument("--slices", type=int, default=64)
    a = ap.parse_args()

    filenames = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(a.checkpoint_dir, "*.safetensors"))
    )
    if not filenames:
        print(f"FATAL: no *.safetensors files under {a.checkpoint_dir}", file=sys.stderr)
        raise SystemExit(1)

    print(f"staging {len(filenames)} file(s) from {a.checkpoint_dir} ...", flush=True)
    t0 = time.perf_counter()
    fd, total_len, files_meta, buf = stager.stage(a.checkpoint_dir, filenames, a.slices)
    dt = time.perf_counter() - t0
    print(f"staged {total_len/1e9:.2f} GB in {dt:.2f}s ({total_len/dt/1e9:.2f} GB/s)", flush=True)

    if os.path.exists(a.sock):
        os.unlink(a.sock)
    srv = broker.FdBrokerServer(a.sock, fd, total_len, files_meta)
    srv.start()
    print(f"broker listening on {a.sock}", flush=True)

    if a.ready_file:
        with open(a.ready_file, "w") as f:
            f.write("ready\n")
        print(f"wrote ready-file {a.ready_file}", flush=True)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    while not _stop:
        time.sleep(1)

    print("stopping", flush=True)
    srv.stop()


if __name__ == "__main__":
    main()
