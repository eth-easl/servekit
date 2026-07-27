#!/usr/bin/env python3
"""Standalone smoke test for hugepage_stage_daemon.py + hugepage_client.py,
no sglang involved. Fetches rank 0's files from an already-running daemon,
checksums the first tensor of the first file against a plain read of the
source checkpoint file.

Usage: smoke_test_daemon.py <checkpoint_dir> --sock <path>
"""
import argparse
import hashlib
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hugepage_client as client
import hugepage_safetensors as hsf


def source_tensor_bytes(src_path, meta):
    start, end = meta["data_offsets"]
    with open(src_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + n + start)
        return f.read(end - start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--sock", required=True)
    ap.add_argument("--rank", type=int, default=0)
    a = ap.parse_args()

    pattern = f"model-rank-{a.rank}-part-*.safetensors"
    result = client.get_rank_files_meta(a.sock, pattern)
    if result is None:
        print("FAIL: get_rank_files_meta returned None", flush=True)
        raise SystemExit(1)
    buf, ordered = result
    print(f"rank {a.rank}: matched {len(ordered)} files: "
          f"{[f['filename'] for f in ordered]}", flush=True)

    checked, mismatches = 0, []
    mv = memoryview(buf)
    for f in ordered:
        src_path = os.path.join(a.checkpoint_dir, f["filename"])
        for name in f["header"]:
            start, end = f["header"][name]["data_offsets"]
            got = bytes(mv[f["data_start"] + start:f["data_start"] + end])
            want = source_tensor_bytes(src_path, f["header"][name])
            checked += 1
            if hashlib.md5(got).digest() != hashlib.md5(want).digest():
                mismatches.append((f["filename"], name))
            _ = hsf.tensor_view(buf, f["header"], f["data_start"], name)
        print(f"  {f['filename']}: span_len={f['span_len']/1e9:.3f} GB, "
              f"{len(f['header'])} tensors OK", flush=True)

    print(f"checked={checked} mismatches={len(mismatches)}", flush=True)
    if mismatches:
        print(f"FAIL: {mismatches[:5]}", flush=True)
        raise SystemExit(1)

    # PipelinedRegistrar smoke test (wait_for/unregister API only, no GPU/
    # cudaMemcpyAsync here -- that needs a CUDA context, exercised by the
    # real e2e sbatch instead). Exercises the same per-tensor wait_for()
    # calls the patched loader.py loop makes, single-threaded.
    reg = client.PipelinedRegistrar(buf, ordered)
    for i, f in enumerate(ordered):
        for name in f["header"]:
            reg.wait_for(i, client.tensor_end_offset_in_file(f, name))
        reg.unregister_file(i)
    print("PipelinedRegistrar: wait_for/unregister_file cycle OK", flush=True)

    print("RESULT: PASS", flush=True)


if __name__ == "__main__":
    main()
