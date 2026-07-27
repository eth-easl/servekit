#!/usr/bin/env python3
"""End-to-end validation of the merged Phase 1.5 design in STEP4_PLAN.md,
before touching SGLang itself:

  1. Stage real safetensors shard(s) into ONE memfd(MFD_HUGETLB) buffer
     (hugepage_stager.stage) -- no filesystem path, no safe_open.
  2. Hand that fd to N SEPARATELY SPAWNED processes (multiprocessing.spawn,
     matching how SGLang's TP ranks are actually launched) via a Unix-socket
     + SCM_RIGHTS broker (hugepage_fd_broker) -- proves the fd-sharing
     mechanism spawned children actually need, since they don't inherit an
     anonymous mapping the way a forked child would.
  3. Each worker mmaps the received fd, reconstructs tensor views directly
     from the staged header metadata (hugepage_safetensors) -- NOT via
     safetensors.safe_open -- and checksums them against a plain read of the
     source file to prove the bytes are bit-correct.
  4. If --gpu is given, each worker also measures cudaHostRegister throughput
     on its file's span and does one real H2D copy of a sample tensor, so the
     35.71 GB/s number from the Phase 0 gate is confirmed on REAL staged
     model data, not a synthetic buffer.

Usage:
  test_hugepage_shared_pipeline.py <src_dir> <file1> [file2 ...] \
      [--nworkers 4] [--gpu]
"""
import argparse
import hashlib
import multiprocessing
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "hugepage-sharded-loading-exp", "scripts"))
import hugepage_fd_broker as broker
import hugepage_safetensors as hsf
import hugepage_stager as stager


def source_tensor_bytes(src_path, meta):
    import struct
    start, end = meta["data_offsets"]
    with open(src_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        data_start_in_file = 8 + n
        f.seek(data_start_in_file + start)
        return f.read(end - start)


def worker_main(sock_path, src_dir, filenames, gpu, result_q, worker_idx=0):
    try:
        fd, total_len, files_meta = broker.fetch(sock_path)
        buf = stager.mmap_from_fd(fd, total_len)

        mv = memoryview(buf)
        checked, mismatches = 0, []
        for f in filenames:
            base, header, data_start = files_meta[f]
            src_path = os.path.join(src_dir, f)
            for name in header:
                # Raw bytes straight from the staged buffer -- sidesteps
                # numpy's lack of a bfloat16 dtype, and independently checks
                # the same offsets tensor_view() uses (called once below just
                # to sanity-check shape/dtype construction doesn't error).
                start, end = header[name]["data_offsets"]
                got = bytes(mv[data_start + start:data_start + end])
                want = source_tensor_bytes(src_path, header[name])
                checked += 1
                if hashlib.md5(got).digest() != hashlib.md5(want).digest():
                    mismatches.append(name)
                _ = hsf.tensor_view(buf, header, data_start, name)

        gpu_report = None
        if gpu:
            import torch
            gpu_id = worker_idx % torch.cuda.device_count()
            torch.cuda.set_device(gpu_id)
            dev = torch.device(f"cuda:{gpu_id}")
            torch.empty(1, device=dev)
            cudart = torch.cuda.cudart()

            f = filenames[0]
            base, header, data_start = files_meta[f]
            # Largest tensor by data_offsets end -- covers the file's full real
            # extent (header order is not size order; picking the first key
            # understated the span to near-zero on the first run).
            biggest_name = max(header, key=lambda n: header[n]["data_offsets"][1])
            span_len = data_start - base + header[biggest_name]["data_offsets"][1]
            ptr = ctypes_addr(buf, base)
            first_name = biggest_name

            t0 = time.perf_counter()
            rc = cudart.cudaHostRegister(ptr, span_len, 0)
            reg_dt = time.perf_counter() - t0
            reg_gbps = span_len / reg_dt / 1e9 if int(rc) == 0 else None

            t = hsf.tensor_view(buf, header, data_start, first_name)
            gpu_t = torch.empty(t.shape, dtype=t.dtype, device=dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gpu_t.copy_(t, non_blocking=True)
            torch.cuda.synchronize()
            copy_dt = time.perf_counter() - t0
            copy_gbps = (t.numel() * t.element_size()) / copy_dt / 1e9

            if int(rc) == 0:
                cudart.cudaHostUnregister(ptr)
            gpu_report = dict(register_gbps=reg_gbps, copy_gbps=copy_gbps,
                               span_gb=span_len / 1e9)

        result_q.put(dict(pid=os.getpid(), checked=checked, mismatches=mismatches,
                           gpu=gpu_report))
    except Exception as e:
        result_q.put(dict(pid=os.getpid(), error=f"{type(e).__name__}: {e}"))


def ctypes_addr(buf, offset):
    import ctypes
    return ctypes.addressof(buf) + offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--nworkers", type=int, default=4)
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    print(f"staging {a.files} from {a.src_dir} ...", flush=True)
    t0 = time.perf_counter()
    fd, total_len, files_meta, buf = stager.stage(a.src_dir, a.files)
    print(f"staged {total_len/1e9:.2f} GB in {time.perf_counter()-t0:.2f}s", flush=True)

    with tempfile.TemporaryDirectory() as d:
        sock_path = os.path.join(d, "broker.sock")
        srv = broker.FdBrokerServer(sock_path, fd, total_len, files_meta)
        srv.start()

        ctx = multiprocessing.get_context("spawn")
        result_q = ctx.Queue()
        procs = [ctx.Process(target=worker_main,
                              args=(sock_path, a.src_dir, a.files, a.gpu, result_q, i))
                 for i in range(a.nworkers)]
        for p in procs:
            p.start()

        results = [result_q.get(timeout=120) for _ in procs]
        for p in procs:
            p.join(timeout=30)
        srv.stop()

    ok = True
    for r in results:
        if "error" in r:
            print(f"worker pid={r['pid']}: ERROR {r['error']}", flush=True)
            ok = False
            continue
        status = "OK" if not r["mismatches"] else f"MISMATCH {r['mismatches']}"
        print(f"worker pid={r['pid']}: checked={r['checked']} tensors -> {status}", flush=True)
        if r["mismatches"]:
            ok = False
        if r.get("gpu"):
            g = r["gpu"]
            print(f"  gpu: register={g['register_gbps']:.2f} GB/s over {g['span_gb']:.3f} GB, "
                  f"H2D copy={g['copy_gbps']:.2f} GB/s", flush=True)

    print(f"\nRESULT: {'PASS' if ok and len(results) == a.nworkers else 'FAIL'}", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
