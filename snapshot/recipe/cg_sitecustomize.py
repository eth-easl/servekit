# sitecustomize.py — monkeypatches torch.cuda.CUDAGraph to time the ONE layer
# every capture path funnels through (vLLM FULL, breakable, torch.compile
# cudagraph_trees all call capture_begin/capture_end/replay on this class).
# Enabled only when VLLM_CG_INSTRUMENT=<path> is set. CSV columns:
#   idx, kind, capture_begin_ms, capture_end_ms, replay_ms, origin
# "origin" = a short stack token identifying the driving path so we also learn
# WHICH path is active. Flushes every row (the harness SIGKILLs vllm, which
# bypasses atexit).
import os
import time

_LOG = os.environ.get("VLLM_CG_INSTRUMENT")
if _LOG:
    import atexit
    import threading

    _lock = threading.Lock()
    _idx = [0]
    _header_written = [False]

    def _origin_from_stack():
        import traceback
        for fr in traceback.extract_stack()[::-1]:
            fn = fr.filename
            nm = fr.name
            if "breakable_cudagraph" in fn:
                return "breakable:" + (nm or "?")
            if "cudagraph_trees" in fn:
                return "ctrees:" + (nm or "?")
            if "cudagraph_utils" in fn:
                return "vllm:" + (nm or "?")
        return "unknown"

    def _emit(body):
        """Append + flush immediately (harness uses kill -9; no atexit)."""
        with _lock:
            _idx[0] += 1
            with open(_LOG, "a", buffering=1) as f:
                if not _header_written[0]:
                    f.write("idx,kind,capture_begin_ms,capture_end_ms,replay_ms,origin\n")
                    _header_written[0] = True
                f.write(f"{_idx[0]},{body}\n")

    atexit.register(lambda: None)  # keep import; flushing is per-row

    def _install():
        import torch
        CG = torch.cuda.CUDAGraph

        _cb = CG.capture_begin
        _ce = CG.capture_end
        _rp = CG.replay

        def capture_begin(self, *a, **kw):
            t0 = time.perf_counter()
            r = _cb(self, *a, **kw)
            ms = (time.perf_counter() - t0) * 1000.0
            _emit(f"begin,{ms:.3f},,,{_origin_from_stack()}")
            return r

        def capture_end(self, *a, **kw):
            t0 = time.perf_counter()
            r = _ce(self, *a, **kw)
            ms = (time.perf_counter() - t0) * 1000.0
            _emit(f"end,,{ms:.3f},,,{_origin_from_stack()}")
            return r

        def replay(self, *a, **kw):
            t0 = time.perf_counter()
            r = _rp(self, *a, **kw)
            ms = (time.perf_counter() - t0) * 1000.0
            _emit(f"replay,,,{ms:.3f},{_origin_from_stack()}")
            return r

        CG.capture_begin = capture_begin
        CG.capture_end = capture_end
        CG.replay = replay

    # torch may not be imported yet at sitecustomize time; hook its import.
    try:
        _install()
    except Exception:
        import builtins

        _real_import = builtins.__import__

        def _wrap(name, *a, **kw):
            m = _real_import(name, *a, **kw)
            if name == "torch" and not getattr(_wrap, "_done", False):
                _wrap._done = True
                try:
                    _install()
                except Exception:
                    pass
            return m

        builtins.__import__ = _wrap
