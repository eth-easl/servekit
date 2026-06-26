# sitecustomize.py — monkeypatches torch.cuda.CUDAGraph to time the ONE layer
# every capture path funnels through. CSV columns:
#   idx, kind, capture_begin_ms, capture_end_ms, replay_ms, capture_wall_ms, origin
# capture_wall_ms (on "end" rows) = wall time from capture_begin to capture_end
#   for the SAME graph instance = the forward compute (the real capture cost).
# Flushes every row (harness uses kill -9; bypasses atexit).
import os
import time

_LOG = os.environ.get("VLLM_CG_INSTRUMENT")
_DEBUG = os.environ.get("VLLM_CG_DEBUG", "")  # if set, dump full stacks here
if _LOG:
    import threading

    _lock = threading.Lock()
    _idx = [0]
    _header_written = [False]
    _begin_ts = {}  # id(g) -> perf_counter at capture_begin (forward start)
    _stack_dumped = [0]

    def _classify(fn):
        # fn = frame filename
        b = os.path.basename(fn)
        if "breakable_cudagraph" in b:
            return "breakable"
        if "cudagraph_trees" in b:
            return "ctrees"
        if b in ("cudagraph_utils.py", "cuda_graph.py"):
            return "vllm-cg"
        if "encoder_cudagraph" in b:
            return "encoder-cg"
        if "gpu_ubatch_wrapper" in b:
            return "ubatch"
        return None

    def _origin_from_stack():
        import traceback
        frames = traceback.extract_stack()
        # walk innermost -> outermost, classify each frame
        for fr in frames[::-1]:
            o = _classify(fr.filename)
            if o:
                return f"{o}:{fr.name}"
        return "unknown"

    def _dump_full_stack(tag):
        if not _DEBUG or _stack_dumped[0] >= 5:
            return
        import traceback
        with _lock:
            _stack_dumped[0] += 1
            n = _stack_dumped[0]
        with open(_DEBUG, "a") as f:
            f.write(f"\n===== {tag} #{n} =====\n")
            f.write("".join(traceback.format_stack()))

    def _emit(body):
        with _lock:
            _idx[0] += 1
            with open(_LOG, "a", buffering=1) as f:
                if not _header_written[0]:
                    f.write(
                        "idx,kind,capture_begin_ms,capture_end_ms,replay_ms,"
                        "capture_wall_ms,origin\n"
                    )
                    _header_written[0] = True
                f.write(f"{_idx[0]},{body}\n")

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
            _begin_ts[id(self)] = t0
            _dump_full_stack("begin")
            _emit(f"begin,{ms:.3f},,,,{_origin_from_stack()}")
            return r

        def capture_end(self, *a, **kw):
            t0 = time.perf_counter()
            r = _ce(self, *a, **kw)
            ms = (time.perf_counter() - t0) * 1000.0
            wall = ""
            bts = _begin_ts.pop(id(self), None)
            if bts is not None:
                wall = f"{(time.perf_counter() - bts) * 1000.0:.1f}"
            _dump_full_stack("end")
            _emit(f"end,,{ms:.3f},,{wall},{_origin_from_stack()}")
            return r

        def replay(self, *a, **kw):
            t0 = time.perf_counter()
            r = _rp(self, *a, **kw)
            ms = (time.perf_counter() - t0) * 1000.0
            _emit(f"replay,,,{ms:.3f},,{_origin_from_stack()}")
            return r

        CG.capture_begin = capture_begin
        CG.capture_end = capture_end
        CG.replay = replay

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
