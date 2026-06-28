# Instrumentation patch for vllm/compilation/breakable_cudagraph.py
# This vLLM build uses BreakableCUDAGraphCapture (its OWN path) for PIECEWISE,
# NOT torch.compile's cudagraph_trees -- so the real capture cost lives here.
# Decomposes each PIECEWISE _capture into:
#   - setup_ms:     gc.collect + empty_cache + offloader sync
#   - forward_ms:   the `with capture:` block (model forward + segment begin/ends)
#   - seg_begin_ms: sum of capture_begin() across N segments (stream begin capture)
#   - seg_end_ms:   sum of capture_end() across N segments (graph finalization)
#   - num_segments: N (break points + 1)
# The first replay() (instantiate) is timed separately on the first _replay.
# Enabled only when VLLM_CG_INSTRUMENT is set; CSV to $VLLM_CG_INSTRUMENT.
import re
from pathlib import Path

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/compilation/breakable_cudagraph.py")
src = P.read_text()
MARK = "# <<<VLLM_CG_INSTRUMENT>>>"

# Idempotent: strip prior patch markers.
for m in list(re.finditer(r"[^\n]*" + re.escape(MARK) + r"[^\n]*\n", src)):
    pass
src = re.sub(r"\n?[^\n]*" + re.escape(MARK) + r"[^\n]*\n", "\n", src)

# 1) Module-level helpers (insert after first import block).
helper = '''import os as _cg_os
import time as _cg_time
def _cg_log_path():
    return _cg_os.environ.get("VLLM_CG_INSTRUMENT")
def _cg_emit(kind, **kw):
    p = _cg_log_path()
    if not p: return
    with open(p, "a", buffering=1) as f:
        if _cg_os.path.getsize(p) == 0 and kind == "capture":
            f.write("kind,num_segments,setup_ms,forward_ms,seg_begin_ms,seg_end_ms\\n")
        if kind == "capture":
            f.write(f"capture,{kw['nseg']},{kw['setup']:.1f},{kw['forward']:.1f},"
                    f"{kw['begin']:.1f},{kw['end']:.1f}\\n")
        elif kind == "replay":
            f.write(f"replay,{kw['nseg']},,,,{kw['replay']:.1f}\\n")  # <<<VLLM_CG_INSTRUMENT>>>
'''
# anchor: insert after the top-of-file imports. Use the `import dataclasses` line.
anchor = "import dataclasses"
assert anchor in src, "dataclasses import anchor not found"
src = src.replace(anchor, helper.strip("\n") + "\n" + anchor, 1)

# 2) Time capture_begin in _begin_segment and accumulate on the capture object.
b_old = '''    def _begin_segment(self) -> None:
        assert not self._capturing
        g = torch.cuda.CUDAGraph()
        if self.pool is not None:
            g.capture_begin(pool=self.pool)
        else:
            g.capture_begin()
        self._current_graph = g
        self._capturing = True'''
b_new = '''    def _begin_segment(self) -> None:
        assert not self._capturing
        _t0 = _cg_time.perf_counter()  # <<<VLLM_CG_INSTRUMENT>>>
        g = torch.cuda.CUDAGraph()
        if self.pool is not None:
            g.capture_begin(pool=self.pool)
        else:
            g.capture_begin()
        self._seg_begin_ms = getattr(self, "_seg_begin_ms", 0.0) + (_cg_time.perf_counter() - _t0) * 1000.0  # <<<VLLM_CG_INSTRUMENT>>>
        self._current_graph = g
        self._capturing = True'''
assert b_old in src, "_begin_segment anchor not found"
src = src.replace(b_old, b_new, 1)

# 3) Time capture_end in _end_segment and accumulate.
e_old = '''    def _end_segment(self) -> None:
        if not self._capturing:
            return
        assert self._current_graph is not None
        self._current_graph.capture_end()
        self.segments.append(self._current_graph.replay)
        self._num_graphs += 1
        self._current_graph = None
        self._capturing = False'''
e_new = '''    def _end_segment(self) -> None:
        if not self._capturing:
            return
        assert self._current_graph is not None
        _t0 = _cg_time.perf_counter()  # <<<VLLM_CG_INSTRUMENT>>>
        self._current_graph.capture_end()
        self._seg_end_ms = getattr(self, "_seg_end_ms", 0.0) + (_cg_time.perf_counter() - _t0) * 1000.0  # <<<VLLM_CG_INSTRUMENT>>>
        self.segments.append(self._current_graph.replay)
        self._num_graphs += 1
        self._current_graph = None
        self._capturing = False'''
assert e_old in src, "_end_segment anchor not found"
src = src.replace(e_old, e_new, 1)

# 4) Time replay() (first call per capture = hipGraphInstantiate x N).
r_old = '''    def replay(self) -> None:
        for r in self.segments:
            r()'''
r_new = '''    def replay(self) -> None:
        _t0 = _cg_time.perf_counter()  # <<<VLLM_CG_INSTRUMENT>>>
        for r in self.segments:
            r()
        _cg_emit("replay", nseg=self._num_graphs, replay=(_cg_time.perf_counter() - _t0) * 1000.0)  # <<<VLLM_CG_INSTRUMENT>>>'''
assert r_old in src, "replay anchor not found"
src = src.replace(r_old, r_new, 1)

# 5) Time setup vs the with-block in _capture, emit one capture row.
c_old = '''        gc.collect()
        torch.accelerator.empty_cache()
        # Sync the offloader's copy stream before capture so any in-flight
        # pre-capture prefetches are complete and don't leak into the graph.
        get_offloader().sync_prev_onload()'''
c_new = '''        _cg_setup_t0 = _cg_time.perf_counter()  # <<<VLLM_CG_INSTRUMENT>>>
        gc.collect()
        torch.accelerator.empty_cache()
        # Sync the offloader's copy stream before capture so any in-flight
        # pre-capture prefetches are complete and don't leak into the graph.
        get_offloader().sync_prev_onload()
        _cg_setup_ms = (_cg_time.perf_counter() - _cg_setup_t0) * 1000.0  # <<<VLLM_CG_INSTRUMENT>>>'''
assert c_old in src, "_capture setup anchor not found"
src = src.replace(c_old, c_new, 1)

cap_old = '''        capture = BreakableCUDAGraphCapture(pool=self.graph_pool)
        with capture:
            output = self.runnable(*args, **kwargs)
            # Join the offloader's copy stream while we still hold the last
            # segment open, so the join is captured into the graph (otherwise
            # we get an "unjoined stream" error on subsequent forwards).
            get_offloader().join_after_forward()
            # Convert output to a weak ref *inside* the capture context so the
            # strong ref is dropped before the last segment closes, letting
            # the cudagraph pool reclaim/reuse that memory immediately for
            # the next batch descriptor's capture.
            output = weak_ref_tensors(output)

        entry.capture = capture'''
cap_new = '''        capture = BreakableCUDAGraphCapture(pool=self.graph_pool)
        _cg_fwd_t0 = _cg_time.perf_counter()  # <<<VLLM_CG_INSTRUMENT>>>
        with capture:
            output = self.runnable(*args, **kwargs)
            # Join the offloader's copy stream while we still hold the last
            # segment open, so the join is captured into the graph (otherwise
            # we get an "unjoined stream" error on subsequent forwards).
            get_offloader().join_after_forward()
            # Convert output to a weak ref *inside* the capture context so the
            # strong ref is dropped before the last segment closes, letting
            # the cudagraph pool reclaim/reuse that memory immediately for
            # the next batch descriptor's capture.
            output = weak_ref_tensors(output)
        _cg_emit("capture", nseg=capture._num_graphs, setup=_cg_setup_ms,
                 forward=(_cg_time.perf_counter() - _cg_fwd_t0) * 1000.0,
                 begin=getattr(capture, "_seg_begin_ms", 0.0),
                 end=getattr(capture, "_seg_end_ms", 0.0))  # <<<VLLM_CG_INSTRUMENT>>>

        entry.capture = capture'''
assert cap_old in src, "_capture with-block anchor not found"
src = src.replace(cap_old, cap_new, 1)

P.write_text(src)
print("patched OK:", P)
print("instrument markers:", src.count("VLLM_CG_INSTRUMENT"))
