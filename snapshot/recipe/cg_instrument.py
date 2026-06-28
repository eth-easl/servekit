# Instrumentation patch for torch/_inductor/cudagraph_trees.py
# Decomposes the per-subgraph PIECEWISE capture cost into:
#   - capture block:  the `with torch.cuda.graph(...)` (forward + capture overhead)
#   - metadata setup: tensor-metadata bookkeeping after capture
#   - first replay:   self.graph.replay() (triggers lazy hipGraphInstantiate)
# Goal: confirm the prize — that the capture block + replay dominate, so a
# snapshot that injects a pre-built graph and skips these two recovers most of
# the ~17.5s/it PIECEWISE cost. Enabled only when VLLM_CG_INSTRUMENT is set;
# writes one CSV line per CUDAGraphNode to $VLLM_CG_INSTRUMENT.
#
# Apply with: python apply_cg_instrument.py  (idempotent: strips prior patch)
import re
import sys
from pathlib import Path

CT = Path("/usr/local/lib/python3.12/dist-packages/torch/_inductor/cudagraph_trees.py")
src = CT.read_text()

MARK = "# <<<VLLM_CG_INSTRUMENT>>>"

# Idempotent: strip any previous patch.
src = re.sub(
    r"\n?[^#\n]*" + re.escape(MARK) + r"[^\n]*\n",
    "\n",
    src,
)
src = src.replace(
    "# <<<VLLM_CG_INSTRUMENT_HELPERS>>>",
    "",
)
# remove the block of helper lines we may have inserted
src = re.sub(
    r"    # ---VLLM_CG_INSTRUMENT helpers---.*?    # ---end helpers---\n",
    "",
    src,
    flags=re.DOTALL,
)

# 1) Inject helpers + counters near the top of the module (after imports).
helper = '''
    # ---VLLM_CG_INSTRUMENT helpers---
    import os as _cg_os
    import time as _cg_time
    _CG_LOG = None
    _CG_N = [0]
    def _cg_open():
        global _CG_LOG
        if _CG_LOG is not None: return _CG_LOG
        p = _cg_os.environ.get("VLLM_CG_INSTRUMENT")
        if not p: return None
        _CG_LOG = open(p, "a", buffering=1)
        if _CG_LOG.tell() == 0:
            _CG_LOG.write("idx,capture_ms,meta_ms,replay_ms,total_ms,depth\\n")
        return _CG_LOG
    def _cg_emit(**kw):
        f = _cg_open()
        if f is None: return
        _CG_N[0] += 1
        f.write(f"{_CG_N[0]},{kw['capture']:.3f},{kw['meta']:.3f},"
                f"{kw['replay']:.3f},{kw['total']:.3f},{kw['depth']}\\n")
    # ---end helpers---
'''
# Insert helper block right after the CUDAGraphNode class docstring/imports area:
# anchor on the first method line of CUDAGraphNode to keep indentation valid.
anchor = "class CUDAGraphNode:"
assert anchor in src, "CUDAGraphNode anchor not found"
src = src.replace(anchor, helper.strip("\n") + "\n\n" + anchor, 1)

# 2) Time the capture block inside _record. Replace the `with (...) as _:` that
#    wraps the capture, adding timers around it. We match the model(inputs) call
#    inside and bracket the whole `with` block.
rec_old = '''        with (
            preserve_rng_state(),
            torch.cuda.device(self.device),
            clear_cublas_manager(),
            torch.cuda.graph(
                self.graph,
                stream=self.stream,
                pool=self.cuda_graphs_pool,
                capture_error_mode="thread_local",
            ),
            get_history_recording(),
        ):
            static_outputs = model(inputs)'''
rec_new = '''        _cg_t_cap0 = _cg_time.perf_counter()  # <<<VLLM_CG_INSTRUMENT>>>
        with (
            preserve_rng_state(),
            torch.cuda.device(self.device),
            clear_cublas_manager(),
            torch.cuda.graph(
                self.graph,
                stream=self.stream,
                pool=self.cuda_graphs_pool,
                capture_error_mode="thread_local",
            ),
            get_history_recording(),
        ):
            static_outputs = model(inputs)
        torch.cuda.synchronize()  # <<<VLLM_CG_INSTRUMENT>>>
        self._cg_capture_ms = (_cg_time.perf_counter() - _cg_t_cap0) * 1000.0  # <<<VLLM_CG_INSTRUMENT>>>'''
assert rec_old in src, "_record capture block not found verbatim"
src = src.replace(rec_old, rec_new, 1)

# 3) Time metadata + replay in __init__. The init does _record, then metadata,
#    then self.graph.replay(). We bracket the replay and emit one CSV row.
init_old = '''        with dynamo_timed_cudagraph("CUDAGraphNode.record", compile_id, mode):
            self.recording_outputs: Optional[OutputType] = self._record(
                wrapped_function.model, recording_inputs
            )
        self.outputs_metadata: OutputList[Union[dict[str, Any], int, None]] = []'''
init_new = '''        _cg_t_tot0 = _cg_time.perf_counter()  # <<<VLLM_CG_INSTRUMENT>>>
        with dynamo_timed_cudagraph("CUDAGraphNode.record", compile_id, mode):
            self.recording_outputs: Optional[OutputType] = self._record(
                wrapped_function.model, recording_inputs
            )
        _cg_t_meta0 = _cg_time.perf_counter()  # <<<VLLM_CG_INSTRUMENT>>>
        self.outputs_metadata: OutputList[Union[dict[str, Any], int, None]] = []'''
assert init_old in src, "__init__ _record anchor not found"
src = src.replace(init_old, init_new, 1)

# Replace the trailing `self.graph.replay()` in __init__ with timed version.
repl_old = '''            else:
                assert isinstance(out, (int, type(None))), type(out)
                self.outputs_metadata.append(out)

        self.graph.replay()

    def _copy_inputs_and_remove_from_src('''
repl_new = '''            else:
                assert isinstance(out, (int, type(None))), type(out)
                self.outputs_metadata.append(out)

        _cg_meta_ms = (_cg_time.perf_counter() - _cg_t_meta0) * 1000.0  # <<<VLLM_CG_INSTRUMENT>>>
        _cg_t_rep0 = _cg_time.perf_counter()  # <<<VLLM_CG_INSTRUMENT>>>
        self.graph.replay()
        torch.cuda.synchronize()  # <<<VLLM_CG_INSTRUMENT>>>
        _cg_rep_ms = (_cg_time.perf_counter() - _cg_t_rep0) * 1000.0  # <<<VLLM_CG_INSTRUMENT>>>
        _cg_depth = len(self.path_weakrefs) if getattr(self, "path_weakrefs", None) else 0  # <<<VLLM_CG_INSTRUMENT>>>
        _cg_emit(capture=getattr(self, "_cg_capture_ms", 0.0),
                 meta=_cg_meta_ms, replay=_cg_rep_ms,
                 total=(_cg_time.perf_counter() - _cg_t_tot0) * 1000.0,
                 depth=_cg_depth)  # <<<VLLM_CG_INSTRUMENT>>>

    def _copy_inputs_and_remove_from_src('''
assert repl_old in src, "__init__ replay anchor not found"
src = src.replace(repl_old, repl_new, 1)

CT.write_text(src)
print("patched OK:", CT)
print("helper block lines:", src.count("# ---VLLM_CG_INSTRUMENT"))
print("instrument markers:", src.count("VLLM_CG_INSTRUMENT"))
