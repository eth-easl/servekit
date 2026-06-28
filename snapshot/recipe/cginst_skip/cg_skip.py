"""
cg_skip.py — Skip the model forward during CUDA graph capture while keeping
the capture FUNCTIONAL, by feeding dummy tensors of recorded shapes.

Why shapes matter: PIECEWISE capture interleaves graph-captured regions with
EAGER ops (attention between regions, _dummy_run's output indexing). Returning
None from a skipped region crashes the eager data flow. But CUDA graph capture
is DATA-INDEPENDENT — it records the kernel sequence + arg pointers, not the
tensor values. So feeding a zero tensor of the correct shape captures an
identical graph.

Strategy:
  1. The profiling forward (profile_run, NONE/eager mode) runs the real model
     ONCE. We hook CUDAGraphWrapper.__call__ to record each region's output
     shape + dtype (keyed by wrapper instance, which persists across runs).
  2. During capture (inside `with torch.cuda.graph()`), we replace runnable's
     output with a zero tensor of the recorded shape, scaled to the current
     num_tokens (the batch dim). The eager attention runs on zeros (produces
     zeros), the next region captures correctly, and the HIP interposer
     rebuilds the real graph topology + kernels from the snapshot.

Cost: one real forward (~17s) instead of ~340s of per-size capture forwards.

Modes (env VLLM_CG_SKIP_CAPTURE):
  measure - skip ALL _dummy_run calls (measurement only, non-functional).
  shim    - shape-dummy capture (functional + fast, needs HIP interposer).
"""
import os
import threading
import time

_MODE = os.environ.get("VLLM_CG_SKIP_CAPTURE", "")
# True only for shim_pw: PIECEWISE-only skip (FULL gets real forward).
# In plain shim mode, ALL captures (PIECEWISE + FULL) are dummied.
_SHIM_PW = _MODE == "shim_pw"

_tls = threading.local()


def _log(msg):
    print(f"[cg_skip] {msg}", flush=True)


def _in_capture():
    return getattr(_tls, "capture", False)


def _set_phase_env(mode_name):
    """Signal the current capture phase (PIECEWISE/FULL) to the HIP
    interposer in-process. The interposer reads SNAPSHOT_RESTORE_PHASE at each
    hipStreamEndCapture and serves a graph from the matching PW/FULL pool
    (mode-aware serve; see snapshot_record.cpp). os.environ.__setitem__ calls
    libc setenv(), which updates the same process environ that the
    interposer's std::getenv reads — no IPC needed."""
    try:
        os.environ["SNAPSHOT_RESTORE_PHASE"] = str(mode_name or "")
    except Exception:
        pass


def _install():
    import torch
    from vllm.compilation import cuda_graph as cgmod
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    CUDAGraphWrapper = cgmod.CUDAGraphWrapper

    # --- weak_ref_tensors: tolerate non-tensor dummies ---
    _orig_wrt = cgmod.weak_ref_tensors

    def _safe_wrt(x):
        if x is None:
            return None
        try:
            return _orig_wrt(x)
        except Exception:
            return x

    cgmod.weak_ref_tensors = _safe_wrt

    # --- track per-wrapper recorded shapes from the profiling (eager) run ---
    # key=id(wrapper) -> (ref_num_tokens, shape, dtype, device)
    _shapes = {}
    _caps = {"n_dummy": 0, "n_real": 0}

    # --- set _in_capture flag around `with torch.cuda.graph()` ---
    _Graph = torch.cuda.graph
    _orig_enter = _Graph.__enter__
    _orig_exit = _Graph.__exit__

    def _enter(self):
        _tls.capture = True
        return _orig_enter(self)

    def _exit(self, *a):
        _tls.capture = False
        return _orig_exit(self, *a)

    _Graph.__enter__ = _enter
    _Graph.__exit__ = _exit

    # --- patch CUDAGraphWrapper.__call__ ---
    _orig_call = CUDAGraphWrapper.__call__

    @torch.no_grad()
    def _dummy_like(ref_nt, shape, dtype, device, cur_nt):
        """Build a zero tensor of `shape` with the batch dim scaled from
        ref_nt (profiling size) to cur_nt (current capture size)."""
        new_shape = list(shape)
        if ref_nt and ref_nt != cur_nt:
            for i, d in enumerate(new_shape):
                if d == ref_nt:
                    new_shape[i] = cur_nt
        return torch.zeros(new_shape, dtype=dtype, device=device)

    def _as_dummy(out, cur_nt):
        """Recursively replace tensors with shape-matched dummies."""
        if out is None:
            return None
        if isinstance(out, torch.Tensor):
            return torch.zeros_like(out)
        if isinstance(out, (list, tuple)):
            keep = type(out)
            return keep(_as_dummy(x, cur_nt) for x in out)
        if isinstance(out, dict):
            return {k: _as_dummy(v, cur_nt) for k, v in out.items()}
        return out

    def _record(out):
        """Record output metadata from the eager/profiling run."""
        if isinstance(out, torch.Tensor) and out.numel() > 0:
            return (out.shape[0] if out.dim() > 0 else 1,
                    tuple(out.shape), out.dtype, out.device)
        return None

    def _call(self, *args, **kwargs):
        key = id(self)
        real = self.runnable

        def _wrapped_runnable(*a, **k):
            if _in_capture():
                # In shim_pw mode, only PIECEWISE captures are skipped
                # (rebuilt from snapshot); FULL captures run the real forward
                # so the live-captured FULL graph is valid (no fault).
                mode = getattr(_tls, "cg_mode", "")
                # shim: dummy ALL captures. shim_pw: dummy only PIECEWISE.
                pw_skip = (mode == "PIECEWISE" or mode == "" or
                           not _SHIM_PW)
                if pw_skip:
                    rec = _shapes.get(key)
                    if rec is not None:
                        ref_nt, shape, dtype, device = rec
                        cur_nt = ref_nt
                        for x in a:
                            if isinstance(x, torch.Tensor) and x.dim() > 0:
                                cur_nt = x.shape[0]
                                break
                        _caps["n_dummy"] += 1
                        return _dummy_like(ref_nt, shape, dtype, device, cur_nt)
                _caps["n_real"] += 1
                return real(*a, **k)
            out = real(*a, **k)
            if key not in _shapes:
                rec = _record(out)
                if rec is not None:
                    _shapes[key] = rec
            return out

        self.runnable = _wrapped_runnable
        try:
            return _orig_call(self, *args, **kwargs)
        finally:
            self.runnable = real

    CUDAGraphWrapper.__call__ = _call

    if _MODE == "measure":
        # measurement: skip everything (non-functional)
        def _skip(self, desc, cudagraph_runtime_mode, **kw):
            pass

        _orig_capture = GPUModelRunner._capture_cudagraphs

        def _timed(self, batch_descriptors, cudagraph_runtime_mode):
            n = len(batch_descriptors)
            t0 = time.perf_counter()
            _orig_capture(self, batch_descriptors, cudagraph_runtime_mode)
            el = time.perf_counter() - t0
            print(f"[cg_skip] CAPTURE_PHASE_S={el:.1f} "
                  f"mode={cudagraph_runtime_mode.name} n={n}", flush=True)

        GPUModelRunner._warmup_and_capture = _skip
        GPUModelRunner._capture_cudagraphs = _timed
        _log("installed MEASURE (non-functional, skip all forwards)")

    elif _MODE in ("shim", "shim_pw"):
        # shim    : skip ALL captures (PIECEWISE + FULL) — dummies everywhere.
        #          Fast, but a FULL graph with no snapshot is dummy-captured
        #          (invalid) → hangs vLLM startup. Only safe when EVERY graph
        #          (incl. FULL) has a restorable snapshot.
        # shim_pw : skip ONLY PIECEWISE captures (rebuilt from snapshot); run
        #          the REAL forward for FULL captures (live-captured, valid).
        #          This avoids the FULL-decode-graph fault and only needs
        #          PIECEWISE snapshots in the restore dir. FULL is cheap to
        #          live-capture (~7s each) so the cold-start win is preserved.
        _PW_ONLY = _MODE == "shim_pw"

        def _skip_warmup_keep_capture(self, desc, cudagraph_runtime_mode, **kw):
            self._dummy_run(
                desc.num_tokens,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                uniform_decode=desc.uniform,
                allow_microbatching=kw.get("allow_microbatching", False),
                skip_eplb=True,
                remove_lora=False,
                num_active_loras=desc.num_active_loras,
                is_graph_capturing=True,
                profile_seq_lens=kw.get("profile_seq_lens"),
            )

        _orig_capture = GPUModelRunner._capture_cudagraphs

        def _timed(self, batch_descriptors, cudagraph_runtime_mode):
            n = len(batch_descriptors)
            _tls.cg_mode = getattr(cudagraph_runtime_mode, "name", "")
            _set_phase_env(_tls.cg_mode)
            t0 = time.perf_counter()
            _orig_capture(self, batch_descriptors, cudagraph_runtime_mode)
            el = time.perf_counter() - t0
            print(f"[cg_skip] CAPTURE_PHASE_S={el:.1f} "
                  f"mode={cudagraph_runtime_mode.name} n={n} "
                  f"(dummy={_caps['n_dummy']} real={_caps['n_real']} "
                  f"shapes_recorded={len(_shapes)})", flush=True)

        GPUModelRunner._warmup_and_capture = _skip_warmup_keep_capture
        GPUModelRunner._capture_cudagraphs = _timed
        if _PW_ONLY:
            _log("installed SHIM_PW: PIECEWISE skipped (snapshot rebuild), "
                 "FULL real-forward (live-capture)")
        else:
            _log("installed SHIM: shapes recorded on profile run, dummies on capture")

    elif _MODE == "record_pw":
        # Record mode: run REAL forward for PIECEWISE captures (graph is
        # captured normally + recorded by the snapshot interposer), but SKIP
        # FULL captures entirely (no graph produced, no snapshot file). This
        # yields a PIECEWISE-only snapshot dir that shim_pw restore consumes
        # directly — FULL graphs are live-captured at restore time (cheap,
        # ~7s each, avoids the FULL-decode-graph trajectory fault).
        _orig_warmup = GPUModelRunner._warmup_and_capture

        def _record_pw(self, desc, cudagraph_runtime_mode, **kw):
            mode = getattr(cudagraph_runtime_mode, "name", "")
            if mode == "FULL":
                n = len(getattr(desc, "batch_descriptors", [])) if hasattr(
                    desc, "batch_descriptors") else "?"
                _log(f"record_pw: SKIP FULL capture (no snapshot)")
                return
            t0 = time.perf_counter()
            _orig_warmup(self, desc, cudagraph_runtime_mode, **kw)
            _log(f"record_pw: PIECEWISE capture done "
                 f"({time.perf_counter() - t0:.1f}s, real forward + recorded)")

        GPUModelRunner._warmup_and_capture = _record_pw
        _log("installed RECORD_PW: PIECEWISE recorded (real), FULL skipped")


try:
    _install()
except Exception:
    import builtins
    _real = builtins.__import__

    def _wrap(name, *a, **kw):
        m = _real(name, *a, **kw)
        if name.startswith("vllm") and not getattr(_wrap, "_done", False):
            try:
                _install()
                _wrap._done = True
            except Exception:
                pass
        return m

    builtins.__import__ = _wrap
