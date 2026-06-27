"""
cg_meta_cuda.py — CUDA port of cginst_skip/cg_meta.py: persist + reconstruct
CUDAGraphWrapper entry.output across cold starts, so the capture-phase forward
can be skipped on restore.

THE INSIGHT (unchanged from the HIP prototype)
----------------------------------------------
In snapshot restore mode, the C-interposer (snapshot_record_cuda) FAKES
cudaStreamBeginCapture / cudaStreamEndCapture: EndCapture returns the next
pre-built graph rebuilt from the `.snap` files regardless of what ran between
begin and end. So the forward inside `with torch.cuda.graph(): output =
self.runnable(...)` is NOT needed to attach the graph. Its only essential
product is `entry.output` — a tensor vLLM reads after replay.

The rebuilt graph writes its final result to region_base + recorded_offset.
entry.output must therefore be a tensor viewing EXACTLY that address. We build
such a tensor with a hand-crafted DLPack capsule (ctypes) — the stable
cross-framework path for wrapping a raw device pointer, since PyTorch exposes no
public raw-ptr ctor.

CUDA vs HIP (why this is simpler than the AMD prototype)
--------------------------------------------------------
Under snapshot_redirect_cuda's FIXED base, live_base == snap_base (Δ=0), so the
recorded device pointers are valid UNMODIFIED at restore — no relocation. Two
port deltas from cg_meta.py:
  1. DLPack device type is kDLCUDA (2), not kDLROCM (10).
  2. _region_base() resolves snapshot_record_cuda_region_base() (the N5b Task 3
     export from the record .so), not the HIP redirect symbol.
_reconstruct keeps `region_base + off` (Δ=0 → equals the recorded data_ptr).

vLLM 0.23.0 binding
-------------------
Hooks the same surface the N3 cginst instrumentation patches successfully:
  vllm.compilation.cuda_graph.CUDAGraphWrapper / CUDAGraphEntry / weak_ref_tensors
  vllm.v1.worker.gpu_model_runner.GPUModelRunner (_warmup_and_capture)
  vllm.forward_context (is_forward_context_available / get_forward_context)
  vllm.config.CUDAGraphMode
  vllm.utils.torch_utils.current_stream

MODES (env-driven, loaded via sitecustomize.py)
  VLLM_CG_RECORD_META=<path>   RECORD: observe each capture's output, write JSON.
  VLLM_CG_RESTORE_META=<path>  RESTORE (lazy): skip forward, wrap entry.output.
"""
from __future__ import annotations

import ctypes
import json
import os


def _log(msg):
    print(f"[cg_meta_cuda] {msg}", flush=True)


def _region_base():
    # snapshot_record_cuda (LD_PRELOAD'd) exports snapshot_record_cuda_region_base()
    # (N5b Task 3), which reads the redirect's fixed VMM base. Resolve it directly
    # via ctypes: os.environ would not reflect a C-level setenv made after Python
    # started, and the symbol lives in the interposer .so (global scope via
    # CDLL(None) == RTLD_DEFAULT).
    try:
        lib = ctypes.CDLL(None)
        fn = getattr(lib, "snapshot_record_cuda_region_base", None)
        if fn is not None:
            fn.restype = ctypes.c_uint64
            v = fn()
            if v:
                return v
    except Exception:
        pass
    v = os.environ.get("SNAPSHOT_RECORD_REGION_BASE", "0")
    return int(v, 0) if v else 0


# --------------------------------------------------------------------------
# DLPack raw-pointer wrap (kDLCUDA = 2 for this CUDA build)
# --------------------------------------------------------------------------
class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device_device_type", ctypes.c_int),
        ("device_device_id", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("dtype_code", ctypes.c_uint8),
        ("dtype_bits", ctypes.c_uint8),
        ("dtype_lanes", ctypes.c_uint16),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_capsule_new = ctypes.pythonapi.PyCapsule_New
_capsule_new.restype = ctypes.py_object
_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

_DT_MAP = {
    "torch.float32": (2, 32, 4),
    "torch.float16": (2, 16, 2),
    "torch.bfloat16": (4, 16, 2),
    "torch.float64": (2, 64, 8),
    "torch.int32": (0, 32, 4),
    "torch.int64": (0, 64, 8),
    "torch.int16": (0, 16, 2),
    "torch.int8": (0, 8, 1),
    "torch.uint8": (1, 8, 1),
    "torch.bool": (1, 1, 1),
}

_KEEPLIVE = []  # hold cshapes + DLManagedTensor alive for tensor lifetime


def wrap_device_ptr(ptr, shape, dtype):
    import torch

    key = str(dtype)
    if key not in _DT_MAP:
        raise ValueError(f"unsupported dtype {key}")
    dcode, dbits, _ = _DT_MAP[key]
    cshape = (ctypes.c_int64 * len(shape))(*shape)
    m = _DLManagedTensor()
    m.dl_tensor.data = ptr
    m.dl_tensor.device_device_type = 2  # kDLCUDA (was 10 / kDLROCM in the HIP port)
    m.dl_tensor.device_device_id = 0
    m.dl_tensor.ndim = len(shape)
    m.dl_tensor.dtype_code = dcode
    m.dl_tensor.dtype_bits = dbits
    m.dl_tensor.dtype_lanes = 1
    m.dl_tensor.shape = ctypes.cast(cshape, ctypes.POINTER(ctypes.c_int64))
    m.dl_tensor.strides = None  # NULL => C-contiguous
    m.dl_tensor.byte_offset = 0
    m.manager_ctx = None
    m.deleter = None
    cap = _capsule_new(ctypes.addressof(m), b"dltensor", None)
    t = torch.utils.dlpack.from_dlpack(cap)
    _KEEPLIVE.append((cshape, m))  # keep alive while t lives
    return t


# --------------------------------------------------------------------------
# Recursive output (de)serialization
# --------------------------------------------------------------------------
def _serialize(out, region_base):
    import torch

    if isinstance(out, torch.Tensor):
        if out.numel() == 0:
            return {"k": "empty", "shape": list(out.shape), "dtype": str(out.dtype)}
        return {
            "k": "tensor",
            "off": out.data_ptr() - region_base,
            "shape": list(out.shape),
            "dtype": str(out.dtype),
        }
    if isinstance(out, (list, tuple)):
        return {"k": "seq", "items": [_serialize(x, region_base) for x in out]}
    if isinstance(out, dict):
        return {"k": "dict", "items": {kk: _serialize(v, region_base) for kk, v in out.items()}}
    return {"k": "other", "repr": repr(out)[:120]}


def _reconstruct(meta, live_base):
    import torch

    k = meta["k"]
    if k == "tensor":
        ptr = live_base + meta["off"]
        return wrap_device_ptr(ptr, meta["shape"], _dtype(meta["dtype"]))
    if k == "empty":
        return torch.empty(meta["shape"], dtype=_dtype(meta["dtype"]), device="cuda")
    if k == "seq":
        return [_reconstruct(x, live_base) for x in meta["items"]]
    if k == "dict":
        return {kk: _reconstruct(v, live_base) for kk, v in meta["items"].items()}
    return None  # best-effort for scalars/other


_DTYPE_CACHE = {}


def _dtype(s):
    import torch

    if s not in _DTYPE_CACHE:
        _DTYPE_CACHE[s] = getattr(torch, s.replace("torch.", ""))
    return _DTYPE_CACHE[s]


# --------------------------------------------------------------------------
# Common: hook CUDAGraphWrapper.__call__ capture path
# --------------------------------------------------------------------------
def _install():
    import torch
    from vllm.compilation import cuda_graph as cgmod
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    CUDAGraphWrapper = cgmod.CUDAGraphWrapper
    _orig_call = CUDAGraphWrapper.__call__
    _orig_wrt = cgmod.weak_ref_tensors

    RECORD_PATH = os.environ.get("VLLM_CG_RECORD_META")
    RESTORE_PATH = os.environ.get("VLLM_CG_RESTORE_META")

    # --- weak_ref_tensors: tolerate non-tensor (wrapped/dummy) values ---
    def _safe_wrt(x):
        if x is None:
            return None
        try:
            return _orig_wrt(x)
        except Exception:
            return x

    cgmod.weak_ref_tensors = _safe_wrt

    if RECORD_PATH:
        _install_record(CUDAGraphWrapper, _orig_call, RECORD_PATH)
    elif RESTORE_PATH:
        _install_lazy(CUDAGraphWrapper, _orig_call, GPUModelRunner, RESTORE_PATH)


# --------------------------------------------------------------------------
# RECORD mode
# --------------------------------------------------------------------------
def _install_record(CUDAGraphWrapper, _orig_call, path):
    state = {"n": 0}  # region_base read lazily at capture time (redirect inits
                       # on first cudaMalloc, which is after Python startup)

    def _write(entries):
        # Flush on EVERY capture: the harness kills the server with SIGKILL
        # (kill -9), which bypasses atexit, so a final flush would be lost.
        try:
            with open(path, "w") as f:
                _json_dump = json.dump(
                    {"region_base": _region_base(), "entries": entries}, f
                )
        except Exception as e:
            _log(f"RECORD write failed: {e!r}")

    entries = []

    def _call(self, *args, **kwargs):
        # Record ONLY real captures: mode must match THIS wrapper's runtime_mode
        # AND the entry must be fresh (cudagraph is None). This excludes the
        # eager calls region-wrappers make during a FULL capture (mode=FULL !=
        # their PIECEWISE runtime_mode), which would otherwise be over-counted.
        from vllm.forward_context import (
            is_forward_context_available,
            get_forward_context,
        )

        is_capture = False
        fc = get_forward_context() if is_forward_context_available() else None
        if fc is not None and fc.cudagraph_runtime_mode == self.runtime_mode:
            bd = fc.batch_descriptor
            ent = self.concrete_cudagraph_entries.get(bd)
            if ent is None or ent.cudagraph is None:
                is_capture = True

        out = _orig_call(self, *args, **kwargs)

        if is_capture:
            entries.append(
                {
                    "idx": state["n"],
                    "runtime_mode": fc.cudagraph_runtime_mode.name,
                    "output": _serialize(out, _region_base()),
                }
            )
            state["n"] += 1
            _write(entries)  # immediate flush (SIGKILL-safe)
        return out

    CUDAGraphWrapper.__call__ = _call
    _log(f"RECORD mode -> {path} (incremental flush, SIGKILL-safe)")


# --------------------------------------------------------------------------
# RESTORE (lazy) mode: skip forward, reconstruct entry.output
# --------------------------------------------------------------------------
def _install_lazy(CUDAGraphWrapper, _orig_call, GPUModelRunner, path):
    import torch

    with open(path) as f:
        meta = json.load(f)
    snap_base = meta["region_base"]
    entries_by_idx = {e["idx"]: e for e in meta["entries"]}
    n_expected = len(meta["entries"])

    # live_base is read lazily at first capture: the redirect inits on the
    # first cudaMalloc (during model load), which is AFTER this install runs
    # at Python startup, so _region_base() is 0 here.
    state = {"idx": 0, "skipped": 0, "live_base": None, "logged": False}

    def _call(self, *args, **kwargs):
        # Dispatch exactly like the original __call__, but override the CAPTURE
        # branch to skip the forward (the pre-built graph attaches at EndCapture
        # regardless) and reconstruct entry.output from recorded metadata.
        from vllm.forward_context import (
            is_forward_context_available,
            get_forward_context,
        )
        from vllm.config import CUDAGraphMode

        if not is_forward_context_available():
            # outside the inference path (e.g. vision encoder) -> run for real
            return _orig_call(self, *args, **kwargs)
        fc = get_forward_context()
        mode = fc.cudagraph_runtime_mode
        if mode == CUDAGraphMode.NONE or mode != self.runtime_mode:
            # eager: profiling / warmup -> run for real
            return _orig_call(self, *args, **kwargs)

        bd = fc.batch_descriptor
        entries = self.concrete_cudagraph_entries
        if bd in entries and entries[bd].cudagraph is not None:
            # REPLAY: the original code replays + returns entry.output
            return _orig_call(self, *args, **kwargs)

        # CAPTURE: skip the forward. Create the graph context so EndCapture
        # attaches the pre-built graph, then reconstruct entry.output.
        from vllm.compilation.cuda_graph import CUDAGraphEntry
        from vllm.utils.torch_utils import current_stream

        if bd not in entries:
            entries[bd] = CUDAGraphEntry(batch_descriptor=bd)
        entry = entries[bd]

        cudagraph = torch.cuda.CUDAGraph()
        meta_e = entries_by_idx.get(state["idx"])
        # Empty graph context: the C-interposer fakes begin/end, attaching the
        # next pre-built graph from the .snap at EndCapture.
        with torch.cuda.graph(
            cudagraph, pool=self.graph_pool, stream=current_stream()
        ):
            pass
        # Resolve live_base now (redirect has initialized by capture time).
        # CUDA Δ=0: live_base == snap_base, so reconstructed pointers equal the
        # recorded data_ptrs exactly.
        if state["live_base"] is None:
            state["live_base"] = _region_base()
            if not state["logged"]:
                state["logged"] = True
                delta = state["live_base"] - snap_base
                _log(
                    f"RESTORE lazy: snap_base=0x{snap_base:x} "
                    f"live_base=0x{state['live_base']:x} delta=0x{delta:x}"
                    + (" (Δ=0, pointers valid unmodified)" if delta == 0 else "")
                )
        out = None
        if meta_e is not None:
            try:
                out = _reconstruct(meta_e["output"], state["live_base"])
            except Exception as e:
                _log(f"reconstruct idx={state['idx']} failed: {e!r}")
        entry.output = out
        entry.cudagraph = cudagraph
        state["idx"] += 1
        state["skipped"] += 1
        if state["idx"] == n_expected:
            _log(
                f"RESTORE lazy: all {n_expected} entries reconstructed "
                f"(skipped {state['skipped']} forwards)"
            )
        return out

    CUDAGraphWrapper.__call__ = _call

    # Skip warmup _dummy_run calls; only the capture _dummy_run reaches the
    # wrapper capture path above (where the forward is skipped).
    def _skip_warmups(self, desc, cudagraph_runtime_mode, **kw):
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

    GPUModelRunner._warmup_and_capture = _skip_warmups
    _log(f"RESTORE lazy mode installed")


try:
    _install()
except Exception as e:
    _log(f"install deferred: {e!r}")
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
