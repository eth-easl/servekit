# sitecustomize.py — auto-imported by Python. Loads the CUDA cg_meta_cuda hook
# when either of its env vars is set. Mirrors cginst_skip/sitecustomize.py.
import os

# cg_meta_cuda: persist + reconstruct entry.output across cold starts.
#   RECORD:  VLLM_CG_RECORD_META=<path>  (observe capture outputs, write JSON)
#   RESTORE: VLLM_CG_RESTORE_META=<path> (skip forward, wrap entry.output)
if os.environ.get("VLLM_CG_RECORD_META", "") or os.environ.get(
    "VLLM_CG_RESTORE_META", ""
):
    try:
        import cg_meta_cuda  # noqa: F401
    except Exception as e:
        print(f"[cginst_cuda/sitecustomize] cg_meta_cuda failed: {e!r}", flush=True)
