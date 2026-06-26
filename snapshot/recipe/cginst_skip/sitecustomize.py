# sitecustomize.py — auto-imported by Python. Loads whichever cg* hook is
# requested via env vars. Each module self-gates on its own env var.
import os

# cg_skip: measurement / shim prototypes (VLLM_CG_SKIP_CAPTURE=measure|shim)
if os.environ.get("VLLM_CG_SKIP_CAPTURE", ""):
    try:
        import cg_skip  # noqa: F401
    except Exception as e:
        print(f"[cginst_skip/sitecustomize] cg_skip failed: {e!r}", flush=True)

# cg_meta: persist + reconstruct entry.output across cold starts.
#   RECORD:  VLLM_CG_RECORD_META=<path>  (observe capture outputs, write JSON)
#   RESTORE: VLLM_CG_RESTORE_META=<path> (skip forward, wrap entry.output)
if os.environ.get("VLLM_CG_RECORD_META", "") or os.environ.get(
    "VLLM_CG_RESTORE_META", ""
):
    try:
        import cg_meta  # noqa: F401
    except Exception as e:
        print(f"[cginst_skip/sitecustomize] cg_meta failed: {e!r}", flush=True)
