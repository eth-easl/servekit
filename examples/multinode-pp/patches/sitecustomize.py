"""Raise sglang's post-load barrier timeout, which is a module constant.

`dist_barrier_after_load` calls `monitored_barrier(timeout=
UNBALANCED_MODEL_LOADING_TIMEOUT_S)` per TP group. The constant is hardcoded at
480 s, only rank 0 observes it, and a cold 1.5 TB read off Lustre has shown
within-node spreads well past that -- so whenever the slowest rank in a group is
not rank 0, rank 0 raises and kills the whole engine.

Python auto-imports sitecustomize at interpreter startup, before sglang runs, so
this reaches the scheduler subprocesses too; putting the directory on PYTHONPATH
is the whole installation.
"""
import os

try:
    import sglang.srt.model_executor.model_runner_components.load_model_utils as _lm

    _lm.UNBALANCED_MODEL_LOADING_TIMEOUT_S = int(
        os.environ.get("UNBALANCED_MODEL_LOADING_TIMEOUT_S", "3600")
    )
except Exception:
    pass
