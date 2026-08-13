"""Pipeline-parallel filenames for SGLang's `sharded_state` checkpoints.

`sharded_state` keys shard files on `tp_rank`, which restarts at 0 on every
pipeline stage, so stages collide on one filename and overwrite each other.

An SGLang plugin rather than a monkeypatch because SGLang forces the `spawn`
start method: the schedulers that read and write weights are fresh interpreters
that never see a patch made in the parent. Filenames match SGLang's own
pipeline-parallel branch, so if that lands upstream these hooks go redundant
rather than conflicting.

SGLang imports stay inside function bodies -- `prepare` imports
`wait_for_writes`, and servekit must not pull SGLang into its own process.
"""
from __future__ import annotations

import os
import time

PP_PATTERN = "model-pp-{pp_rank}-rank-{rank}-part-{part}.safetensors"
DONE_PREFIX = ".sharded-state-done"

SAVE_TARGET = "sglang.srt.model_loader.loader.ShardedStateLoader.save_model"
LOAD_TARGET = "sglang.srt.model_loader.loader.ShardedStateLoader.load_model"


def done_marker(path: str, pp_rank: int, tp_rank: int) -> str:
    return os.path.join(path, f"{DONE_PREFIX}-pp{pp_rank}-tp{tp_rank}")


def wait_for_writes(
    path: str, pp_size: int, tp_size: int, timeout_s: float = 1800, poll_s: float = 0.5
) -> None:
    """Block until every rank has written its shards, then clear the markers.

    Only the caller of the save rpc may wait: a stage receives a control request
    only after the stage ahead of it ran it, so a rank blocking in the handler is
    the very rank that still has to relay it onward.
    """
    markers = [
        done_marker(path, pp, tp) for pp in range(pp_size) for tp in range(tp_size)
    ]
    deadline = time.monotonic() + timeout_s
    while True:
        pending = [m for m in markers if not os.path.exists(m)]
        if not pending:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"timed out after {timeout_s}s waiting for sharded state writes in "
                f"{path}; {len(pending)} rank(s) never reported done: "
                f"{', '.join(sorted(os.path.basename(m) for m in pending))}. Those "
                f"ranks most likely failed -- check their logs."
            )
        time.sleep(poll_s)
    for marker in markers:
        os.unlink(marker)


def _stage_pattern(template: str, parallel) -> str:
    """Fill in {pp_rank}, leaving {rank} and {part} for SGLang to format."""
    if "{pp_rank}" not in template:
        raise ValueError(
            f"sharded_state pattern {template!r} must contain '{{pp_rank}}' when "
            f"pp_size > 1, otherwise pipeline stages overwrite each other's shards"
        )
    return template.replace("{pp_rank}", str(parallel.pp_rank))


def _around_load_model(original, self, **kwargs):
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if parallel.pp_size > 1:
        chosen = self.pattern != type(self).DEFAULT_PATTERN
        self.pattern = _stage_pattern(self.pattern if chosen else PP_PATTERN, parallel)
    return original(self, **kwargs)


def _around_save_model(original, model, path, pattern=None, max_size=None):
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if parallel.pp_size == 1:
        return original(model, path, pattern, max_size)
    original(model, path, _stage_pattern(pattern or PP_PATTERN, parallel), max_size)
    # Never block here; the caller waits on these markers instead.
    with open(done_marker(path, parallel.pp_rank, parallel.tp_rank), "w"):
        pass


def install() -> None:
    """SGLang plugin entry point; runs in every engine and scheduler process."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(SAVE_TARGET, _around_save_model, HookType.AROUND)
    HookRegistry.register(LOAD_TARGET, _around_load_model, HookType.AROUND)
