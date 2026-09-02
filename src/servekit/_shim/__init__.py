"""SGLang plugin: pipeline-parallel checkpoint filenames, and the --overlap
stage barrier.

`sharded_state` keys shard files on `tp_rank`, which restarts at 0 on every
pipeline stage, so stages collide on one filename and overwrite each other.

An SGLang plugin rather than a monkeypatch because SGLang forces the `spawn`
start method: the schedulers that read and write weights are fresh interpreters
that never see a patch made in the parent. Filenames match SGLang's own
pipeline-parallel branch, so if that lands upstream these hooks go redundant
rather than conflicting.

The same `spawn` boundary is why --overlap needs a filesystem marker rather
than a lock or event: `launch.py` publishes stage progress at
`stage_marker(dest)`, and `_around_run_load` blocks each scheduler there until
its node's stage is done, right before the weight read.

SGLang imports stay inside function bodies -- `prepare` imports
`wait_for_writes`, and servekit must not pull SGLang into its own process.
"""
from __future__ import annotations

import os
import sys
import time

PP_PATTERN = "model-pp-{pp_rank}-rank-{rank}-part-{part}.safetensors"
DONE_PREFIX = ".sharded-state-done"

SAVE_TARGET = "sglang.srt.model_loader.loader.ShardedStateLoader.save_model"
LOAD_TARGET = "sglang.srt.model_loader.loader.ShardedStateLoader.load_model"
RUN_TARGET = "sglang.srt.model_executor.model_runner.ModelRunner.load_model"

STAGE_OK = "ok"
STAGE_PENDING = "pending"


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


def stage_marker(dest: str) -> str:
    p = os.path.normpath(dest)
    parent, name = os.path.dirname(p), os.path.basename(p)
    return os.path.join(parent, f".{name}.stage")


def publish_stage(dest: str, state: str) -> None:
    marker = stage_marker(dest)
    tmp = marker + ".tmp"
    with open(tmp, "w") as f:
        f.write(state)
    os.replace(tmp, marker)


def clear_stage(dest: str) -> None:
    try:
        os.unlink(stage_marker(dest))
    except FileNotFoundError:
        pass


def wait_for_stage(dest: str, timeout_s: float = 1800, poll_s: float = 0.1) -> None:
    marker = stage_marker(dest)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with open(marker) as f:
                state = f.read()
        except FileNotFoundError:
            return
        if state == STAGE_OK:
            return
        if state != STAGE_PENDING:
            raise RuntimeError(f"servekit stage into {dest} failed: {state}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"timed out after {timeout_s}s waiting for the stage into {dest}")
        time.sleep(poll_s)


def _stage_pattern(template: str, parallel) -> str:
    """Fill in {pp_rank}, leaving {rank} and {part} for SGLang to format."""
    if "{pp_rank}" not in template:
        raise ValueError(
            f"sharded_state pattern {template!r} must contain '{{pp_rank}}' when "
            f"pp_size > 1, otherwise pipeline stages overwrite each other's shards"
        )
    return template.replace("{pp_rank}", str(parallel.pp_rank))


def _pp_requested() -> bool:
    """Best-effort check of the engine's own argv, independent of runtime_context:
    used only to tell a real pp>1 request apart from no pp at all when that
    module is missing, so the fallback below fails loud on the former instead
    of silently mis-sharding it as pp_size=1.
    """
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg in ("--pipeline-parallel-size", "--pp-size"):
            return i + 1 < len(argv) and argv[i + 1] not in ("1", "0")
        if arg.startswith("--pipeline-parallel-size=") or arg.startswith("--pp-size="):
            return arg.split("=", 1)[1] not in ("1", "0")
    return False


def _get_parallel():
    """`None` on sglang builds that predate `runtime_context` -- these hooks run
    on every load regardless of pp_size, and that module is where pipeline
    parallel support (and this shim's pp_size introspection) lives. Older than
    that, servekit only has TP to shim: callers treat `None` as pp_size=1, which
    is safe only because a real pp>1 request raises here instead of silently
    falling through and mis-sharding.
    """
    try:
        from sglang.srt.runtime_context import get_parallel
    except ModuleNotFoundError:
        if _pp_requested():
            raise RuntimeError(
                "servekit's pipeline-parallel checkpoint sharding needs "
                "sglang.srt.runtime_context, which this sglang build does not have "
                "(needs sglang >= 0.5.11, and some 0.5.11 builds still lack it -- "
                "if so, use a newer point release). TP-only launches are unaffected."
            ) from None
        return None
    return get_parallel()


def _around_load_model(original, self, **kwargs):
    parallel = _get_parallel()
    if parallel is not None and parallel.pp_size > 1:
        chosen = self.pattern != type(self).DEFAULT_PATTERN
        self.pattern = _stage_pattern(self.pattern if chosen else PP_PATTERN, parallel)
    return original(self, **kwargs)


def _around_save_model(original, model, path, pattern=None, max_size=None):
    parallel = _get_parallel()
    if parallel is None or parallel.pp_size == 1:
        return original(model, path, pattern, max_size)
    original(model, path, _stage_pattern(pattern or PP_PATTERN, parallel), max_size)
    # Never block here; the caller waits on these markers instead.
    with open(done_marker(path, parallel.pp_rank, parallel.tp_rank), "w"):
        pass


def _around_run_load(original, self):
    wait_for_stage(self.server_args.model_path)
    return original(self)


def install() -> None:
    """SGLang plugin entry point; runs in every engine and scheduler process."""
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(SAVE_TARGET, _around_save_model, HookType.AROUND)
    HookRegistry.register(LOAD_TARGET, _around_load_model, HookType.AROUND)
    HookRegistry.register(RUN_TARGET, _around_run_load, HookType.AROUND)
