from __future__ import annotations

_MODES = {"record", "restore"}


def plan(dep, mode: str | None) -> tuple[dict[str, str], str | None]:
    """Translate --snapshot <mode> into extra env and an optional serve script."""
    if not mode:
        return {}, None
    if mode not in _MODES:
        raise ValueError(f"unknown snapshot mode {mode!r}; expected one of {sorted(_MODES)}")
    if not dep.snapshot.enabled:
        return {}, None
    env = dep.snapshot.record_env if mode == "record" else dep.snapshot.restore_env
    return dict(env), dep.snapshot.serve
