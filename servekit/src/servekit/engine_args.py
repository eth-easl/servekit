"""Finding and replacing the model path in an engine launch command.

Pure: no filesystem, no processes. Everything engine-specific lives in
`FrameworkSpec.model_flags`, so adding an engine is a table entry, not a branch.
"""
from __future__ import annotations

from typing import List, Tuple

from .profile import FrameworkSpec


def find_model_path(command: List[str], spec: FrameworkSpec) -> Tuple[int, str]:
    """Return (index of the argv element holding the path, the path).

    For `--model-path=/x` the index is that single element; for
    `--model-path /x` it is the value that follows.
    """
    if not spec.model_flags:
        raise ValueError(
            f"servekit launch does not know where the model path is in a {spec.name} command yet; "
            f"use `servekit profile` instead"
        )
    for i, arg in enumerate(command):
        for flag in spec.model_flags:
            if arg == flag:
                if i + 1 >= len(command):
                    raise ValueError(f"{flag} has no value")
                return i + 1, command[i + 1]
            if arg.startswith(flag + "="):
                return i, arg[len(flag) + 1 :]
    flags = " / ".join(spec.model_flags)
    raise ValueError(f"no {flags} in the {spec.name} command")


def replace_model_path(command: List[str], spec: FrameworkSpec, new_path: str) -> List[str]:
    """`command` with the model path swapped for `new_path`. Nothing else changes."""
    idx, old = find_model_path(command, spec)
    out = list(command)
    out[idx] = new_path if out[idx] == old else out[idx].replace("=" + old, "=" + new_path, 1)
    return out
