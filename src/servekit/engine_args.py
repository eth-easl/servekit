"""Reading and rewriting an engine launch command.

Pure: no filesystem, no processes. Everything engine-specific lives in
`FrameworkSpec`, so adding an engine is a table entry, not a branch.
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

from .manifest import Manifest
from .profile import FrameworkSpec

# Checked only when the command sets them: left off, the engine resolves them
# from the checkpoint's own config, which is what prepare recorded.
_CONFIG_FLAGS = {"dtype": "--dtype", "quantization": "--quantization"}


def _find_flag(command: List[str], flags: List[str]) -> Optional[Tuple[int, str]]:
    for i, arg in enumerate(command):
        for flag in flags:
            if arg == flag:
                if i + 1 >= len(command):
                    raise ValueError(f"{flag} has no value")
                return i + 1, command[i + 1]
            if arg.startswith(flag + "="):
                return i, arg[len(flag) + 1 :]
    return None


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
    found = _find_flag(command, spec.model_flags)
    if found is None:
        flags = " / ".join(spec.model_flags)
        raise ValueError(f"no {flags} in the {spec.name} command")
    return found


def replace_model_path(command: List[str], spec: FrameworkSpec, new_path: str) -> List[str]:
    """`command` with the model path swapped for `new_path`. Nothing else changes."""
    idx, old = find_model_path(command, spec)
    out = list(command)
    out[idx] = new_path if out[idx] == old else out[idx].replace("=" + old, "=" + new_path, 1)
    return out


def replace_presharded_root(command: List[str], new_root: str) -> List[str]:
    """`command` with `presharded_path` repointed at the staged copy.

    The model path is deliberately left alone: `presharded` still reads the real
    checkpoint's config, and only the dump moves to /dev/shm.
    """
    from .topology import LOADER_CONFIG_FLAGS, loader_config

    found = _find_flag(command, LOADER_CONFIG_FLAGS)
    if found is None:
        raise ValueError(f"no {LOADER_CONFIG_FLAGS[0]} in the command")
    config = dict(loader_config(command))
    config["presharded_path"] = new_root
    idx, old = found
    out = list(command)
    out[idx] = json.dumps(config) if out[idx] == old else out[idx].replace(old, json.dumps(config), 1)
    return out


def parallel_sizes(command: List[str], spec: FrameworkSpec) -> dict:
    """What the command asks for, in `shard_config`'s vocabulary."""
    sizes = {}
    for name, flags in spec.parallel_flags.items():
        found = _find_flag(command, flags)
        sizes[name.replace("_size", "")] = int(found[1]) if found else 1
    return sizes


def check_manifest(command: List[str], spec: FrameworkSpec, manifest: Manifest) -> List[str]:
    """Ways `command` cannot load `manifest`'s checkpoint, in plain words.

    Empty means go ahead. Parallelism the engine catches late and cryptically;
    dtype and quantization it does not catch at all.
    """
    problems = []

    if manifest.engine != spec.name:
        problems.append(
            f"engine: the checkpoint was prepared for {manifest.engine}, "
            f"and {spec.name} cannot load another engine's shards"
        )

    for name, flags in spec.parallel_flags.items():
        found = _find_flag(command, flags)
        asked = int(found[1]) if found else 1
        # A manifest written before servekit knew about this axis cannot speak to it.
        prepared = getattr(manifest, name, None)
        if prepared is not None and asked != prepared:
            problems.append(
                f"{name}: the checkpoint is sharded for {prepared}, the command asks for {asked}"
            )

    for name, flag in _CONFIG_FLAGS.items():
        found = _find_flag(command, [flag])
        if found is None or found[1] == "auto":
            continue
        prepared = getattr(manifest, name) or "none"
        if found[1] != prepared:
            problems.append(f"{name}: the checkpoint was written as {prepared}, {flag} says {found[1]}")

    return problems
