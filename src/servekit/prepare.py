"""Writing a TP-presharded checkpoint plus its manifest, for `servekit launch`.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from . import jit_cache
from ._shim import PP_PATTERN
from .manifest import Manifest

SAVE_SCRIPT = Path(__file__).parent / "_prepare" / "save_sharded_state.py"

FORMAT = "sharded_state"

TP_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

PLUGIN_FLOOR = "0.5.11"

NO_PLUGINS = (
    "the installed sglang has no plugin framework, so servekit cannot keep "
    "pipeline stages from overwriting each other's shards. It arrived in "
    f"v{PLUGIN_FLOOR}; upgrade sglang, or prepare with --pp 1"
)


def _plugins_available() -> bool:
    """Whether the installed sglang carries the plugin framework.

    Read off disk rather than imported: `import sglang` costs tens of seconds,
    and `find_spec` on a top-level name does not execute the package.
    """
    try:
        spec = importlib.util.find_spec("sglang")
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.submodule_search_locations:
        return False
    root = Path(list(spec.submodule_search_locations)[0])
    return (root / "srt" / "plugins" / "hook_registry.py").is_file()


def _missing_shards(out: Path, tp_size: int, pp_size: int = 1) -> List[str]:
    pattern = PP_PATTERN if pp_size > 1 else TP_PATTERN
    names = [
        pattern.format(pp_rank=pp, rank=tp, part=0)
        for pp in range(pp_size)
        for tp in range(tp_size)
    ]
    return [name for name in names if not (out / name).is_file()]


def prepare(
    model: Path,
    out: Path,
    tp: int,
    engine_args: Sequence[str] = (),
    python: Optional[str] = None,
    nnodes: int = 1,
    node_rank: int = 0,
    dist_init_addr: Optional[str] = None,
    pp: int = 1,
    cache_root: Path = jit_cache.NODE_LOCAL_ROOT,
) -> int:
    if not model.is_dir():
        print(f"error: model path {model} is not a directory", file=sys.stderr)
        return 2
    if pp > 1 and not _plugins_available():
        print(f"error: {NO_PLUGINS}", file=sys.stderr)
        return 2
    if nnodes > 1:
        if dist_init_addr is None:
            print("error: --nnodes > 1 needs --dist-init-addr so every node rendezvouses", file=sys.stderr)
            return 2
        if not 0 <= node_rank < nnodes:
            print(f"error: --node-rank {node_rank} is outside the {nnodes} nodes asked for", file=sys.stderr)
            return 2
    out.mkdir(parents=True, exist_ok=True)
    # Built where `launch` will read them: the engine bakes this path into its
    # ninja build dirs, so a cache compiled elsewhere rebuilds and buys nothing.
    cache_build = jit_cache.node_local(cache_root, out.name)
    env = {**os.environ, **jit_cache.create(cache_build)}

    dist_args: List[str] = []
    if nnodes > 1:
        dist_args = [
            "--nnodes", str(nnodes),
            "--node-rank", str(node_rank),
            "--dist-init-addr", str(dist_init_addr),
        ]

    with tempfile.TemporaryDirectory() as tmp:
        resolved_path = Path(tmp) / "resolved.json"
        command = [
            python or sys.executable,
            str(SAVE_SCRIPT),
            "--model-path", str(model),
            "--output", str(out),
            "--tensor-parallel-size", str(tp),
            "--pipeline-parallel-size", str(pp),
            "--servekit-resolved-out", str(resolved_path),
            *dist_args,
            *engine_args,
        ]
        where = f" (node {node_rank} of {nnodes})" if nnodes > 1 else ""
        print(f"[SERVEKIT] preparing {model} -> {out} (tp={tp}, pp={pp}){where}", flush=True)
        print(f"[SERVEKIT] JIT caches will be built in {cache_build}", flush=True)
        rc = subprocess.call(command, env=env)
        if node_rank != 0:
            # A worker only ends when the job tears its task down, so its exit
            # code says nothing. The head gates.
            print(f"[SERVEKIT] node {node_rank}: shards written; the head gates the result", flush=True)
            return 0
        if rc != 0:
            print(f"error: sharding failed (rc={rc}); {out} is incomplete", file=sys.stderr)
            return 1
        if not resolved_path.is_file():
            print("error: the sharding run wrote no resolved args; refusing to write a manifest", file=sys.stderr)
            return 1
        resolved = json.loads(resolved_path.read_text())

    missing = _missing_shards(out, tp, pp)
    if missing:
        print(f"error: {out} is missing shards: {', '.join(missing)}", file=sys.stderr)
        return 1
    stale = sorted(p.name for p in out.glob("*.index.json"))
    if stale:
        # ShardedStateLoader prefers the index and then looks for files a
        # presharded checkpoint does not have.
        print(f"error: stale weight index left in {out}: {', '.join(stale)}", file=sys.stderr)
        return 1

    cached = jit_cache.copy_into(cache_build, out / jit_cache.CACHE_DIR_NAME)
    if cached is not None:
        n = sum(1 for p in cached.rglob("*") if p.is_file())
        print(f"[SERVEKIT] kept {n} JIT cache files in {cached}", flush=True)

    manifest = Manifest(format=FORMAT, source=str(model), **resolved)
    path = manifest.write(out)
    print(f"[SERVEKIT] prepared {tp * pp} ranks in {out}; manifest written to {path}", flush=True)
    print(
        f"[SERVEKIT] launch it with: servekit launch --servekit-artifact-path {out} -- "
        f"python -m sglang.launch_server --model-path {model} "
        f"--tensor-parallel-size {tp} --pipeline-parallel-size {pp}",
        flush=True,
    )
    return 0
