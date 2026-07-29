"""`servekit launch -- <engine command>`: stage into /dev/shm, run the engine
against the copy, free the copy when the server reports ready.

Sequential, not overlapped: the stage completes before the engine starts, so a
partially-written file cannot be read and no engine-side barrier is needed. The
only thing servekit does to the engine is hand it a different directory.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

from .engine_args import find_model_path, replace_model_path
from .profile import Phase, ProfileReport, detect_framework, render_table, run_profile, save_json
from .stage import DEFAULT_SLICES, stage

DEFAULT_ROOT = Path("/dev/shm/servekit")


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def free(path: Path) -> int:
    """Remove a staged copy, returning the bytes unlinked.

    The point is to give the RAM back to the running job while it serves, not to
    tidy up afterwards: the weights are on the GPU by now, and the node needs
    that memory for its own work (KV cache offloading and the like).

    Freeing is `unlink`, and tmpfs pages only return once nothing maps them.
    Measured on Clariden: SGLang drops its mappings by ready under both the
    `sharded_state` and the default mmap loader, so ~139.5 GB of a 141.1 GB
    staged 70B comes back within a minute.
    """
    if not path.is_dir():
        return 0
    freed = _dir_bytes(path)
    shutil.rmtree(path)
    return freed


def launch(
    command: List[str],
    out: Optional[Path] = None,
    shm_root: Path = DEFAULT_ROOT,
    slices: int = DEFAULT_SLICES,
    timeout: float = 1800.0,
) -> int:
    spec = detect_framework(command)
    _, src = find_model_path(command, spec)
    src_path = Path(src)
    if not src_path.is_dir():
        print(f"error: model path {src} is not a directory", file=sys.stderr)
        return 2

    dest = shm_root / src_path.name
    t0 = time.time()
    print(f"[SERVEKIT] staging {src} -> {dest}", flush=True)
    try:
        staged = stage(src_path, dest, slices=slices)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        print(f"       anything left behind: rm -r {dest}", file=sys.stderr)
        return 1
    print(f"[SERVEKIT] staged {staged.bytes / 1e9:.1f} GB in {staged.wall_s:.2f} s ({staged.gbps:.2f} GB/s)", flush=True)

    engine_command = replace_model_path(command, spec, str(dest))
    print(f"[SERVEKIT] launching: {' '.join(engine_command)}", flush=True)

    emitted = {"done": False}

    def emit(report: ProfileReport) -> None:
        # Stage is part of the cold start: prepend it as a phase and move the
        # start time back to t0, so total covers wrapper start to ready.
        report.command = " ".join(command)
        report.phases.insert(0, Phase("stage", round(staged.wall_s, 2), "wall_clock"))
        report.started_at = t0
        print()
        print(render_table(report))
        out_path = out or Path(f"servekit-launch-{int(t0)}.json")
        save_json(report, out_path)
        print(f"\nreport written to {out_path}", flush=True)

    def on_ready(report: ProfileReport) -> None:
        emitted["done"] = True
        emit(report)
        freed = free(dest)
        print(f"[SERVEKIT] freed {freed / 1e9:.1f} GB from {dest}", flush=True)

    report = run_profile(engine_command, ready_timeout=timeout, on_ready=on_ready)
    if not emitted["done"]:
        # Never reached ready: the copy is left behind on purpose, since the
        # engine may still be holding it and we have no way to know.
        emit(report)
        print(f"[SERVEKIT] server never reported ready; {dest} left in place (rm -r {dest})", file=sys.stderr)
    return 0 if report.success else 1
