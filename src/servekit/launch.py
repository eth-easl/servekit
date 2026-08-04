"""`servekit launch -- <engine command>`: stage into /dev/shm, run the engine
against the copy, free the copy when the server reports ready.

`--overlap` starts the engine alongside the stage. It is unsafe and opt-in: the
stager truncates every destination to full size before writing, so an engine
that opens a file too early reads zeros with no error.

Multi-node is the same command on every node, under one srun task per node, with
the engine's own `--nnodes / --node-rank / --dist-init-addr` set. Each node
stages only the shards its own ranks read, profiles itself, and frees its own
copy; the nodes never talk to each other and each writes its own report.
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from . import manifest as manifest_mod
from .engine_args import check_manifest, find_model_path, replace_model_path
from .profile import Phase, ProfileReport, detect_framework, render_table, run_profile, save_json
from .stage import DEFAULT_SLICES, stage
from .topology import Topology, read_topology, shard_glob, wants_sharded_state

DEFAULT_ROOT = Path("/dev/shm/servekit")

OVERLAP_WARNING = (
    "[SERVEKIT] --overlap is UNSAFE: the engine starts before staging finishes, with no barrier "
    "stopping it reading a file that is at full size but still zero-filled. Corrupt weights are "
    "silent. Check the output before trusting the run."
)


def _dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def free(path: Path) -> int:
    """Remove a staged copy, returning the bytes unlinked.

    The point is to give the RAM back to the job while it serves: the weights are
    on the GPU by now, and the node wants that memory for its own work.
    """
    if not path.is_dir():
        return 0
    freed = _dir_bytes(path)
    shutil.rmtree(path)
    return freed


def _copy_metadata(src: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(src.iterdir()) if f.is_file() and f.suffix != ".safetensors"]
    for f in files:
        shutil.copy2(f, dest / f.name)
    return len(files)


def _out_path(out: Optional[Path], t0: float, topo: Topology) -> Path:
    """Where this node writes its report.

    Every node is handed the same --out, so the rank goes in here rather than
    being left to the caller: a forgotten $SLURM_PROCID would silently leave one
    report instead of two.
    """
    path = out or Path(f"servekit-launch-{int(t0)}.json")
    if not topo.is_multinode:
        return path
    return path.with_name(f"{path.stem}.node{topo.node_rank}{path.suffix}")


def launch(
    command: List[str],
    out: Optional[Path] = None,
    shm_root: Path = DEFAULT_ROOT,
    slices: int = DEFAULT_SLICES,
    timeout: float = 1800.0,
    overlap: bool = False,
) -> int:
    spec = detect_framework(command)
    topo = read_topology(command, spec)
    if not topo.is_head and spec.worker_ready_pattern is None:
        print(
            f"error: servekit does not know what a {spec.name} worker node prints when it is up",
            file=sys.stderr,
        )
        return 2
    _, src = find_model_path(command, spec)
    src_path = Path(src)
    if not src_path.is_dir():
        print(f"error: model path {src} is not a directory", file=sys.stderr)
        return 2

    prepared = manifest_mod.read(src_path)
    if prepared is not None:
        problems = check_manifest(command, spec, prepared)
        if problems:
            print(f"error: {src} cannot be loaded by this command:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print("       re-run `servekit prepare` for these settings, or fix the command", file=sys.stderr)
            return 2

    dest = shm_root / src_path.name
    engine_command = replace_model_path(command, spec, str(dest))
    t0 = time.time()

    # A presharded checkpoint gives each rank its own files, so a node stages
    # only its own ranks' -- two nodes pull disjoint halves off Lustre at once
    # instead of the whole checkpoint twice.
    sliced = topo.is_multinode and wants_sharded_state(command)
    # The engine reads config.json and the tokenizer within seconds of starting,
    # so with --overlap those cannot be left to a stage still running underneath
    # it: the stager truncates every destination to full size first, so an early
    # reader would get a zero-filled config. Copy them up front and overlap only
    # the shards, as experiments/clariden-loading-exp does.
    split = overlap and any(src_path.glob("*.safetensors"))
    if sliced:
        pattern = shard_glob(topo)
    elif split:
        pattern = "*.safetensors"
    else:
        pattern = "*"
    # Either way the stage skips the non-weight files, which every node needs.
    replicate_metadata = sliced or split

    staged: dict = {}

    def do_stage() -> None:
        try:
            staged["result"] = stage(src_path, dest, slices=slices, file_pattern=pattern)
            # The stager's byte gate passes trivially for a glob that matched
            # the wrong files, so count them too.
            if pattern != "*":
                want = len(list(src_path.glob(pattern)))
                got = len(list(dest.glob(pattern)))
                if got != want:
                    staged["error"] = RuntimeError(
                        f"staged {got} of {want} files matching {pattern!r}; refusing to trust {dest}"
                    )
        except RuntimeError as e:
            staged["error"] = e

    def report_stage() -> None:
        result = staged["result"]
        print(
            f"[SERVEKIT] staged {result.bytes / 1e9:.1f} GB in {result.wall_s:.2f} s ({result.gbps:.2f} GB/s)",
            flush=True,
        )

    if sliced:
        lo, hi = topo.local_rank_range
        print(
            f"[SERVEKIT] node {topo.node_rank} of {topo.nnodes}: staging ranks {lo}-{hi} ({pattern})",
            flush=True,
        )

    # Not a daemon: if the engine dies early we still wait for the stage rather
    # than leaving its ~1700 `dd` processes running on the node.
    thread: Optional[threading.Thread] = None
    if replicate_metadata:
        n = _copy_metadata(src_path, dest)
        print(f"[SERVEKIT] copied {n} metadata files up front", flush=True)
    if overlap:
        print(OVERLAP_WARNING, file=sys.stderr, flush=True)
        print(f"[SERVEKIT] staging {src} -> {dest} (overlapped, {pattern})", flush=True)
        thread = threading.Thread(target=do_stage)
        thread.start()
    else:
        print(f"[SERVEKIT] staging {src} -> {dest}", flush=True)
        do_stage()
        if "error" in staged:
            print(f"error: {staged['error']}", file=sys.stderr)
            print(f"       anything left behind: rm -r {dest}", file=sys.stderr)
            return 1
        report_stage()

    print(f"[SERVEKIT] launching: {' '.join(engine_command)}", flush=True)

    emitted = {"done": False}
    joined = {"done": False}

    def join_stage() -> None:
        if joined["done"]:
            return
        joined["done"] = True
        if thread is not None:
            thread.join()
            if "result" in staged:
                report_stage()

    def emit(report: ProfileReport) -> None:
        report.command = " ".join(command)
        if topo.is_multinode:
            report.node_rank = topo.node_rank
            report.nnodes = topo.nnodes
        if not overlap and "result" in staged:
            # Serial: the stage is part of the cold start. Overlapped it runs
            # concurrently with the phases below and would double-count.
            report.phases.insert(0, Phase("stage", round(staged["result"].wall_s, 2), "wall_clock"))
            report.started_at = t0
        print()
        print(render_table(report))
        out_path = _out_path(out, t0, topo)
        save_json(report, out_path)
        print(f"\nreport written to {out_path}", flush=True)

    def on_ready(report: ProfileReport) -> None:
        # On a worker this fires on its own readiness line, not the head's.
        # Either way the free needs no cross-node barrier: a node stages only
        # the shards its own ranks read.
        emitted["done"] = True
        join_stage()
        emit(report)
        freed = free(dest)
        print(f"[SERVEKIT] freed {freed / 1e9:.1f} GB from {dest}", flush=True)

    report = run_profile(
        engine_command,
        ready_timeout=timeout,
        on_ready=on_ready,
        head=topo.is_head,
    )

    if not emitted["done"]:
        join_stage()
        emit(report)
        print(f"[SERVEKIT] server never reported ready; {dest} left in place (rm -r {dest})", file=sys.stderr)

    if "error" in staged:
        print(f"error: the stage failed: {staged['error']}", file=sys.stderr)
        return 1
    return 0 if report.success else 1
