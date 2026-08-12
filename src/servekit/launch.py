"""`servekit launch -- <engine command>`: stage into /dev/shm, run the engine
against the copy, free the copy when the server reports ready.

`--overlap` starts the engine alongside the stage. It is unsafe and opt-in: the
stager truncates every destination to full size before writing, so an engine
that opens a file too early reads zeros with no error.

Multi-node is the same command on every node, under one srun task per node, with
the engine's own `--nnodes / --node-rank / --dist-init-addr` set. Each node
stages only the shards its own ranks read, profiles itself, and frees its own
copy; the nodes never talk to each other and each writes its own report.

Pipeline parallelism takes a different route, chosen here rather than named by
the caller: at pp > 1 the model path is the dump `prepare` wrote, so servekit
stages the dump's own subdirectory alongside the config and tokenizer, and writes
`--load-format presharded` and `presharded_path` itself. Which files a node needs
comes from the dump's `checksum.json` rather than a glob, which is what lets
pipeline stages -- whose ranks hold different layers -- each stage only their own.
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from . import manifest as manifest_mod
from . import presharded as presharded_mod
from . import quant_guard
from .engine_args import (
    PRESHARDED_FORMAT,
    check_manifest,
    find_model_path,
    parallel_sizes,
    replace_model_path,
    with_presharded_loader,
)
from .profile import Phase, ProfileReport, detect_framework, render_table, run_profile, save_json
from .stage import DEFAULT_SLICES, copy_metadata, stage
from .topology import (
    Topology,
    load_format,
    read_topology,
    shard_glob,
    wants_sharded_state,
)

DEFAULT_ROOT = Path("/dev/shm/servekit")

OVERLAP_WARNING = (
    "[SERVEKIT] --overlap is UNSAFE: the engine starts before staging finishes, with no barrier "
    "stopping it reading a file that is at full size but still zero-filled. Corrupt weights are "
    "silent. Check the output before trusting the run."
)

# presharded has a barrier sharded_state does not: the loader will not touch a dump
# without its READY sentinel, and servekit writes that last. Arriving early is a
# cache miss, not corruption -- expensive, but it cannot produce a wrong model.
PRESHARDED_OVERLAP_NOTE = (
    "[SERVEKIT] --overlap on a presharded dump is gated by the READY sentinel, written last and "
    "only on a clean stage. An engine that looks too early takes a cache MISS -- a full source "
    "load plus a re-dump into /dev/shm -- which shows up as a slow weight_loading phase."
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


@dataclass
class _Staging:
    """What this node copies, and the order it must happen in."""

    src: Path
    dest: Path
    pattern: str = "*"
    files: Optional[List[str]] = None
    # Copied synchronously before the bulk: the loader reads the plan first, and it
    # is kilobytes, so it must never be the thing an overlapped engine waits on.
    presync: Sequence[str] = ()
    # Created last and only on a clean stage. This is the loader's barrier.
    ready_marker: Optional[Path] = None


def _presharded_dump(
    command: List[str], spec, topo: Topology, model_path: Path
) -> Optional[presharded_mod.ShardPlan]:
    """The dump this command should load, or None to stage the checkpoint whole.

    servekit picks the format rather than making the caller name it: pipeline
    parallelism needs `presharded`, and the dump `prepare` wrote *is* the model
    path, so there is nothing left to point at. An explicit --load-format wins.
    """
    asked = load_format(command)
    if asked is not None and asked != PRESHARDED_FORMAT:
        return None
    if asked is None and topo.pp_size <= 1:
        return None

    if not presharded_mod.find_dumps(model_path):
        if asked is not None:
            raise ValueError(
                f"no completed presharded dump under {model_path}; run `servekit prepare` first"
            )
        # Asked for pipeline parallelism without preparing one. The plain path
        # still serves, just off the source checkpoint.
        print(
            f"[SERVEKIT] pp={topo.pp_size} but no presharded dump under {model_path}; staging the "
            f"checkpoint whole. `servekit prepare` writes one, which loads far faster.",
            flush=True,
        )
        return None

    if presharded_mod.loader_available() is False:
        raise ValueError(presharded_mod.MISSING_LOADER)
    return presharded_mod.select_dump(model_path, parallel_sizes(command, spec))


def _presharded_staging(dump: presharded_mod.ShardPlan, topo: Topology, dest: Path) -> _Staging:
    """What this node copies out of `dump`, into the staged model directory.

    The dump keeps its own subdirectory under `dest`, because that is the layout
    the loader globs for and `dest` is also the model path now.
    """
    lo, hi = topo.local_rank_range
    if hi < lo:
        # tp*pp < nnodes floors ranks_per_node to zero, and an empty file list
        # stages cleanly -- the node would just have no weights.
        raise ValueError(
            f"tp*pp = {topo.tp_size * topo.pp_size} leaves no ranks for node {topo.node_rank} "
            f"of {topo.nnodes}"
        )
    if hi >= dump.world_size:
        raise ValueError(
            f"this node expects ranks {lo}-{hi} but {dump.directory.name} was dumped for "
            f"{dump.world_size} ranks"
        )
    staged_dump = dest / dump.directory.name
    return _Staging(
        src=dump.directory,
        dest=staged_dump,
        files=dump.files_for_ranks(lo, hi),
        presync=(presharded_mod.CHECKSUM_NAME,),
        ready_marker=staged_dump / presharded_mod.READY_NAME,
    )


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

    # A checkpoint written before this check existed is still unusable, and so
    # is one whose serve-time flags reach a path the dump did not. Either way it
    # looks like a healthy server answering from dead weights.
    if prepared is not None or wants_sharded_state(command):
        unsupported = quant_guard.check_command(src_path, command)
        if unsupported:
            print(quant_guard.refusal(unsupported, src), file=sys.stderr)
            return 2

    try:
        dump = _presharded_dump(command, spec, topo, src_path)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    dest = shm_root / src_path.name
    if dump is not None:
        try:
            plan = _presharded_staging(dump, topo, dest)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        # The staged copy is the model directory too, so it needs the config and
        # tokenizer beside the dump, and the command points at it wholesale.
        engine_command = with_presharded_loader(
            replace_model_path(command, spec, str(dest)), str(dest)
        )
        replicate_metadata = True
        sliced = topo.is_multinode
        lo, hi = topo.local_rank_range
        print(
            f"[SERVEKIT] {dump.directory.name}: ranks {lo}-{hi} of {dump.world_size} read "
            f"{len(plan.files or [])} files, {dump.bytes_for_ranks(lo, hi) / 1e9:.1f} GB",
            flush=True,
        )
    else:
        engine_command = replace_model_path(command, spec, str(dest))
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
        plan = _Staging(src=src_path, dest=dest, pattern=pattern)

    t0 = time.time()
    staged: dict = {}

    def do_stage() -> None:
        try:
            if plan.presync:
                plan.dest.mkdir(parents=True, exist_ok=True)
            for name in plan.presync:
                shutil.copy2(plan.src / name, plan.dest / name)
            staged["result"] = stage(
                plan.src, plan.dest, slices=slices, file_pattern=plan.pattern, files=plan.files
            )
            # The stager's byte gate passes trivially for a glob that matched
            # the wrong files, so count them too. An explicit list it checks itself.
            if plan.files is None and plan.pattern != "*":
                want = len(list(plan.src.glob(plan.pattern)))
                got = len(list(plan.dest.glob(plan.pattern)))
                if got != want:
                    staged["error"] = RuntimeError(
                        f"staged {got} of {want} files matching {plan.pattern!r}; refusing to trust {dest}"
                    )
        except (OSError, RuntimeError) as e:
            staged["error"] = e
        if "error" not in staged and plan.ready_marker is not None:
            plan.ready_marker.touch()

    def report_stage() -> None:
        result = staged["result"]
        print(
            f"[SERVEKIT] staged {result.bytes / 1e9:.1f} GB in {result.wall_s:.2f} s ({result.gbps:.2f} GB/s)",
            flush=True,
        )

    if sliced and dump is None:
        lo, hi = topo.local_rank_range
        print(
            f"[SERVEKIT] node {topo.node_rank} of {topo.nnodes}: staging ranks {lo}-{hi} ({plan.pattern})",
            flush=True,
        )

    # Not a daemon: if the engine dies early we still wait for the stage rather
    # than leaving its ~1700 `dd` processes running on the node.
    thread: Optional[threading.Thread] = None
    if replicate_metadata:
        n = copy_metadata(src_path, dest)
        print(f"[SERVEKIT] copied {n} metadata files up front", flush=True)
    if overlap:
        print(
            PRESHARDED_OVERLAP_NOTE if dump is not None else OVERLAP_WARNING,
            file=sys.stderr,
            flush=True,
        )
        print(f"[SERVEKIT] staging {plan.src} -> {dest} (overlapped)", flush=True)
        thread = threading.Thread(target=do_stage)
        thread.start()
    else:
        print(f"[SERVEKIT] staging {plan.src} -> {dest}", flush=True)
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
