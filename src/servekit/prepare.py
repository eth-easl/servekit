"""`servekit prepare`: write a presharded checkpoint the engine can load back.

Two formats, because SGLang has two. `sharded_state` keys its files on the TP
rank, which is the same on every pipeline stage, so it is TP-only -- at pp > 1
the stages overwrite each other's files. `presharded` keys on the world rank and
is what pipeline parallelism needs.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from . import presharded
from .manifest import Manifest

SAVE_SCRIPT = Path(__file__).parent / "_prepare" / "save_sharded_state.py"

FORMAT = "sharded_state"


def _missing_ranks(out: Path, tp_size: int) -> List[int]:
    return [r for r in range(tp_size) if not (out / f"model-rank-{r}-part-0.safetensors").is_file()]


def prepare_presharded(
    out: Path,
    command: List[str],
    timeout: float = 3600.0,
    node_rank: int = 0,
) -> int:
    """Write a `presharded` dump by starting the engine once against an empty root.

    There is no save API for this format: the dump is a side effect of a normal
    load, so the job is a server start we tear down as soon as it is up.

    The dump is keyed by the full parallel and quantization config, so what comes
    out is locked to the command that made it -- a different tp/pp/ep will not
    pick it up, it will silently miss and re-dump.
    """
    from .engine_args import parallel_sizes, replace_presharded_root
    from .profile import detect_framework, run_profile
    from .topology import wants_presharded

    if presharded.loader_available() is False:
        print(f"error: {presharded.MISSING_LOADER}", file=sys.stderr)
        return 2
    if not wants_presharded(command):
        print("error: the command must pass --load-format presharded", file=sys.stderr)
        return 2

    existing = presharded.find_dumps(out)
    if existing:
        print(f"error: {out} already holds a completed dump: {', '.join(d.name for d in existing)}", file=sys.stderr)
        print(f"       refusing to write over it; remove it by hand (rm -r {out})", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    spec = detect_framework(command)
    sizes = parallel_sizes(command, spec)
    engine_command = replace_presharded_root(command, str(out))
    print(f"[SERVEKIT] dumping to {out} (tp={sizes['tp']} pp={sizes['pp']} ep={sizes['ep']})", flush=True)

    report = run_profile(engine_command, ready_timeout=timeout, head=node_rank == 0, stop_on_ready=True)
    if node_rank != 0:
        # Checked before success on purpose: every rank writes its own manifest,
        # but only the head builds the plan, and a worker's engine goes down with
        # the head's rather than announcing anything. Its readiness says nothing.
        print(f"[SERVEKIT] node {node_rank}: shards written; the head gates the result", flush=True)
        return 0
    if not report.success:
        print(f"error: the engine never reported ready; {out} is incomplete", file=sys.stderr)
        return 1

    dumps = presharded.find_dumps(out)
    if not dumps:
        print(f"error: the engine came up but wrote no completed dump under {out}", file=sys.stderr)
        return 1
    if len(dumps) > 1:
        print(f"error: {out} holds {len(dumps)} dumps; expected one", file=sys.stderr)
        return 1

    plan = presharded.read_plan(dumps[0])
    problems = presharded.verify(plan, sizes["tp"], sizes["pp"])
    if problems:
        print(f"error: {dumps[0].name} is not a valid {sizes['tp']}x{sizes['pp']} split:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    total = sum(f.stat().st_size for f in plan.directory.glob("*.safetensor"))
    print(
        f"[SERVEKIT] verified {dumps[0].name}: {plan.world_size} ranks, "
        f"{sizes['pp']} pipeline stages tiling the layers, {total / 1e9:.1f} GB",
        flush=True,
    )
    return 0


def prepare(
    model: Path,
    out: Path,
    tp: int,
    engine_args: Sequence[str] = (),
    python: Optional[str] = None,
    nnodes: int = 1,
    node_rank: int = 0,
    dist_init_addr: Optional[str] = None,
) -> int:
    if not model.is_dir():
        print(f"error: model path {model} is not a directory", file=sys.stderr)
        return 2
    if nnodes > 1:
        if dist_init_addr is None:
            print("error: --nnodes > 1 needs --dist-init-addr so every node rendezvouses", file=sys.stderr)
            return 2
        if not 0 <= node_rank < nnodes:
            print(f"error: --node-rank {node_rank} is outside the {nnodes} nodes asked for", file=sys.stderr)
            return 2
    out.mkdir(parents=True, exist_ok=True)

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
            "--servekit-resolved-out", str(resolved_path),
            *dist_args,
            *engine_args,
        ]
        where = f" (node {node_rank} of {nnodes})" if nnodes > 1 else ""
        print(f"[SERVEKIT] preparing {model} -> {out} (tp={tp}){where}", flush=True)
        rc = subprocess.call(command)
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

    missing = _missing_ranks(out, tp)
    if missing:
        print(f"error: ranks {missing} produced no shards; {out} is unusable", file=sys.stderr)
        return 1
    stale = sorted(p.name for p in out.glob("*.index.json"))
    if stale:
        # ShardedStateLoader prefers the index and then looks for files a
        # presharded checkpoint does not have.
        print(f"error: stale weight index left in {out}: {', '.join(stale)}", file=sys.stderr)
        return 1

    manifest = Manifest(format=FORMAT, source=str(model), **resolved)
    path = manifest.write(out)
    print(f"[SERVEKIT] prepared {tp} ranks in {out}; manifest written to {path}", flush=True)
    print(
        f"[SERVEKIT] launch it with: servekit launch -- python -m sglang.launch_server "
        f"--model-path {out} --load-format {FORMAT} --tensor-parallel-size {tp}",
        flush=True,
    )
    return 0
