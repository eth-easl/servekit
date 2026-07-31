"""`servekit prepare`: write a TP-presharded checkpoint plus its manifest.

Offline and one-off (~5.5 min for the 70B, and it needs the GPUs). The sharding
runs as a subprocess so `sglang` stays out of servekit's import graph: `launch`,
`profile` and `bench` keep working in a container that has no sglang.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from .manifest import Manifest

SAVE_SCRIPT = Path(__file__).parent / "_prepare" / "save_sharded_state.py"

FORMAT = "sharded_state"


def _missing_ranks(out: Path, tp_size: int) -> List[int]:
    return [r for r in range(tp_size) if not (out / f"model-rank-{r}-part-0.safetensors").is_file()]


def prepare(
    model: Path,
    out: Path,
    tp: int,
    engine_args: Sequence[str] = (),
    python: Optional[str] = None,
) -> int:
    if not model.is_dir():
        print(f"error: model path {model} is not a directory", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        resolved_path = Path(tmp) / "resolved.json"
        command = [
            python or sys.executable,
            str(SAVE_SCRIPT),
            "--model-path", str(model),
            "--output", str(out),
            "--tensor-parallel-size", str(tp),
            "--servekit-resolved-out", str(resolved_path),
            *engine_args,
        ]
        print(f"[SERVEKIT] preparing {model} -> {out} (tp={tp})", flush=True)
        rc = subprocess.call(command)
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

    manifest = Manifest(format=FORMAT, source=str(model), **resolved)
    path = manifest.write(out)
    print(f"[SERVEKIT] prepared {tp} ranks in {out}; manifest written to {path}", flush=True)
    print(
        f"[SERVEKIT] launch it with: servekit launch -- python -m sglang.launch_server "
        f"--model-path {out} --load-format {FORMAT} --tensor-parallel-size {tp}",
        flush=True,
    )
    return 0
