"""`servekit.json`: what a prepared checkpoint can only be loaded back as.

Parallelism mismatches are shape mismatches, which the engine catches ~40 s in
without naming the checkpoint. dtype and quantization it does not catch at all:
ShardedStateLoader does `param.data.copy_(tensor)`, which casts silently.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

MANIFEST_NAME = "servekit.json"


@dataclass
class Manifest:
    format: str
    engine: str
    engine_version: str
    tp_size: int
    pp_size: int
    dp_size: int
    dtype: str
    quantization: Optional[str]
    source: str

    def write(self, directory: Path) -> Path:
        path = Path(directory) / MANIFEST_NAME
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path


def read(directory: Path) -> Optional[Manifest]:
    """The manifest in `directory`, or None if there is none.

    None is not an error: an unprepared checkpoint still stages and still loads,
    it just cannot be checked.
    """
    path = Path(directory) / MANIFEST_NAME
    if not path.is_file():
        return None
    return Manifest(**json.loads(path.read_text()))
