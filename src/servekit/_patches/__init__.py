"""Vendored sglang backports; `servekit launch` runs them all before it prepares or serves."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _drop(name: str, reason: object) -> None:
    """One line: the patch is off, here is why, moving on."""
    text = " ".join(str(reason).split()) if reason is not None else ""
    print(f"[SERVEKIT] dropping {name}: {text or 'unknown error'}", file=sys.stderr, flush=True)


def apply_all() -> None:
    """Run every patch in this package before anything imports sglang, dropping failures with a log line."""
    package = Path(__file__).resolve().parent
    for path in sorted(package.glob("sglang_*.py")):
        name = path.stem
        try:
            module = importlib.import_module(f".{name}", __name__)
        except Exception as e:  # a broken patch file is a log line, not a crash
            _drop(name, e)
            continue
        run = getattr(module, "main", None)
        if not callable(run):
            _drop(name, "no main()")
            continue
        try:
            rc = run()
        except SystemExit as e:
            _drop(name, e.code if isinstance(e.code, int) else f"exit code {e.code}")
            continue
        except Exception as e:
            _drop(name, e)
            continue
        if rc:
            _drop(name, f"exit code {rc}")
