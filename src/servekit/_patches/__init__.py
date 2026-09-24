"""Vendored backports applied to the installed sglang.

`servekit launch` runs every patch in this package before it prepares or
starts the engine (see apply_all). A patch is a file named
``sglang_<upstream-issue>.py`` with a ``main()`` that applies idempotently
and fails loudly when it does not fit the installed sglang. One that fails
is dropped with a log line, never a refusal: the engine then runs unpatched,
exactly like it would without servekit.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _drop(name: str, reason: object) -> None:
    """One line: the patch is off, here is why, moving on."""
    text = " ".join(str(reason).split()) if reason is not None else ""
    print(f"[PATCH] dropping {name}: {text or 'unknown error'}", file=sys.stderr, flush=True)


def apply_all() -> None:
    """Run every patch in this package, dropping the ones that fail.

    Called by `servekit launch` before anything imports sglang: a backport
    like sglang#35715 guards the dump path of ShardedStateLoader, so prepare
    needs the patch on disk just as much as serve does. A patch that does not
    fit the installed sglang is a log line, not a refusal.
    """
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
