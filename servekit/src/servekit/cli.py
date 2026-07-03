from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .profile import ProfileReport, render_table, run_profile, save_json


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] != "profile":
        print("usage: servekit profile [--out PATH] [--timeout SECONDS] -- <command...>", file=sys.stderr)
        return 2

    rest = argv[1:]
    if "--" in rest:
        idx = rest.index("--")
        options, command = rest[:idx], rest[idx + 1 :]
    else:
        options, command = [], rest

    parser = argparse.ArgumentParser(prog="servekit profile")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for the ready signal")
    args = parser.parse_args(options)

    if not command:
        print("error: no command given after --", file=sys.stderr)
        return 2

    emitted = False

    def emit(report: ProfileReport) -> None:
        nonlocal emitted
        emitted = True
        report.command = " ".join(command)
        print()
        print(render_table(report))
        out_path = args.out or Path(f"servekit-profile-{int(report.started_at)}.json")
        save_json(report, out_path)
        print(f"\nreport written to {out_path}", flush=True)

    report = run_profile(command, ready_timeout=args.timeout, on_ready=emit)
    if not emitted:
        emit(report)

    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
