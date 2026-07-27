from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .bench import BenchConfig, run_benchmark, render_bench
from .profile import ProfileReport, render_table, run_profile, save_json

USAGE = """usage:
  servekit profile [--out PATH] [--timeout SECONDS] -- <command...>
  servekit bench --url URL (--into PATH | --out PATH) [--wait-ready SECONDS] [...]

`profile` launches a server and measures its cold start; `bench` loads any live
server, whether or not servekit launched it. To do both, run them side by side
and join them on the report file:

  servekit profile --out run.json -- python -m sglang.launch_server ... &
  PROF=$!
  servekit bench --url http://127.0.0.1:8080 --into run.json --requests 64
  kill $PROF; wait $PROF
"""


def _split_command(rest: List[str]) -> Tuple[List[str], List[str]]:
    if "--" in rest:
        idx = rest.index("--")
        return rest[:idx], rest[idx + 1 :]
    return [], rest


def _profile(argv: List[str]) -> int:
    options, command = _split_command(argv)

    parser = argparse.ArgumentParser(prog="servekit profile")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for the ready signal")
    args = parser.parse_args(options)

    if not command:
        print("error: no command given after --", file=sys.stderr)
        return 2

    print(f"[SERVEKIT] profiling cold start of: {' '.join(command)}", flush=True)

    def emit(report: ProfileReport) -> None:
        report.command = " ".join(command)
        print()
        print(render_table(report))
        out_path = args.out or Path(f"servekit-profile-{int(report.started_at)}.json")
        save_json(report, out_path)
        print(f"\nreport written to {out_path}", flush=True)

    # The report is written the moment the server reports ready, not when it
    # exits -- so a concurrent `servekit bench` (and the user) can see it while
    # the server is still up. A run that never gets ready never fires the
    # callback, so we emit the failure ourselves.
    emitted = {"done": False}

    def emit_once(report: ProfileReport) -> None:
        emitted["done"] = True
        emit(report)

    report = run_profile(command, ready_timeout=args.timeout, on_ready=emit_once)
    if not emitted["done"]:
        emit(report)
    return 0 if report.success else 1


def _wait_for_report(path: Path, timeout_s: float, interval_s: float = 0.5) -> bool:
    """Block until `profile` has written its report, or `timeout_s` elapses.

    This sequences the two subcommands, and it exists because an engine starts
    accepting traffic *before* it announces itself: SGLang binds Uvicorn and
    runs its warmup request several seconds before logging "fired up and ready
    to roll". A bench that probed purely over HTTP could therefore win the race
    and start loading the server while the profiler is still timing the warmup
    phase -- corrupting the very cold-start measurement it runs alongside.

    Profile writes the report the instant it detects ready, so the report
    appearing is the signal that measurement is over and the server is ours.
    Only used with --into; a standalone bench (a CRIU-restored server, say) has
    no report to wait on and goes straight to the HTTP probe.
    """
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 >= timeout_s:
            return False
        time.sleep(interval_s)
    return True


def _merge_into(path: Path, benchmark: dict) -> int:
    """Write `benchmark` into an existing profile report, in place.

    This is the whole linking mechanism between the two subcommands: the report
    file is the run identity. No run ids, no daemon, no IPC -- and the on-disk
    schema stays exactly what `profile --bench` used to produce.

    The report is expected to exist by now even though it may not have when we
    started: `profile` writes it the instant the server reports ready, which is
    necessarily before this server could answer our requests. If it is missing
    anyway (profile crashed, wrong path), the benchmark we just spent minutes on
    is still salvaged to a sibling file rather than thrown away.
    """
    if not path.exists():
        fallback = path.with_suffix(".bench.json")
        fallback.write_text(json.dumps(benchmark, indent=2))
        print(
            f"error: --into {path} does not exist; benchmark written to {fallback} instead",
            file=sys.stderr,
        )
        return 1
    report = json.loads(path.read_text())
    report["benchmark"] = benchmark
    path.write_text(json.dumps(report, indent=2))
    print(f"\nbenchmark merged into {path}", flush=True)
    return 0


def _bench(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="servekit bench")
    parser.add_argument("--url", required=True, help="server base URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--into", type=Path, default=None, help="merge results into an existing profile report")
    parser.add_argument("--out", type=Path, default=None, help="write a standalone bench report here")
    parser.add_argument(
        "--wait-ready",
        type=float,
        default=0.0,
        help="poll the server until it serves a request, up to this many seconds (0: fail immediately)",
    )
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--input-len", type=int, default=512, help="approx prompt length in words")
    parser.add_argument("--output-len", type=int, default=128, help="max_new_tokens per request")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-correctness", action="store_true", help="skip the correctness probe")
    args = parser.parse_args(argv)

    if args.into is None and args.out is None:
        print("error: one of --into (merge into a profile report) or --out is required", file=sys.stderr)
        return 2
    # Only the *directory* is checked up front. The report file itself usually
    # does not exist yet: the normal flow starts this bench alongside a still
    # booting `servekit profile`, which writes the report when the server is
    # ready -- i.e. after we start, but before we finish. See _merge_into.
    for dest in (args.into, args.out):
        if dest is not None and not dest.parent.is_dir():
            print(f"error: directory {dest.parent} does not exist", file=sys.stderr)
            return 2

    cfg = BenchConfig(
        requests=args.requests,
        input_len=args.input_len,
        output_len=args.output_len,
        concurrency=args.concurrency,
        seed=args.seed,
        correctness=not args.no_correctness,
        wait_ready_s=args.wait_ready,
    )

    if args.into is not None and args.wait_ready > 0 and not args.into.exists():
        print(f"[SERVEKIT] waiting for the profile report {args.into}", flush=True)
        if not _wait_for_report(args.into, args.wait_ready):
            print(f"error: no profile report at {args.into} after {args.wait_ready:g}s", file=sys.stderr)
            return 1

    if args.wait_ready > 0:
        print(f"[SERVEKIT] waiting up to {args.wait_ready:g}s for {args.url} to serve", flush=True)
    print(f"[SERVEKIT] benchmarking {args.url}", flush=True)

    report = run_benchmark(args.url, cfg)
    print()
    print(render_bench(report))

    rc = 1 if report.errors else 0
    if args.into is not None:
        rc = _merge_into(args.into, report.to_dict()) or rc
    if args.out is not None:
        args.out.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nbench report written to {args.out}", flush=True)
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("profile", "bench"):
        print(USAGE, file=sys.stderr)
        return 2
    return _profile(argv[1:]) if argv[0] == "profile" else _bench(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
