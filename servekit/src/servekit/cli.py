from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .bench import BenchConfig, run_benchmark, render_bench
from .launch import DEFAULT_ROOT, launch
from .profile import ProfileReport, detect_framework, render_table, run_profile, save_json
from .stage import DEFAULT_SLICES

USAGE = """usage:
  servekit launch [--out PATH] [--slices N] -- <command...>
  servekit profile [--out PATH] [--timeout SECONDS] -- <command...>
  servekit bench --url URL (--into PATH | --out PATH) [--wait-ready SECONDS] [...]

`launch` wraps an engine command: it copies the model into /dev/shm, starts the
engine against the copy, and frees the copy once the server reports ready --
returning the RAM to the running job while it serves. It behaves like the
command it wraps, so removing the wrapper gives the baseline back. `profile` is
`launch` without the staging. `bench` loads any live server over the OpenAI
protocol, whether or not servekit launched it.

  servekit launch -- python -m sglang.launch_server --model-path /store/llama70b-tp4 \\
      --tensor-parallel-size 4 --load-format sharded_state

To profile and bench one run, put them side by side and join on the report file:

  servekit launch --out run.json -- python -m sglang.launch_server ... &
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

    try:
        spec = detect_framework(command)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"[SERVEKIT] profiling cold start of {spec.name}: {' '.join(command)}", flush=True)

    def emit(report: ProfileReport) -> None:
        report.command = " ".join(command)
        print()
        print(render_table(report))
        out_path = args.out or Path(f"servekit-profile-{int(report.started_at)}.json")
        save_json(report, out_path)
        print(f"\nreport written to {out_path}", flush=True)

    # Fires the moment ready is detected, not at process exit, so a concurrent
    # `servekit bench` can see the report while the server is still up.
    emitted = {"done": False}

    def emit_once(report: ProfileReport) -> None:
        emitted["done"] = True
        emit(report)

    report = run_profile(command, ready_timeout=args.timeout, on_ready=emit_once)
    if not emitted["done"]:
        emit(report)
    return 0 if report.success else 1


def _launch(argv: List[str]) -> int:
    options, command = _split_command(argv)

    parser = argparse.ArgumentParser(prog="servekit launch")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for the ready signal")
    parser.add_argument("--shm-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--slices", type=int, default=DEFAULT_SLICES, help="concurrent read slices per file")
    args = parser.parse_args(options)

    if not command:
        print("error: no command given after --", file=sys.stderr)
        return 2

    try:
        return launch(
            command,
            out=args.out,
            shm_root=args.shm_root,
            slices=args.slices,
            timeout=args.timeout,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _wait_for_report(path: Path, timeout_s: float, interval_s: float = 0.5) -> bool:
    """Block until `profile` has written its report, or `timeout_s` elapses.

    An engine accepts traffic before it announces ready, so probing over HTTP
    alone could start benching mid-warmup and corrupt the cold-start
    measurement. The report's appearance is the ready signal instead.
    """
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 >= timeout_s:
            return False
        time.sleep(interval_s)
    return True


def _merge_into(path: Path, benchmark: dict) -> int:
    """Write `benchmark` into an existing profile report, in place.

    If the report is missing (profile crashed, wrong path), the benchmark is
    salvaged to a sibling file instead of being thrown away.
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
    # Only the *directory* is checked up front -- the report file itself usually
    # doesn't exist yet (see _merge_into).
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
    commands = {"launch": _launch, "profile": _profile, "bench": _bench}
    if not argv or argv[0] not in commands:
        print(USAGE, file=sys.stderr)
        return 2
    return commands[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
