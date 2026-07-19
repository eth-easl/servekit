from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .bench import BenchConfig, BenchReport, render_bench
from .profile import (
    ProfileReport,
    render_table,
    run_profile,
    run_profile_with_bench,
    save_json,
)


def _derive_base_url(command: List[str]) -> str:
    """Pull --host/--port out of the wrapped launch command for the bench target."""
    host, port = "127.0.0.1", "8080"
    for i, tok in enumerate(command[:-1]):
        if tok == "--host":
            host = command[i + 1]
        elif tok == "--port":
            port = command[i + 1]
    if host in ("0.0.0.0", "::", ""):
        host = "127.0.0.1"
    return f"http://{host}:{port}"


def _split_command(rest: List[str]) -> Tuple[List[str], List[str]]:
    if "--" in rest:
        idx = rest.index("--")
        return rest[:idx], rest[idx + 1 :]
    return [], rest


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] != "profile":
        print(
            "usage: servekit profile [--out PATH] [--timeout SECONDS] [--bench [--keep-alive] ...] -- <command...>",
            file=sys.stderr,
        )
        return 2

    options, command = _split_command(argv[1:])

    parser = argparse.ArgumentParser(prog="servekit profile")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for the ready signal")
    parser.add_argument("--bench", action="store_true", help="benchmark the live server after it reports ready")
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="with --bench: leave the server running after the benchmark (report is written as soon as the bench ends)",
    )
    parser.add_argument("--bench-url", default=None, help="server base URL (default: derived from --host/--port)")
    parser.add_argument("--bench-requests", type=int, default=50)
    parser.add_argument("--bench-input-len", type=int, default=512, help="approx prompt length in words")
    parser.add_argument("--bench-output-len", type=int, default=128, help="max_new_tokens per request")
    parser.add_argument("--bench-concurrency", type=int, default=16)
    parser.add_argument("--bench-seed", type=int, default=42)
    parser.add_argument("--no-correctness", action="store_true", help="skip the correctness probe")
    args = parser.parse_args(options)

    if not command:
        print("error: no command given after --", file=sys.stderr)
        return 2

    print(f"[SERVEKIT] profiling cold start of: {' '.join(command)}", flush=True)

    def emit(report: ProfileReport) -> None:
        report.command = " ".join(command)
        print()
        print(render_table(report))
        if report.benchmark is not None:
            print()
            print(render_bench(BenchReport(**report.benchmark)))
        out_path = args.out or Path(f"servekit-profile-{int(report.started_at)}.json")
        save_json(report, out_path)
        print(f"\nreport written to {out_path}", flush=True)

    if args.bench:
        base_url = args.bench_url or _derive_base_url(command)
        print(f"[SERVEKIT] will benchmark {base_url} after ready", flush=True)
        cfg = BenchConfig(
            requests=args.bench_requests,
            input_len=args.bench_input_len,
            output_len=args.bench_output_len,
            concurrency=args.bench_concurrency,
            seed=args.bench_seed,
            correctness=not args.no_correctness,
        )
        if args.keep_alive:
            # The server outlives the bench, so the report must be written the
            # moment the bench ends -- not on return, which is now "when the
            # server exits". emit fires from the callback on a good run; a run
            # that never got ready has no bench and nothing to keep alive, so
            # the callback never fires and we emit the failure here.
            report = run_profile_with_bench(
                command, base_url, cfg, ready_timeout=args.timeout,
                keep_alive=True, on_bench_done=emit,
            )
            if not report.success:
                emit(report)
        else:
            report = run_profile_with_bench(command, base_url, cfg, ready_timeout=args.timeout)
            emit(report)
    else:
        emitted = {"done": False}

        def emit_once(report: ProfileReport) -> None:
            emitted["done"] = True
            emit(report)

        report = run_profile(command, ready_timeout=args.timeout, on_ready=emit_once)
        if not emitted["done"]:
            emit(report)

    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
