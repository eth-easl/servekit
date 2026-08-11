from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .bench import BenchConfig, run_benchmark, render_bench
from .launch import DEFAULT_ROOT, launch
from .prepare import prepare
from .profile import ProfileReport, detect_framework, render_table, run_profile, save_json
from .stage import DEFAULT_SLICES
from .topology import read_topology
from .verify import capture, compare, discover_model, load_prompt_set, render_verify, wait_for_ready

USAGE = """usage:
  servekit launch [--out PATH] [--slices N] [--overlap] -- <command...>
  servekit prepare --model PATH --out PATH [--tp N] [-- <extra engine args>]
                   [--nnodes N --node-rank N --dist-init-addr HOST:PORT]
  servekit profile [--out PATH] [--timeout SECONDS] -- <command...>
  servekit bench --url URL (--into PATH | --out PATH) [--wait-ready SECONDS] [...]
  servekit verify --url URL (--record PATH | --reference PATH) [--wait-ready SECONDS] [...]

`launch` wraps an engine command: it copies the model into /dev/shm, starts the
engine against the copy, and frees the copy once the server reports ready --
returning the RAM to the running job while it serves. It behaves like the
command it wraps, so removing the wrapper gives the baseline back. `profile` is
`launch` without the staging. `bench` loads any live server over the OpenAI
protocol, whether or not servekit launched it. `prepare` is the one offline
step: it writes a TP-presharded checkpoint that `launch` stages and loads with
`--load-format sharded_state`. `verify` checks that a served model produces the
same per-token logprobs it did before: `--record` captures a reference from a
trusted server, `--reference` checks a later server against it.

  servekit launch -- python -m sglang.launch_server --model-path /store/llama70b-tp4 \\
      --tensor-parallel-size 4 --load-format sharded_state

To profile and bench one run, put them side by side and join on the report file:

  servekit launch --out run.json -- python -m sglang.launch_server ... &
  PROF=$!
  servekit bench --url http://127.0.0.1:8080 --into run.json --requests 64
  kill $PROF; wait $PROF

Multi-node is the same command on every node, under an srun of one task per
node, with the engine's own distributed flags set. Each node stages the shards
its own ranks read and writes its own run.node<rank>.json:

  srun --ntasks=$NNODES --ntasks-per-node=1 servekit launch --out run.json -- \\
      python -m sglang.launch_server --model-path /store/llama70b-tp8 --tp-size 8 \\
      --nnodes $NNODES --node-rank $SLURM_PROCID --dist-init-addr $HEAD:20000 \\
      --load-format sharded_state
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
        topo = read_topology(command, spec)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"[SERVEKIT] profiling cold start of {spec.name}: {' '.join(command)}", flush=True)

    def emit(report: ProfileReport) -> None:
        report.command = " ".join(command)
        if topo.is_multinode:
            report.node_rank = topo.node_rank
            report.nnodes = topo.nnodes
        print()
        print(render_table(report))
        out_path = args.out or Path(f"servekit-profile-{int(report.started_at)}.json")
        if topo.is_multinode:
            out_path = out_path.with_name(f"{out_path.stem}.node{topo.node_rank}{out_path.suffix}")
        save_json(report, out_path)
        print(f"\nreport written to {out_path}", flush=True)

    # Fires the moment ready is detected, not at process exit, so a concurrent
    # `servekit bench` can see the report while the server is still up.
    emitted = {"done": False}

    def emit_once(report: ProfileReport) -> None:
        emitted["done"] = True
        emit(report)

    report = run_profile(
        command,
        ready_timeout=args.timeout,
        on_ready=emit_once,
        head=topo.is_head,
    )
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
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="UNSAFE: start the engine concurrently with the stage; no barrier, corruption is silent",
    )
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
            overlap=args.overlap,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _prepare(argv: List[str]) -> int:
    options, engine_args = _split_command(argv)

    parser = argparse.ArgumentParser(prog="servekit prepare")
    parser.add_argument("--model", type=Path, required=True, help="source checkpoint (a local directory)")
    parser.add_argument("--out", type=Path, required=True, help="where to write the presharded checkpoint")
    parser.add_argument("--tp", type=int, default=1, help="tensor-parallel size the shards will be locked to")
    parser.add_argument("--nnodes", type=int, default=1, help="nodes to shard across (TP > GPUs per node)")
    parser.add_argument("--node-rank", type=int, default=0, help="this node's rank, e.g. $SLURM_PROCID")
    parser.add_argument("--dist-init-addr", default=None, help="rendezvous address, HOST:PORT on the head")
    args = parser.parse_args(options)

    return prepare(
        args.model,
        args.out,
        args.tp,
        engine_args=engine_args,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        dist_init_addr=args.dist_init_addr,
    )


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


def _verify(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="servekit verify")
    parser.add_argument("--url", default=None, help="server base URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--record", type=Path, default=None, help="capture a reference from this server")
    parser.add_argument("--reference", type=Path, default=None, help="check this server against a reference")
    parser.add_argument("--prompts", type=Path, default=None, help="override the built-in prompt set (--record only)")
    parser.add_argument("--out", type=Path, default=None, help="write the machine-readable check result here")
    parser.add_argument("--wait-ready", type=float, default=0.0, help="seconds to wait for the server to serve")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request timeout")
    parser.add_argument("--token-tol", type=float, default=1e-6, help="max allowed |delta| per token logprob")
    parser.add_argument("--nll-tol", type=float, default=1e-6, help="max allowed |delta| in mean NLL")
    args = parser.parse_args(argv)

    if not args.url:
        print("error: --url is required", file=sys.stderr)
        return 2
    if bool(args.record) == bool(args.reference):
        print("error: exactly one of --record or --reference is required", file=sys.stderr)
        return 2
    if args.reference and args.prompts:
        print("error: --prompts only applies to --record; the reference carries its own prompts", file=sys.stderr)
        return 2

    if args.wait_ready > 0:
        print(f"[SERVEKIT] waiting up to {args.wait_ready:g}s for {args.url} to serve", flush=True)
        try:
            _, model = wait_for_ready(args.url, args.wait_ready)
        except TimeoutError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
    else:
        model = discover_model(args.url)

    if args.record:
        prompt_set, prompts = load_prompt_set(args.prompts)
        print(f"[SERVEKIT] recording {len(prompts)} prompts from {args.url}", flush=True)
        result = capture(args.url, model, prompts, prompt_set, timeout=args.timeout)
        args.record.write_text(json.dumps(result, indent=1))
        print(f"[SERVEKIT] reference written to {args.record}", flush=True)
        return 0

    reference = json.loads(args.reference.read_text())
    prompt_set, prompts = load_prompt_set(None)
    ref_prompts = [(p["key"], p["text"]) for p in reference["prompts"]]
    print(f"[SERVEKIT] checking {len(ref_prompts)} prompts from {args.url} against {args.reference}", flush=True)
    captured = capture(
        args.url,
        model,
        ref_prompts,
        reference.get("prompt_set", prompt_set),
        timeout=args.timeout,
        greedy_prompts=reference.get("greedy_prompts", 8),
        greedy_tokens=reference.get("greedy_tokens", 32),
    )
    result = compare(reference, captured, token_tol=args.token_tol, nll_tol=args.nll_tol)
    print()
    print(render_verify(args.url, result))

    if args.out:
        args.out.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nresult written to {args.out}", flush=True)

    return 0 if result.passed else 1


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    commands = {
        "launch": _launch,
        "prepare": _prepare,
        "profile": _profile,
        "bench": _bench,
        "verify": _verify,
    }
    if not argv or argv[0] not in commands:
        print(USAGE, file=sys.stderr)
        return 2
    return commands[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
