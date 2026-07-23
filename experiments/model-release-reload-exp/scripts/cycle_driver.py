"""Release the weights from a live SGLang server, resume, reload from disk - and time it.

Runs *after* servekit reports ready (the profile JSON's existence is the ready
signal - servekit writes it the moment the post-ready bench finishes, and
--keep-alive leaves the server up for us).

The sequence, with a memory snapshot around every step:

    S1 post_bench -> release(weights) -> S2 -> resume(weights) -> S3
       -> update_weights_from_disk -> S4 -> re-bench

WHY THE THREE CALLS ARE ALL NEEDED. `enable_weights_cpu_backup` defaults False
(model_runner.py:1148), so release genuinely FREES the weights rather than
stashing them in host RAM. Resume therefore hands back garbage physical pages at
the same virtual addresses, and update_weights_from_disk is what makes the model
real again. If VmRSS jumps by ~35 GB/rank on release, that assumption broke and
the reload timing below measures nothing - hence VmRSS is in every snapshot.

NO /generate BETWEEN RELEASE AND RELOAD. Between S2 and S4 the weight VAs are
unmapped or garbage; a query would segfault the worker, not return an error.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


def _post(base_url: str, path: str, body: dict, timeout: float) -> tuple[int, object]:
    """POST json, return (status, parsed_or_text). A 4xx/5xx is data, not an exception."""
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode(), e.code
    try:
        return status, json.loads(raw) if raw.strip() else None
    except json.JSONDecodeError:
        return status, raw


def _csv_rows(cmd: str, ncols: int) -> tuple[list, Optional[str]]:
    """Run an nvidia-smi --format=csv query -> (rows, error).

    nvidia-smi prints prose on failure ("couldn't communicate with the NVIDIA
    driver") and can emit warnings alongside real rows, so anything that isn't
    an ncols-wide row is reported as an error rather than parsed. A snapshot
    must never crash the cycle: a lost run costs a 70B cold start.
    """
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    rows, junk = [], []
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        (rows if len(parts) == ncols else junk).append(parts if len(parts) == ncols else line.strip())
    err = None
    if p.returncode != 0 or junk or (not rows and p.stderr.strip()):
        err = " | ".join(junk + ([p.stderr.strip()] if p.stderr.strip() else [])) or f"rc={p.returncode}"
    return rows, err


def _proc_mem(pid: str) -> dict:
    """Host-side memory for one PID. VmPin/VmLck are absent on some kernels -> omitted."""
    out: dict = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            k = line.split(":")[0]
            if k in ("VmRSS", "VmPin", "VmLck"):
                out[k] = line.split(":", 1)[1].strip()
    except OSError as e:
        # nvidia-smi reports host PIDs; if the container had a PID namespace they
        # would not resolve here. Say so rather than silently reporting nothing.
        out["error"] = f"/proc/{pid}/status unreadable: {e}"
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            k = line.split(":")[0]
            if k in ("Rss", "Pss", "Locked"):
                out[f"smaps_{k}"] = line.split(":", 1)[1].strip()
    except OSError:
        pass
    return out


def snapshot(label: str) -> dict:
    """GPU + host memory, as seen from outside the engine. The evidence."""
    app_rows, app_err = _csv_rows(
        "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader", 2
    )
    apps = [{"pid": pid, "gpu_used": used, **_proc_mem(pid)} for pid, used in app_rows]

    gpu_rows, gpu_err = _csv_rows(
        "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader", 3
    )
    gpus = [{"index": i, "used": u, "total": t} for i, u, t in gpu_rows]

    # Page cache: the cold-start load is buffered pread (no GDS here), so the
    # model is probably still cached when we reload -> the reload time is a
    # warm-cache lower bound. Record it instead of pretending otherwise.
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k = line.split(":")[0]
        if k in ("MemTotal", "MemAvailable", "Cached"):
            meminfo[k] = line.split(":", 1)[1].strip()

    snap = {
        "label": label,
        "t": time.time(),
        "compute_apps": apps,
        "gpus": gpus,
        "meminfo": meminfo,
    }
    if app_err or gpu_err:
        snap["nvidia_smi_error"] = app_err or gpu_err
        print(f"[CYCLE] WARNING: nvidia-smi at {label}: {app_err or gpu_err}", file=sys.stderr, flush=True)
    return snap


def render(snaps: list, timings: dict) -> str:
    lines = ["", "=" * 78, "release/resume/reload cycle", "=" * 78]
    lines.append(f"{'step':<26}{'seconds':>10}")
    lines.append("-" * 36)
    for k, v in timings.items():
        lines.append(f"{k:<26}{v:>10.2f}")
    lines.append("")
    hdr = f"{'snapshot':<18}{'gpu_used(all)':>18}{'per-proc gpu':>34}{'VmRSS(sum)':>14}"
    lines += [hdr, "-" * len(hdr)]
    for s in snaps:
        gpu_tot = " ".join(g["used"].replace(" MiB", "") for g in s["gpus"]) or "-"
        per = " ".join(a["gpu_used"].replace(" MiB", "") for a in s["compute_apps"]) or "-"
        rss = sum(int(a["VmRSS"].split()[0]) for a in s["compute_apps"] if "VmRSS" in a)
        lines.append(f"{s['label']:<18}{gpu_tot:>18}{per:>34}{rss/1024/1024:>13.1f}G")
    lines.append("")
    for s in snaps:
        lines.append(f"  {s['label']:<18} Cached={s['meminfo'].get('Cached','?')}"
                     f"  MemAvailable={s['meminfo'].get('MemAvailable','?')}")
    lines.append("=" * 78)
    return "\n".join(lines)


def wait_for_ready(profile_json: Path, timeout: float) -> Optional[dict]:
    """servekit writes the profile JSON on ready -> its existence IS the ready signal."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if profile_json.exists():
            time.sleep(2)  # let the write land
            try:
                return json.loads(profile_json.read_text())
            except json.JSONDecodeError:
                time.sleep(2)
                return json.loads(profile_json.read_text())
        time.sleep(5)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-json", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--load-format", default="fastsafetensors")
    ap.add_argument("--tags", default="weights")
    ap.add_argument(
        "--no-release",
        action="store_true",
        help="skip release+resume and only call update_weights_from_disk -- the control "
        "that says whether the reload works at all, independently of the memory saver",
    )
    ap.add_argument("--ready-timeout", type=float, default=1800.0)
    ap.add_argument("--call-timeout", type=float, default=1800.0)
    args = ap.parse_args()

    tags = [] if args.no_release else [t for t in args.tags.split(",") if t]
    report: dict = {
        "tags": tags,
        "no_release": args.no_release,
        "model": args.model,
        "load_format": args.load_format,
    }
    snaps: list = []
    timings: dict = {}

    print(f"[CYCLE] waiting for {args.profile_json}", flush=True)
    profile = wait_for_ready(args.profile_json, args.ready_timeout)
    if profile is None:
        print("[CYCLE] FAIL: profile JSON never appeared -> server never got ready", file=sys.stderr)
        return 1
    if not profile.get("success"):
        print("[CYCLE] FAIL: cold start did not reach ready", file=sys.stderr)
        return 1
    print("[CYCLE] server is up and benched", flush=True)

    snaps.append(snapshot("S1_post_bench"))

    def step(name: str, path: str, body: dict) -> object:
        print(f"[CYCLE] -> {name} {body}", flush=True)
        t0 = time.time()
        status, resp = _post(args.base_url, path, body, args.call_timeout)
        timings[name] = time.time() - t0
        print(f"[CYCLE] <- {name} status={status} in {timings[name]:.2f}s resp={resp}", flush=True)
        report.setdefault("responses", {})[name] = {"status": status, "body": resp}
        return status

    if args.no_release:
        print("[CYCLE] --no-release: skipping release/resume, reload only", flush=True)
    else:
        st = step("release_memory_occupation", "/release_memory_occupation", {"tags": tags})
        snaps.append(snapshot("S2_post_release"))
        if st != 200:
            report["error"] = "release failed"

        step("resume_memory_occupation", "/resume_memory_occupation", {"tags": tags})
        snaps.append(snapshot("S3_post_resume"))

    st = step(
        "update_weights_from_disk",
        "/update_weights_from_disk",
        {"model_path": args.model, "load_format": args.load_format},
    )
    snaps.append(snapshot("S4_post_reload"))
    if st != 200:
        report["error"] = "update_weights_from_disk failed"

    # Re-verify with the SAME prompts and the SAME seeded workload the cold-start
    # bench used, by calling the same code -> apples-to-apples by construction.
    try:
        from servekit.bench import BenchConfig, render_bench, run_benchmark

        old = (profile.get("benchmark") or {}).get("throughput") or {}
        cfg = BenchConfig(
            requests=old.get("requests", 64),
            input_len=old.get("input_len", 512),
            output_len=old.get("output_len", 128),
            concurrency=old.get("concurrency", 16),
        )
        t0 = time.time()
        after = run_benchmark(args.base_url, cfg)
        timings["rebench"] = time.time() - t0
        report["bench_after_reload"] = after.to_dict()
        print(render_bench(after), flush=True)

        # Qualitative, not hash-equality: SGLang is not bit-deterministic across
        # runs (see bench.py) - exact comparison would false-alarm.
        before_c = ((profile.get("benchmark") or {}).get("correctness") or {}).get("results", [])
        after_c = (after.correctness or {}).get("results", [])
        print("\n[CYCLE] correctness, cold start vs after reload:", flush=True)
        for b, a in zip(before_c, after_c):
            same = "SAME" if b["output"].strip() == a["output"].strip() else "DIFF"
            print(f"  [{same}] {b['prompt'][:38]!r}", flush=True)
            if same == "DIFF":
                print(f"      before: {' '.join(b['output'].split())[:90]!r}", flush=True)
                print(f"      after : {' '.join(a['output'].split())[:90]!r}", flush=True)
        report["correctness_identical"] = [
            b["output"].strip() == a["output"].strip() for b, a in zip(before_c, after_c)
        ]
    except Exception as e:  # noqa: BLE001
        report["error"] = f"re-bench failed: {e}"
        print(f"[CYCLE] re-bench failed: {e}", file=sys.stderr, flush=True)

    report["snapshots"] = snaps
    report["timings_s"] = {k: round(v, 2) for k, v in timings.items()}
    args.out.write_text(json.dumps(report, indent=2))
    print(render(snaps, timings), flush=True)
    print(f"[CYCLE] report written to {args.out}", flush=True)
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
