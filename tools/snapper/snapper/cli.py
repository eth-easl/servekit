from __future__ import annotations

import typer

from . import config, rcc, service, slurm, snapshot

app = typer.Typer(help="Lifecycle CLI for LLM deployments on CSCS clusters.")


@app.command("list")
def list_cmd():
    """List discovered deployments (deploy/*/snapper.toml)."""
    found = config.discover(config.repo_root() / "deploy")
    if not found:
        typer.echo("no deployments found (no deploy/*/snapper.toml)")
        return
    for name, dep in sorted(found.items()):
        typer.echo(
            f"{name}\tprofile={dep.profile}\tengine={dep.engine}\tnodes={dep.default_nodes}"
        )


def _status_one(dep) -> None:
    typer.echo(f"== {dep.name} ({dep.profile}) ==")
    env = service.read_service_env(dep.profile)
    if env:
        for key, value in env.items():
            typer.echo(f"{key}={value}")
    else:
        typer.echo("(no last_service.env; not started?)")
    sq = rcc.run(dep.profile, 'squeue -u "$USER" -o "%.18i %.9P %.30j %.2t %.10M %.6D %R"')
    typer.echo(sq.stdout.rstrip())


def _parse_set(pairs: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise typer.BadParameter(f"--set expects KEY=VAL, got {pair!r}")
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"--set expects KEY=VAL, got {pair!r}")
        out[key] = value.strip()
    return out


def _resolve_serve_script(dep, variant: str | None, snap_override: str | None) -> str:
    if variant:
        if variant not in dep.variants:
            known = ", ".join(sorted(dep.variants)) or "(none)"
            raise typer.BadParameter(f"unknown variant {variant!r}; known: {known}")
        return dep.variants[variant]
    if snap_override:
        return snap_override
    return dep.serve_script


@app.command("status")
def status_cmd(name: str | None = typer.Argument(None, help="deployment name; omit for all")):
    """Show squeue + last_service.env for one or all deployments."""
    root = config.repo_root()
    if name:
        _status_one(config.load(name, root=root))
        return
    for _, dep in sorted(config.discover(root / "deploy").items()):
        _status_one(dep)
        typer.echo("")


@app.command("up")
def up_cmd(
    name: str = typer.Argument(..., help="deployment name"),
    nodes: int | None = typer.Option(
        None,
        "--nodes",
        help="node count (default: manifest default_nodes)",
    ),
    variant: str | None = typer.Option(None, "--variant", help="serve-script variant"),
    snapshot_mode: str | None = typer.Option(
        None,
        "--snapshot",
        help="record|restore (opt-in seam)",
    ),
    sets: list[str] | None = typer.Option(None, "--set", help="extra env KEY=VAL"),
    no_push: bool = typer.Option(False, "--no-push"),
    no_probe: bool = typer.Option(False, "--no-probe"),
    no_wait: bool = typer.Option(False, "--no-wait"),
    timeout: float = typer.Option(1800, "--timeout", help="ready-wait timeout seconds"),
):
    """Bring a deployment up: push -> probe -> serve -> wait-for-ready -> endpoint."""
    dep = config.load(name, root=config.repo_root())
    node_count = nodes or dep.default_nodes
    snap_env, snap_serve = snapshot.plan(dep, snapshot_mode)
    serve_script = _resolve_serve_script(dep, variant, snap_serve)
    env = {**_parse_set(sets), **snap_env}

    if not no_push:
        typer.echo(f"[push] {dep.profile}")
        pushed = rcc.push(dep.profile)
        if pushed.returncode != 0:
            typer.echo(pushed.stderr or pushed.stdout)
            raise typer.Exit(1)

    if not no_probe and dep.probe_script:
        typer.echo("[probe] submitting")
        probe_jobid = slurm.probe(dep)
        state = slurm.wait_in_queue(dep.profile, probe_jobid)
        log = rcc.run(dep.profile, f"tail -n 40 slurm-{probe_jobid}.out 2>/dev/null || true")
        if state not in ("COMPLETED", "") or log.returncode != 0:
            typer.echo(f"[probe] FAILED (state={state}). Last log lines:")
            typer.echo(log.stdout)
            typer.echo("See .agents/skills/deploy/references/gotchas.md")
            raise typer.Exit(1)
        typer.echo("[probe] ok")

    typer.echo(f"[serve] submitting {serve_script} on {node_count} node(s)")
    jobid = slurm.submit(dep, nodes=node_count, serve_script=serve_script, env=env)
    typer.echo(f"[serve] job {jobid}")

    if no_wait:
        typer.echo("[serve] submitted; not waiting (--no-wait)")
        return

    typer.echo("[serve] waiting for all ranks ready (cold start can take 10-20 min)...")
    if not slurm.wait_for_ready(dep, jobid, node_count, timeout=timeout):
        typer.echo("[serve] TIMEOUT before ready; job left running. Use `snapper logs`/`snapper down`.")
        raise typer.Exit(1)

    env_out = service.read_service_env(dep.profile)
    endpoints = service.endpoints(env_out)
    typer.echo("[ready] endpoints:")
    for endpoint in endpoints:
        typer.echo(f"  {endpoint}")
    if endpoints:
        typer.echo(f"\nverify: snapper verify {dep.name}")


@app.command("down")
def down_cmd(name: str = typer.Argument(..., help="deployment name")):
    """Cancel the running serve job for a deployment."""
    dep = config.load(name, root=config.repo_root())
    env = service.read_service_env(dep.profile)
    jobid = env.get("JOBID") or env.get("JOB_ID")
    if not jobid:
        typer.echo(f"{dep.name}: nothing to cancel (no JOBID in last_service.env)")
        return
    res = rcc.run(dep.profile, f"scancel {jobid}")
    if res.returncode != 0:
        typer.echo(res.stderr or res.stdout)
        raise typer.Exit(1)
    typer.echo(f"{dep.name}: scancel {jobid}")


@app.command("logs")
def logs_cmd(
    name: str = typer.Argument(..., help="deployment name"),
    rank: int = typer.Option(0, "--rank"),
    follow: bool = typer.Option(False, "--follow", "-f"),
    lines: int = typer.Option(80, "--lines", "-n"),
):
    """Tail a per-rank OpenTela log."""
    dep = config.load(name, root=config.repo_root())
    env = service.read_service_env(dep.profile)
    jobid = env.get("JOBID") or env.get("JOB_ID")
    if not jobid:
        typer.echo(f"{dep.name}: no JOBID in last_service.env; not started?")
        raise typer.Exit(1)
    path = dep.log_pattern.format(job=jobid, rank=rank)
    flag = "-f " if follow else ""
    res = rcc.run(dep.profile, f"tail {flag}-n {lines} {path}", stream=follow)
    if res.returncode != 0:
        typer.echo(res.stderr or res.stdout)
        raise typer.Exit(1)
    if not follow:
        typer.echo(res.stdout.rstrip())


@app.command("verify")
def verify_cmd(name: str = typer.Argument(..., help="deployment name")):
    """Smoke-test the service: /v1/models + one chat request."""
    dep = config.load(name, root=config.repo_root())
    env = service.read_service_env(dep.profile)
    endpoints = service.endpoints(env)
    if not endpoints:
        typer.echo(f"{dep.name}: no endpoints in last_service.env; not ready?")
        raise typer.Exit(1)
    res = service.verify(dep, endpoints[0])
    typer.echo(res.stdout.rstrip())
    if res.returncode != 0:
        typer.echo(res.stderr)
        raise typer.Exit(1)


def main():
    app()
