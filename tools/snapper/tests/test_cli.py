from typer.testing import CliRunner

from snapper import cli, config, rcc, service as service_mod

runner = CliRunner()


def test_list_shows_discovered(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    result = runner.invoke(cli.app, ["list"])
    assert result.exit_code == 0
    assert "glm-47-flash-bristen" in result.stdout
    assert "sglang" in result.stdout


def test_status_one_deployment_reads_cluster(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)

    def fake_run(profile, command, **kw):
        if "last_service.env" in command:
            return rcc.RunResult(0, "ENDPOINT_0=nid001:8080\nJOBID=7\n", "")
        return rcc.RunResult(0, "  7  normal job RUN\n", "")

    monkeypatch.setattr(rcc, "run", fake_run)
    result = runner.invoke(cli.app, ["status", "glm-47-flash-bristen"])
    assert result.exit_code == 0
    assert "nid001:8080" in result.stdout


def test_parse_set_pairs():
    from snapper.cli import _parse_set

    assert _parse_set(["TP_SIZE=2", "X=y"]) == {"TP_SIZE": "2", "X": "y"}


def test_resolve_serve_script_variant_wins(deploy_tree):
    from snapper.cli import _resolve_serve_script

    dep = config.load("glm-47-flash-bristen", root=deploy_tree)
    assert (
        _resolve_serve_script(dep, "router", None)
        == "serve_glm_47_flash_sglang_router.sbatch"
    )
    assert _resolve_serve_script(dep, None, None) == dep.serve_script
    assert _resolve_serve_script(dep, None, "serve_snap.sbatch") == "serve_snap.sbatch"


def test_up_runs_full_sequence(deploy_tree, monkeypatch):
    from snapper import service as service_mod
    from snapper import slurm

    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    calls = []
    monkeypatch.setattr(
        rcc,
        "push",
        lambda profile: calls.append(("push", profile)) or rcc.RunResult(0, "", ""),
    )
    monkeypatch.setattr(
        slurm,
        "probe",
        lambda dep, **k: calls.append(("probe", dep.name)) or 100,
    )
    monkeypatch.setattr(slurm, "wait_in_queue", lambda *a, **k: "COMPLETED")
    monkeypatch.setattr(rcc, "run", lambda profile, command, **k: rcc.RunResult(0, "PROBE OK", ""))
    monkeypatch.setattr(
        slurm,
        "submit",
        lambda dep, **k: calls.append(("submit", k.get("nodes"))) or 200,
    )
    monkeypatch.setattr(slurm, "wait_for_ready", lambda *a, **k: True)
    monkeypatch.setattr(
        service_mod,
        "read_service_env",
        lambda profile, **k: {"ENDPOINT_0": "nid001:8080", "JOBID": "200"},
    )

    result = runner.invoke(cli.app, ["up", "glm-47-flash-bristen", "--nodes", "3"])
    assert result.exit_code == 0, result.stdout
    names = [call[0] for call in calls]
    assert names == ["push", "probe", "submit"]
    assert ("submit", 3) in calls
    assert "nid001:8080" in result.stdout


def test_up_aborts_when_probe_fails(deploy_tree, monkeypatch):
    from snapper import slurm

    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    monkeypatch.setattr(rcc, "push", lambda profile: rcc.RunResult(0, "", ""))
    monkeypatch.setattr(slurm, "probe", lambda dep, **k: 100)
    monkeypatch.setattr(slurm, "wait_in_queue", lambda *a, **k: "FAILED")
    monkeypatch.setattr(rcc, "run", lambda profile, command, **k: rcc.RunResult(1, "boom traceback", ""))
    submitted = []
    monkeypatch.setattr(slurm, "submit", lambda dep, **k: submitted.append(1) or 200)

    result = runner.invoke(cli.app, ["up", "glm-47-flash-bristen", "--nodes", "1"])
    assert result.exit_code != 0
    assert submitted == []


def test_down_scancels_jobid(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    monkeypatch.setattr(service_mod, "read_service_env", lambda profile, **k: {"JOBID": "321"})
    seen = {}
    monkeypatch.setattr(
        rcc,
        "run",
        lambda profile, command, **k: seen.update(cmd=command) or rcc.RunResult(0, "", ""),
    )
    result = runner.invoke(cli.app, ["down", "glm-47-flash-bristen"])
    assert result.exit_code == 0
    assert "scancel 321" in seen["cmd"]


def test_down_no_job_is_friendly(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    monkeypatch.setattr(service_mod, "read_service_env", lambda profile, **k: {})
    result = runner.invoke(cli.app, ["down", "glm-47-flash-bristen"])
    assert result.exit_code == 0
    assert "nothing" in result.stdout.lower()


def test_down_accepts_legacy_job_id_key(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    monkeypatch.setattr(service_mod, "read_service_env", lambda profile, **k: {"JOB_ID": "654"})
    seen = {}
    monkeypatch.setattr(
        rcc,
        "run",
        lambda profile, command, **k: seen.update(cmd=command) or rcc.RunResult(0, "", ""),
    )
    result = runner.invoke(cli.app, ["down", "glm-47-flash-bristen"])
    assert result.exit_code == 0
    assert "scancel 654" in seen["cmd"]


def test_logs_tails_resolved_rank_path(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    monkeypatch.setattr(service_mod, "read_service_env", lambda profile, **k: {"JOBID": "77"})
    seen = {}
    monkeypatch.setattr(
        rcc,
        "run",
        lambda profile, command, **k: seen.update(cmd=command) or rcc.RunResult(0, "loglines", ""),
    )
    result = runner.invoke(cli.app, ["logs", "glm-47-flash-bristen", "--rank", "2"])
    assert result.exit_code == 0
    assert "logs/opentela-77-2.log" in seen["cmd"]


def test_verify_hits_first_endpoint(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    monkeypatch.setattr(
        service_mod,
        "read_service_env",
        lambda profile, **k: {"ENDPOINT_0": "nid001:8080"},
    )
    seen = {}
    monkeypatch.setattr(
        service_mod,
        "verify",
        lambda dep, ep, **k: seen.update(ep=ep) or rcc.RunResult(0, "{}", ""),
    )
    result = runner.invoke(cli.app, ["verify", "glm-47-flash-bristen"])
    assert result.exit_code == 0
    assert seen["ep"] == "nid001:8080"
