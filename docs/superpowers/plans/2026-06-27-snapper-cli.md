# snapper CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `snapper`, a small Python CLI that drives the `docs/start_model.md` runbook — full deployment lifecycle (`up`/`down`/`status`/`logs`/`verify`/`list`) over `rcc` transport, with an inert opt-in snapshot fast-launch seam.

**Architecture:** A Python package under `tools/snapper/`. The **only** I/O boundary to the cluster is `rcc.py` (it shells out to the `rcc` CLI); every other module is pure, fixture-testable logic. Deployments are described by a per-dir `deploy/<model>-<cluster>/snapper.toml`, discovered by scanning. Runtime state is derived from the cluster (`squeue` + the `last_service.env` the serve job writes), never from a local state file.

**Tech Stack:** Python 3.11+ (stdlib `tomllib`), `typer` (CLI), `pytest` (tests). No `rich` required (optional later). Build backend `hatchling`. The reference spec is `docs/superpowers/specs/2026-06-27-snapper-cli-design.md`.

## Global Constraints

- **Python 3.11+** — relies on stdlib `tomllib`. Copy verbatim into `pyproject.toml`: `requires-python = ">=3.11"`.
- **Single cluster I/O boundary:** only `snapper/rcc.py` may invoke the `rcc` CLI / `subprocess`. No other module shells out. Tests mock at this boundary.
- **No local state file:** `status`/`down`/`logs` derive runtime state from `squeue` + `last_service.env` only.
- **Tests must not hit the network or a cluster.** Unit tests mock the `rcc` boundary and monkeypatch `time.sleep`.
- **Snapshot stays inert in v1:** `snapshot.plan()` is a no-op whenever `[snapshot].enabled` is false (which is every shipped manifest in v1). snapper never implements record/restore logic itself.
- **CLI override for node count:** the serve job is always submitted with `--nodes=N` on the `sbatch` command line (an env var does not work — Slurm honors the directive).
- **Run tests from `tools/snapper/`:** `pyproject.toml` sets `[tool.pytest.ini_options] pythonpath = ["."]` so the package imports without an install.

---

## File Structure

```
tools/snapper/
  pyproject.toml          # project metadata, deps, pytest pythonpath, console_script
  snapper/
    __init__.py           # version
    config.py             # repo_root, Deployment + SnapshotCfg dataclasses, load/discover
    rcc.py                # ONLY shell-out: RunResult, push(), run()
    slurm.py              # parse_jobid, parse_squeue_state, parse_rank_dump,
                          #   count_ready_ranks, submit(), probe(), wait_in_queue(), wait_for_ready()
    service.py            # parse_last_service_env, endpoints, read_service_env, verify
    snapshot.py           # plan(): the opt-in seam (inert in v1)
    cli.py                # Typer app: list/status/up/down/logs/verify
  tests/
    conftest.py           # shared fixtures (sample snapper.toml, fake rcc)
    test_config.py
    test_rcc.py
    test_slurm.py
    test_service.py
    test_snapshot.py
    test_cli.py
  README.md               # install + documented end-to-end manual cluster test
```

Plus repo edits (Task 9–10): a `snapper.toml` in each existing `deploy/<model>-<cluster>/`, a `JOBID=$SLURM_JOB_ID` line in each serve script, and a pointer from `docs/start_model.md`.

---

### Task 1: Package scaffold + `config.py`

**Files:**
- Create: `tools/snapper/pyproject.toml`
- Create: `tools/snapper/snapper/__init__.py`
- Create: `tools/snapper/snapper/config.py`
- Test: `tools/snapper/tests/test_config.py`
- Test: `tools/snapper/tests/conftest.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SnapshotCfg(enabled: bool, record_env: dict[str,str], restore_env: dict[str,str], serve: str | None)`
  - `Deployment(name, profile, engine, served_model_name, port, default_nodes, deploy_dir: Path, rel_dir: str, probe_script: str | None, serve_script: str, variants: dict[str,str], ready_markers: list[str], log_pattern: str, verify_models_path: str, verify_chat_path: str, snapshot: SnapshotCfg)`
  - `repo_root(start: Path | None = None) -> Path` — walks up to the dir containing `.rcc/config.toml`.
  - `load_deployment(toml_path: Path) -> Deployment`
  - `discover(deploy_root: Path) -> dict[str, Deployment]` — name → Deployment, from `<deploy_root>/*/snapper.toml`.
  - `load(name: str, *, root: Path | None = None) -> Deployment` — raises `KeyError` listing known names if absent.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "snapper"
version = "0.1.0"
description = "Lifecycle CLI for LLM deployments on CSCS clusters (drives docs/start_model.md)"
requires-python = ">=3.11"
dependencies = ["typer>=0.12"]

[project.scripts]
snapper = "snapper.cli:main"

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.hatch.build.targets.wheel]
packages = ["snapper"]

[tool.pytest.ini_options]
pythonpath = ["."]
```

- [ ] **Step 2: Write `snapper/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Write the shared fixtures in `tests/conftest.py`**

```python
from pathlib import Path
import textwrap
import pytest

SAMPLE_TOML = textwrap.dedent("""
    name    = "glm-47-flash-bristen"
    profile = "glm-47-flash-bristen"
    engine  = "sglang"
    served_model_name = "zai-org/GLM-4.7-Flash"
    port = 8080
    default_nodes = 5

    [scripts]
    probe = "probe_sglang.sbatch"
    serve = "serve_glm_47_flash_sglang.sbatch"

    [ready]
    markers = ["fired up and ready to roll", "vmagent_started"]
    log_pattern = "logs/opentela-{job}-{rank}.log"

    [verify]
    models_path = "/v1/models"
    chat_path   = "/v1/chat/completions"

    [variants.router]
    serve = "serve_glm_47_flash_sglang_router.sbatch"

    [snapshot]
    enabled = false
""")


@pytest.fixture
def deploy_tree(tmp_path: Path) -> Path:
    """A fake repo root with .rcc/ and one deploy dir holding a snapper.toml."""
    (tmp_path / ".rcc").mkdir()
    (tmp_path / ".rcc" / "config.toml").write_text("default = 'x'\n")
    d = tmp_path / "deploy" / "glm-47-flash-bristen"
    d.mkdir(parents=True)
    (d / "snapper.toml").write_text(SAMPLE_TOML)
    return tmp_path
```

- [ ] **Step 4: Write the failing test `tests/test_config.py`**

```python
from pathlib import Path
import pytest
from snapper import config


def test_load_deployment_parses_core_fields(deploy_tree: Path):
    dep = config.load_deployment(deploy_tree / "deploy" / "glm-47-flash-bristen" / "snapper.toml")
    assert dep.name == "glm-47-flash-bristen"
    assert dep.profile == "glm-47-flash-bristen"
    assert dep.engine == "sglang"
    assert dep.port == 8080
    assert dep.default_nodes == 5
    assert dep.serve_script == "serve_glm_47_flash_sglang.sbatch"
    assert dep.probe_script == "probe_sglang.sbatch"
    assert dep.rel_dir == "deploy/glm-47-flash-bristen"
    assert dep.ready_markers == ["fired up and ready to roll", "vmagent_started"]
    assert dep.variants == {"router": "serve_glm_47_flash_sglang_router.sbatch"}
    assert dep.snapshot.enabled is False


def test_discover_and_load_by_name(deploy_tree: Path):
    found = config.discover(deploy_tree / "deploy")
    assert set(found) == {"glm-47-flash-bristen"}
    dep = config.load("glm-47-flash-bristen", root=deploy_tree)
    assert dep.served_model_name == "zai-org/GLM-4.7-Flash"


def test_load_unknown_name_lists_known(deploy_tree: Path):
    with pytest.raises(KeyError) as exc:
        config.load("nope", root=deploy_tree)
    assert "glm-47-flash-bristen" in str(exc.value)
```

- [ ] **Step 5: Run the test, verify it fails**

Run: `cd tools/snapper && python -m pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'snapper.config'`).

- [ ] **Step 6: Write `snapper/config.py`**

```python
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SnapshotCfg:
    enabled: bool = False
    record_env: dict[str, str] = field(default_factory=dict)
    restore_env: dict[str, str] = field(default_factory=dict)
    serve: str | None = None


@dataclass
class Deployment:
    name: str
    profile: str
    engine: str
    served_model_name: str
    port: int
    default_nodes: int
    deploy_dir: Path
    rel_dir: str
    probe_script: str | None
    serve_script: str
    variants: dict[str, str]
    ready_markers: list[str]
    log_pattern: str
    verify_models_path: str
    verify_chat_path: str
    snapshot: SnapshotCfg


def repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for d in (cur, *cur.parents):
        if (d / ".rcc" / "config.toml").is_file():
            return d
    raise FileNotFoundError("not inside a serving-stack repo (no .rcc/config.toml found)")


def load_deployment(toml_path: Path) -> Deployment:
    toml_path = Path(toml_path)
    data = tomllib.loads(toml_path.read_text())
    deploy_dir = toml_path.parent
    scripts = data.get("scripts", {})
    ready = data.get("ready", {})
    verify = data.get("verify", {})
    snap = data.get("snapshot", {})
    variants = {k: v["serve"] for k, v in data.get("variants", {}).items()}
    if "serve" not in scripts:
        raise ValueError(f"{toml_path}: missing required [scripts].serve")
    return Deployment(
        name=data.get("name", deploy_dir.name),
        profile=data["profile"],
        engine=data["engine"],
        served_model_name=data["served_model_name"],
        port=int(data["port"]),
        default_nodes=int(data.get("default_nodes", 1)),
        deploy_dir=deploy_dir,
        rel_dir=f"deploy/{deploy_dir.name}",
        probe_script=scripts.get("probe"),
        serve_script=scripts["serve"],
        variants=variants,
        ready_markers=list(ready.get("markers", [])),
        log_pattern=ready.get("log_pattern", "logs/opentela-{job}-{rank}.log"),
        verify_models_path=verify.get("models_path", "/v1/models"),
        verify_chat_path=verify.get("chat_path", "/v1/chat/completions"),
        snapshot=SnapshotCfg(
            enabled=bool(snap.get("enabled", False)),
            record_env=dict(snap.get("record_env", {})),
            restore_env=dict(snap.get("restore_env", {})),
            serve=snap.get("serve"),
        ),
    )


def discover(deploy_root: Path) -> dict[str, Deployment]:
    out: dict[str, Deployment] = {}
    for toml_path in sorted(Path(deploy_root).glob("*/snapper.toml")):
        dep = load_deployment(toml_path)
        out[dep.name] = dep
    return out


def load(name: str, *, root: Path | None = None) -> Deployment:
    root = root or repo_root()
    found = discover(root / "deploy")
    if name not in found:
        known = ", ".join(sorted(found)) or "(none)"
        raise KeyError(f"unknown deployment {name!r}; known: {known}")
    return found[name]
```

- [ ] **Step 7: Run the tests, verify they pass**

Run: `cd tools/snapper && python -m pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add tools/snapper/pyproject.toml tools/snapper/snapper/__init__.py \
        tools/snapper/snapper/config.py tools/snapper/tests/conftest.py \
        tools/snapper/tests/test_config.py
git commit -m "feat(snapper): package scaffold + snapper.toml config loader"
```

---

### Task 2: `rcc.py` — the cluster I/O boundary

**Files:**
- Create: `tools/snapper/snapper/rcc.py`
- Test: `tools/snapper/tests/test_rcc.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `RunResult(returncode: int, stdout: str, stderr: str)`
  - `_run_argv(argv: list[str], *, stream: bool = False) -> RunResult` — the single `subprocess` call; **tests monkeypatch this**.
  - `push(profile: str) -> RunResult` — runs `rcc --profile <profile> push`.
  - `run(profile: str, command: str, *, stream: bool = False) -> RunResult` — runs `rcc --profile <profile> run bash -lc "<command>"`.

- [ ] **Step 1: Write the failing test `tests/test_rcc.py`**

```python
from snapper import rcc


def test_push_builds_expected_argv(monkeypatch):
    seen = {}
    def fake(argv, *, stream=False):
        seen["argv"] = argv
        return rcc.RunResult(0, "", "")
    monkeypatch.setattr(rcc, "_run_argv", fake)
    rcc.push("glm-47-flash-bristen")
    assert seen["argv"] == ["rcc", "--profile", "glm-47-flash-bristen", "push"]


def test_run_wraps_command_in_bash_lc(monkeypatch):
    seen = {}
    def fake(argv, *, stream=False):
        seen["argv"] = argv
        return rcc.RunResult(0, "ok", "")
    monkeypatch.setattr(rcc, "_run_argv", fake)
    res = rcc.run("p", "squeue -j 5 -h -o %T")
    assert seen["argv"] == ["rcc", "--profile", "p", "run", "bash", "-lc", "squeue -j 5 -h -o %T"]
    assert res.stdout == "ok"
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `cd tools/snapper && python -m pytest tests/test_rcc.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'snapper.rcc'`).

- [ ] **Step 3: Write `snapper/rcc.py`**

```python
from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str


def _run_argv(argv: list[str], *, stream: bool = False) -> RunResult:
    """The single shell-out boundary. Tests monkeypatch this function."""
    if stream:
        proc = subprocess.run(argv)
        return RunResult(proc.returncode, "", "")
    proc = subprocess.run(argv, capture_output=True, text=True)
    return RunResult(proc.returncode, proc.stdout, proc.stderr)


def push(profile: str) -> RunResult:
    return _run_argv(["rcc", "--profile", profile, "push"])


def run(profile: str, command: str, *, stream: bool = False) -> RunResult:
    return _run_argv(
        ["rcc", "--profile", profile, "run", "bash", "-lc", command],
        stream=stream,
    )
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `cd tools/snapper && python -m pytest tests/test_rcc.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/snapper/snapper/rcc.py tools/snapper/tests/test_rcc.py
git commit -m "feat(snapper): rcc transport boundary (push/run)"
```

---

### Task 3: `slurm.py` — parsers + submit/probe/wait

**Files:**
- Create: `tools/snapper/snapper/slurm.py`
- Test: `tools/snapper/tests/test_slurm.py`

**Interfaces:**
- Consumes: `config.Deployment`; `rcc.run` (module passed as `rcc_mod`, default `snapper.rcc`).
- Produces:
  - `parse_jobid(sbatch_stdout: str) -> int`
  - `parse_squeue_state(stdout: str) -> str | None`
  - `parse_rank_dump(text: str) -> dict[int, str]` — inverse of the per-rank cat dump (delimiter `===RANK <n>===`).
  - `count_ready_ranks(logs_by_rank: dict[int,str], markers: list[str]) -> int`
  - `submit(dep, *, nodes: int, serve_script: str, env: dict[str,str], rcc_mod=rcc) -> int`
  - `probe(dep, *, rcc_mod=rcc) -> int`
  - `wait_in_queue(profile: str, jobid: int, *, poll: float = 5.0, timeout: float = 1800, rcc_mod=rcc, sleep=time.sleep) -> str`
  - `wait_for_ready(dep, jobid: int, nodes: int, *, poll: float = 15.0, timeout: float = 1800, rcc_mod=rcc, sleep=time.sleep) -> bool`

- [ ] **Step 1: Write the failing test `tests/test_slurm.py`**

```python
from snapper import slurm, config


def _dep(deploy_tree):
    return config.load("glm-47-flash-bristen", root=deploy_tree)


def test_parse_jobid():
    assert slurm.parse_jobid("Submitted batch job 123456\n") == 123456


def test_parse_squeue_state_running_then_absent():
    assert slurm.parse_squeue_state("RUNNING\n") == "RUNNING"
    assert slurm.parse_squeue_state("\n") is None


def test_parse_rank_dump_and_count():
    dump = (
        "===RANK 0===\nstuff fired up and ready to roll\nvmagent_started\n"
        "===RANK 1===\nstill loading\n"
    )
    logs = slurm.parse_rank_dump(dump)
    assert set(logs) == {0, 1}
    markers = ["fired up and ready to roll", "vmagent_started"]
    assert slurm.count_ready_ranks(logs, markers) == 1


def test_submit_builds_command_with_env_and_nodes(deploy_tree, monkeypatch):
    dep = _dep(deploy_tree)
    seen = {}
    class FakeRcc:
        @staticmethod
        def run(profile, command, **kw):
            seen["profile"] = profile
            seen["command"] = command
            from snapper.rcc import RunResult
            return RunResult(0, "Submitted batch job 999\n", "")
    jobid = slurm.submit(dep, nodes=5, serve_script=dep.serve_script,
                         env={"TP_SIZE": "2"}, rcc_mod=FakeRcc)
    assert jobid == 999
    assert seen["profile"] == "glm-47-flash-bristen"
    assert "TP_SIZE=2 sbatch --nodes=5 deploy/glm-47-flash-bristen/serve_glm_47_flash_sglang.sbatch" in seen["command"]


def test_wait_for_ready_returns_true_when_all_ranks_ready(deploy_tree, monkeypatch):
    dep = _dep(deploy_tree)
    class FakeRcc:
        @staticmethod
        def run(profile, command, **kw):
            from snapper.rcc import RunResult
            return RunResult(0, "===RANK 0===\nfired up and ready to roll\nvmagent_started\n", "")
    ok = slurm.wait_for_ready(dep, 999, 1, poll=0, timeout=1, rcc_mod=FakeRcc, sleep=lambda s: None)
    assert ok is True
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `cd tools/snapper && python -m pytest tests/test_slurm.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'snapper.slurm'`).

- [ ] **Step 3: Write `snapper/slurm.py`**

```python
from __future__ import annotations

import re
import time

from . import rcc

_JOBID_RE = re.compile(r"Submitted batch job (\d+)")
_RANK_RE = re.compile(r"^===RANK (\d+)===$")


def parse_jobid(sbatch_stdout: str) -> int:
    m = _JOBID_RE.search(sbatch_stdout)
    if not m:
        raise ValueError(f"could not parse jobid from sbatch output: {sbatch_stdout!r}")
    return int(m.group(1))


def parse_squeue_state(stdout: str) -> str | None:
    state = stdout.strip()
    return state or None


def parse_rank_dump(text: str) -> dict[int, str]:
    logs: dict[int, str] = {}
    cur: int | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = _RANK_RE.match(line)
        if m:
            if cur is not None:
                logs[cur] = "\n".join(buf)
            cur = int(m.group(1))
            buf = []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        logs[cur] = "\n".join(buf)
    return logs


def count_ready_ranks(logs_by_rank: dict[int, str], markers: list[str]) -> int:
    return sum(
        1 for text in logs_by_rank.values()
        if all(marker in text for marker in markers)
    )


def _env_prefix(env: dict[str, str]) -> str:
    return "".join(f"{k}={v} " for k, v in env.items())


def submit(dep, *, nodes: int, serve_script: str, env: dict[str, str], rcc_mod=rcc) -> int:
    cmd = f"{_env_prefix(env)}sbatch --nodes={nodes} {dep.rel_dir}/{serve_script}"
    res = rcc_mod.run(dep.profile, cmd)
    if res.returncode != 0:
        raise RuntimeError(f"sbatch failed: {res.stderr or res.stdout}")
    return parse_jobid(res.stdout)


def probe(dep, *, rcc_mod=rcc) -> int:
    if not dep.probe_script:
        raise ValueError(f"{dep.name}: no probe script declared")
    res = rcc_mod.run(dep.profile, f"sbatch {dep.rel_dir}/{dep.probe_script}")
    if res.returncode != 0:
        raise RuntimeError(f"probe sbatch failed: {res.stderr or res.stdout}")
    return parse_jobid(res.stdout)


def wait_in_queue(profile: str, jobid: int, *, poll: float = 5.0,
                  timeout: float = 1800, rcc_mod=rcc, sleep=time.sleep) -> str:
    """Block until the job leaves the queue; return its last observed state."""
    waited = 0.0
    last = "PENDING"
    while True:
        res = rcc_mod.run(profile, f"squeue -j {jobid} -h -o %T")
        state = parse_squeue_state(res.stdout)
        if state is None:
            return last
        last = state
        if waited >= timeout:
            return last
        sleep(poll)
        waited += poll


def _fetch_rank_logs(dep, jobid: int, nodes: int, rcc_mod=rcc) -> dict[int, str]:
    ranks = " ".join(str(r) for r in range(nodes))
    pat = dep.log_pattern.format(job=jobid, rank="$r")
    cmd = f'for r in {ranks}; do echo "===RANK $r==="; cat {pat} 2>/dev/null; done'
    res = rcc_mod.run(dep.profile, cmd)
    return parse_rank_dump(res.stdout)


def wait_for_ready(dep, jobid: int, nodes: int, *, poll: float = 15.0,
                   timeout: float = 1800, rcc_mod=rcc, sleep=time.sleep) -> bool:
    waited = 0.0
    while True:
        logs = _fetch_rank_logs(dep, jobid, nodes, rcc_mod=rcc_mod)
        if count_ready_ranks(logs, dep.ready_markers) >= nodes:
            return True
        if waited >= timeout:
            return False
        sleep(poll)
        waited += poll
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `cd tools/snapper && python -m pytest tests/test_slurm.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/snapper/snapper/slurm.py tools/snapper/tests/test_slurm.py
git commit -m "feat(snapper): slurm submit/probe/wait + log parsers"
```

---

### Task 4: `service.py` — `last_service.env` + verify

**Files:**
- Create: `tools/snapper/snapper/service.py`
- Test: `tools/snapper/tests/test_service.py`

**Interfaces:**
- Consumes: `config.Deployment`; `rcc.run` (as `rcc_mod`).
- Produces:
  - `parse_last_service_env(text: str) -> dict[str, str]`
  - `endpoints(env: dict[str,str]) -> list[str]` — every `ENDPOINT_*` value, ordered by index.
  - `read_service_env(profile: str, *, rcc_mod=rcc) -> dict[str,str]` — `cat last_service.env` in the remote dir.
  - `verify(dep, endpoint: str, *, rcc_mod=rcc) -> RunResult` — runs `curl` **on the cluster** (endpoints are cluster-internal) against `/v1/models` then a chat request.

- [ ] **Step 1: Write the failing test `tests/test_service.py`**

```python
from snapper import service, config
from snapper.rcc import RunResult


def test_parse_last_service_env():
    text = "ENDPOINT_0=nid001:8080\nENDPOINT_1=nid002:8080\nJOBID=555\n"
    env = service.parse_last_service_env(text)
    assert env["JOBID"] == "555"
    assert service.endpoints(env) == ["nid001:8080", "nid002:8080"]


def test_read_service_env_uses_cat(monkeypatch):
    seen = {}
    class FakeRcc:
        @staticmethod
        def run(profile, command, **kw):
            seen["command"] = command
            return RunResult(0, "ENDPOINT_0=nid001:8080\nJOBID=7\n", "")
    env = service.read_service_env("p", rcc_mod=FakeRcc)
    assert "cat last_service.env" in seen["command"]
    assert env["JOBID"] == "7"


def test_verify_curls_models_and_chat(deploy_tree):
    dep = config.load("glm-47-flash-bristen", root=deploy_tree)
    cmds = []
    class FakeRcc:
        @staticmethod
        def run(profile, command, **kw):
            cmds.append(command)
            return RunResult(0, '{"data":[]}', "")
    service.verify(dep, "nid001:8080", rcc_mod=FakeRcc)
    joined = "\n".join(cmds)
    assert "nid001:8080/v1/models" in joined
    assert "nid001:8080/v1/chat/completions" in joined
    assert "zai-org/GLM-4.7-Flash" in joined  # served_model_name in the chat body
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `cd tools/snapper && python -m pytest tests/test_service.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'snapper.service'`).

- [ ] **Step 3: Write `snapper/service.py`**

```python
from __future__ import annotations

import json

from . import rcc


def parse_last_service_env(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip()
    return env


def endpoints(env: dict[str, str]) -> list[str]:
    items = [(k, v) for k, v in env.items() if k.startswith("ENDPOINT_")]
    items.sort(key=lambda kv: int(kv[0].split("_", 1)[1]))
    return [v for _, v in items]


def read_service_env(profile: str, *, rcc_mod=rcc) -> dict[str, str]:
    res = rcc_mod.run(profile, "cat last_service.env 2>/dev/null || true")
    return parse_last_service_env(res.stdout)


def verify(dep, endpoint: str, *, rcc_mod=rcc):
    base = f"http://{endpoint}"
    body = json.dumps({
        "model": dep.served_model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    })
    cmd = (
        f"set -e; "
        f"curl -sS {base}{dep.verify_models_path}; echo; "
        f"curl -sS {base}{dep.verify_chat_path} "
        f"-H 'Content-Type: application/json' -d '{body}'"
    )
    return rcc_mod.run(dep.profile, cmd)
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `cd tools/snapper && python -m pytest tests/test_service.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/snapper/snapper/service.py tools/snapper/tests/test_service.py
git commit -m "feat(snapper): last_service.env parsing + verify (curl on cluster)"
```

---

### Task 5: `snapshot.py` — the opt-in seam (inert in v1)

**Files:**
- Create: `tools/snapper/snapper/snapshot.py`
- Test: `tools/snapper/tests/test_snapshot.py`

**Interfaces:**
- Consumes: `config.Deployment` / `config.SnapshotCfg`.
- Produces:
  - `plan(dep, mode: str | None) -> tuple[dict[str,str], str | None]` — returns `(extra_env, serve_script_override)`. No-op `({}, None)` when `mode` is falsy or `dep.snapshot.enabled` is false. Raises `ValueError` for an unknown mode.

- [ ] **Step 1: Write the failing test `tests/test_snapshot.py`**

```python
import pytest
from snapper import snapshot
from snapper.config import Deployment, SnapshotCfg
from pathlib import Path


def _dep(snap: SnapshotCfg) -> Deployment:
    return Deployment(
        name="d", profile="p", engine="sglang", served_model_name="m",
        port=8080, default_nodes=1, deploy_dir=Path("."), rel_dir="deploy/d",
        probe_script=None, serve_script="serve.sbatch", variants={},
        ready_markers=[], log_pattern="logs/x-{job}-{rank}.log",
        verify_models_path="/v1/models", verify_chat_path="/v1/chat/completions",
        snapshot=snap,
    )


def test_plan_noop_when_disabled():
    dep = _dep(SnapshotCfg(enabled=False))
    assert snapshot.plan(dep, "record") == ({}, None)


def test_plan_noop_when_mode_none():
    dep = _dep(SnapshotCfg(enabled=True, record_env={"SNAPSHOT_MODE": "record"}))
    assert snapshot.plan(dep, None) == ({}, None)


def test_plan_record_when_enabled():
    dep = _dep(SnapshotCfg(enabled=True, record_env={"SNAPSHOT_MODE": "record"},
                           serve="serve_snap.sbatch"))
    assert snapshot.plan(dep, "record") == ({"SNAPSHOT_MODE": "record"}, "serve_snap.sbatch")


def test_plan_unknown_mode_raises():
    dep = _dep(SnapshotCfg(enabled=True))
    with pytest.raises(ValueError):
        snapshot.plan(dep, "bogus")
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `cd tools/snapper && python -m pytest tests/test_snapshot.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'snapper.snapshot'`).

- [ ] **Step 3: Write `snapper/snapshot.py`**

```python
from __future__ import annotations

_MODES = {"record", "restore"}


def plan(dep, mode: str | None) -> tuple[dict[str, str], str | None]:
    """Translate --snapshot <mode> into (extra_env, serve_script_override).

    Inert in v1: returns ({}, None) whenever the deployment has snapshot
    disabled or no mode was requested. snapper never implements record/restore
    itself — it only injects env and optionally selects a snapshot serve script.
    """
    if not mode:
        return {}, None
    if mode not in _MODES:
        raise ValueError(f"unknown snapshot mode {mode!r}; expected one of {sorted(_MODES)}")
    if not dep.snapshot.enabled:
        return {}, None
    env = dep.snapshot.record_env if mode == "record" else dep.snapshot.restore_env
    return dict(env), dep.snapshot.serve
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `cd tools/snapper && python -m pytest tests/test_snapshot.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/snapper/snapper/snapshot.py tools/snapper/tests/test_snapshot.py
git commit -m "feat(snapper): inert snapshot seam (plan)"
```

---

### Task 6: `cli.py` scaffold + `list` + `status`

**Files:**
- Create: `tools/snapper/snapper/cli.py`
- Test: `tools/snapper/tests/test_cli.py`

**Interfaces:**
- Consumes: `config.discover`/`config.load`/`config.repo_root`; `rcc.run`; `service`.
- Produces:
  - `app` — the `typer.Typer()` instance.
  - `main()` — console-script entry (`app()`).
  - `list` command — prints discovered deployment names + profile + engine.
  - `status` command — for one or all deployments, prints `last_service.env` + a filtered `squeue`.

- [ ] **Step 1: Write the failing test `tests/test_cli.py`**

```python
from typer.testing import CliRunner
from snapper import cli, config, rcc

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
        return rcc.RunResult(0, "  7  normal job RUN\n", "")  # squeue
    monkeypatch.setattr(rcc, "run", fake_run)
    result = runner.invoke(cli.app, ["status", "glm-47-flash-bristen"])
    assert result.exit_code == 0
    assert "nid001:8080" in result.stdout
```

- [ ] **Step 2: Run the test, verify it fails**

Run: `cd tools/snapper && python -m pytest tests/test_cli.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'snapper.cli'`).

- [ ] **Step 3: Write `snapper/cli.py`**

```python
from __future__ import annotations

import typer

from . import config, rcc, service

app = typer.Typer(help="Lifecycle CLI for LLM deployments on CSCS clusters.")


@app.command("list")
def list_cmd():
    """List discovered deployments (deploy/*/snapper.toml)."""
    found = config.discover(config.repo_root() / "deploy")
    if not found:
        typer.echo("no deployments found (no deploy/*/snapper.toml)")
        return
    for name, dep in sorted(found.items()):
        typer.echo(f"{name}\tprofile={dep.profile}\tengine={dep.engine}\tnodes={dep.default_nodes}")


def _status_one(dep) -> None:
    typer.echo(f"== {dep.name} ({dep.profile}) ==")
    env = service.read_service_env(dep.profile)
    if env:
        for k, v in env.items():
            typer.echo(f"{k}={v}")
    else:
        typer.echo("(no last_service.env — not started?)")
    sq = rcc.run(dep.profile, 'squeue -u "$USER" -o "%.18i %.9P %.30j %.2t %.10M %.6D %R"')
    typer.echo(sq.stdout.rstrip())


@app.command("status")
def status_cmd(name: str = typer.Argument(None, help="deployment name; omit for all")):
    """Show squeue + last_service.env for one or all deployments."""
    root = config.repo_root()
    if name:
        _status_one(config.load(name, root=root))
        return
    for _, dep in sorted(config.discover(root / "deploy").items()):
        _status_one(dep)
        typer.echo("")


def main():
    app()
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `cd tools/snapper && python -m pytest tests/test_cli.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/snapper/snapper/cli.py tools/snapper/tests/test_cli.py
git commit -m "feat(snapper): cli scaffold + list + status"
```

---

### Task 7: `up` orchestration

**Files:**
- Modify: `tools/snapper/snapper/cli.py`
- Test: `tools/snapper/tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `rcc.push`; `slurm.probe`/`slurm.wait_in_queue`/`slurm.submit`/`slurm.wait_for_ready`; `snapshot.plan`; `service.read_service_env`/`service.endpoints`.
- Produces:
  - `up` command — resolves serve script (`--variant` > snapshot override > default), runs push → probe(+wait+check) → serve → wait-for-ready → print endpoints.
  - `_resolve_serve_script(dep, variant, snap_override) -> str` — selection helper (variant wins, then snapshot override, then default).
  - `_parse_set(pairs: list[str]) -> dict[str,str]` — `["A=1","B=2"] -> {"A":"1","B":"2"}`.

- [ ] **Step 1: Write the failing tests (append to `tests/test_cli.py`)**

```python
from snapper import slurm, snapshot, service as service_mod


def test_parse_set_pairs():
    from snapper.cli import _parse_set
    assert _parse_set(["TP_SIZE=2", "X=y"]) == {"TP_SIZE": "2", "X": "y"}


def test_resolve_serve_script_variant_wins(deploy_tree):
    from snapper.cli import _resolve_serve_script
    dep = config.load("glm-47-flash-bristen", root=deploy_tree)
    assert _resolve_serve_script(dep, "router", None) == "serve_glm_47_flash_sglang_router.sbatch"
    assert _resolve_serve_script(dep, None, None) == dep.serve_script
    assert _resolve_serve_script(dep, None, "serve_snap.sbatch") == "serve_snap.sbatch"


def test_up_runs_full_sequence(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    calls = []
    monkeypatch.setattr(rcc, "push", lambda profile: calls.append(("push", profile)) or rcc.RunResult(0, "", ""))
    monkeypatch.setattr(slurm, "probe", lambda dep, **k: calls.append(("probe", dep.name)) or 100)
    monkeypatch.setattr(slurm, "wait_in_queue", lambda *a, **k: "COMPLETED")
    monkeypatch.setattr(rcc, "run", lambda profile, command, **k: rcc.RunResult(0, "PROBE OK", ""))
    monkeypatch.setattr(slurm, "submit", lambda dep, **k: calls.append(("submit", k.get("nodes"))) or 200)
    monkeypatch.setattr(slurm, "wait_for_ready", lambda *a, **k: True)
    monkeypatch.setattr(service_mod, "read_service_env", lambda profile, **k: {"ENDPOINT_0": "nid001:8080", "JOBID": "200"})

    result = runner.invoke(cli.app, ["up", "glm-47-flash-bristen", "--nodes", "3"])
    assert result.exit_code == 0, result.stdout
    names = [c[0] for c in calls]
    assert names == ["push", "probe", "submit"]
    assert ("submit", 3) in calls
    assert "nid001:8080" in result.stdout


def test_up_aborts_when_probe_fails(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    monkeypatch.setattr(rcc, "push", lambda profile: rcc.RunResult(0, "", ""))
    monkeypatch.setattr(slurm, "probe", lambda dep, **k: 100)
    monkeypatch.setattr(slurm, "wait_in_queue", lambda *a, **k: "FAILED")
    monkeypatch.setattr(rcc, "run", lambda profile, command, **k: rcc.RunResult(1, "boom traceback", ""))
    submitted = []
    monkeypatch.setattr(slurm, "submit", lambda dep, **k: submitted.append(1) or 200)

    result = runner.invoke(cli.app, ["up", "glm-47-flash-bristen", "--nodes", "1"])
    assert result.exit_code != 0
    assert submitted == []  # serve never submitted
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `cd tools/snapper && python -m pytest tests/test_cli.py -k "up or resolve or parse_set" -v`
Expected: FAIL (`AttributeError`/`ImportError` for `_parse_set` / `up` not defined).

- [ ] **Step 3: Add the `up` command + helpers to `snapper/cli.py`**

Add the imports `from . import slurm, snapshot` at the top (alongside the existing imports), then append:

```python
def _parse_set(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs or []:
        if "=" not in p:
            raise typer.BadParameter(f"--set expects KEY=VAL, got {p!r}")
        k, _, v = p.partition("=")
        out[k.strip()] = v.strip()
    return out


def _resolve_serve_script(dep, variant: str | None, snap_override: str | None) -> str:
    if variant:
        if variant not in dep.variants:
            raise typer.BadParameter(
                f"unknown variant {variant!r}; known: {', '.join(dep.variants) or '(none)'}")
        return dep.variants[variant]
    if snap_override:
        return snap_override
    return dep.serve_script


@app.command("up")
def up_cmd(
    name: str = typer.Argument(..., help="deployment name"),
    nodes: int = typer.Option(None, "--nodes", help="node count (default: manifest default_nodes)"),
    variant: str = typer.Option(None, "--variant", help="serve-script variant"),
    snapshot_mode: str = typer.Option(None, "--snapshot", help="record|restore (opt-in seam)"),
    sets: list[str] = typer.Option(None, "--set", help="extra env KEY=VAL (repeatable)"),
    no_push: bool = typer.Option(False, "--no-push"),
    no_probe: bool = typer.Option(False, "--no-probe"),
    no_wait: bool = typer.Option(False, "--no-wait"),
    timeout: float = typer.Option(1800, "--timeout", help="ready-wait timeout seconds"),
):
    """Bring a deployment up: push -> probe -> serve -> wait-for-ready -> endpoint."""
    dep = config.load(name, root=config.repo_root())
    n = nodes or dep.default_nodes
    snap_env, snap_serve = snapshot.plan(dep, snapshot_mode)
    serve_script = _resolve_serve_script(dep, variant, snap_serve)
    env = {**_parse_set(sets), **snap_env}

    if not no_push:
        typer.echo(f"[push] {dep.profile}")
        if rcc.push(dep.profile).returncode != 0:
            raise typer.Exit(1)

    if not no_probe and dep.probe_script:
        typer.echo("[probe] submitting")
        pid = slurm.probe(dep)
        state = slurm.wait_in_queue(dep.profile, pid)
        log = rcc.run(dep.profile, f"tail -n 40 slurm-{pid}.out 2>/dev/null || true")
        if state not in ("COMPLETED", "") or log.returncode != 0:
            typer.echo(f"[probe] FAILED (state={state}). Last log lines:")
            typer.echo(log.stdout)
            typer.echo("See .agents/skills/deploy/references/gotchas.md")
            raise typer.Exit(1)
        typer.echo("[probe] ok")

    typer.echo(f"[serve] submitting {serve_script} on {n} node(s)")
    jobid = slurm.submit(dep, nodes=n, serve_script=serve_script, env=env)
    typer.echo(f"[serve] job {jobid}")

    if no_wait:
        typer.echo("[serve] submitted; not waiting (--no-wait)")
        return

    typer.echo("[serve] waiting for all ranks ready (cold start can take 10-20 min)...")
    if not slurm.wait_for_ready(dep, jobid, n, timeout=timeout):
        typer.echo("[serve] TIMEOUT before ready; job left running. Use `snapper logs`/`snapper down`.")
        raise typer.Exit(1)

    env_out = service.read_service_env(dep.profile)
    eps = service.endpoints(env_out)
    typer.echo("[ready] endpoints:")
    for ep in eps:
        typer.echo(f"  {ep}")
    if eps:
        typer.echo(f"\nverify: snapper verify {dep.name}")
```

- [ ] **Step 4: Run the tests, verify they pass**

Run: `cd tools/snapper && python -m pytest tests/test_cli.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tools/snapper/snapper/cli.py tools/snapper/tests/test_cli.py
git commit -m "feat(snapper): up orchestration (push/probe/serve/wait)"
```

---

### Task 8: `down` + `logs` + `verify`

**Files:**
- Modify: `tools/snapper/snapper/cli.py`
- Test: `tools/snapper/tests/test_cli.py` (append)

**Interfaces:**
- Consumes: `service.read_service_env`/`service.endpoints`/`service.verify`; `rcc.run`.
- Produces:
  - `down` command — `scancel` the running job (from `last_service.env` JOBID; friendly no-op if none).
  - `logs` command — `tail [-f] -n N` the per-rank log resolved from `log_pattern` + JOBID.
  - `verify` command — runs `service.verify` against the first endpoint.

- [ ] **Step 1: Write the failing tests (append to `tests/test_cli.py`)**

```python
def test_down_scancels_jobid(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    monkeypatch.setattr(service_mod, "read_service_env", lambda profile, **k: {"JOBID": "321"})
    seen = {}
    monkeypatch.setattr(rcc, "run", lambda profile, command, **k: seen.update(cmd=command) or rcc.RunResult(0, "", ""))
    result = runner.invoke(cli.app, ["down", "glm-47-flash-bristen"])
    assert result.exit_code == 0
    assert "scancel 321" in seen["cmd"]


def test_down_no_job_is_friendly(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    monkeypatch.setattr(service_mod, "read_service_env", lambda profile, **k: {})
    result = runner.invoke(cli.app, ["down", "glm-47-flash-bristen"])
    assert result.exit_code == 0
    assert "nothing" in result.stdout.lower()


def test_logs_tails_resolved_rank_path(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    monkeypatch.setattr(service_mod, "read_service_env", lambda profile, **k: {"JOBID": "77"})
    seen = {}
    monkeypatch.setattr(rcc, "run", lambda profile, command, **k: seen.update(cmd=command) or rcc.RunResult(0, "loglines", ""))
    result = runner.invoke(cli.app, ["logs", "glm-47-flash-bristen", "--rank", "2"])
    assert result.exit_code == 0
    assert "logs/opentela-77-2.log" in seen["cmd"]


def test_verify_hits_first_endpoint(deploy_tree, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda *a, **k: deploy_tree)
    monkeypatch.setattr(service_mod, "read_service_env", lambda profile, **k: {"ENDPOINT_0": "nid001:8080"})
    seen = {}
    monkeypatch.setattr(service_mod, "verify", lambda dep, ep, **k: seen.update(ep=ep) or rcc.RunResult(0, "{}", ""))
    result = runner.invoke(cli.app, ["verify", "glm-47-flash-bristen"])
    assert result.exit_code == 0
    assert seen["ep"] == "nid001:8080"
```

- [ ] **Step 2: Run the tests, verify they fail**

Run: `cd tools/snapper && python -m pytest tests/test_cli.py -k "down or logs or verify" -v`
Expected: FAIL (commands not defined).

- [ ] **Step 3: Add `down`, `logs`, `verify` to `snapper/cli.py`**

```python
@app.command("down")
def down_cmd(name: str = typer.Argument(..., help="deployment name")):
    """Cancel the running serve job for a deployment."""
    dep = config.load(name, root=config.repo_root())
    env = service.read_service_env(dep.profile)
    jobid = env.get("JOBID")
    if not jobid:
        typer.echo(f"{dep.name}: nothing to cancel (no JOBID in last_service.env)")
        return
    rcc.run(dep.profile, f"scancel {jobid}")
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
    jobid = env.get("JOBID")
    if not jobid:
        typer.echo(f"{dep.name}: no JOBID in last_service.env — not started?")
        raise typer.Exit(1)
    path = dep.log_pattern.format(job=jobid, rank=rank)
    flag = "-f " if follow else ""
    res = rcc.run(dep.profile, f"tail {flag}-n {lines} {path}", stream=follow)
    if not follow:
        typer.echo(res.stdout.rstrip())


@app.command("verify")
def verify_cmd(name: str = typer.Argument(..., help="deployment name")):
    """Smoke-test the service: /v1/models + one chat request."""
    dep = config.load(name, root=config.repo_root())
    env = service.read_service_env(dep.profile)
    eps = service.endpoints(env)
    if not eps:
        typer.echo(f"{dep.name}: no endpoints in last_service.env — not ready?")
        raise typer.Exit(1)
    res = service.verify(dep, eps[0])
    typer.echo(res.stdout.rstrip())
    if res.returncode != 0:
        raise typer.Exit(1)
```

- [ ] **Step 4: Run the full suite, verify it passes**

Run: `cd tools/snapper && python -m pytest -v`
Expected: all tests passed (config, rcc, slurm, service, snapshot, cli).

- [ ] **Step 5: Commit**

```bash
git add tools/snapper/snapper/cli.py tools/snapper/tests/test_cli.py
git commit -m "feat(snapper): down + logs + verify commands"
```

---

### Task 9: Per-deploy `snapper.toml` manifests + `JOBID=` serve-script edit

**Files:**
- Create: `deploy/glm-47-flash-bristen/snapper.toml` (worked example, full)
- Create: `snapper.toml` in each other catalog deploy dir (procedure below)
- Modify: each `deploy/<model>-<cluster>/serve_*.sbatch` that writes `last_service.env`

**Interfaces:**
- Consumes: `config.discover` (Task 1) must find every new manifest.
- Produces: discoverable deployments + `JOBID` in `last_service.env` at runtime.

- [ ] **Step 1: Write the worked manifest `deploy/glm-47-flash-bristen/snapper.toml`**

```toml
name    = "glm-47-flash-bristen"
profile = "glm-47-flash-bristen"
engine  = "sglang"
served_model_name = "zai-org/GLM-4.7-Flash"
port = 8080
default_nodes = 5

[scripts]
probe = "probe_sglang.sbatch"
serve = "serve_glm_47_flash_sglang.sbatch"

[ready]
markers = ["fired up and ready to roll", "vmagent_started"]
log_pattern = "logs/opentela-{job}-{rank}.log"

[verify]
models_path = "/v1/models"
chat_path   = "/v1/chat/completions"

[variants.router]
serve = "serve_glm_47_flash_sglang_router.sbatch"

[snapshot]
enabled = false
```

- [ ] **Step 2: Create manifests for the rest of the catalog**

For each row below, run `ls deploy/<dir>` to confirm the exact probe/serve script filenames, then write `deploy/<dir>/snapper.toml` mirroring Step 1 with these values (from `docs/start_model.md` §Deployment catalog and the per-model runbooks). Use `engine`, `port`, and `markers` per the engine:

- vLLM markers: `["APIV1-server", "vmagent_started"]`
- SGLang markers: `["fired up and ready to roll", "vmagent_started"]`

| dir | profile | engine | served_model_name | port | default_nodes |
|-----|---------|--------|-------------------|------|---------------|
| `glm-47-flash-beverin` | `glm-47-flash` | vllm | `zai-org/GLM-4.7-Flash-rocm` | 8000 | 5 |
| `kimi-k25-beverin` | `beverin` | vllm | `moonshotai/Kimi-K2.7-Code` | 8000 | 2 |
| `glm-52-fp8-beverin` | `glm-52-fp8` | vllm | `zai-org/GLM-5.2-FP8-rocm` | 8000 | 4 |
| `glm-47-flash-bristen-xzyao` | `glm-47-flash-bristen-xzyao` | sglang | `zai-org/GLM-4.7-Flash-xzyao` | 8090 | 5 |

For `glm-47-flash-bristen-xzyao` (the router deploy) the public endpoint is the **router** (port 8090), the serve script is the router script, and the dp2 alternate is a variant. After `ls` confirms the filenames, its manifest should look like:

```toml
name    = "glm-47-flash-bristen-xzyao"
profile = "glm-47-flash-bristen-xzyao"
engine  = "sglang"
served_model_name = "zai-org/GLM-4.7-Flash-xzyao"
port = 8090
default_nodes = 5

[scripts]
probe = "probe_router.sbatch"
serve = "serve_glm_47_flash_sglang_router.sbatch"

[ready]
markers = ["router_ready", "vmagent_started"]
log_pattern = "logs/opentela-{job}-{rank}.log"

[verify]
models_path = "/v1/models"
chat_path   = "/v1/chat/completions"

[variants.dp2]
serve = "serve_glm_47_flash_sglang_router_dp2.sbatch"

[snapshot]
enabled = false
```

- [ ] **Step 3: Add `JOBID` to each serve script's `last_service.env` block**

In each serve script, find the block that writes `last_service.env` (it already
emits `ENDPOINT_*`). For the bristen SGLang script it is around
`serve_glm_47_flash_sglang.sbatch:241` — a `{ ... } > "${DEPLOY_DIR}/last_service.env"`
group. Add a `JOBID` line **inside** the braces, e.g.:

```bash
{
  echo "JOBID=${SLURM_JOB_ID}"
  # ... existing ENDPOINT_${i}=... lines ...
} > "${DEPLOY_DIR}/last_service.env"
```

Apply the same one-line addition to every catalog serve script (including the
router and dp2 variants).

- [ ] **Step 4: Verify discovery + JOBID edits**

Run:
```bash
cd tools/snapper && python -c "from snapper import config; print(sorted(config.discover(config.repo_root()/'deploy')))"
grep -rl 'JOBID=${SLURM_JOB_ID}' deploy/*/serve_*.sbatch
```
Expected: the discovery list includes all five catalog deployments; the grep lists every catalog serve script.

- [ ] **Step 5: Commit**

```bash
git add deploy/*/snapper.toml deploy/*/serve_*.sbatch
git commit -m "feat(snapper): per-deploy snapper.toml manifests + JOBID in last_service.env"
```

---

### Task 10: `tools/snapper/README.md` + runbook pointer

**Files:**
- Create: `tools/snapper/README.md`
- Modify: `docs/start_model.md` (add a short pointer)

**Interfaces:**
- Consumes: the finished CLI.
- Produces: install instructions + the documented end-to-end manual cluster test.

- [ ] **Step 1: Write `tools/snapper/README.md`**

````markdown
# snapper

A small CLI that drives the `docs/start_model.md` runbook: full deployment
lifecycle over `rcc`. Deployments are described by `deploy/*/snapper.toml`.

## Install

```bash
cd tools/snapper
uv pip install -e .        # or: pipx install -e .
```

## Commands

```bash
snapper list                       # discovered deployments
snapper up   <dep> [--nodes N] [--variant V] [--no-probe] [--no-wait] [--set K=V]
snapper down <dep>
snapper status [<dep>]
snapper logs <dep> [--rank R] [-f]
snapper verify <dep>
```

`<dep>` is the `name` in a `deploy/<model>-<cluster>/snapper.toml`.

## Tests

```bash
cd tools/snapper && python -m pytest -v   # no cluster needed; mocks the rcc boundary
```

## End-to-end manual test (on the cluster)

Per the repo rule "validate on compute nodes", confirm against a real
deployment once:

```bash
snapper up glm-47-flash-bristen --nodes 1     # push -> probe -> serve -> wait
snapper status glm-47-flash-bristen
snapper verify glm-47-flash-bristen
snapper down  glm-47-flash-bristen
```

Expect: probe passes, serve reaches "ready", `verify` returns a models list
and a chat completion, `down` cancels the job.

## Snapshot seam

`--snapshot record|restore` is wired but inert while every manifest has
`[snapshot].enabled = false`. It will activate once the `snapshot/` subsystem
lands a real record/restore serve path (see the snapshot work / "N5b").
````

- [ ] **Step 2: Add a pointer to `docs/start_model.md`**

Add this just under the top intro paragraph (after the line ending
"`.agents/skills/deploy/`."):

```markdown
> **Automating this runbook:** `tools/snapper/` provides a `snapper` CLI that
> runs these steps as commands (`snapper up/down/status/logs/verify`). It reads
> a per-deploy `deploy/<model>-<cluster>/snapper.toml`. See
> `tools/snapper/README.md`.
```

- [ ] **Step 3: Verify the docs render and links are right**

Run: `grep -n "tools/snapper" docs/start_model.md tools/snapper/README.md`
Expected: the pointer appears in `docs/start_model.md`; the README references the commands.

- [ ] **Step 4: Commit**

```bash
git add tools/snapper/README.md docs/start_model.md
git commit -m "docs(snapper): README + runbook pointer"
```

---

## Self-Review

**Spec coverage** (against `docs/superpowers/specs/2026-06-27-snapper-cli-design.md`):

- §3 Architecture / module layout → Tasks 1–8 create every module at `tools/snapper/`. ✓
- §4 `snapper.toml` manifest + `Deployment` dataclass → Task 1 (loader) + Task 9 (real manifests). ✓
- §5 Command surface (`list`/`up`/`down`/`status`/`logs`/`verify`) → Tasks 6–8. ✓
- §6 `up` orchestration (push→probe→serve→wait→endpoint, `--nodes` on CLI, probe-abort, timeout-leaves-running) → Task 7. ✓
- §7 No local state; JOBID via `last_service.env`, squeue fallback; the additive serve-script edit → Task 8 (`down`/`logs` read JOBID) + Task 9 (write JOBID). ✓
- §8 Snapshot seam inert in v1 → Task 5 + `--snapshot` wiring in Task 7. ✓
- §9 Error handling (unknown name, rcc failure, probe failure, ready timeout, down no-op, verify not-ready) → Tasks 1/7/8. ✓
- §10 Testing (mock the rcc boundary; fixtures; manual e2e) → tests in every task + Task 10 README. ✓
- §11 Rollout → Tasks 1–10 follow the rollout order. ✓

**Placeholder scan:** No "TBD/TODO/handle edge cases". Task 9 Step 2 is a concrete procedure with an explicit `ls` confirm step (filenames must be read from disk — not a guess presented as fact), with full worked manifests for the two non-trivial dirs.

**Type consistency:** `Deployment`/`SnapshotCfg` field names are identical across Tasks 1, 3, 4, 5, 7, 8. `rcc.RunResult(returncode, stdout, stderr)` and `rcc.run(profile, command, *, stream=)` / `rcc.push(profile)` signatures match every call site. `snapshot.plan(dep, mode) -> (env, serve|None)` consumed correctly in Task 7. `service.read_service_env(profile)` / `service.endpoints(env)` / `service.verify(dep, endpoint)` consistent across Tasks 4/6/7/8.
