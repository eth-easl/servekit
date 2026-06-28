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
