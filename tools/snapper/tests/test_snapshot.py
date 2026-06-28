from pathlib import Path

import pytest

from snapper import snapshot
from snapper.config import Deployment, SnapshotCfg


def _dep(snap: SnapshotCfg) -> Deployment:
    return Deployment(
        name="d",
        profile="p",
        engine="sglang",
        served_model_name="m",
        port=8080,
        default_nodes=1,
        deploy_dir=Path("."),
        rel_dir="deploy/d",
        probe_script=None,
        serve_script="serve.sbatch",
        variants={},
        ready_markers=[],
        log_pattern="logs/x-{job}-{rank}.log",
        verify_models_path="/v1/models",
        verify_chat_path="/v1/chat/completions",
        snapshot=snap,
    )


def test_plan_noop_when_disabled():
    dep = _dep(SnapshotCfg(enabled=False))
    assert snapshot.plan(dep, "record") == ({}, None)


def test_plan_noop_when_mode_none():
    dep = _dep(SnapshotCfg(enabled=True, record_env={"SNAPSHOT_MODE": "record"}))
    assert snapshot.plan(dep, None) == ({}, None)


def test_plan_record_when_enabled():
    dep = _dep(
        SnapshotCfg(
            enabled=True,
            record_env={"SNAPSHOT_MODE": "record"},
            serve="serve_snap.sbatch",
        )
    )
    assert snapshot.plan(dep, "record") == (
        {"SNAPSHOT_MODE": "record"},
        "serve_snap.sbatch",
    )


def test_plan_unknown_mode_raises():
    dep = _dep(SnapshotCfg(enabled=True))
    with pytest.raises(ValueError):
        snapshot.plan(dep, "bogus")
