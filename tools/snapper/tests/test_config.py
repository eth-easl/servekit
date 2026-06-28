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
