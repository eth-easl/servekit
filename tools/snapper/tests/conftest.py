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
