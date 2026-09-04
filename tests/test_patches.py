"""Tests for the vendored sglang backports."""
import shutil
import sys
from pathlib import Path

import pytest

from servekit._patches import sglang_35715

REAL = Path("srt") / "models" / "deepseek_v2.py"

SOURCE = """\
class DeepseekV2AttentionMLA(nn.Module):
    def __init__(self):
        self.alt_stream = alt_stream
        self.attn_mha.kv_b_proj = None

        self.w_kc = None

    def forward_prepare(self):
        if self.attn_mha.kv_b_proj is None:
            self.attn_mha.kv_b_proj = self.kv_b_proj

        return self
"""


def _install(root: Path, source: str = SOURCE) -> Path:
    path = root / REAL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


@pytest.fixture
def sys_path(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    return sys.path


def test_a_shadowing_source_tree_does_not_win(tmp_path, sys_path, monkeypatch):
    """The image's workdir is /opt and the sglang checkout sits at /opt/sglang,
    so the first candidate on sys.path holds no package at all."""
    shadow = tmp_path / "opt"
    (shadow / "sglang" / "python").mkdir(parents=True)
    real = _install(tmp_path / "dist-packages" / "sglang")
    sys_path.insert(0, str(shadow))
    sys_path.insert(1, str(tmp_path / "dist-packages"))
    monkeypatch.chdir(shadow)

    assert sglang_35715.target() == real


def test_patch_applies_once_and_is_idempotent(tmp_path, sys_path):
    real = _install(tmp_path / "sglang")
    sys_path.insert(0, str(tmp_path))

    assert sglang_35715.main() == 0
    patched = real.read_text()
    assert "object.__setattr__(self.attn_mha" in patched
    assert "if self.attn_mha.kv_b_proj is None:" not in patched

    assert sglang_35715.main() == 0
    assert real.read_text() == patched


def test_a_build_without_the_anchors_is_refused(tmp_path, sys_path, capsys):
    _install(tmp_path / "sglang", source="class DeepseekV2AttentionMLA:\n    pass\n")
    sys_path.insert(0, str(tmp_path))

    assert sglang_35715.main() == 1
    assert "does not fit this build" in capsys.readouterr().err


def test_no_sglang_at_all_names_what_it_tried(tmp_path, sys_path, monkeypatch):
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    with pytest.raises(SystemExit, match="no sglang holding"):
        sglang_35715.target()
