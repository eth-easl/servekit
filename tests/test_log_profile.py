"""Saved logs use the same parser and report as live Servekit profiles."""
import json
from pathlib import Path

import pytest

from servekit import cli
from servekit.log_profile import SPECS, _load_spec, profile_log
from servekit.profile import ProfileReport, render_table

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("log, version", [
    ("vllm-0.14.1-apertus.log", "0.14.1"),
    ("vllm-0.18.0-qwen3.log", "0.18.0"),
    ("llama70b-vllm-ngc.log", "0.24.0+092c4842.dev"),
    ("vllm-0.28.1rc1-glm-h100.log", "0.28.1rc1.dev580+g385dce36b"),
])
def test_repo_vllm_logs_use_normal_profile_reports(log, version):
    report, approximate = profile_log(FIXTURES / log)
    assert isinstance(report, ProfileReport)
    assert report.success and report.total_duration_s > 0
    assert report.framework_version == version
    assert report.pattern_version == ("0.24.0" if log == "llama70b-vllm-ngc.log" else version)
    assert approximate == (log == "llama70b-vllm-ngc.log")
    assert "[SERVEKIT] Cold-start profile" in render_table(report)
    assert report.to_dict()["phases"] == [vars(phase) for phase in report.phases]


def test_worker_maximum_and_separate_graph_capture_passes():
    old, _ = profile_log(FIXTURES / "llama70b-vllm-ngc.log")
    assert [p.duration_s for p in old.phases if p.name == "weight_loading"] == [233.24]

    newer, _ = profile_log(FIXTURES / "vllm-0.28.1rc1-glm-h100.log")
    assert [p.duration_s for p in newer.phases if p.name == "cuda_graph_capture"] == [142.0, 4.0]
    assert next(p.duration_s for p in newer.phases if p.name == "deepgemm_warmup") > 190


def test_launch_time_adds_container_startup():
    log = FIXTURES / "vllm-0.28.1rc1-glm-h100.log"
    plain, _ = profile_log(log)
    launched, _ = profile_log(log, "2026-10-05T16:18:25.112827Z")
    assert plain.phases[0].name == launched.phases[0].name == "process_startup"
    assert plain.phases[0].duration_s == 0
    assert launched.phases[0].duration_s == pytest.approx(41.55, abs=0.01)
    assert launched.total_duration_s - plain.total_duration_s == pytest.approx(41.55, abs=0.01)


def test_cli_prints_table_writes_standard_json_and_warns_on_fallback(tmp_path, capsys):
    log = tmp_path / "new-version.log"
    log.write_text((FIXTURES / "vllm-0.18.0-qwen3.log").read_text().replace("version 0.18.0", "version 0.18.1", 1))
    out = tmp_path / "report.json"
    assert cli.main(["profile", "--log", str(log), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    data = json.loads(out.read_text())
    assert "Cold-start profile" in captured.out and "weight_loading" in captured.out
    assert "using 0.18.0" in captured.err
    assert data["framework"] == "vllm" and data["framework_version"] == "0.18.1"
    assert data["pattern_version"] == "0.18.0"
    assert isinstance(data["phases"], list) and data["phases"][0]["source"] == "wall_clock"


def test_incomplete_log_fails_instead_of_reporting_ready(tmp_path, capsys):
    log = tmp_path / "incomplete.log"
    log.write_text((FIXTURES / "vllm-0.18.0-qwen3.log").read_text().replace("Application startup complete", "startup failed"))
    assert cli.main(["profile", "--log", str(log)]) == 2
    assert "no vLLM ready line" in capsys.readouterr().err


def test_version_files_load_as_framework_specs():
    assert len(list(SPECS.glob("*.json"))) >= 4
    for path in SPECS.glob("*.json"):
        spec = _load_spec(path)
        assert spec.name == "vllm" and spec.phase_patterns and spec.milestone_patterns
