"""Tests for the stage barrier in servekit._shim: the marker protocol that lets
a scheduler process, started fresh under sglang's `spawn`, wait for a stage
thread in a different process to finish."""
import threading
import time

import pytest

from servekit import _shim


def test_marker_is_a_dotfile_sibling_of_the_staged_dir(tmp_path):
    dest = tmp_path / "llama70b"
    assert _shim.stage_marker(str(dest)) == str(tmp_path / ".llama70b.stage")


def test_no_marker_means_no_barrier(tmp_path):
    _shim.wait_for_stage(str(tmp_path / "model"), timeout_s=0.01)


def test_ok_releases_the_waiter(tmp_path):
    dest = tmp_path / "model"
    _shim.publish_stage(str(dest), _shim.STAGE_OK)
    _shim.wait_for_stage(str(dest), timeout_s=0.01)


def test_a_failed_stage_raises_with_its_message(tmp_path):
    dest = tmp_path / "model"
    _shim.publish_stage(str(dest), "staged 3 of 4 files")
    with pytest.raises(RuntimeError, match="staged 3 of 4 files"):
        _shim.wait_for_stage(str(dest), timeout_s=0.01)


def test_a_stuck_pending_marker_times_out(tmp_path):
    dest = tmp_path / "model"
    _shim.publish_stage(str(dest), _shim.STAGE_PENDING)
    with pytest.raises(TimeoutError):
        _shim.wait_for_stage(str(dest), timeout_s=0.05, poll_s=0.01)


def test_a_stage_finishing_late_releases_a_waiter(tmp_path):
    dest = tmp_path / "model"
    _shim.publish_stage(str(dest), _shim.STAGE_PENDING)

    def finish_later():
        time.sleep(0.05)
        _shim.publish_stage(str(dest), _shim.STAGE_OK)

    threading.Thread(target=finish_later).start()
    _shim.wait_for_stage(str(dest), timeout_s=2, poll_s=0.01)


def test_clear_stage_is_missing_ok(tmp_path):
    _shim.clear_stage(str(tmp_path / "model"))
    dest = tmp_path / "model"
    _shim.publish_stage(str(dest), _shim.STAGE_OK)
    _shim.clear_stage(str(dest))
    _shim.wait_for_stage(str(dest), timeout_s=0.01)


def test_get_parallel_is_none_without_sglang():
    # install() hooks _around_load_model/_around_save_model on every load, TP-only
    # included, and both call this first. An sglang build without runtime_context
    # (pipeline parallel support, and this shim's only way to read pp_size) must
    # not crash a plain TP load over it -- sglang itself isn't installed here,
    # which is the same ModuleNotFoundError a too-old sglang build hits. pytest's
    # own argv has no --pipeline-parallel-size, so this is the pp-not-requested case.
    assert _shim._get_parallel() is None


def test_get_parallel_raises_when_pp_was_actually_requested(monkeypatch):
    # Without runtime_context, a real pp>1 request can't be sharded correctly --
    # falling through to pp_size=1 would silently collide stages on one filename
    # instead of raising, which is worse than a loud failure.
    monkeypatch.setattr("sys.argv", ["sglang.launch_server", "--pipeline-parallel-size", "2"])
    with pytest.raises(RuntimeError, match="runtime_context"):
        _shim._get_parallel()


@pytest.mark.parametrize("argv", [[], ["--pipeline-parallel-size", "1"], ["--pipeline-parallel-size=1"]])
def test_get_parallel_does_not_raise_when_pp_is_1_or_absent(monkeypatch, argv):
    monkeypatch.setattr("sys.argv", ["sglang.launch_server", *argv])
    assert _shim._get_parallel() is None
