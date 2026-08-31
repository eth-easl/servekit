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
