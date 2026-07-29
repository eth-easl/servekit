"""Tests for `servekit launch`: the argv rewrite, the vendored stager, and the free."""
import filecmp
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from servekit.cli import main
from servekit.engine_args import find_model_path, replace_model_path
from servekit.profile import SGLANG, VLLM, detect_framework
from servekit.stage import STAGER, stage

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT_STAGER = REPO / "experiments/lustre-loading-exp/scripts/phase4_shm/stage_to_shm_sliced.sh"

SGLANG_CMD = [
    "python", "-m", "sglang.launch_server",
    "--model-path", "/store/llama70b-tp4",
    "--tensor-parallel-size", "4",
    "--context-length", "32768",
    "--load-format", "sharded_state",
]


# --- engine_args: pure -----------------------------------------------------

@pytest.mark.parametrize(
    "command, expected",
    [
        (SGLANG_CMD, "/store/llama70b-tp4"),
        (["python", "-m", "sglang.launch_server", "--model-path=/store/m"], "/store/m"),
        (["python", "-m", "sglang.launch_server", "--model_path", "/store/m"], "/store/m"),
    ],
)
def test_finds_the_model_path(command, expected):
    assert find_model_path(command, SGLANG)[1] == expected


def test_replacing_the_model_path_changes_nothing_else():
    out = replace_model_path(SGLANG_CMD, SGLANG, "/dev/shm/servekit/llama70b-tp4")
    assert out[4] == "/dev/shm/servekit/llama70b-tp4"
    assert out[:4] == SGLANG_CMD[:4] and out[5:] == SGLANG_CMD[5:]
    assert SGLANG_CMD[4] == "/store/llama70b-tp4"  # caller's list untouched


def test_replacing_an_inline_model_path():
    command = ["python", "-m", "sglang.launch_server", "--model-path=/store/m", "--tp=4"]
    assert replace_model_path(command, SGLANG, "/dev/shm/m") == [
        "python", "-m", "sglang.launch_server", "--model-path=/dev/shm/m", "--tp=4",
    ]


def test_a_command_without_a_model_path_is_rejected():
    with pytest.raises(ValueError, match="no --model-path"):
        find_model_path(["python", "-m", "sglang.launch_server", "--tp", "4"], SGLANG)


def test_a_flag_with_no_value_is_rejected():
    with pytest.raises(ValueError, match="no value"):
        find_model_path(["python", "-m", "sglang.launch_server", "--model-path"], SGLANG)


def test_vllm_staging_is_not_supported_yet():
    """Phase 4's job. Better a clear refusal than a command rewritten by guess."""
    assert VLLM.model_flags == []
    with pytest.raises(ValueError, match="does not know where the model path is"):
        find_model_path(["vllm", "serve", "/store/m"], VLLM)


def test_launch_rejects_an_unusable_command():
    assert main(["launch", "--", "python", "-m", "sglang.launch_server", "--tp", "4"]) == 2
    assert main(["launch", "--", "frobnicate"]) == 2
    assert main(["launch"]) == 2


# --- the stager ------------------------------------------------------------

def test_the_vendored_stager_is_byte_identical_to_the_experiment():
    """The measured code path must stay literally the measured code path."""
    if not EXPERIMENT_STAGER.exists():
        pytest.skip("experiments/ not present")
    assert filecmp.cmp(STAGER, EXPERIMENT_STAGER, shallow=False)


@pytest.fixture
def odirect_tmpdir():
    """A temp dir on a filesystem that supports O_DIRECT, which the stager needs."""
    with tempfile.TemporaryDirectory(dir=REPO) as d:
        probe = Path(d) / ".probe"
        probe.write_bytes(b"x" * 4096)
        try:
            os.close(os.open(probe, os.O_RDONLY | os.O_DIRECT))
        except OSError:
            pytest.skip(f"no O_DIRECT under {REPO}")
        probe.unlink()
        yield Path(d)


def test_stage_copies_every_file_and_reports_its_own_timing(odirect_tmpdir):
    src, dest = odirect_tmpdir / "src", odirect_tmpdir / "dest"
    src.mkdir()
    (src / "config.json").write_text('{"model_type": "llama"}')
    (src / "model-rank-0-part-0.safetensors").write_bytes(os.urandom(3_000_000))

    result = stage(src, dest, slices=4)

    assert sorted(p.name for p in dest.iterdir()) == ["config.json", "model-rank-0-part-0.safetensors"]
    for f in src.iterdir():
        assert (dest / f.name).read_bytes() == f.read_bytes()
    assert result.bytes == sum(f.stat().st_size for f in src.iterdir())
    assert result.wall_s > 0 and result.gbps > 0


def test_stage_fails_loudly_on_a_missing_source(odirect_tmpdir):
    with pytest.raises(RuntimeError, match="stager failed"):
        stage(odirect_tmpdir / "nope", odirect_tmpdir / "dest")


# --- end to end against a fake engine --------------------------------------

# Enough SGLang output to drive run_profile through to ready. Also exercises
# run_profile's subprocess path, which profile.py's own tests never enter.
FAKE_SGLANG = '''
import sys, time
args = sys.argv[1:]
model = args[args.index("--model-path") + 1]
open(sys.argv[0] + ".seen", "w").write(model)
print("Init torch distributed ends. elapsed=1.00 s", flush=True)
print("Load weight end. elapsed=2.00 s", flush=True)
if "--never-ready" not in args:
    print("The server is fired up and ready to roll!", flush=True)
time.sleep(0.3)
'''


def _fake_engine(tmpdir):
    path = tmpdir / "fake-sglang.py"
    path.write_text(FAKE_SGLANG)
    return path


def _launch_argv(tmpdir, src, extra=()):
    engine = _fake_engine(tmpdir)
    return [
        "launch", "--shm-root", str(tmpdir / "shm"), "--slices", "2",
        "--out", str(tmpdir / "run.json"), *extra,
        "--", sys.executable, str(engine), "--model-path", str(src),
    ]


def _src_model(tmpdir):
    src = tmpdir / "llama70b-tp4"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "model-rank-0-part-0.safetensors").write_bytes(os.urandom(200_000))
    return src


def test_launch_stages_rewrites_and_frees_on_ready(odirect_tmpdir):
    src = _src_model(odirect_tmpdir)
    dest = odirect_tmpdir / "shm" / "llama70b-tp4"

    assert main(_launch_argv(odirect_tmpdir, src)) == 0

    # The engine was pointed at the copy, not the original.
    assert (odirect_tmpdir / "fake-sglang.py.seen").read_text() == str(dest)
    assert not dest.exists()

    report = json.loads((odirect_tmpdir / "run.json").read_text())
    assert report["success"]
    phases = {p["name"]: p["duration_s"] for p in report["phases"]}
    assert "stage" in phases and phases["stage"] > 0
    assert phases["weight_loading"] == 2.0
    # The report records what the user asked for, not the rewritten path.
    assert str(src) in report["command"] and str(dest) not in report["command"]


def test_a_run_that_never_reaches_ready_leaves_the_copy_intact(odirect_tmpdir):
    """No leak recovery in Phase 1: the copy stays and `rm -r` is the hatch.

    Freeing exists to give the RAM back to a *serving* job, and a server that
    never got ready is not serving -- so guessing that the engine has let go of
    the files buys nothing and risks pulling them out from under it.
    """
    src = _src_model(odirect_tmpdir)
    dest = odirect_tmpdir / "shm" / "llama70b-tp4"

    rc = main(_launch_argv(odirect_tmpdir, src) + ["--never-ready"])

    assert rc == 1
    assert dest.is_dir()
    assert not json.loads((odirect_tmpdir / "run.json").read_text())["success"]
