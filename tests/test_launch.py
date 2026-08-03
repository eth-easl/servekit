"""Tests for `servekit launch`: the argv rewrite, the vendored stager, and the free()."""
import filecmp
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

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


# The stager's own gate is size parity, and `truncate` already fixed the size
# before any writer ran -- only a checksum proves the content. One row per rank,
# one size per part, the way a real sharded checkpoint lands. Sizes straddle the
# 16 MiB block so parts are tiled across several concurrent dd's, which is the
# part a size check cannot see.
PART_BYTES = [
    [40_000_003, 4_000_000, 999],
    [33_554_432, 17_000_001, 0],
    [20_000_001, 1_048_577, 12],
    [16_777_217, 3_000_000, 7],
]


def _shard(path, nbytes):
    save_file({"w": np.frombuffer(os.urandom(nbytes), dtype=np.uint8)}, str(path))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_is_bitwise_exact_across_concurrent_writers(odirect_tmpdir):
    src, dest = odirect_tmpdir / "src", odirect_tmpdir / "dest"
    src.mkdir()
    (src / "config.json").write_text('{"model_type": "llama"}')
    for rank, parts in enumerate(PART_BYTES):
        for part, nbytes in enumerate(parts):
            _shard(src / f"model-rank-{rank}-part-{part}.safetensors", nbytes)

    result = stage(src, dest, slices=64)

    assert sorted(p.name for p in dest.iterdir()) == sorted(p.name for p in src.iterdir())
    for f in src.iterdir():
        assert _sha256(dest / f.name) == _sha256(f), f.name
    assert result.bytes == sum(f.stat().st_size for f in src.iterdir())
    assert result.wall_s > 0 and result.gbps > 0


def test_stage_fails_loudly_on_a_missing_source(odirect_tmpdir):
    with pytest.raises(RuntimeError, match="stager failed"):
        stage(odirect_tmpdir / "nope", odirect_tmpdir / "dest")


# --- end to end against a fake engine --------------------------------------

# Enough SGLang output to drive run_profile through to ready. Also exercises
# run_profile's subprocess path, which profile.py's own tests never enter.
FAKE_SGLANG = '''
import os, sys, time
args = sys.argv[1:]
model = args[args.index("--model-path") + 1]
open(sys.argv[0] + ".seen", "w").write(model)
# What config.json looked like the instant the engine started, the way SGLang
# reads it seconds in.
cfg = os.path.join(model, "config.json")
open(sys.argv[0] + ".cfg", "w").write(open(cfg).read() if os.path.exists(cfg) else "MISSING")
# --slow-start delays the loader so a concurrent stage finishes first, the way a
# real engine's ~40 s of import/TP-spawn/NCCL init does.
if "--slow-start" in args:
    time.sleep(float(args[args.index("--slow-start") + 1]))
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
    """The default is sequential: the stage completes before the engine starts."""
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


def test_overlap_is_opt_in_and_warns(odirect_tmpdir, capsys):
    src = _src_model(odirect_tmpdir)
    dest = odirect_tmpdir / "shm" / "llama70b-tp4"

    assert main(_launch_argv(odirect_tmpdir, src, extra=["--overlap"]) + ["--slow-start", "1.5"]) == 0

    assert (odirect_tmpdir / "fake-sglang.py.seen").read_text() == str(dest)
    assert not dest.exists()
    assert "UNSAFE" in capsys.readouterr().err

    report = json.loads((odirect_tmpdir / "run.json").read_text())
    assert report["success"]
    # Concurrent with the engine's phases, so not a serial phase of its own.
    assert "stage" not in {p["name"] for p in report["phases"]}


def test_overlap_copies_metadata_before_starting_the_engine(odirect_tmpdir, monkeypatch):
    """config.json must be readable at t=0, not left to a stage still running.

    The stager truncates every destination to full size before writing, so an
    engine that reads config.json mid-stage gets zeros. The stage is slowed here
    so the engine would lose that race if the metadata were not copied up front.
    """
    import servekit.launch as launch_mod

    real_stage = launch_mod.stage

    def slow_stage(*a, **kw):
        time.sleep(1.0)
        return real_stage(*a, **kw)

    monkeypatch.setattr(launch_mod, "stage", slow_stage)

    src = _src_model(odirect_tmpdir)
    assert main(_launch_argv(odirect_tmpdir, src, extra=["--overlap"])) == 0
    assert (odirect_tmpdir / "fake-sglang.py.cfg").read_text() == "{}"


def test_a_failed_stage_fails_the_run_even_when_overlapped(odirect_tmpdir, monkeypatch):
    """The engine can reach ready on a half-staged directory; rc must still be 1."""
    import servekit.launch as launch_mod

    def broken_stage(*a, **kw):
        raise RuntimeError("stager failed (rc=2) staging x -> y")

    monkeypatch.setattr(launch_mod, "stage", broken_stage)

    src = _src_model(odirect_tmpdir)
    assert main(_launch_argv(odirect_tmpdir, src, extra=["--overlap"])) == 1


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
