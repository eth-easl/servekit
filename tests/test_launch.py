"""Tests for `servekit launch`: the argv rewrite, the vendored stager, and the free()."""
import filecmp
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from servekit import jit_cache
from servekit.cli import main
from servekit.engine_args import find_model_path, replace_model_path
from servekit.manifest import Manifest
from servekit.profile import SGLANG, VLLM, ProfileReport, detect_framework
from servekit.stage import STAGER, StageResult, stage

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
        (["python", "-m", "sglang.launch_server", "--model", "/store/m"], "/store/m"),
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


def test_launch_rejects_an_unusable_command(tmp_path):
    artifact = ["--servekit-artifact-path", str(tmp_path / "a")]
    assert main(["launch", *artifact, "--", "python", "-m", "sglang.launch_server", "--tp", "4"]) == 2
    assert main(["launch", *artifact, "--", "frobnicate"]) == 2
    assert main(["launch"]) == 2


# --- the JIT caches --------------------------------------------------------

@pytest.fixture
def fake_engine(monkeypatch):
    """Run `launch` without a GPU, recording the env it would have handed sglang."""
    seen = {}

    def fake_stage(src, dest, slices=64, file_pattern="*"):
        return StageResult(dest=dest, wall_s=0.1, gbps=1.0, bytes=1)

    def fake_run_profile(command, ready_timeout=1800.0, on_ready=None, head=True, env=None):
        seen["env"] = env or {}
        seen["command"] = command
        return ProfileReport(command="", started_at=0.0, ready_at=1.0, success=True, framework="sglang")

    monkeypatch.setattr("servekit.launch.stage", fake_stage)
    monkeypatch.setattr("servekit.launch.run_profile", fake_run_profile)
    return seen


@pytest.fixture
def model(tmp_path):
    src = tmp_path / "llama70b"
    src.mkdir()
    (src / "config.json").write_text("{}")
    return src


@pytest.fixture
def prepared_checkpoint(tmp_path, model):
    src = tmp_path / "llama70b-tp4"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "model-rank-0-part-0.safetensors").write_bytes(b"weights")
    Manifest(
        format="sharded_state", engine="sglang", engine_version="0.5.10",
        tp_size=1, pp_size=1, dp_size=1, dtype="bfloat16", quantization=None,
        source=str(model),
    ).write(src)
    return src


def _launch(tmp_path, model, artifact, monkeypatch):
    # --shm-root/--cache-root aren't public CLI flags, so isolation from the
    # real /dev/shm and /tmp/servekit/caches goes through the module defaults.
    monkeypatch.setattr("servekit.launch.DEFAULT_ROOT", tmp_path / "shm")
    monkeypatch.setattr("servekit.launch.DEFAULT_CACHE_ROOT", tmp_path / "cache-root")
    # --out or the report lands in the cwd, which is the checkout.
    return main([
        "launch",
        "--servekit-artifact-path", str(artifact),
        "--out", str(tmp_path / "run.json"),
        "--", "python", "-m", "sglang.launch_server", "--model-path", str(model),
    ])


def _with_a_cached_kernel(checkpoint):
    cache = checkpoint / jit_cache.CACHE_DIR_NAME
    (cache / "triton").mkdir(parents=True)
    (cache / "triton" / "kernel.cubin").write_bytes(b"cubin")
    return cache


def test_the_engine_reads_a_node_local_copy_not_the_checkpoint(tmp_path, model, prepared_checkpoint, fake_engine, monkeypatch):
    _with_a_cached_kernel(prepared_checkpoint)

    assert _launch(tmp_path, model, prepared_checkpoint, monkeypatch) == 0

    copy = tmp_path / "cache-root" / prepared_checkpoint.name
    assert fake_engine["env"] == jit_cache.env_for(copy)
    assert fake_engine["command"] == [
        "python", "-m", "sglang.launch_server",
        "--model-path", str(tmp_path / "shm" / prepared_checkpoint.name),
        "--load-format", "sharded_state",
    ]


def test_a_checkpoint_with_no_caches_still_serves(tmp_path, model, prepared_checkpoint, fake_engine, capsys, monkeypatch):
    assert _launch(tmp_path, model, prepared_checkpoint, monkeypatch) == 0

    assert fake_engine["env"] == {}
    assert "JIT from cold" in capsys.readouterr().out


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
