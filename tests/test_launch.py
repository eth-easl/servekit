"""Tests for `servekit launch`: the argv rewrite, the vendored stager, and the free()."""
import filecmp
import hashlib
import os
import tempfile
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


def test_launch_rejects_an_unusable_command():
    assert main(["launch", "--", "python", "-m", "sglang.launch_server", "--tp", "4"]) == 2
    assert main(["launch", "--", "frobnicate"]) == 2
    assert main(["launch"]) == 2


# --- the stager ------------------------------------------------------------

def _hot_path(script: Path) -> str:
    """The tiling and the dd fan-out: the part the throughput numbers measured."""
    text = script.read_text()
    start = text.index("# Full size up front")
    end = text.index('end="$(date +%s.%N)"')
    return text[start:end]


def test_the_vendored_stagers_hot_path_matches_the_experiment():
    """The measured code path must stay literally the measured code path.

    Narrowed from a whole-file compare when the vendored copy gained FILE_LIST:
    a presharded node's file set is a union of its ranks' reads, which no glob
    describes, so selection had to diverge. Selection is not what 11.9/17.0 GB/s
    measured -- the slice tiling and the dd fan-out are, and those must not move.
    """
    if not EXPERIMENT_STAGER.exists():
        pytest.skip("experiments/ not present")
    assert _hot_path(STAGER) == _hot_path(EXPERIMENT_STAGER)


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


def test_stage_copies_exactly_the_listed_files_and_no_others(odirect_tmpdir):
    """A presharded node's file set is a union of its ranks' reads, not a glob.

    The names deliberately include a `-common` file and a multi-rank one, which
    is what a pattern-based stager cannot select without dragging in the rest.
    """
    src, dest = odirect_tmpdir / "src", odirect_tmpdir / "dest"
    src.mkdir()
    names = [
        "model-00000-rank-000.safetensor",
        "model-00001-rank-000,001.safetensor",
        "model-00003-common.safetensor",
        "model-00006-rank-001.safetensor",
        "model-00008-rank-002.safetensor",
    ]
    for name, nbytes in zip(names, [40_000_003, 1_048_577, 999, 16_777_217, 20_000_001]):
        _shard(src / name, nbytes)

    wanted = names[:3]
    result = stage(src, dest, slices=64, files=wanted)

    assert sorted(p.name for p in dest.iterdir()) == sorted(wanted)
    for name in wanted:
        assert _sha256(dest / name) == _sha256(src / name), name
    assert result.bytes == sum((src / n).stat().st_size for n in wanted)


def test_stage_refuses_a_file_list_naming_something_absent(odirect_tmpdir):
    src, dest = odirect_tmpdir / "src", odirect_tmpdir / "dest"
    src.mkdir()
    _shard(src / "model-00000-rank-000.safetensor", 4096)
    with pytest.raises(RuntimeError, match="stager failed"):
        stage(src, dest, files=["model-00000-rank-000.safetensor", "gone.safetensor"])
