"""Tests for `servekit prepare`. The sharding needs 4 GPUs and sglang, so the
child is faked; servekit's half is what is under test."""
import json
import sys

import pytest

from servekit import manifest as manifest_mod
from servekit.cli import main
from servekit.engine_args import check_manifest
from servekit.manifest import Manifest
from servekit.prepare import SAVE_SCRIPT, prepare
from servekit.profile import SGLANG

RESOLVED = {
    "engine": "sglang",
    "engine_version": "0.4.6",
    "tp_size": 4,
    "pp_size": 1,
    "dp_size": 1,
    "dtype": "bfloat16",
    "quantization": None,
}

FAKE_SHARDER = '''
import json, sys
args = sys.argv[1:]
def arg(name):
    return args[args.index(name) + 1]
out, tp = arg("--output"), int(arg("--tensor-parallel-size"))
ranks = tp - 1 if "--drop-a-rank" in args else tp
for r in range(ranks):
    open(f"{out}/model-rank-{r}-part-0.safetensors", "wb").write(b"weights")
if "--no-resolved" not in args:
    resolved = json.load(open(arg("--fake-resolved")))
    resolved["tp_size"] = tp
    json.dump(resolved, open(arg("--servekit-resolved-out"), "w"))
sys.exit(3 if "--fail" in args else 0)
'''


@pytest.fixture
def fake_sharder(tmp_path, monkeypatch):
    script = tmp_path / "fake_sharder.py"
    script.write_text(FAKE_SHARDER)
    monkeypatch.setattr("servekit.prepare.SAVE_SCRIPT", script)
    resolved = tmp_path / "resolved.json"
    resolved.write_text(json.dumps(RESOLVED))
    return ["--fake-resolved", str(resolved)]


@pytest.fixture
def model(tmp_path):
    src = tmp_path / "Llama-3.1-70B-Instruct"
    src.mkdir()
    (src / "config.json").write_text('{"torch_dtype": "bfloat16"}')
    return src


def test_prepare_writes_a_manifest_of_what_the_engine_resolved(tmp_path, model, fake_sharder):
    out = tmp_path / "llama70b-tp4"

    assert prepare(model, out, tp=4, engine_args=fake_sharder) == 0

    written = manifest_mod.read(out)
    assert written == Manifest(
        format="sharded_state", source=str(model), **{**RESOLVED, "tp_size": 4}
    )
    assert written.dtype == "bfloat16"


def test_a_missing_rank_leaves_no_manifest(tmp_path, model, fake_sharder):
    out = tmp_path / "llama70b-tp4"

    assert prepare(model, out, tp=4, engine_args=fake_sharder + ["--drop-a-rank"]) == 1
    assert manifest_mod.read(out) is None


def test_a_failed_sharding_run_leaves_no_manifest(tmp_path, model, fake_sharder):
    out = tmp_path / "llama70b-tp4"
    assert prepare(model, out, tp=4, engine_args=fake_sharder + ["--fail"]) == 1
    assert manifest_mod.read(out) is None


def test_no_resolved_args_is_a_failure_not_a_guess(tmp_path, model, fake_sharder):
    out = tmp_path / "llama70b-tp4"
    assert prepare(model, out, tp=4, engine_args=fake_sharder + ["--no-resolved"]) == 1
    assert manifest_mod.read(out) is None


def test_a_source_that_is_not_a_directory_is_rejected(tmp_path, fake_sharder):
    assert prepare(tmp_path / "nope", tmp_path / "out", tp=4, engine_args=fake_sharder) == 2


def test_no_manifest_reads_as_none(tmp_path):
    assert manifest_mod.read(tmp_path) is None


def test_prepare_cli_forwards_extra_engine_args(tmp_path, model, monkeypatch):
    seen = {}

    def fake_prepare(model_arg, out, tp, engine_args=()):
        seen.update(model=model_arg, out=out, tp=tp, engine_args=list(engine_args))
        return 0

    monkeypatch.setattr("servekit.cli.prepare", fake_prepare)
    rc = main([
        "prepare", "--model", str(model), "--out", str(tmp_path / "o"), "--tp", "4",
        "--", "--trust-remote-code", "--context-length", "32768",
    ])

    assert rc == 0
    assert seen["tp"] == 4
    assert seen["engine_args"] == ["--trust-remote-code", "--context-length", "32768"]


def test_prepare_cli_requires_model_and_out():
    with pytest.raises(SystemExit):
        main(["prepare", "--model", "/store/m"])


PREPARED = Manifest(format="sharded_state", source="/store/m", **RESOLVED)


def _command(*args):
    return ["python", "-m", "sglang.launch_server", "--model-path", "/store/m", *args]


def test_a_matching_command_has_no_problems():
    command = _command("--tensor-parallel-size", "4", "--load-format", "sharded_state")
    assert check_manifest(command, SGLANG, PREPARED) == []


@pytest.mark.parametrize(
    "args, expected",
    [
        (["--tensor-parallel-size", "2"], "tp_size: the checkpoint is sharded for 4, the command asks for 2"),
        (["--tp-size=8"], "tp_size: the checkpoint is sharded for 4, the command asks for 8"),
        (["--tp", "1"], "tp_size: the checkpoint is sharded for 4, the command asks for 1"),
        ([], "tp_size: the checkpoint is sharded for 4, the command asks for 1"),
        (["--tensor-parallel-size", "4", "--pp-size", "2"], "pp_size: the checkpoint is sharded for 1, the command asks for 2"),
        (["--tensor-parallel-size", "4", "--data-parallel-size", "2"], "dp_size: the checkpoint is sharded for 1, the command asks for 2"),
    ],
)
def test_a_parallelism_mismatch_is_refused(args, expected):
    assert check_manifest(_command(*args), SGLANG, PREPARED) == [expected]


def test_a_dtype_the_engine_would_silently_cast_is_refused():
    command = _command("--tensor-parallel-size", "4", "--dtype", "float16")
    assert check_manifest(command, SGLANG, PREPARED) == [
        "dtype: the checkpoint was written as bfloat16, --dtype says float16"
    ]


def test_dtype_auto_and_an_unset_dtype_are_left_to_the_engine():
    assert check_manifest(_command("--tp", "4", "--dtype", "auto"), SGLANG, PREPARED) == []
    assert check_manifest(_command("--tp", "4"), SGLANG, PREPARED) == []


def test_a_checkpoint_prepared_for_another_engine_is_refused():
    other = Manifest(format="sharded_state", source="/store/m", **{**RESOLVED, "engine": "vllm"})
    problems = check_manifest(_command("--tp", "4"), SGLANG, other)
    assert problems == ["engine: the checkpoint was prepared for vllm, and sglang cannot load another engine's shards"]


def test_launch_refuses_a_mismatched_checkpoint_before_staging(tmp_path):
    src = tmp_path / "llama70b-tp4"
    src.mkdir()
    (src / "config.json").write_text("{}")
    (src / "model-rank-0-part-0.safetensors").write_bytes(b"weights")
    PREPARED.write(src)

    shm = tmp_path / "shm"
    rc = main([
        "launch", "--shm-root", str(shm),
        "--", "python", "-m", "sglang.launch_server",
        "--model-path", str(src), "--tensor-parallel-size", "2",
    ])

    assert rc == 2
    assert not shm.exists()


def test_the_vendored_sharder_skips_the_stale_weight_index(tmp_path):
    assert ".index.json" in SAVE_SCRIPT.read_text()


def test_servekit_does_not_import_sglang():
    assert "sglang" not in sys.modules
