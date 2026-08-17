"""Tests for preparing an artifact. The sharding needs 4 GPUs and sglang, so the
child is faked; servekit's half is what is under test."""
import json
import sys

import pytest

from servekit import jit_cache
from servekit import manifest as manifest_mod
from servekit.cli import main
from servekit.engine_args import check_manifest
from servekit.manifest import Manifest
from servekit.prepare import SAVE_SCRIPT, prepare
from servekit.profile import SGLANG, ProfileReport

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
import json, os, sys
args = sys.argv[1:]
def arg(name):
    return args[args.index(name) + 1]
out, tp = arg("--output"), int(arg("--tensor-parallel-size"))
pp = int(arg("--pipeline-parallel-size"))
ranks = tp - 1 if "--drop-a-rank" in args else tp
for p in range(pp):
    for r in range(ranks):
        name = f"model-pp-{p}-rank-{r}-part-0" if pp > 1 else f"model-rank-{r}-part-0"
        open(f"{out}/{name}.safetensors", "wb").write(b"weights")
if "--dump-env" in args:
    json.dump(dict(os.environ), open(arg("--dump-env"), "w"))
if "--jit-a-kernel" in args:
    open(arg("--jit-a-kernel"), "wb").write(b"cubin")
if "--no-resolved" not in args:
    resolved = json.load(open(arg("--fake-resolved")))
    resolved["tp_size"], resolved["pp_size"] = tp, pp
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


def test_the_sharding_run_jits_where_launch_will_read(tmp_path, model, fake_sharder):
    """Not into `out`: a cache built anywhere but where launch reads it
    rebuilds, and buys nothing."""
    out = tmp_path / "llama70b-tp4"
    cache_root = tmp_path / "node-local"
    dumped = tmp_path / "env.json"

    rc = prepare(
        model, out, tp=4, cache_root=cache_root,
        engine_args=fake_sharder + ["--dump-env", str(dumped)],
    )

    assert rc == 0
    child_env = json.loads(dumped.read_text())
    assert {k: child_env.get(k) for k in jit_cache.CACHES} == jit_cache.env_for(cache_root / out.name)


def test_what_the_sharding_run_jitted_is_kept_with_the_checkpoint(tmp_path, model, fake_sharder):
    out = tmp_path / "llama70b-tp4"
    cache_root = tmp_path / "node-local"

    rc = prepare(
        model, out, tp=4, cache_root=cache_root,
        engine_args=fake_sharder + ["--jit-a-kernel", str(cache_root / out.name / "triton" / "k.cubin")],
    )

    assert rc == 0
    assert (out / jit_cache.CACHE_DIR_NAME / "triton" / "k.cubin").read_bytes() == b"cubin"


def test_a_missing_rank_leaves_no_manifest(tmp_path, model, fake_sharder):
    out = tmp_path / "llama70b-tp4"

    assert prepare(model, out, tp=4, engine_args=fake_sharder + ["--drop-a-rank"]) == 1
    assert manifest_mod.read(out) is None


def test_prepare_records_the_pipeline_stages_it_sharded_for(tmp_path, model, fake_sharder, monkeypatch):
    monkeypatch.setattr("servekit.prepare._plugins_available", lambda: True)
    out = tmp_path / "llama70b-tp4pp2"

    assert prepare(model, out, tp=4, pp=2, engine_args=fake_sharder) == 0

    assert manifest_mod.read(out).pp_size == 2
    assert (out / "model-pp-1-rank-3-part-0.safetensors").is_file()
    assert not (out / "model-rank-0-part-0.safetensors").exists()


def test_pipeline_parallel_is_refused_without_the_plugin_framework(tmp_path, model, fake_sharder, monkeypatch, capsys):
    monkeypatch.setattr("servekit.prepare._plugins_available", lambda: False)
    out = tmp_path / "llama70b-tp4pp2"

    assert prepare(model, out, tp=4, pp=2, engine_args=fake_sharder) == 2
    assert "0.5.11" in capsys.readouterr().err
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


def test_launch_prepares_with_the_engine_args_it_will_serve_with(tmp_path, model, monkeypatch):
    seen = {}

    def fake_prepare(model_arg, out, tp, engine_args=(), **kwargs):
        seen.update(model=model_arg, out=out, tp=tp, engine_args=list(engine_args), **kwargs)
        return 0

    monkeypatch.setattr("servekit.launch.prepare", fake_prepare)
    rc = main([
        "launch", "--servekit-artifact-path", str(tmp_path / "o"), "--only-prepare",
        "--", "python", "-m", "sglang.launch_server",
        "--model-path", str(model), "--tensor-parallel-size", "4",
        "--trust-remote-code", "--context-length", "32768",
        "--load-format", "sharded_state",
    ])

    assert rc == 0
    assert seen["tp"] == 4
    assert seen["model"] == model and seen["out"] == tmp_path / "o"
    assert seen["engine_args"] == [
        "--model-path", str(model), "--tensor-parallel-size", "4",
        "--trust-remote-code", "--context-length", "32768",
    ]
    # Single-node, single-stage unless asked otherwise, so the sharder's argv is unchanged.
    assert seen["pp"] == 1
    assert seen["nnodes"] == 1 and seen["node_rank"] == 0 and seen["dist_init_addr"] is None


def test_launch_requires_an_artifact_path():
    with pytest.raises(SystemExit):
        main(["launch", "--", "python", "-m", "sglang.launch_server", "--model-path", "/store/m"])


def test_prepare_is_not_a_subcommand():
    assert main(["prepare", "--model", "/store/m", "--out", "/scratch/o"]) == 2


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


def test_launch_warns_about_a_mismatched_artifact_and_serves_the_source(tmp_path, model, monkeypatch, capsys):
    artifact = tmp_path / "llama70b-tp4"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    (artifact / "model-rank-0-part-0.safetensors").write_bytes(b"weights")
    Manifest(format="sharded_state", source=str(model), **RESOLVED).write(artifact)

    launched = {}

    def fake_run_profile(command, ready_timeout=1800.0, on_ready=None, head=True, env=None):
        launched["command"] = command
        return ProfileReport(command="", started_at=0.0, ready_at=1.0, success=True, framework="sglang")

    def no_stage(*a, **k):
        raise AssertionError("the slow path must not stage")

    shm = tmp_path / "shm"
    monkeypatch.setattr("servekit.launch.DEFAULT_ROOT", shm)
    monkeypatch.setattr("servekit.launch.stage", no_stage)
    monkeypatch.setattr("servekit.launch.run_profile", fake_run_profile)
    rc = main([
        "launch", "--servekit-artifact-path", str(artifact), "--out", str(tmp_path / "run.json"),
        "--", "python", "-m", "sglang.launch_server",
        "--model-path", str(model), "--tensor-parallel-size", "2",
    ])

    assert rc == 0
    assert not shm.exists()
    assert "tp_size: the checkpoint is sharded for 4, the command asks for 2" in capsys.readouterr().out
    assert launched["command"] == [
        "python", "-m", "sglang.launch_server",
        "--model-path", str(model), "--tensor-parallel-size", "2",
    ]


def test_the_vendored_sharder_skips_the_stale_weight_index(tmp_path):
    assert ".index.json" in SAVE_SCRIPT.read_text()


def test_servekit_does_not_import_sglang():
    assert "sglang" not in sys.modules
