"""Tests for the quantization guard: the SGLang postprocess paths whose weights
a presharded checkpoint cannot carry, and the paths that must stay quiet."""
import json

import pytest

from servekit.quant_guard import Setup, check, check_command


def model_dir(tmp_path, config, gguf=False):
    src = tmp_path / "checkpoint"
    src.mkdir(exist_ok=True)
    (src / "config.json").write_text(json.dumps(config))
    if gguf:
        (src / "model.gguf").write_bytes(b"")
    return src


@pytest.mark.parametrize(
    "setup",
    [
        Setup(quant_method="mxfp4", moe_runner_backend="triton_kernel"),
        Setup(quant_config={"quant_algo": "NVFP4"}, architectures=("DeepseekV3ForCausalLM",)),
        Setup(quant_method="gguf"),
        Setup(gguf_files=True),
        # --moe-runner-backend megamoe is an alias sglang folds into the a2a backend.
        Setup(moe_a2a_backend="megamoe"),
        Setup(moe_runner_backend="megamoe"),
        Setup(quant_method="gptq", quant_config={"desc_act": True}),
        Setup(quant_method="compressed-tensors",
              quant_config={"config_groups": {"g": {"weights": {"actorder": "group"}}}}),
        Setup(moe_runner_backend="flashinfer_mxfp4", architectures=("DeepseekV3ForCausalLM",)),
    ],
)
def test_unsupported_paths_are_refused(setup):
    assert check(setup)


@pytest.mark.parametrize(
    "setup",
    [
        Setup(),
        # Every other mxfp4 backend re-registers w13_weight and round-trips.
        Setup(quant_method="mxfp4"),
        Setup(quant_method="mxfp4", moe_runner_backend="marlin"),
        Setup(quant_config={"quant_algo": "NVFP4"}, architectures=("LlamaForCausalLM",)),
        Setup(quant_method="fp8", moe_a2a_backend="deepep"),
        Setup(quant_method="gptq", quant_config={"desc_act": False}),
        Setup(quant_method="compressed-tensors",
              quant_config={"config_groups": {"g": {"weights": {}}}}),
        Setup(quant_method="awq"),
    ],
)
def test_supported_paths_are_not_refused(setup):
    assert check(setup) == []


def test_reads_the_checkpoint_config_and_the_engine_flags(tmp_path):
    src = model_dir(tmp_path, {"quantization_config": {"quant_method": "mxfp4"}})
    command = ["python", "-m", "sglang.launch_server", "--model-path", str(src),
               "--moe-runner-backend", "triton_kernel"]

    assert check_command(src, command)
    assert check_command(src, command[:-2]) == []


def test_the_quantization_flag_wins_over_the_checkpoint_config(tmp_path):
    src = model_dir(tmp_path, {})
    command = ["python", "--model-path", str(src),
               "--quantization=mxfp4", "--moe-runner-backend=triton_kernel"]

    assert check_command(src, command)


def test_a_missing_or_unreadable_config_refuses_nothing(tmp_path):
    src = tmp_path / "no-config"
    src.mkdir()
    assert check_command(src, ["python", "--model-path", str(src)]) == []

    (src / "config.json").write_text("{not json")
    assert check_command(src, ["python", "--model-path", str(src)]) == []


def test_a_gguf_directory_is_caught_without_a_config(tmp_path):
    src = model_dir(tmp_path, {}, gguf=True)
    assert check_command(src, ["python", "--model-path", str(src)])
