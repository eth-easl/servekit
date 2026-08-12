"""Quantization paths a presharded checkpoint cannot round-trip.

SGLang's presharding savers serialize a model *after*
`process_weights_after_loading()` has run. The methods below move
inference-critical data off the module in that hook -- onto the quant-method
object, onto a bare attribute, or into a non-persistent buffer -- so it is
absent from the dump and never recomputed on reload. Nothing raises: the server
comes up and answers fluently on partly-zeroed weights, which is why these are
refused rather than warned about.

Diagnosis for the mxfp4 case: <https://github.com/eth-easl/servekit/issues/14>,
reported upstream as <https://github.com/sgl-project/sglang/issues/34448>.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from .engine_args import _find_flag


@dataclass(frozen=True)
class Setup:
    """The checkpoint's quantization and the engine flags that pick a kernel.

    Every field defaults to the value that triggers nothing, so a caller that
    cannot determine one is no worse off than with no check at all.
    """

    quant_method: Optional[str] = None
    quant_config: Mapping = field(default_factory=dict)
    architectures: Tuple[str, ...] = ()
    gguf_files: bool = False
    moe_runner_backend: str = "auto"
    moe_a2a_backend: str = "none"

    @property
    def quant_algo(self) -> str:
        return str(self.quant_config.get("quant_algo", "")).upper()

    @property
    def is_deepseek(self) -> bool:
        return any("deepseek" in a.lower() for a in self.architectures)

    @property
    def act_order(self) -> bool:
        """Act-order GPTQ, spelled two ways depending on the checkpoint format."""
        if self.quant_config.get("desc_act"):
            return True
        groups = self.quant_config.get("config_groups") or {}
        return any(
            (g or {}).get("weights", {}).get("actorder") == "group" for g in groups.values()
        )


# (what servekit will not preshard, what makes it so). Kept as short names: the
# rationale is one hook away, in the module docstring and issue #14.
RULES = [
    ("mxfp4 with --moe-runner-backend triton_kernel",
     lambda s: s.quant_method == "mxfp4" and s.moe_runner_backend == "triton_kernel"),
    ("NVFP4 on DeepSeek (SwiGLU-fused shared experts)",
     lambda s: "NVFP4" in s.quant_algo and s.is_deepseek),
    ("GGUF",
     lambda s: s.quant_method == "gguf" or s.gguf_files),
    ("FP8 MoE with the MegaMoE all-to-all backend",
     lambda s: "megamoe" in (s.moe_a2a_backend, s.moe_runner_backend)),
    ("act-order GPTQ / compressed-tensors on Marlin",
     lambda s: s.act_order),
    ("FP4 experts with --moe-runner-backend flashinfer_mxfp4",
     lambda s: s.moe_runner_backend == "flashinfer_mxfp4" and s.is_deepseek),
]


def check(setup: Setup) -> List[str]:
    """What about this setup servekit will not preshard. Empty means go ahead."""
    return [name for name, triggers in RULES if triggers(setup)]


def check_dir(model_dir: Path, **engine) -> List[str]:
    """`check` for a checkpoint directory plus whatever the caller resolved.

    An absent or unreadable config is not an error: it means no rule fires,
    which is where servekit was before this check existed.
    """
    config = {}
    path = Path(model_dir) / "config.json"
    if path.is_file():
        try:
            config = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            config = {}
    quant_config = config.get("quantization_config") or {}
    return check(Setup(
        quant_method=engine.pop("quantization", None) or quant_config.get("quant_method"),
        quant_config=quant_config,
        architectures=tuple(config.get("architectures") or ()),
        gguf_files=any(Path(model_dir).glob("*.gguf")),
        **{k: v for k, v in engine.items() if v is not None},
    ))


def check_command(model_dir: Path, command: Sequence[str]) -> List[str]:
    """`check_dir` for an engine command line, for the launch side."""

    def flag(name: str, default: Optional[str] = None) -> Optional[str]:
        found = _find_flag(list(command), [name])
        return found[1] if found else default

    return check_dir(
        model_dir,
        quantization=flag("--quantization"),
        moe_runner_backend=flag("--moe-runner-backend"),
        moe_a2a_backend=flag("--moe-a2a-backend"),
    )


def refusal(unsupported: Sequence[str], model_dir) -> str:
    # Prefixed, because this one surfaces in the middle of the engine's own output.
    return (
        f"[SERVEKIT] error: servekit does not support {', '.join(unsupported)} as a presharded checkpoint: "
        f"these weights do not survive the sharded_state round-trip, and the loss is silent "
        f"({model_dir})"
    )
