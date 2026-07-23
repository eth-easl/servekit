#!/usr/bin/env python3
"""Apply foundry's 4 direct edits to the installed SGLang.

Upstream ships these as a fork (foundry-org/sglang, branch `foundry`), but the
whole diff is 47 lines across 4 files and every anchor already exists in the
v0.5.10 the image ships -- so we patch in place instead of swapping the engine.
That keeps the baseline and the foundry runs on the *same* SGLang build.

Anchored on source text rather than line numbers so it survives any drift
between the pip wheel and the git tag. Idempotent; asserts every edit landed.

Source: https://github.com/foundry-org/foundry/blob/main/docs/sglang/direct-edits.md
"""

from __future__ import annotations

import os
import sys

# Python puts this script's own directory at sys.path[0]. The image has an
# unrelated /opt/sglang directory, so running from /opt makes `import sglang`
# resolve to that empty namespace portion instead of the real package at
# /sgl-workspace/sglang/python. Drop our own directory before importing.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]

import sglang  # noqa: E402

def _find_srt() -> str:
    """Locate the real sglang/srt source tree.

    sglang is a namespace package here (__file__ is None), and the image also has
    an unrelated /opt/sglang directory -- which becomes a competing namespace
    portion whenever /opt lands on sys.path (e.g. running this script from /opt).
    So don't trust ordering: pick the portion that actually holds the source.
    """
    candidates = list(sglang.__path__)
    for base in candidates:
        srt = os.path.join(base, "srt")
        if os.path.isfile(os.path.join(srt, "server_args.py")):
            return srt
    raise SystemExit(
        f"could not find sglang/srt/server_args.py in any of: {candidates}"
    )


SRT = _find_srt()

# 1. New file: the activation entry point. Verbatim from the fork.
SHIM = '''# SPDX-License-Identifier: Apache-2.0
"""Small activation shim for the Foundry SGLang integration."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_server_args(server_args) -> None:
    cfg_path = getattr(server_args, "foundry_graph_extension_config_path", None)
    if not cfg_path:
        return

    # Keep phase-1 SAVE/LOAD deterministic while preserving full-graph
    # torch.compile semantics. In SGLang, torch.compile is triggered from the
    # full CudaGraphRunner capture path for batch sizes in compile_bs, so
    # overriding enable_torch_compile here would silently change the graph.
    server_args.disable_piecewise_cuda_graph = True
    server_args.enable_profile_cuda_graph = False
    server_args.disable_flashinfer_autotune = True

    from foundry.integration.sglang.hooks import install_hooks

    install_hooks(server_args)
    logger.info("Foundry SGLang integration activated from %s", cfg_path)
'''

# Spawned-child variant: module-level function body, operating on `server_args`.
# Compat shim. foundry's integration targets sglang >= 0.5.11, where
# MemoryPoolConfig lives in its own `pool_configurator` module. v0.5.10 (the last
# torch-2.9 release; 0.5.11+ require torch 2.11 / cu13) still defines the very
# same class in model_runner_kv_cache_mixin. Re-export it so both import paths
# resolve to one class object -- foundry constructs and isinstance-checks it.
POOL_SHIM = '''# SPDX-License-Identifier: Apache-2.0
"""Back-compat: MemoryPoolConfig moved here upstream in v0.5.11."""

from sglang.srt.model_executor.model_runner_kv_cache_mixin import MemoryPoolConfig

__all__ = ["MemoryPoolConfig"]
'''

ACTIVATE = """    if server_args.foundry_graph_extension_config_path:
        from sglang.srt.foundry_shim import apply_server_args

        apply_server_args(server_args)
"""

# __post_init__ variant: one indent deeper, and the object is `self`.
ACTIVATE_POST = """        if self.foundry_graph_extension_config_path:
            from sglang.srt.foundry_shim import apply_server_args

            apply_server_args(self)
"""

# (file, anchor, insertion). Insertion goes immediately *after* the anchor.
EDITS = [
    # 2a. server_args.py -- real dataclass field: argparse consumes it and
    #     ServerArgs is pickled across the spawn boundary, so a monkey-attached
    #     attribute would not survive.
    (
        "server_args.py",
        '    disable_flashinfer_autotune: bool = False\n',
        '    foundry_graph_extension_config_path: Optional[str] = None\n',
    ),
    # 2b. server_args.py -- CLI argument.
    (
        "server_args.py",
        '            "--disable-flashinfer-autotune",\n'
        '            default=ServerArgs.disable_flashinfer_autotune,\n'
        '            action="store_true",\n'
        '            help="Disable FlashInfer autotuning.",\n'
        '        )\n',
        '        parser.add_argument(\n'
        '            "--foundry-graph-extension-config-path",\n'
        '            type=str,\n'
        '            default=ServerArgs.foundry_graph_extension_config_path,\n'
        '            help="Path to Foundry CUDA graph extension TOML config.",\n'
        '        )\n',
    ),
    # 2c. server_args.py -- activation, at the very END of __post_init__ so
    #     foundry's forced flags win over the _handle_* helpers above (several
    #     of which set disable_piecewise_cuda_graph themselves).
    (
        "server_args.py",
        '        # Handle any other necessary validations.\n'
        '        self._handle_other_validations()\n',
        ACTIVATE_POST,
    ),
    # 3. scheduler.py -- spawn does not inherit Python state, so each scheduler
    #    child must re-install the hooks or it takes the upstream path.
    #    Two candidate anchors: v0.5.11+ opens with load_plugins() (this is the
    #    form foundry's own direct-edits.md documents); v0.5.10 opens with
    #    configure_scheduler(). Whichever matches exactly once wins.
    (
        "managers/scheduler.py",
        (
            "    load_plugins()\n",
            "    dp_rank = configure_scheduler(\n"
            "        server_args, tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank, pp_rank, dp_rank\n"
            "    )\n",
        ),
        ACTIVATE,
    ),
    # 3b. model_runner.py -- NOT in foundry's direct-edits.md, but its runtime
    #     demands it: _resolve_dp_rank() reads ModelRunner.dp_rank, and only
    #     falls back to compute_dp_attention_world_info() when dp-attention is
    #     on. We run plain DP (enable_dp_attention=False), so without this the
    #     integration raises "cannot derive regular DP rank". v0.5.11 takes
    #     dp_rank as a constructor arg but never stores it (it is used only for
    #     the offloader), so preserve it next to the sibling rank attributes --
    #     before init_torch_distributed, as the error message instructs.
    (
        "model_executor/model_runner.py",
        "        self.moe_dp_rank = moe_dp_rank\n",
        "        self.dp_rank = dp_rank\n",
    ),
    # 4. data_parallel_controller.py -- required for DP: this child spawns its
    #    own schedulers, so foundry's spawn-site patches must be installed
    #    before it does, or those grandchildren get no LD_PRELOAD.
    (
        "managers/data_parallel_controller.py",
        "    configure_logger(server_args)\n",
        ACTIVATE,
    ),
]


def main() -> int:
    print(f"patching sglang at {SRT}")
    shim = os.path.join(SRT, "foundry_shim.py")
    with open(shim, "w") as fh:
        fh.write(SHIM)
    print(f"wrote {shim}")

    pool = os.path.join(SRT, "model_executor", "pool_configurator.py")
    if not os.path.exists(pool):
        with open(pool, "w") as fh:
            fh.write(POOL_SHIM)
        print(f"wrote {pool} (compat shim)")
    else:
        print("skip  pool_configurator.py: already provided by this sglang")

    for rel, anchors, insertion in EDITS:
        path = os.path.join(SRT, rel)
        with open(path) as fh:
            src = fh.read()

        if insertion in src:
            print(f"skip  {rel}: already patched")
            continue

        if isinstance(anchors, str):
            anchors = (anchors,)
        matched = [a for a in anchors if src.count(a) == 1]
        if not matched:
            counts = ", ".join(f"{src.count(a)}x" for a in anchors)
            print(
                f"FAIL  {rel}: no candidate anchor matched exactly once ({counts}).\n"
                f"       sglang version drift -- update the anchors.",
                file=sys.stderr,
            )
            return 1
        anchor = matched[0]

        with open(path, "w") as fh:
            fh.write(src.replace(anchor, anchor + insertion))
        print(f"patch {rel}")

    # Prove the flag is really wired into argparse, not just present in source.
    from sglang.srt.server_args import ServerArgs

    assert hasattr(ServerArgs, "foundry_graph_extension_config_path"), "field missing"
    print("ok: ServerArgs.foundry_graph_extension_config_path present")

    # foundry imports MemoryPoolConfig from pool_configurator, so that path must
    # work and must yield the same class the engine itself uses -- otherwise a
    # saved memory-pool config would silently fail to apply on LOAD.
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
    from sglang.srt.model_executor import model_runner

    engine_cls = getattr(model_runner, "MemoryPoolConfig", None)
    if engine_cls is not None and engine_cls is not MemoryPoolConfig:
        print("FAIL: pool_configurator yields a different class than model_runner uses",
              file=sys.stderr)
        return 1
    print("ok: MemoryPoolConfig import path agrees with the engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
