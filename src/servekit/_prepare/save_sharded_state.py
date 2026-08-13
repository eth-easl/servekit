"""Write a TP-sharded checkpoint (drop-in for --load-format sharded_state).

Fork of experiments/clariden-loading-exp/scripts/shared/save_sharded_state_fixed.py,
adding --servekit-resolved-out and the stale-index skip.

A TP size larger than one node's GPU count needs one of these per node, with
--nnodes/--node-rank/--dist-init-addr set. On node_rank>0 SGLang's Engine
constructor never returns -- it joins its scheduler processes, which is what
keeps the worker alive to answer the head's RPC. The guard below only matters if
a future SGLang lets it return.
"""

import dataclasses
import inspect
import json
import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

from sglang import Engine, ServerArgs

from servekit import quant_guard
from servekit._shim import wait_for_writes

parser = ArgumentParser()
ServerArgs.add_cli_args(parser)
parser.add_argument(
    "--output", "-o", required=True, type=str, help="path to output checkpoint"
)
parser.add_argument(
    "--file-pattern", type=str, default=None, help="string pattern of saved filenames"
)
parser.add_argument(
    "--max-file-size",
    type=int,
    default=5 * 1024**3,
    help="max size (in bytes) of each safetensors file",
)
parser.add_argument("--servekit-resolved-out", type=str, default=None)


def resolved(llm, engine_args) -> dict:
    """What the engine decided, not what argv asked for.

    `--dtype auto` is why: ServerArgs keeps the string, only ModelConfig turns it
    into bfloat16. Out of reach, report the unresolved value -- "auto" is honest,
    a guessed bfloat16 over a float16 checkpoint is not.
    """
    import sglang

    model_config = getattr(getattr(llm, "tokenizer_manager", None), "model_config", None)

    dtype = engine_args.dtype
    if dtype == "auto":
        resolved_dtype = getattr(model_config, "dtype", None)
        if resolved_dtype is not None:
            dtype = str(resolved_dtype).replace("torch.", "")

    quantization = engine_args.quantization
    if quantization is None:
        quantization = getattr(model_config, "quantization", None)

    return {
        "engine": "sglang",
        "engine_version": getattr(sglang, "__version__", "unknown"),
        "tp_size": engine_args.tp_size,
        "pp_size": getattr(engine_args, "pp_size", 1),
        "dp_size": getattr(engine_args, "dp_size", 1),
        "dtype": dtype,
        "quantization": quantization,
    }


def main(args):
    engine_args = ServerArgs.from_cli_args(args)
    model_path = engine_args.model_path
    if not Path(model_path).is_dir():
        raise ValueError("model path must be a local directory")

    node_rank = getattr(engine_args, "node_rank", 0)

    # Before the Engine, so an unsupported checkpoint costs a second rather than
    # a full load followed by a dump nobody can trust.
    unsupported = quant_guard.check_dir(
        Path(model_path),
        quantization=engine_args.quantization,
        moe_runner_backend=getattr(engine_args, "moe_runner_backend", None),
        moe_a2a_backend=getattr(engine_args, "moe_a2a_backend", None),
    )
    if unsupported:
        print(quant_guard.refusal(unsupported, model_path), file=sys.stderr)
        raise SystemExit(2)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    llm = Engine(**dataclasses.asdict(engine_args))

    if node_rank != 0:
        print(f"node_rank={node_rank}: engine constructor returned; nothing to issue here")
        return

    payload = {
        "path": args.output,
        "pattern": args.file_pattern,
        "max_size": args.max_file_size,
    }
    # The handler took one `params` dict up to v0.5.12 and flat kwargs from
    # v0.5.13, where the dict moved a layer down into WeightUpdater. Sending the
    # wrong shape surfaces as KeyError 'path' after a full model load.
    from sglang.srt.managers.scheduler import Scheduler

    if "params" in inspect.signature(Scheduler.save_sharded_model).parameters:
        payload = {"params": payload}
    llm.collective_rpc("save_sharded_model", **payload)

    pp_size = getattr(engine_args, "pp_size", 1)
    if pp_size > 1:
        # The rpc reply comes from pp0/tp0 before later stages have even received
        # the request, so without this the copy below exits out from under them.
        wait_for_writes(args.output, pp_size, engine_args.tp_size)

    for file in os.listdir(model_path):
        src = os.path.join(model_path, file)
        if os.path.isdir(src):
            print(f"  skipping directory {file}/")
            continue
        if os.path.splitext(file)[1] in (".bin", ".pt", ".safetensors"):
            continue
        if file.endswith(".index.json"):
            print(f"  skipping stale weight index {file}")
            continue
        shutil.copy(src, os.path.join(args.output, file))

    if args.servekit_resolved_out:
        with open(args.servekit_resolved_out, "w") as f:
            json.dump(resolved(llm, engine_args), f)

    print(f"saved sharded checkpoint to {args.output}")
    llm.shutdown()


if __name__ == "__main__":
    main(parser.parse_args())
