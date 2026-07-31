"""Write a TP-sharded checkpoint (drop-in for --load-format sharded_state).

Fork of experiments/clariden-loading-exp/scripts/shared/save_sharded_state_fixed.py,
adding --servekit-resolved-out and the stale-index skip.
"""

import dataclasses
import json
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

from sglang import Engine, ServerArgs

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

    llm = Engine(**dataclasses.asdict(engine_args))
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Flattening these hits a handler that takes one positional `params` dict.
    llm.collective_rpc(
        "save_sharded_model",
        params={
            "path": args.output,
            "pattern": args.file_pattern,
            "max_size": args.max_file_size,
        },
    )

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
