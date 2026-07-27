"""Write a TP-sharded checkpoint (drop-in for --load-format sharded_state).

Why this exists instead of upstream examples/runtime/engine/save_sharded_state.py:
that example is broken against this SGLang. The chain is

    Engine.save_sharded_model(**kwargs)
      -> collective_rpc(method, **kwargs)   # parameters = {path, pattern, max_size}
      -> Scheduler.handle_rpc_request       # func(**recv_req.parameters)
      -> SchedulerUpdateWeightsMixin.save_sharded_model(self, params)   # ONE dict

so expanding {path, pattern, max_size} as keywords hits a handler that accepts a
single positional `params`, giving:

    got an unexpected keyword argument 'path'

Sending the dict nested under `params` makes the expansion produce exactly the
one argument the handler wants. No engine patch needed.
"""

import dataclasses
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


def main(args):
    engine_args = ServerArgs.from_cli_args(args)
    model_path = engine_args.model_path
    if not Path(model_path).is_dir():
        raise ValueError("model path must be a local directory")

    llm = Engine(**dataclasses.asdict(engine_args))
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # The nested-dict form; see module docstring.
    llm.collective_rpc(
        "save_sharded_model",
        params={
            "path": args.output,
            "pattern": args.file_pattern,
            "max_size": args.max_file_size,
        },
    )

    # Copy metadata (config.json, tokenizer, ...) so the output is a drop-in
    # --model-path. Weight files are excluded; the shards replace them.
    #
    # Directories are skipped entirely. The upstream example copytree's them,
    # which for Llama repos drags in original/ (132 GB of consolidated .pth) --
    # doubling the checkpoint and, worse, doubling the /dev/shm stage time at
    # serve time for bytes nothing reads.
    for file in os.listdir(model_path):
        src = os.path.join(model_path, file)
        if os.path.isdir(src):
            print(f"  skipping directory {file}/")
            continue
        if os.path.splitext(file)[1] in (".bin", ".pt", ".safetensors"):
            continue
        shutil.copy(src, os.path.join(args.output, file))

    print(f"saved sharded checkpoint to {args.output}")
    llm.shutdown()


if __name__ == "__main__":
    main(parser.parse_args())
