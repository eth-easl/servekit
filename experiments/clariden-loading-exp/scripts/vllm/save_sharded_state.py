"""Write a TP-sharded checkpoint with vLLM (drop-in for --load-format sharded_state).

The vLLM counterpart of scripts/shared/save_sharded_state_fixed.py, and NOT
interchangeable with it. vLLM's model carries four attention scale parameters
per layer -- _k_scale, _v_scale, _q_scale, _prob_scale -- that SGLang never
writes, so vLLM's ShardedStateLoader rejects an SGLang-written checkpoint with
`Missing keys (...) in loaded state!` (320 of them for the 80-layer 70B; job
2918412). One presharded checkpoint per engine is therefore mandatory, not a
tidiness preference.

Engine args are passed explicitly rather than via EngineArgs.add_cli_args: only
the handful that must match the serve command matter, and the sbatch reads all
of them from scripts/shared/models.sh so the two cannot drift.
"""

import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

from vllm import LLM

parser = ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--output", "-o", required=True)
parser.add_argument("--tensor-parallel-size", type=int, required=True)
parser.add_argument("--max-model-len", type=int, required=True)
parser.add_argument("--gpu-memory-utilization", type=float, required=True)
parser.add_argument("--max-file-size", type=int, default=5 * 1024**3)
parser.add_argument("--file-pattern", default=None)


def save(llm, kwargs):
    """Call whichever save path this vLLM build exposes.

    The entry point moved between releases (LLM.collective_rpc, then
    LLMEngine.collective_rpc, and model_executor.save_sharded_state before
    that). Trying them in order beats pinning to one and finding out inside a
    4-GPU allocation; the winner is printed so the run record says which ran.
    """
    attempts = [
        ("LLM.collective_rpc",
         lambda: llm.collective_rpc("save_sharded_state", kwargs=kwargs)),
        ("LLMEngine.collective_rpc",
         lambda: llm.llm_engine.collective_rpc("save_sharded_state", kwargs=kwargs)),
        ("model_executor.save_sharded_state",
         lambda: llm.llm_engine.model_executor.save_sharded_state(**kwargs)),
    ]
    errors = []
    for name, fn in attempts:
        try:
            fn()
        except (AttributeError, TypeError) as e:
            errors.append(f"  {name}: {type(e).__name__}: {e}")
            continue
        print(f"saved via {name}")
        return
    raise RuntimeError("no working save_sharded_state entry point:\n" + "\n".join(errors))


def main(args):
    if not Path(args.model_path).is_dir():
        raise ValueError("model path must be a local directory")

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        disable_custom_all_reduce=True,
    )
    Path(args.output).mkdir(parents=True, exist_ok=True)

    save(llm, dict(path=args.output, pattern=args.file_pattern,
                   max_size=args.max_file_size))

    # Copy metadata (config.json, tokenizer, ...) so the output is a drop-in
    # model path. Weight files are excluded; the shards replace them.
    #
    # Directories are skipped entirely: for Llama repos copying them drags in
    # original/ (132 GB of consolidated .pth), doubling the checkpoint and the
    # /dev/shm stage time at serve time for bytes nothing reads.
    for file in os.listdir(args.model_path):
        src = os.path.join(args.model_path, file)
        if os.path.isdir(src):
            print(f"  skipping directory {file}/")
            continue
        if os.path.splitext(file)[1] in (".bin", ".pt", ".safetensors"):
            continue
        shutil.copy(src, os.path.join(args.output, file))

    print(f"saved sharded checkpoint to {args.output}")


if __name__ == "__main__":
    main(parser.parse_args())
