"""SPEC.md §5 gate 1: every loaded parameter is byte-identical to the source.

Called from a hook placed immediately AFTER model_runner's "Load weight end"
log line, so the phase timing is already computed and this cannot inflate the
number it validates. It does add wall time to the job (a full readback plus a
re-read of the source), which is why it lives outside the timed region rather
than being skipped.

Compares raw bytes, not values: `torch.equal` treats NaN as unequal and would
report a false failure on any checkpoint containing one, while a uint8 view
answers the question actually being asked -- did every byte arrive.

The failure this exists to catch is a pipelined async H2D that recycles or
frees a staging buffer while a copy is still draining. That corrupts weights
nondeterministically, so a design can pass a throughput benchmark and still be
wrong; only checking on every run catches it.
"""

import glob
import logging
import os
import time

logger = logging.getLogger(__name__)


def _enabled():
    return os.environ.get("SGLANG_EXP2_VERIFY", "0").lower() not in ("0", "false", "")


def verify_sharded_state(model, model_path, rank):
    """Returns (checked, mismatches, elapsed_s). Raises nothing; logs loudly."""
    import torch
    from safetensors import safe_open

    params = {name: p.data for name, p in model.named_parameters()}
    params.update({name: b for name, b in model.named_buffers()})

    files = sorted(glob.glob(os.path.join(model_path, f"model-rank-{rank}-part-*.safetensors")))
    checked, mismatches = 0, []
    t0 = time.perf_counter()

    for path in files:
        with safe_open(path, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118
                src = f.get_tensor(key)
                dst = params.get(key)
                if dst is None:
                    mismatches.append(f"{key}: not present in model")
                    continue
                # The loader narrows into a padded parameter (LoRA case); mirror
                # that here or we would compare against padding.
                for dim, size in enumerate(src.shape):
                    if size < dst.shape[dim]:
                        dst = dst.narrow(dim, 0, size)
                got = dst.contiguous().cpu().view(torch.uint8)
                want = src.contiguous().view(torch.uint8)
                if got.shape != want.shape or not torch.equal(got, want):
                    mismatches.append(key)
                checked += 1

    return checked, mismatches, time.perf_counter() - t0


def maybe_verify(model_runner):
    """Hook entry point. Never raises out into the engine unless weights are
    actually wrong -- an infrastructure problem here must not be reported as a
    passing run, but must also not be confused with corruption."""
    if not _enabled():
        return
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        path = model_runner.model_config.model_path
        checked, mismatches, elapsed = verify_sharded_state(
            model_runner.model, path, rank
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("EXP2_VERIFY ERROR rank=? could not run: %r", exc)
        return

    if mismatches:
        logger.error(
            "EXP2_VERIFY FAIL rank=%d checked=%d mismatched=%d first=%s elapsed=%.2fs",
            rank, checked, len(mismatches), mismatches[:5], elapsed,
        )
        raise RuntimeError(
            f"exp2 bit-exactness gate failed on rank {rank}: "
            f"{len(mismatches)}/{checked} tensors differ from the source"
        )

    logger.info(
        "EXP2_VERIFY PASS rank=%d checked=%d tensors bit-exact elapsed=%.2fs",
        rank, checked, elapsed,
    )
