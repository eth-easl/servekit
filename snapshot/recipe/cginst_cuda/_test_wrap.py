# _test_wrap.py — N5b Task 4 host-side unit test for cg_meta_cuda.
#
# Verifies the two port deltas from the HIP prototype WITHOUT vLLM:
#   1. wrap_device_ptr builds a DLPack capsule with device type kDLCUDA (2) and
#      a torch tensor viewing the exact device pointer.
#   2. _serialize / _reconstruct round-trip nested tensor structures through
#      JSON; under Δ=0 (live_base + off == recorded data_ptr) the reconstructed
#      tensor views the same storage.
#
# Run: python3 _test_wrap.py   (from this dir, or with it on PYTHONPATH).
# Needs torch + CUDA. Prints CG_META_WRAP_OK=1 on success.
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from cg_meta_cuda import _serialize, _reconstruct, wrap_device_ptr  # noqa: E402


def main() -> int:
    # --- (1) wrap_device_ptr: exact pointer / shape / dtype / values ---
    t0 = torch.arange(16, device="cuda", dtype=torch.bfloat16)
    t1 = wrap_device_ptr(t0.data_ptr(), [16], torch.bfloat16)
    assert t1.data_ptr() == t0.data_ptr(), (
        f"data_ptr mismatch: {t1.data_ptr()} != {t0.data_ptr()}"
    )
    assert list(t1.shape) == [16], f"shape mismatch: {t1.shape}"
    assert t1.dtype == torch.bfloat16, f"dtype mismatch: {t1.dtype}"
    assert torch.equal(t1, t0), "wrapped tensor values differ from source"
    print(f"wrap_device_ptr OK: ptr=0x{t0.data_ptr():x} shape={list(t0.shape)} "
          f"dtype={t0.dtype}")

    # --- (2) round-trip a nested {a: [t0, t0]} through JSON ---
    # region_base=0 → off == data_ptr; reconstruct with live_base=0 so
    # ptr = 0 + off == t0.data_ptr() (the Δ=0 case).
    meta = _serialize({"a": [t0, t0]}, region_base=0)
    meta_js = json.loads(json.dumps(meta))  # must be JSON-serializable
    off0 = meta_js["items"]["a"]["items"][0]["off"]
    assert off0 == t0.data_ptr(), (
        f"offset mismatch (region_base=0): off=0x{off0:x} != data_ptr=0x{t0.data_ptr():x}"
    )
    got = _reconstruct(meta_js, live_base=0)
    assert isinstance(got, dict) and isinstance(got["a"], list), "structure lost"
    a0, a1 = got["a"]
    assert a0.data_ptr() == t0.data_ptr(), (
        f"reconstructed ptr 0x{a0.data_ptr():x} != 0x{t0.data_ptr():x}"
    )
    assert a1.data_ptr() == t0.data_ptr(), "second list element ptr wrong"
    assert torch.equal(a0, t0), "reconstructed tensor values differ"
    print(f"round-trip OK: nested dict/list reconstructed viewing t0 storage "
          f"(Δ=0, ptr=0x{a0.data_ptr():x})")

    # --- (3) Δ≠0 simulation: live_base shifted by +4096 reconstructs a shifted ptr ---
    meta_shift = _serialize(t0, region_base=t0.data_ptr())  # off == 0
    got_shift = _reconstruct(meta_shift, live_base=t0.data_ptr())
    assert got_shift.data_ptr() == t0.data_ptr(), (
        f"shifted reconstruct ptr 0x{got_shift.data_ptr():x} != 0x{t0.data_ptr():x}"
    )
    print(f"delta-relocate OK: off=0 + live_base=0x{t0.data_ptr():x} -> "
          f"0x{got_shift.data_ptr():x}")

    print("CG_META_WRAP_OK=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
