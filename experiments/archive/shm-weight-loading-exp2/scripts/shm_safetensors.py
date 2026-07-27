"""Minimal safetensors header parser + tensor-view constructor.

Bypasses `safetensors.safe_open`'s path-based file-size validation, which
hugetlbfs cannot satisfy (STEP4_PLAN.md Phase 0/1: ftruncate on the only
writable hugetlbfs mount in this container only accepts exact 2 MB multiples,
and safe_open rejects a padded file with "incomplete metadata, file not fully
covered"). This reads the same on-disk format directly from a buffer we
already control (an mmap'd memfd(MFD_HUGETLB) region), so any trailing
padding is simply never referenced.

Format (https://github.com/huggingface/safetensors#format):
  bytes 0:8      u64 LE header length N
  bytes 8:8+N    UTF-8 JSON header: {name: {dtype, shape, data_offsets}, ...,
                 optional "__metadata__"}
  bytes 8+N:     raw tensor bytes, back-to-back, offsets relative to 8+N

Verified against a real shard (model-00030-of-00030.safetensors, single
lm_head.weight tensor): 8 + header_len(120) + data_offsets[1](2101346304) ==
actual file size (2101346432) exactly.
"""
import json
import struct

import torch

_DTYPE = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
    "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
    "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
}


def parse_header(buf, base_offset=0):
    """buf: anything exposing the buffer protocol (memoryview/mmap/ctypes array).

    Returns (header_without_metadata, data_start), where data_start is the
    absolute offset (from the start of buf) of byte 0 of the data section.
    """
    mv = memoryview(buf)
    n = struct.unpack_from("<Q", mv, base_offset)[0]
    header = json.loads(bytes(mv[base_offset + 8: base_offset + 8 + n]))
    header.pop("__metadata__", None)
    return header, base_offset + 8 + n


def tensor_view(buf, header, data_start, name):
    meta = header[name]
    dtype = _DTYPE[meta["dtype"]]
    shape = meta["shape"]
    start, end = meta["data_offsets"]
    raw = torch.frombuffer(buf, dtype=torch.uint8, count=end - start,
                            offset=data_start + start)
    return raw.view(dtype).view(shape) if shape else raw.view(dtype).view(())


def iter_tensors(buf, header, data_start):
    for name in header:
        yield name, tensor_view(buf, header, data_start, name)
