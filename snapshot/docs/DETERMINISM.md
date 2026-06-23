# Determinism Notes

The prototype treats device virtual addresses as `region_base + offset`.
Allocation offsets are deterministic because the allocator rounds every request
to the backend granularity and appends an ordered `AllocEvent` log.

Restore asks the backend to reserve the captured base as a hint. If the backend
returns a different base, graph IR is relocated by the constant delta:

```text
delta = restored_base - captured_base
```

Known pointer offsets recorded during graph capture are patched first. Blind
8-byte scanning is used only when no known offsets patched a kernel parameter
blob. Values are patched only when they fall inside the captured VMM region.

The host STUB backend intentionally shifts the second reservation so unit tests
exercise relocation without requiring ROCm or CUDA hardware.
