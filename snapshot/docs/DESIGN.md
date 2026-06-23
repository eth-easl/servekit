# Snapshot Prototype Status

The root [DESIGN.md](../DESIGN.md) is the source design. This implementation
currently provides:

- C++17 HAL interfaces and opaque handles.
- Deterministic VMM allocator with replayable allocation event logs.
- Pointer relocation by known offsets plus bounded blind scan fallback.
- Little-endian sectioned snapshot format with per-section CRC32.
- Host STUB backend for local tests and synthetic CLI workflows.
- HIP/CUDA backend scaffolds behind the HAL.
- Slurm/RCC recipe files for ROCm builds on beverin.

The GPU graph introspection/rebuild path is intentionally still gated. The HIP
backend contains VMM/module/stream entry points, but graph IR extraction and
rebuild require the next milestone before real two-process GPU replay can pass.
