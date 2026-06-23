# Snapshot Format

Files are little-endian and section-tagged.

Header:

```text
magic[8] = "SNAPSHOT"
u32 version
u32 vendor
u64 captured_base
u64 region_size
u8  fixed_base_honored
u64 granularity
u32 section_count
string arch
```

Each section has:

```text
u32 tag
u64 payload_size
u32 payload_crc32
payload[payload_size]
```

Required sections:

- `SEC_ALLOC_LOG`: allocator metadata plus ordered allocation events.
- `SEC_MODULES`: module hash, image bytes, and entry names.
- `SEC_GRAPH_NODES`: graph node records, kernel launch blobs, and pointer offsets.
- `SEC_GRAPH_EDGES`: graph dependency edges.

The reader validates magic, version, duplicate/missing sections, and per-section
CRC before parsing payloads.
