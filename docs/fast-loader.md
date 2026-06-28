# Cheap mount/protocol tuning. 

The single highest-yield knob is nconnect — by default a Linux NFS client multiplexes everything over one TCP connection per client-server pair, so a single mount is capped by one stream's throughput regardless of how many threads you fork. nconnect=8 (or 16) opens parallel connections and often gives a near-linear bump until you hit the server. Pair that with max rsize/wsize (1 MB on most servers), vers=4.2, noatime, and bumping sunrpc.tcp_max_slot_table_entries so the client can keep enough RPCs in flight. Raising NFS readahead helps purely sequential reads.

# Fix the read pattern. 

mmap over NFS is a classic trap: safetensors' lazy mmap path looks elegant but on NFS each page fault becomes a small synchronous read, and you get latency-bound trickle instead of bandwidth. For NFS specifically you're usually better off doing explicit large sequential reads (with posix_fadvise(WILLNEED/SEQUENTIAL)) into a buffer and then moving to GPU, and reading distinct shards from parallel threads/processes so you actually exploit nconnect. If you're double-buffering into the page cache pointlessly, O_DIRECT can help, though for repeated loads you may want the page cache.

# Kill the N× duplication. This is often the real story at scale. If you have 64 ranks each reading the full checkpoint, you're demanding 64× the bytes out of one server — no amount of client tuning saves you. Two standard fixes: read once per node (or once per data-parallel group) and broadcast over NCCL/the interconnect, which is dramatically faster than NFS; or stage the weights to node-local NVMe on first touch so subsequent loads (and restarts) hit local disk, not the network. A caching layer (JuiceFS, Alluxio, or even a manual rsync-to-scratch) formalizes this.

# Move fewer bytes.

Streaming loaders like CoreWeave's tensorizer or the run:ai model streamer issue many concurrent reads straight into GPU memory and overlap I/O with H2D copy, which tends to beat the naive load_state_dict path. GPUDirect Storage (cuFile) over NFS-RDMA can bypass the CPU bounce buffer entirely if your hardware supports it. And the cheapest byte is the one you don't send: quantization, or delta-compression against a shared base in the multi-tenant fine-tuned case — exactly the DeltaZip-style redundancy exploitation — cuts the NFS transfer proportionally, which is often a bigger win than any transport tuning.
# When NFS is genuinely the wall. 

If the aggregate demand exceeds a single server's NIC, no client trick fixes it — you need horizontal scale-out: pNFS (vers=4.1) with multiple data servers, or a parallel filesystem (Lustre, GPFS, BeeGFS, WekaFS), or the fan-out caching approach above so the steady-state reads never touch the central server at all.

