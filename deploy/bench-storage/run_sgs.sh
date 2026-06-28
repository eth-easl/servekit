#!/bin/bash
# run_sgs.sh — storage characterization for sgs-gpu07 (ETH, run directly via ssh).
# Compares node-local NVMe (/tmp, on system-root NVMe) vs the NFS tiers (home +
# scratch). The NFS mounts have NO nconnect (verified), so this directly tests
# fast-loader §1 (single TCP-stream cap) vs §5 (server wall) and the NVMe
# staging ceiling (§3).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BC="$HERE/bench_core.sh"

echo "############################################################"
echo "# SGS-GPU07 STORAGE CHARACTERIZATION ($(date --iso-8601=seconds))"
echo "############################################################"
hostname; uname -r
echo
echo "--- GPUs (type + PCIe link = the H2D ceiling) ---"
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv
echo "--- NVLink topology (GPU<->GPU) ---"
nvidia-smi topo -m 2>/dev/null | head -8
echo "--- CPU / MEM ---"
lscpu | grep -E "Model name|^CPU\(s\)"
free -g | head -2
echo "--- node-local block devices (§3 staging candidates) ---"
lsblk -d -o NAME,SIZE,MODEL,ROTA 2>/dev/null | head
echo "--- sunrpc slot table (NFS RPCs in flight) ---"
echo "tcp_max_slot_table_entries=$(cat /proc/sys/sunrpc/tcp_max_slot_table_entries 2>/dev/null)"
echo "--- /tmp backing device (node-local NVMe for this run) ---"
df -h /tmp | tail -1
echo

# 8 × 1GiB dense (non-sparse) synthetic files per tier.
NF=8; SZM=1024
mkfiles(){ # $1 = dir  -> prints file paths to STDOUT (status msgs go to STDERR)
  local d="$1" i f
  mkdir -p "$d" || return 1
  for ((i=0;i<NF;i++)); do
    f="$d/syn-$i.bin"
    if [ ! -f "$f" ] || [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" -ne $((SZM*1024*1024)) ]; then
      echo "  (creating $f)" >&2
      dd if=/dev/zero of="$f" bs=1M count=$SZM status=none oflag=direct 2>/dev/null \
        || dd if=/dev/zero of="$f" bs=1M count=$SZM status=none
    fi
    echo "$f"
  done
}

# Tier 1: node-local NVMe via /tmp (on system-root NVMe) — the §3 staging ceiling
echo "########## TIER: NVMe /tmp (node-local) ##########"
mapfile -t F_NVME < <(mkfiles /tmp/xiayao_bench)
bash "$BC" "NVMe-/tmp" "${F_NVME[@]}"

# Tier 2: NFS scratch (where HF cache lives; HF downloads land here)
echo; echo "########## TIER: NFS scratch (/home/xiayao/.cache) ##########"
mapfile -t F_NFSS < <(mkfiles /home/xiayao/.cache/bench_sgs)
bash "$BC" "NFS-scratch" "${F_NFSS[@]}"

# Tier 3: NFS home
echo; echo "########## TIER: NFS home (/home/xiayao) ##########"
mapfile -t F_NFSH < <(mkfiles /home/xiayao/bench_sgs)
bash "$BC" "NFS-home" "${F_NFSH[@]}"

echo; echo "############################################################"
echo "# DONE ($(date --iso-8601=seconds))"
echo "############################################################"
