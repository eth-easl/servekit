#!/bin/bash
#SBATCH --job-name=bench-loader
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/xyao/bench-loader/logs/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/bench-loader/logs/%x-%j.err
#
# bristen_bench.sbatch — storage characterization on a CSCS bristen compute node.
# Reads the REAL GLM-4.7-Flash safetensors (same files beverin measured 93s on)
# from /capstor Lustre at 1..32 parallel streams to expose the single-OST cap
# (stripe_count=1) vs the parallel-FS ceiling. Also: GPU type, node-local disk.

set -uo pipefail
HERE="/capstor/scratch/cscs/xyao/bench-loader"
BC="$HERE/bench_core.sh"

echo "############################################################"
echo "# BRISTEN STORAGE CHARACTERIZATION ($(date --iso-8601=seconds))"
echo "############################################################"
echo "job=${SLURM_JOB_ID:-none} host=$(hostname) node=${SLURM_NODELIST:-$(hostname)}"
echo

echo "--- GPU type (confirm A100 vs H100) + PCIe link (H2D ceiling) ---"
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv 2>&1 | head
echo "--- NVLink topology ---"
nvidia-smi topo -m 2>&1 | head -8
echo "--- CPU / MEM ---"
lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)"
free -g | head -2
echo "--- node-local block devices (§3 staging candidates on a Cray node) ---"
lsblk -d -o NAME,SIZE,MODEL,ROTA 2>/dev/null | head
echo "--- Lustre client stats / RPC in flight ---"
cat /proc/fs/lustre/llite/*/stats 2>/dev/null | grep -E "read_bytes|stats" | head -3 || echo "(no llite stats)"
echo

MODEL=/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash

echo "########## TIER: Lustre /capstor/store (REAL model shards) ##########"
mapfile -t SHARDS < <(ls "$MODEL"/model-*.safetensors 2>/dev/null | sort -V)
echo "shard_count=${#SHARDS[@]}  (GLM-4.7-Flash, same files beverin measured)"
# Use the first 32 shards for the parallel sweep (one per potential stream).
bash "$BC" "Lustre-store-GLM47" "${SHARDS[@]}"

echo
echo "########## TIER: Lustre /capstor/scratch (synthetic) ##########"
SCR=/capstor/scratch/cscs/xyao/bench-loader/syn
mkdir -p "$SCR"
SF=()
for i in $(seq 0 7); do
  f="$SCR/syn-$i.bin"
  if [ ! -f "$f" ] || [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" -ne 1073741824 ]; then
    echo "  (creating $f)"
    lfs setstripe -c 4 "$f" 2>/dev/null   # give scratch files a fighting chance (4 OSTs)
    dd if=/dev/zero of="$f" bs=1M count=1024 status=none
  fi
  SF+=("$f")
done
bash "$BC" "Lustre-scratch(stripe4)" "${SF[@]}"

echo
echo "############################################################"
echo "# DONE ($(date --iso-8601=seconds))"
echo "# outputs: /capstor/scratch/cscs/xyao/bench-loader/logs/"
echo "############################################################"
