#!/bin/bash
# exp2 first measurement: what per-rank /dev/shm -> GPU rate is actually
# available to 4 concurrent ranks, and does NUMA placement matter.
#
# SPEC.md needs 7.06 GB/s per rank sustained. Today's stock path delivers
# 35.3/11.08 = 3.2 GB/s. This says whether the gap is closable on the host
# side before any engine patch is written.
#
#   sbatch shm-weight-loading-exp2/scripts/shm_h2d_bench.sbatch
#
#SBATCH --job-name=e2-h2dbench
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:35:00
#SBATCH --output=shm-weight-loading-exp2/results/%x-%j.out

set -uo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
EXP_DIR="${SLURM_SUBMIT_DIR}/shm-weight-loading-exp2"
BENCH="${EXP_DIR}/scripts/shm_h2d_bench.py"
NODE="$(hostname)"
JSON="${EXP_DIR}/results/h2dbench-${SLURM_JOB_ID}-${NODE}.jsonl"

echo "job=${SLURM_JOB_ID} node=${NODE} started=$(date --iso-8601=seconds)"

INNER=$(cat <<EOF
set -uo pipefail

echo "=== node topology ==="
lscpu | grep -Ei '^(Model name|Socket|NUMA|Thread|Core)' || true
echo
(numactl --hardware 2>/dev/null || echo "numactl unavailable") | head -20
echo
nvidia-smi topo -m 2>/dev/null | head -12 || true
echo
for d in /sys/bus/pci/devices/*/; do
  nn=\$(cat "\$d/numa_node" 2>/dev/null || echo -1)
  cls=\$(cat "\$d/class" 2>/dev/null || echo)
  [ "\$cls" = "0x030200" ] && echo "GPU \$(basename \$d) numa_node=\$nn"
done
echo
df -h /dev/shm
echo

echo "=== sweep (4 concurrent ranks, 8 GiB each) ==="
run() { python3 "${BENCH}" --json "${JSON}" "\$@" || echo "  ^ FAILED: \$*"; }

# reference points
run --mode pageable --threads 16 --dir /dev/shm/exp2bench
run --mode dma_only --threads 16 --dir /dev/shm/exp2bench

# thread sweep on the staged path -- the exp1 microbench showed this is
# worth more than any other single knob
run --mode pinned --threads 8  --dir /dev/shm/exp2bench
run --mode pinned --threads 16 --dir /dev/shm/exp2bench
run --mode pinned --threads 32 --dir /dev/shm/exp2bench

# chunk size and pipeline depth at the best-guess thread count
run --mode pinned --threads 16 --chunk-mib 256  --dir /dev/shm/exp2bench
run --mode pinned --threads 16 --chunk-mib 1024 --dir /dev/shm/exp2bench
run --mode pinned --threads 16 --depth 3        --dir /dev/shm/exp2bench

# NUMA: separate dir so source pages are first-touched while bound
run --mode pinned   --threads 16 --numa --dir /dev/shm/exp2bench_numa
run --mode pageable --threads 16 --numa --dir /dev/shm/exp2bench_numa
run --mode dma_only --threads 16 --numa --dir /dev/shm/exp2bench_numa

rm -rf /dev/shm/exp2bench /dev/shm/exp2bench_numa
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER}"

echo
echo "=== target: 7.06 GB/s per rank (35.3 GB / 5 s); today's stock path = 3.2 GB/s ==="
echo "jsonl=${JSON}"
echo "finished=$(date --iso-8601=seconds)"
