#!/bin/bash
# Why does dd collapse on the sick OST while fastsafetensors does not?
#
# OST 8 answers a read RPC in ~480 ms regardless of size (a bs sweep at QD=1
# gives 2.1/7.7/33.4/81.6 MB/s at bs=1/4/16/64M -- throughput tracks bs exactly,
# so the cost is per-REQUEST, not per-byte). The client is allowed
# max_rpcs_in_flight=68 to that OST; max_pages_per_rpc=4096 makes a 16 MiB read
# exactly ONE RPC. So `dd bs=16M` keeps exactly 1 of its 68 slots busy and earns
# 16 MiB / 0.48 s = 33 MB/s. Anything that keeps N RPCs in flight earns N times
# that.
#
# This job tests that claim on the ONE file that lives on OST 8, reading it two
# ways on the same node minutes apart, and -- the point -- measures the actual
# RPC queue depth of each reader from Lustre's own counters
# (osc.*OST0008*.rpc_stats "read RPCs in flight" histogram), which is not an
# inference but a direct observation of what each reader asked the OST for.
#
#   sbatch experiments/lustre-contention-exp/scripts/ost8_queue_depth.sbatch
#
#SBATCH --job-name=ost8-qd
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --output=experiments/lustre-contention-exp/results/%x-%j.out

set -uo pipefail
export PATH=$PATH:/usr/sbin

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SICK="${MODEL}/model-00018-of-00030.safetensors"   # OST 8
CTRL="${MODEL}/model-00017-of-00030.safetensors"   # OST 16, healthy control
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
RES="${SLURM_SUBMIT_DIR}/experiments/lustre-contention-exp/results"
NODE="$(hostname)"

echo "job=${SLURM_JOB_ID} node=${NODE} started=$(date --iso-8601=seconds)"

# The OSC param path is per-MOUNT, and /capstor/store is a different client
# instance from /capstor/scratch -- resolving it by hand once cost a wrong
# reading already. Derive it from where the model actually lives.
FSNAME="$(lfs getname "${MODEL}" 2>/dev/null | awk '{print $1}' | head -1)"
INST="${FSNAME##*-}"
P8="osc.capstor-OST0008-osc-${INST}"
echo "store mount instance=${INST}"
echo "ost of sick shard : $(lfs getstripe -i "${SICK}")"
echo "ost of ctrl shard : $(lfs getstripe -i "${CTRL}")"
echo "max_rpcs_in_flight: $(lctl get_param -n ${P8}.max_rpcs_in_flight 2>/dev/null)"
echo "max_pages_per_rpc : $(lctl get_param -n ${P8}.max_pages_per_rpc 2>/dev/null)"

# "read RPCs in flight" histogram: rows are queue depth, cols are rpcs/%/cum%.
# This is the whole answer -- dd should pile up at depth 1, fastsafetensors deep.
qd_hist() {
  lctl get_param -n "${P8}.rpc_stats" 2>/dev/null \
    | awk '/^read RPCs in flight:/{next}
           /rpcs in flight/{f=1; next}
           f && /^[0-9]+:/{print "      depth "$0}
           f && /^$/{exit}' | head -12
}

echo
echo "############ A. dd, bs=16M, QD=1 (the probe that reported 0.6 GB/s)"
lctl set_param -n "${P8}.rpc_stats"=clear >/dev/null 2>&1
t0=$(date +%s.%N)
dd if="${SICK}" of=/dev/null bs=16M count=64 iflag=direct status=none
t1=$(date +%s.%N)
awk -v s=$t0 -v e=$t1 'BEGIN{printf "  1024 MiB in %.1f s -> %.1f MB/s\n", e-s, 1024/(e-s)}'
echo "  read-RPCs-in-flight histogram on OST 8:"; qd_hist

echo
echo "############ B. same file, same O_DIRECT, N parallel readers (queue depth only)"
printf "  %8s %12s\n" "readers" "MB/s"
for n in 1 2 4 8 16 32; do
  per=$(( 64 / n ))            # 64 x 16MiB = 1024 MiB total, evenly divisible
  t0=$(date +%s.%N)
  for ((i=0;i<n;i++)); do
    dd if="${SICK}" of=/dev/null bs=16M skip=$((i*per)) count=$per iflag=direct status=none &
  done
  wait
  t1=$(date +%s.%N)
  awk -v s=$t0 -v e=$t1 -v n=$n 'BEGIN{printf "  %8s %12.1f\n", n, 1024/(e-s)}'
done

echo
echo "############ C. fastsafetensors reading THE SAME sick shard"
# The container mounts only /capstor and /iopsstor -- a script in the host's
# /tmp is invisible inside it (this cost one run). Stage it on a mounted path.
PY_SRC="${RES}/fst_one_${SLURM_JOB_ID}.py"
cat > "${PY_SRC}" <<'PY'
import inspect, os, sys, time, torch
from fastsafetensors import SafeTensorsFileLoader, SingleGroup
import fastsafetensors as F

shard = sys.argv[1]
print(f"  fastsafetensors {getattr(F,'__version__','?')}")
# The defaults ARE the answer to "how deep is its queue" -- print them rather
# than assume: bbuf_size_kb is the bounce buffer, max_copier_threads the fan-out.
sig = inspect.signature(SafeTensorsFileLoader.__init__)
print("  SafeTensorsFileLoader defaults:")
for k, v in sig.parameters.items():
    if v.default is not inspect.Parameter.empty:
        print(f"      {k} = {v.default}")

kw = {}
if "nogds" in sig.parameters:
    kw["nogds"] = True          # bristen has no nvidia_fs; force the CPU path
n = os.path.getsize(shard)
t0 = time.time()
loader = SafeTensorsFileLoader(SingleGroup(), torch.device("cuda:0"), **kw)
loader.add_filenames({0: [shard]})
fb = loader.copy_files_to_device()
dt = time.time() - t0
print(f"  RESULT fastsafetensors: {n/1e6:.0f} MB in {dt:.1f} s -> {n/dt/1e6:.1f} MB/s")
try:
    fb.close(); loader.close()
except Exception:
    pass
PY

# Install and run in ONE srun: each srun is a fresh container and the pip install
# lives in its ephemeral overlay, so a separate install srun is simply lost (this
# cost a run). pip fetches over the network, not from capstor, so it does not
# disturb OST 8's counters.
lctl set_param -n "${P8}.rpc_stats"=clear >/dev/null 2>&1
srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=1 \
  --environment="${EDF}" bash -c "pip install fastsafetensors -q 2>&1 | tail -1; python '${PY_SRC}' '${SICK}'"
echo "  read-RPCs-in-flight histogram on OST 8:"; qd_hist
rm -f "${PY_SRC}"

echo
echo "finished=$(date --iso-8601=seconds)"
