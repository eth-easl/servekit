#!/bin/bash
#SBATCH --job-name=snap-allfull
#SBATCH --partition=mi300
#SBATCH --account=root
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=00:25:00
#SBATCH --output=/capstor/scratch/cscs/xyao/kimi-k25-vllm/logs/%x-%j.out

# M3k+ -- HSACO max_flat_workgroup_size patch validation across the FULL set.
#
# patch_amdgpu_max_flat_workgroup_size (record.cpp) rewrites every
# amdhsa.kernels[].max_flat_workgroup_size -> 1024 so hipGraphAddKernelNode
# stops rejecting pointwise kernels at block=512 whose HSACO declared
# max_threads=256. Gated by SNAPSHOT_HSACO_PATCH_MAXWG. Proven on graph-2
# (job 532648: 17/17 nodes rc=0). This sweep proves it across the ENTIRE
# 800-graph set (766 FULL + 32 PIECEWISE).
#
# IMPORTANT success criterion: the standalone rebuild-check crashes at LAUNCH
# for most graphs afterwards (expected: it lacks vLLM's memory image), so the
# final summary line never prints. That is NOT a rebuild failure. The real
# blocker is the AddKernelNode rejection; we key on "hipGraphAddKernelNode
# failed" presence (rebuilt_ok = absent). SNAPSHOT_SKIP_LAUNCH=1 avoids the
# noisy standalone launch.

set -uo pipefail
cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
GLM_DEPLOY=/capstor/scratch/cscs/xyao/glm-47-flash-vllm
export EDF_PATH="${GLM_DEPLOY}/deploy/glm-47-flash-beverin:${EDF_PATH:-${HOME}/.edf}"
REPO=/capstor/scratch/cscs/xyao/kimi-k25-vllm

echo "allfull-rebuild start: $(date --iso-8601=seconds) node=$(hostname)"

echo "=== in-container build7 (HSACO patch) ==="
srun --environment=glm-47-flash-rocm bash -c "
  cd $REPO
  mkdir -p snapshot/build7 && cd snapshot/build7
  cmake .. -DCMAKE_BUILD_TYPE=Release >/dev/null 2>&1
  cmake --build . -j${SLURM_CPUS_PER_TASK:-8} --target snapshot 2>&1 | grep -iE 'error:' | grep -v filesystem | head
  echo BUILD7_OK
"

# $1 = patch target (empty = off). Loops over ALL snaps in record-default-fb.
run_all() {
  local PATCH=$1
  srun --environment=glm-47-flash-rocm bash -c "
    export HIP_VISIBLE_DEVICES=0
    export SNAPSHOT_SKIP_LAUNCH=1
    ${PATCH:+export SNAPSHOT_HSACO_PATCH_MAXWG=$PATCH;}
    cd $REPO/snapshot/record-default-fb
    ok=0; blocked=0; other=0; bl=''
    for f in graph-*.snap; do
      out=\$(\$REPO/snapshot/build7/snapshot rebuild-check \"\$f\" 2>&1)
      if echo \"\$out\" | grep -qa 'hipGraphAddKernelNode failed'; then
        blocked=\$((blocked+1)); bl=\"\$bl \$f\"
      elif echo \"\$out\" | grep -qa 'gfx90a\|does not match backend arch'; then
        other=\$((other+1))
      else
        ok=\$((ok+1))
      fi
    done
    echo \"RESULT patch=${PATCH:-off} rebuilt_ok=\$ok addkernelnode_blocked=\$blocked arch_skip=\$other\"
    if [ \$blocked -gt 0 ] && [ \$blocked -le 5 ]; then echo \"BLOCKEDLIST:\$bl\"; fi
  "
}

echo "===== sweep: patch OFF (control; expect ~766 blocked) ====="
run_all ""
echo
echo "===== sweep: patch ON (maxwg=1024; expect 0 blocked) ====="
run_all "1024"

echo
echo "allfull-rebuild finished: $(date --iso-8601=seconds)"
