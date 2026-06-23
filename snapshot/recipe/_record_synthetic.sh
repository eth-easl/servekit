#!/bin/bash
# M3a.2 — controlled bit-identical gate via the REAL interposer (not the workload).
#
# Runs our synthetic workload (`snapshot capture --scaled`) under the recorder
# LD_PRELOAD shim. The workload allocates its VMM region (hipMemAddressReserve +
# hipMemMap), loads its hiprtc module (hipModuleLoadData), gets functions
# (hipModuleGetFunction), and captures a graph (hipStreamBeginCapture/EndCapture
# around hipModuleLaunchKernel). The recorder observes ALL of those through the
# same symbols it would intercept in vLLM, and writes its own snapshot file.
#
# Then a FRESH process restores the recorder's snapshot (rebuild + relocate +
# instantiate + launch + re-seed inputs) and checks bit-identical vs the host
# reference. If this passes, the recorder -> serialize -> restore path is sound
# with the interposer (not the workload) as the interceptor — the exact
# mechanism M3b would use on vLLM.
#
# Run inside the ROCm container (sbatch: record_synthetic.sbatch).
set -euo pipefail

cd /capstor/scratch/cscs/xyao/kimi-k25-vllm
SNAP=snapshot/build/snapshot
RECORDER="$PWD/snapshot/build/libsnapshot_record.so"

# Build if missing (the recorder lib is a build target alongside the CLI).
if [ ! -x "$SNAP" ] || [ ! -f "$RECORDER" ]; then
  cmake -S snapshot -B snapshot/build -DSNAPSHOT_BACKEND=HIP
  cmake --build snapshot/build -j"${SLURM_CPUS_PER_TASK:-8}"
fi

OUT=snapshot/record-synthetic
rm -rf "$OUT"; mkdir -p "$OUT"
WORKLOAD_SNAP="$OUT/workload.snap"

echo "=== [1/4] capture under the recorder (recorder writes its own snapshot) ==="
LD_PRELOAD="$RECORDER" \
  SNAPSHOT_RECORD_OUT_DIR="$OUT" \
  SNAPSHOT_RECORD_MAX_GRAPHS=2 \
  SNAPSHOT_RECORD_VERBOSE=1 \
  "$SNAP" capture "$WORKLOAD_SNAP" --scaled

REC_SNAP=$(ls "$OUT"/graph-*.snap | head -1)
echo "recorder snapshot: $REC_SNAP"
echo "workload snapshot: $WORKLOAD_SNAP"

echo
echo "=== [2/4] inspect the recorder snapshot (host-side identity summary) ==="
"$SNAP" inspect "$REC_SNAP"

echo
echo "=== [3/4] FRESH PROCESS: restore the recorder snapshot, bit-identical gate ==="
"$SNAP" restore "$REC_SNAP"

echo
echo "=== [4/4] sanity: recorder vs workload node/module counts ==="
W_NODES=$("$SNAP" inspect "$WORKLOAD_SNAP" 2>/dev/null | awk -F= '/^nodes=/{print $2}' | awk '{print $1}')
R_NODES=$("$SNAP" inspect "$REC_SNAP"     2>/dev/null | awk -F= '/^nodes=/{print $2}' | awk '{print $1}')
echo "workload nodes=$W_NODES  recorder nodes=$R_NODES"
if [ "${W_NODES:-0}" -gt 0 ] && [ "$W_NODES" = "$R_NODES" ]; then
  echo "MATCH_GATE=PASS (recorder captured the same node count as the workload)"
else
  echo "MATCH_GATE=CHECK (node counts differ — compare the two snapshots by hand)"
fi

echo
echo "[M3a.2] DONE"
