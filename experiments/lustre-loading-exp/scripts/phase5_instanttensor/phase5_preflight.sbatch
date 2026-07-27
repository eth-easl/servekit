#!/bin/bash
# Phase 5 preflight - prove the InstantTensor backport works, and ANSWER THE GDS
# QUESTION, before spending GPU nodes.
#
# Three jobs in one, all cheap:
#   1. HARNESS: clone sglang @ the pinned SHA, verify all 4 files the backport
#      touches are byte-identical to the image, apply, swap, compile.
#   2. WIRING:  assert `instanttensor` is now a legal --load-format, that the
#      LoadFormat enum has it, and that the iterator is importable.
#   3. GDS:     InstantTensor's headline numbers (35-45 GB/s) are GPUDirect
#      Storage. On Lustre/capstor GDS may simply not be available, in which case
#      it falls back to direct I/O + a bounce buffer and the ceiling is what we
#      already measured with fastsafetensors. This is THE fork in the road, and
#      it costs nothing to check: libcufile, nvidia-fs device nodes, cufile.json.
#
# Also introspects instanttensor's own API (safe_open signature, env knobs) so
# we know whether there is a concurrency/pipeline knob worth sweeping the way
# phase3 swept files-per-rank. We cannot know that from the PR alone.
#
# No GPUs, ~3 min. The sweep chain gates on this.
#
# Submit:
#   sbatch lustre-loading-exp/scripts/phase5_instanttensor/phase5_preflight.sbatch
#
#SBATCH --job-name=p5-preflight
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=lustre-loading-exp/results/phase5_instanttensor/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
S="${EXP_DIR}/scripts/phase5_instanttensor"

echo "job=${SLURM_JOB_ID} node=$(hostname) tag=preflight"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail

echo "=============== 1. instanttensor package ==============="
pip install 'instanttensor>=0.1.9' -q
python -c 'import importlib.metadata as m; print("instanttensor version:", m.version("instanttensor"))'

echo
echo "=============== 2. backport harness (BOTH patches, one call) ==============="
# Must match exactly what the sweep jobs do: phase3's files-per-rank knob AND
# the InstantTensor backport, applied to ONE clone. Both touch weight_utils.py,
# so two separate calls would trip the version check (the first attempt at this
# sweep failed that way - correctly).
bash "${EXP_DIR}/scripts/lib/patch_sglang_in_container.sh" \
  "${EXP_DIR}/scripts/phase3_loader_concurrency/fst_files_per_rank.patch" \
  "${S}/instanttensor_backport.patch"

echo
echo "=============== 3. wiring asserts ==============="
python - <<'PY'
import importlib.util, os
# find_spec gives .../sglang/__init__.py -> the package root. Everything the
# backport touches lives under sglang/srt/.
srt = os.path.join(os.path.dirname(importlib.util.find_spec("sglang").origin), "srt")

src = open(os.path.join(srt, "configs", "load_config.py")).read()
assert 'INSTANTTENSOR = "instanttensor"' in src, "LoadFormat.INSTANTTENSOR missing"

src = open(os.path.join(srt, "server_args.py")).read()
assert '"instanttensor",' in src, "instanttensor not an allowed --load-format"

src = open(os.path.join(srt, "model_loader", "weight_utils.py")).read()
assert "def instanttensor_weights_iterator(" in src, "iterator missing"
assert "get_world_group" in src, "get_world_group not imported"

src = open(os.path.join(srt, "model_loader", "loader.py")).read()
assert "instanttensor_weights_iterator," in src, "iterator not imported in loader"
assert "LoadFormat.INSTANTTENSOR" in src, "loader has no INSTANTTENSOR branch"

# Both patches must survive co-application: phase3's knob AND phase5's iterator
# have to coexist in the one weight_utils.py the engine imports.
src = open(os.path.join(srt, "model_loader", "weight_utils.py")).read()
assert 'os.getenv("SGLANG_FST_FILES_PER_RANK", "1")' in src, "phase3 knob lost"
assert 'SGLANG_IT_CONCURRENCY' in src, "instanttensor knob missing"
print("[preflight] all 4 files wired; both patches coexist in weight_utils.py")
PY

echo
echo "=============== 4. instanttensor API / knobs ==============="
python - <<'PY'
import inspect, instanttensor
print("safe_open signature:", inspect.signature(instanttensor.safe_open))
print("public API:", [n for n in dir(instanttensor) if not n.startswith("_")])
PY
echo "--- INSTANTTENSOR*/IT_* env knobs currently set ---"
env | grep -iE '^(instanttensor|it)_' || echo "(none set; check the signature above for args)"

echo
echo "=============== 5. GDS availability (THE FORK IN THE ROAD) ==============="
echo "--- libcufile ---"
python - <<'PY'
import ctypes
for so in ("libcufile.so", "libcufile.so.0"):
    try:
        ctypes.CDLL(so); print("  LOADED", so); break
    except OSError as e:
        print("  cannot load", so, "->", e)
PY
echo "--- nvidia-fs device nodes ---"
ls -l /dev/nvidia-fs* 2>/dev/null || echo "  NONE (/dev/nvidia-fs* absent -> GDS kernel driver not exposed)"
echo "--- /proc/driver/nvidia-fs ---"
ls /proc/driver/nvidia-fs 2>/dev/null || echo "  NONE"
echo "--- cufile.json ---"
cat /etc/cufile.json 2>/dev/null | head -20 || echo "  NONE at /etc/cufile.json"
echo "--- is the model on a GDS-capable mount? ---"
stat -f -c '  capstor fstype=%T' /capstor/store/cscs/swissai/infra01/hf_models 2>/dev/null || true

echo
echo "[preflight] ALL CHECKS PASSED"
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"

echo "finished=$(date --iso-8601=seconds)"
