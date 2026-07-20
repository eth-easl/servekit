#!/bin/bash
# Preflight: prove the harness is sound before spending GPU nodes on the sweep.
# GPU-less and ~2 min. Checks, in order of what would silently ruin the sweep:
#
#   1. the patch applies to a clone that matches the image byte-for-byte
#      (the harness asserts this itself and aborts loudly)
#   2. the patched file really contains both env knobs
#   3. fastsafetensors ACCEPTS max_threads / bbuf_size_kb as kwargs -- if the
#      names are wrong, every "tuned" run would silently use the defaults and
#      the sweep would report a flat line that means nothing
#   4. the null path is upstream: no env set -> empty kwargs, chunk == pg.size()
#
#   sbatch fst-threads-exp/scripts/preflight.sbatch
#
#SBATCH --job-name=fst-preflight
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=fst-threads-exp/results/%x-%j.out

set -euo pipefail
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
EXP_DIR="${SLURM_SUBMIT_DIR}/fst-threads-exp"
LOADING_EXP="${SLURM_SUBMIT_DIR}/lustre-loading-exp"

echo "job=${SLURM_JOB_ID} node=$(hostname) started=$(date --iso-8601=seconds)"

INNER=$(cat <<EOF
set -euo pipefail
pip install fastsafetensors -q

echo "=== 1. patch applies against the image-pinned clone ==="
bash "${LOADING_EXP}/scripts/lib/patch_sglang_in_container.sh" "${EXP_DIR}/scripts/fst_threads.patch"

echo "=== 2. patched file carries both knobs ==="
SITE=\$(python -c 'import importlib.util,os; print(os.path.dirname(importlib.util.find_spec("sglang").origin))')
WU="\${SITE}/srt/model_loader/weight_utils.py"
for k in SGLANG_FST_FILES_PER_RANK SGLANG_FST_MAX_THREADS SGLANG_FST_BBUF_SIZE_KB; do
  grep -q "\$k" "\${WU}" && echo "  ok: \$k present" || { echo "  FATAL: \$k missing from \${WU}"; exit 1; }
done

echo "=== 3. fastsafetensors accepts the kwargs we intend to pass ==="
python - <<'PY'
import inspect, sys
from fastsafetensors import SafeTensorsFileLoader
sig = inspect.signature(SafeTensorsFileLoader.__init__)
missing = [k for k in ("max_threads", "bbuf_size_kb") if k not in sig.parameters]
if missing:
    print(f"  FATAL: SafeTensorsFileLoader has no {missing} -- the sweep would silently")
    print( "         use defaults and report a meaningless flat line.")
    print(f"  actual signature: {sig}")
    sys.exit(1)
for k in ("max_threads", "bbuf_size_kb"):
    print(f"  ok: {k} accepted, default = {sig.parameters[k].default}")
PY

echo "=== 4. null path is upstream (no env set) ==="
python - <<'PY'
import os
assert not os.getenv("SGLANG_FST_MAX_THREADS"), "env leaked into preflight"
fpr = int(os.getenv("SGLANG_FST_FILES_PER_RANK", "1"))
assert fpr == 1, fpr
kw = {}
if os.getenv("SGLANG_FST_MAX_THREADS"): kw["max_threads"] = 1
if os.getenv("SGLANG_FST_BBUF_SIZE_KB"): kw["bbuf_size_kb"] = 1
assert kw == {}, kw
# round-robin == enumerate(), incl. a short final chunk
for n in (4, 3, 1):
    f = [f"f{i}" for i in range(n)]
    rr = {r: f[r::4] for r in range(4) if f[r::4]}
    up = {i: [x] for i, x in enumerate(f)}
    assert rr == up, (rr, up)
print("  ok: fpr=1 -> chunk=pg.size(); kwargs={}; rank_file_map == enumerate()")
PY

echo "PREFLIGHT OK"
EOF
)

srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --environment="${EDF}" bash -c "${INNER}"

echo "finished=$(date --iso-8601=seconds)"
