#!/bin/bash
# Does persisting SGLang's JIT caches (FlashInfer, Triton, tvm-ffi -- all under
# HOME, which the EDF pins to the container's ephemeral overlay) cut cold start?
#
# One launch per job, one knob:
#
#   WARM=0 sbatch experiments/graph-compile-cache-exp/scripts/jit_cache.sbatch
#   WARM=1 EXCLUDE=<node of the cold run> sbatch ... jit_cache.sbatch
#
# Submit SERIALLY. Never --dependency: it hands back the same node.
#
# Fresh node per run, because OS page cache survives across container runs -- a
# second launch on the same node is faster in phases the JIT cache cannot touch
# (job 76455 measured process_startup -16.3 s that way). It also makes the warm
# run a genuine cross-node test, which is the question a deployment turns on.
#
#SBATCH --job-name=jit-cache
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=experiments/graph-compile-cache-exp/results/jit_cache/%x-%j.out

set -euo pipefail

WARM="${WARM:?set WARM=0 (wipe the cache) or WARM=1 (reuse it)}"
LABEL=$([ "${WARM}" = 1 ] && echo warm || echo cold)

SRC=/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-tp4-sharded
SHM_DEST=/dev/shm/llama70b
CACHE=/iopsstor/scratch/cscs/yboughizane/jit-cache
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
STAGE="${SLURM_SUBMIT_DIR}/experiments/lustre-loading-exp/scripts/phase4_shm/stage_to_shm_sliced.sh"
OUT="${SLURM_SUBMIT_DIR}/experiments/graph-compile-cache-exp/results/jit_cache/${LABEL}-${SLURM_JOB_ID}-$(hostname)-profile.json"

[ "${WARM}" = 1 ] || rm -rf "${CACHE}"
mkdir -p "${CACHE}" "$(dirname "${OUT}")"

echo "job=${SLURM_JOB_ID} node=$(hostname) label=${LABEL} cache=${CACHE}"
echo "started=$(date --iso-8601=seconds)"

srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" bash -euo pipefail -c "
pip install -e '${SLURM_SUBMIT_DIR}/servekit' --no-build-isolation --no-deps -q
bash '${STAGE}' '${SRC}' '${SHM_DEST}' 64

# Not /dev/shm: it is mounted noexec and Triton dlopen()s a .so it compiles into
# its own cache dir. /root is the container overlay, which is what HOME normally
# uses, so it is known to work. Copied in, not pointed at, for the same reason.
export HOME=/root/jit-home
rm -rf \$HOME && mkdir -p \$HOME
# cp always exits 1 here: it copies the file data fine, then fails setting
# permissions on the destination DIRECTORY ('preserving permissions for
# .../jit-home/.: Invalid argument') because GNU cp routes that through
# copy_acl(), which EINVALs between Lustre and the container overlay. The old
# scripts hid it with 2>/dev/null || true. Tolerate it, but verify the copy by
# counting files rather than trusting the exit code.
time cp -r --preserve=mode,timestamps '${CACHE}/.' \$HOME/ || true
have=\$(find \$HOME -type f | wc -l); want=\$(find '${CACHE}' -type f | wc -l)
[ \"\$have\" -ge \"\$want\" ] || { echo \"seed incomplete: \$have/\$want files\" >&2; exit 1; }
echo \"seeded \$have files, \$(du -sh \$HOME | cut -f1), from ${CACHE}\"

# --nccl-port is pinned deliberately. Left to itself SGLang picks nccl_port via
# get_free_port() (server_args.py:6574), which handed back 8080 in jobs 76440 and
# 76446 -- its own scheduler then held the port Uvicorn needed, and the run sat
# out its wait-ready timeout in silence, indistinguishable from a hang. This is
# the only random allocation in the launch; nothing else about the config changes.
servekit profile --out '${OUT}' --timeout 420 -- \
  python -m sglang.launch_server \
    --model-path '${SHM_DEST}' --host 0.0.0.0 --port 8080 \
    --served-model-name meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 --trust-remote-code \
    --context-length 32768 --mem-fraction-static 0.85 \
    --max-running-requests 256 --load-format sharded_state \
    --nccl-port 29500 \
    --enable-metrics &
PROF=\$!

servekit bench --url http://127.0.0.1:8080 --into '${OUT}' \
  --wait-ready 420 --requests 64 --concurrency 16 \
  --input-len 512 --output-len 128 || true
kill \$PROF 2>/dev/null || true; wait \$PROF 2>/dev/null || true

cp -r --preserve=mode,timestamps \$HOME/. '${CACHE}/' || true
# No du here: on Lustre it reports pre-flush allocation right after a write and
# under-reports wildly (111K for what is really 3.7M). File count is reliable.
echo \"saved \$(find '${CACHE}' -type f | wc -l) files to ${CACHE}\"
" || true

echo
python3 - "${OUT}" <<'PY'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print("  no profile JSON -- run did not reach ready"); raise SystemExit
d = json.load(open(p))
for ph in d["phases"]:
    print("  %-32s %8.2f" % (ph["name"], ph["duration_s"]))
print("  %-32s %8.2f" % ("TOTAL", d["total_duration_s"]))
b = d.get("benchmark", {}).get("throughput", {})
print("  %s tok/s, %s/%s ok, errors=%s"
      % (b.get("output_tok_per_s"), b.get("completed"), b.get("requests"), b.get("errors")))
PY
echo "finished=$(date --iso-8601=seconds)"
