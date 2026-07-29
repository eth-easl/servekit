#!/bin/bash
# Preflight - the gate. Nothing else in this round runs until this passes.
#
# Four assumptions, all of them new this round and all of them capable of
# silently ruining a measured run:
#
#   1. the arm64 nightly is usable on GH200 at all -- through CUDA graph
#      capture and a served request. This is exactly where vLLM broke last
#      round, and a nightly is a worse bet than a release.
#   2. --load-format presharded exists and PP=4 reaches ready.
#   3. servekit's profile regexes still match. profile.py scrapes LITERAL
#      SGLang log strings ("Load weight end. elapsed=", "The server is fired up
#      and ready to roll!"). This image is many months ahead of the v0.5.10 the
#      regexes were written against. If a string moved, every downstream number
#      comes back empty and the run looks fine.
#   4. BASH_ENV is unset. In NVIDIA's NGC image it is /etc/bash.bashrc, so each
#      of the stager's ~1700 `bash -c` slice workers ran nvidia-smi at startup:
#      load average 1658 and a 348 s stage at 0.41 GB/s. Report it here; the
#      stager call site uses `env -u BASH_ENV` regardless.
#
#   cd experiments/pp-loading-exp && ./submit.sh apertus8b preflight
#
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=results/%x-%j.out

set -euo pipefail

source "${SLURM_SUBMIT_DIR}/scripts/models.sh"

EDF="${SLURM_SUBMIT_DIR}/scripts/sglang-nightly.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../../servekit"
RESULTS_DIR="${SLURM_SUBMIT_DIR}/results/${RESULTS_SUBDIR}"
PROFILE_OUT="${RESULTS_DIR}/preflight-${SLURM_JOB_ID}-profile.json"

mkdir -p "${RESULTS_DIR}"
echo "job=${SLURM_JOB_ID} node=$(hostname) cpus=${SLURM_CPUS_PER_TASK}"
echo "preset=${MODEL_PRESET:-apertus8b} tp=${TP_SIZE} pp=${PP_SIZE}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail

echo "--- node ---"
echo "arch=\$(uname -m)  nproc=\$(nproc)"
df -h /dev/shm
echo "BASH_ENV='\${BASH_ENV:-}'"
if [ -n "\${BASH_ENV:-}" ]; then
  echo "  NOTE: BASH_ENV is set in this image -- the stager must run under env -u BASH_ENV"
fi

echo "--- engine ---"
python - <<'PY'
import platform, sys, torch, sglang
from sglang.srt.configs.load_config import LoadFormat

print("python     =", sys.version.split()[0])
print("machine    =", platform.machine())
print("sglang     =", sglang.__version__)
print("torch      =", torch.__version__)
import sgl_kernel
print("sgl_kernel =", sgl_kernel.__file__)
print("gpus       =", torch.cuda.device_count())
print("capability =", torch.cuda.get_device_capability())
print("device     =", torch.cuda.get_device_name(0))

ok = True
if platform.machine() != "aarch64":
    print("FAIL: not aarch64"); ok = False
if torch.cuda.get_device_capability() != (9, 0):
    print("FAIL: capability %s, expected (9, 0)" % (torch.cuda.get_device_capability(),)); ok = False
if torch.cuda.device_count() != 4:
    print("FAIL: %d GPUs, expected 4" % torch.cuda.device_count()); ok = False

# The whole reason for this image.
if not hasattr(LoadFormat, "PRESHARDED"):
    print("FAIL: this build has no LoadFormat.PRESHARDED -- wrong image tag"); ok = False
else:
    from sglang.srt.model_loader.loader import PreshardedModelLoader
    print("presharded loader ok:", PreshardedModelLoader.DEFAULT_SUBDIR)

print("ENGINE CHECKS", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

echo "--- PP=${PP_SIZE} server, default loader ---"
servekit profile --out "${PROFILE_OUT}" --timeout "${READY_TIMEOUT}" \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${SERVER_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --pipeline-parallel-size "${PP_SIZE}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --trust-remote-code \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --enable-metrics &
PROF_PID=\$!

# servekit writes the report the moment ready is detected, and also on failure
# with success=false. Without this guard a dead server leaves \`bench
# --wait-ready\` polling for the rest of the job -- which is exactly what
# happened on job 2922510 and blocked the queue.
while kill -0 \${PROF_PID} 2>/dev/null && [ ! -f "${PROFILE_OUT}" ]; do sleep 2; done
if python -c "import json,sys; sys.exit(0 if json.load(open('${PROFILE_OUT}'))['success'] else 1)" 2>/dev/null; then
  servekit bench --url "http://127.0.0.1:${SERVER_PORT}" --into "${PROFILE_OUT}" \
    --wait-ready 300 \
    --requests 8 --concurrency 4 --input-len 128 --output-len 32 || true
else
  echo "SERVER NEVER REACHED READY -- skipping bench"
fi

kill \${PROF_PID} 2>/dev/null || true
wait \${PROF_PID} 2>/dev/null || true
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== preflight verdict ==="
python3 - "${PROFILE_OUT}" <<'PY'
import json, os, sys

path = sys.argv[1]
if not os.path.exists(path):
    print("FAIL: no profile written -- the server never reached ready")
    sys.exit(1)

d = json.load(open(path))
phases = {p['name']: p['duration_s'] for p in d['phases']}
print("  phases parsed:", ", ".join("%s=%.2f" % kv for kv in phases.items()) or "(none)")
print("  total = %.2f s" % d['total_duration_s'])

ok = True
# servekit's regexes are the thing under test here, not the engine's speed.
if 'weight_loading' not in phases:
    print("  FAIL: no weight_loading phase -- the 'Load weight end. elapsed=' string moved")
    print("        fix servekit/src/servekit/profile.py:45 before measuring anything")
    ok = False
if not d.get('ready_at'):
    print("  FAIL: ready never detected -- the ready_pattern string moved (profile.py:49)")
    ok = False

# PP forces overlap schedule off and piecewise CUDA graph off, so this phase is
# expected to be ABSENT. Its presence would mean pp_size did not take effect.
if 'piecewise_cuda_graph_capture' in phases:
    print("  WARN: piecewise_cuda_graph_capture present -- expected absent at pp_size>1;")
    print("        check the server actually ran with --pipeline-parallel-size 4")

b = (d.get('benchmark') or {}).get('throughput') or {}
print("  bench: %s tok/s, %s/%s ok, errors=%s"
      % (b.get('output_tok_per_s'), b.get('completed'), b.get('requests'), b.get('errors')))
if b.get('errors') not in (0, '0', None) or not b.get('completed'):
    print("  FAIL: the server did not serve cleanly at PP=4")
    ok = False

print("PREFLIGHT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
echo "finished=$(date --iso-8601=seconds)"
