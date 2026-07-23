#!/bin/bash
# Phase 1 - instrument the sliced /dev/shm end-to-end cold start with py-spy,
# to find where the 18.6-19.6s weight_loading window (reading an
# already-in-tmpfs model) actually goes. See ../PLAN.md.
#
# Launch vehicle: same stage + server invocation as
# lustre-loading-exp/scripts/phase4_shm/phase4_shm_sliced_e2e.sbatch (sliced
# stage + mmap loader, native source), left unmodified. The only difference
# here is that servekit is launched in the background so py-spy can attach to
# it (and --subprocesses follow the TP worker children) for the startup-to-
# ready window, instead of servekit exec'ing into the foreground.
#
#   sbatch lustre-loading-exp... no -- run from repo root:
#   sbatch shm-weight-loading-exp/scripts/phase1_pyspy_e2e.sbatch
#
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=shm-weight-loading-exp/results/%x-%j.out

set -euo pipefail

SRC="${SRC:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
SLICES="${SLICES:-64}"
SHM_DEST="${SHM_DEST:-/dev/shm/llama70b}"
PYSPY_DURATION="${PYSPY_DURATION:-150}"   # target-time seconds; weight_loading ends ~111s in. Two runs at 200s
                                           # crashed the whole container right as py-spy's timer expired, right
                                           # when the post-ready benchmark was hammering the GPUs with concurrent
                                           # requests -- ptrace-detaching from CUDA-active processes mid-benchmark
                                           # is the suspected cause. 150s ends py-spy well before ready/bench so
                                           # detach happens on an idle-ish process instead.
PYSPY_RATE="${PYSPY_RATE:-20}"            # samples/sec; 100Hz across ~80 forked processes fell 146s behind real time and blew the job's time limit

SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct
EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/servekit"
LUSTRE_EXP_DIR="${SLURM_SUBMIT_DIR}/lustre-loading-exp"
EXP_DIR="${SLURM_SUBMIT_DIR}/shm-weight-loading-exp"
SGLANG_PORT=8080
TP_SIZE=4
MEM_FRACTION_STATIC=0.85
MAX_MODEL_LEN=32768
MAX_RUNNING_REQUESTS=256

NODE="$(hostname)"
PROFILE_OUT="${EXP_DIR}/results/phase1-${SLURM_JOB_ID}-${NODE}-profile.json"
STAGE_LOG="${EXP_DIR}/results/phase1-${SLURM_JOB_ID}-${NODE}-stage.txt"
PYSPY_OUT="${EXP_DIR}/results/phase1-${SLURM_JOB_ID}-${NODE}-pyspy.speedscope"
META_OUT="${EXP_DIR}/results/phase1-${SLURM_JOB_ID}-${NODE}-meta.txt"

echo "job=${SLURM_JOB_ID} node=${NODE} src=${SRC} slices=${SLICES} pyspy_duration=${PYSPY_DURATION}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
bash "${LUSTRE_EXP_DIR}/scripts/phase4_shm/stage_to_shm_sliced.sh" "${SRC}" "${SHM_DEST}" "${SLICES}" | tee "${STAGE_LOG}"
pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
pip install fastsafetensors -q

servekit profile --out "${PROFILE_OUT}" \
  --bench --bench-requests 64 --bench-concurrency 16 \
  --bench-input-len 512 --bench-output-len 128 \
  -- \
  python -m sglang.launch_server \
    --model-path "${SHM_DEST}" \
    --host 0.0.0.0 \
    --port "${SGLANG_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --trust-remote-code \
    --context-length "${MAX_MODEL_LEN}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --enable-metrics &
SERVEKIT_PID=\$!
echo "servekit_pid=\${SERVEKIT_PID}"

PYSPY_START=\$(date +%s.%N)
echo "pyspy_start_epoch=\${PYSPY_START}"
# setsid isolates py-spy's process group -- kept as cheap insurance, though it
# did NOT explain two prior crashes (both --duration 200, both died the
# instant py-spy's timer expired, "task 0: Terminated" / step CANCELLED, no
# time-limit or OOM signature). Those runs also had --keep-alive, so py-spy's
# 200s window ran past "ready" into the concurrent-load benchmark; the
# leading theory is ptrace-detaching from CUDA-active processes mid-benchmark
# crashing the container. Fix: PYSPY_DURATION=150 stops py-spy comfortably
# before ready (~190-270s observed) instead, and --keep-alive is dropped so
# servekit tears the server down itself after bench, as in the one run
# (75699) that completed cleanly.
setsid py-spy record --pid "\${SERVEKIT_PID}" --subprocesses --nonblocking --rate "${PYSPY_RATE}" \
  --duration "${PYSPY_DURATION}" --output "${PYSPY_OUT}" --format speedscope &
PYSPY_PID=\$!

# py-spy exits on its own after PYSPY_DURATION, well before ready -- that's
# the deliverable, and the run keeps going undisturbed. Without --keep-alive,
# servekit reaches ready, benches, writes the profile JSON, then terminates
# the server itself (proc.terminate() in profile.py) -- so just let the
# whole run finish naturally; do not touch the server before then.
wait "\${PYSPY_PID}" || true
echo "pyspy_done=\$(date --iso-8601=seconds)"

wait "\${SERVEKIT_PID}" || echo "servekit exited nonzero: \$?"
echo "servekit_done=\$(date --iso-8601=seconds)"

# Write this immediately, not after any more shell logic: once the sglang
# server's process tree exits, the container gets torn down (pyxis/enroot
# reacting to the workload finishing) and kills the rest of this script too
# -- observed as "task 0: Terminated" right after servekit_done in every run
# so far, even ones that completed cleanly. Nothing after servekit exits is
# guaranteed to run, so the meta file has to land before that happens.
echo "pyspy_start_epoch=\${PYSPY_START}" > "${META_OUT}"

# Safety net only, best-effort: servekit's own proc.terminate() should already
# be done by the time it returns above, and may not even get to run.
pkill -TERM -f "sglang.launch_server" 2>/dev/null || true
sleep 5
pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== result ==="
if [ -f "${PROFILE_OUT}" ] && [ -f "${META_OUT}" ]; then
  python3 "${EXP_DIR}/scripts/compute_weight_loading_window.py" "${PROFILE_OUT}" "${META_OUT}"
else
  echo "  missing profile JSON and/or meta file -- run did not reach ready within PYSPY_DURATION=${PYSPY_DURATION}s"
  echo "  profile: ${PROFILE_OUT} (exists=$([ -f "${PROFILE_OUT}" ] && echo yes || echo no))"
  echo "  meta:    ${META_OUT} (exists=$([ -f "${META_OUT}" ] && echo yes || echo no))"
fi
echo "pyspy trace: ${PYSPY_OUT}"
echo "finished=$(date --iso-8601=seconds)"
