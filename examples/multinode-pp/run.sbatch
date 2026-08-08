#!/bin/bash
# Serve the presharded dump across nodes, one pipeline stage per node, from
# /dev/shm.
#
# What differs from examples/multinode (which is tensor parallel only): each
# node's ranks hold DIFFERENT LAYERS, so each node stages a different set of
# files. servekit reads that set out of the dump's own checksum.json rather than
# globbing -- a glob cannot express it, because files read by ranks on both
# nodes (the -common ones, and TP-replicated tensors content dedup stored once)
# have to land on both.
#
# --overlap is safe here in a way it is not for sharded_state: the loader will
# not touch a dump without its READY sentinel, and servekit writes that last and
# only on a clean stage. An engine that looks too early does not read zeros, it
# takes a cache MISS -- a full source load plus a re-dump into /dev/shm. The
# report's overlap_gate says which happened; check it before trusting a timing.
#
#   ./submit.sh llama70b run
#
# normal, not debug: debug-qos caps a job at 90 node-minutes, which 8 nodes
# exhaust in 11.
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
# --exclusive gets the node's CPUs and GPUs but NOT its memory: the step lands
# in a cgroup sized by DefMemPerNode (450 GiB here), and tmpfs pages are charged
# to whoever wrote them, so a staged shard set counts against it before the
# engine allocates anything. kimi-k3 peaks around 290 GiB/node (190 staged plus
# the ranks), which fits, but not with much left over. Ask for the node.
#
# Read Slurm's memory numbers here with care: on GH200 each GPU's HBM is a
# coherent NUMA node, so MemTotal, MemAvailable and this limit all count 380 GiB
# of device memory as if it were host RAM. Actual Grace DRAM is 474 GiB.
#SBATCH --mem=850000
#SBATCH --exclusive
#SBATCH --time=02:00:00

set -euo pipefail

# See prepare.sbatch: /users is a symlink to /iopsstor and is not mounted in
# the container, so the submit dir must be resolved before it crosses into one.
SLURM_SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"

source "${SLURM_SUBMIT_DIR}/models.sh"

EDF="${SLURM_SUBMIT_DIR}/${EDF_NAME}"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../.."
RUNDIR="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
OUT="${RUNDIR}/run.json"
OUT_HEAD="${RUNDIR}/run.node0.json"
mkdir -p "${RUNDIR}"

if [ -z "$(ls -d "${PRESHARDED_ROOT}"/*/ 2>/dev/null || true)" ]; then
  echo "ERROR: no dump under ${PRESHARDED_ROOT}; run ./submit.sh ${MODEL_PRESET:-llama70b} prepare first" >&2
  exit 1
fi

HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
echo "preset=${MODEL_PRESET:-llama70b} tp=${TP_SIZE} pp=${PP_SIZE} ep=${EP_SIZE} nodes=${NNODES}"
echo "head node: ${HEAD_NODE}, nodes: ${SLURM_JOB_NODELIST}"
echo "dump: ${PRESHARDED_ROOT}"
echo "started=$(date --iso-8601=seconds)"

INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
# The image sets BASH_ENV, which makes every one of the stager's subshells run
# nvidia-smi first: 0.41 GB/s instead of 16.89.
unset BASH_ENV

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q
export PYTHONPATH="${SLURM_SUBMIT_DIR}/patches"
df -h /dev/shm

servekit launch --out "${OUT}" --timeout ${READY_TIMEOUT} --overlap \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --load-format presharded \
    --model-loader-extra-config '{"presharded_path": "${PRESHARDED_ROOT}"}' \
    --tensor-parallel-size ${TP_SIZE} \
    --pipeline-parallel-size ${PP_SIZE} \
    --expert-parallel-size ${EP_SIZE} \
    --nnodes ${NNODES} \
    --node-rank \${SLURM_PROCID} \
    --dist-init-addr ${HEAD_NODE}:${DIST_PORT} \
    --dist-timeout 3600 \
    ${ENGINE_EXTRA[@]} \
    --host 0.0.0.0 \
    --port ${SERVE_PORT} \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --context-length ${MAX_MODEL_LEN} \
    --mem-fraction-static ${MEM_FRACTION_STATIC} \
    --max-running-requests ${MAX_RUNNING_REQUESTS} \
    --enable-metrics &
LAUNCH_PID=\$!

# Only the head binds a port, so only the head benches.
if [ "\${SLURM_PROCID}" = "0" ]; then
  servekit bench --url "http://127.0.0.1:${SERVE_PORT}" --into "${OUT_HEAD}" \
    --wait-ready ${READY_TIMEOUT} --requests 64 --concurrency 16 \
    --input-len 512 --output-len 128 || true
  kill \${LAUNCH_PID} 2>/dev/null || true
fi
wait \${LAUNCH_PID} 2>/dev/null || true
EOF
)

srun \
  --nodes="${NNODES}" \
  --ntasks="${NNODES}" \
  --ntasks-per-node=1 \
  --wait=60 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}" || true

echo
echo "=== per-node reports ==="
python3 - "${RUNDIR}" <<'PY'
import glob, json, os, sys

for path in sorted(glob.glob(os.path.join(sys.argv[1], "run.node*.json"))):
    d = json.load(open(path))
    g = d.get("overlap_gate") or {}
    phases = dict((p["name"], p["duration_s"]) for p in d.get("phases", []))
    print("%-18s total=%7.2fs  weight_loading=%6.2fs  gate=%s" % (
        os.path.basename(path), d.get("total_duration_s", float("nan")),
        phases.get("weight_loading", float("nan")),
        ("VALID slack=%.1fs" % g["slack_s"]) if g.get("valid")
        else ("INVALID %s" % g.get("slack_s", "n/a"))))

    # The bench only runs on the head, so this prints once. A dump loaded at the
    # wrong shape still serves -- it just serves the wrong tensors -- so the
    # greedy completions are the check that the weights actually survived the
    # preshard/stage round trip; timings alone would not catch it.
    bench = d.get("benchmark") or {}
    thr = bench.get("throughput") or {}
    if thr:
        print("  throughput: %.1f tok/s (%s reqs, conc=%s, in=%s out=%s)" % (
            thr.get("output_tok_per_s", float("nan")), thr.get("requests", "?"),
            thr.get("concurrency", "?"), thr.get("input_len", "?"),
            thr.get("output_len", "?")))
    for r in (bench.get("correctness") or {}).get("results", []):
        text = " ".join(r["output"].split())
        print("  correctness: %-46s -> %s" % (r["prompt"][:46], text[:70]))
PY
echo "finished=$(date --iso-8601=seconds)"
