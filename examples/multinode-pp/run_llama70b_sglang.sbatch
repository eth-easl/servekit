#!/bin/bash
# Llama-3.1-70B at TP=4 PP=2 across two 4-GPU Clariden nodes, served from the
# presharded dump staged into /dev/shm.
#
# What differs from examples/multinode (tensor parallel only): each node's ranks
# hold DIFFERENT LAYERS, so each node stages a different set of files. servekit
# reads that set out of the dump's own checksum.json rather than globbing -- a
# glob cannot express it, because files read by ranks on both nodes (the -common
# ones, and TP-replicated tensors content dedup stored once) have to land on
# both.
#
# --overlap is safe here in a way it is not for sharded_state: the loader will
# not touch a dump without its READY sentinel, and servekit writes that last and
# only on a clean stage. An engine that looks too early does not read zeros, it
# takes a cache MISS -- a full source load plus a re-dump into /dev/shm. The
# report's overlap_gate says which happened; check it before trusting a timing.
#SBATCH --job-name=multinode-pp-llama70b
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

# /users is a symlink to /iopsstor and is not mounted in the container, so the
# submit dir must be resolved before it crosses into one.
SLURM_SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd -P)"

MODEL="${MODEL:-/capstor/store/cscs/swissai/infra01/hf_models/models/meta-llama/Llama-3.1-70B-Instruct}"
PRESHARDED_ROOT="${PRESHARDED_ROOT:-/capstor/store/cscs/swissai/infra01/cold-start-experiments/llama70b-pp-presharded-tp4pp2}"
SERVED_MODEL_NAME=meta-llama/Llama-3.1-70B-Instruct

NNODES=2
TP_SIZE=4
PP_SIZE=2
DIST_PORT=20000

EDF="${SLURM_SUBMIT_DIR}/sglang-nightly-clariden.toml"
SERVEKIT_DIR="${SLURM_SUBMIT_DIR}/../.."
RUNDIR="${SLURM_SUBMIT_DIR}/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
OUT="${RUNDIR}/run.json"
# What `servekit launch` writes on the head, once it knows --nnodes > 1.
OUT_HEAD="${RUNDIR}/run.node0.json"
mkdir -p "${RUNDIR}"

if [ -z "$(ls -d "${PRESHARDED_ROOT}"/*/ 2>/dev/null || true)" ]; then
  echo "ERROR: no dump under ${PRESHARDED_ROOT}; sbatch prepare.sbatch first" >&2
  exit 1
fi

# Resolved here, not in the container: both tasks must be handed the same
# literal address, and only the submit host can expand the nodelist.
HEAD_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -1)"
echo "head node: ${HEAD_NODE}, nodes: ${SLURM_JOB_NODELIST}"
echo "dump: ${PRESHARDED_ROOT}"

# No backticks anywhere below, not even in comments: they would be command
# substituted on the submit host while this heredoc is built.
INNER_CMD=$(cat <<EOF
set -euo pipefail
ulimit -c 0
# The image sets BASH_ENV=/etc/bash.bashrc, which makes every one of the
# stager's subshells run nvidia-smi first: 0.41 GB/s instead of 16.89.
unset BASH_ENV

pip install -e "${SERVEKIT_DIR}" --no-build-isolation --no-deps -q

servekit launch --out "${OUT}" --timeout 3000 --overlap \
  -- \
  python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --load-format presharded \
    --model-loader-extra-config '{"presharded_path": "${PRESHARDED_ROOT}"}' \
    --tensor-parallel-size ${TP_SIZE} \
    --pipeline-parallel-size ${PP_SIZE} \
    --nnodes ${NNODES} \
    --node-rank \${SLURM_PROCID} \
    --dist-init-addr ${HEAD_NODE}:${DIST_PORT} \
    --attention-backend flashinfer \
    --host 0.0.0.0 \
    --port 8080 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --trust-remote-code \
    --context-length 32768 \
    --mem-fraction-static 0.85 \
    --max-running-requests 256 \
    --enable-metrics &
LAUNCH_PID=\$!

# Only the head binds a port, so only the head benches. The worker just waits
# for its engine to go down with the head's.
if [ "\${SLURM_PROCID}" = "0" ]; then
  servekit bench --url "http://127.0.0.1:8080" --into "${OUT_HEAD}" \
    --wait-ready 3000 --requests 64 --concurrency 16 --input-len 512 --output-len 128 || true
  kill \${LAUNCH_PID} 2>/dev/null || true
fi
wait \${LAUNCH_PID} 2>/dev/null || true
EOF
)

# --wait=60: once the head's task ends, give the worker a minute to fall over on
# its own rather than holding the allocation to the time limit.
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
    # greedy completions are the check that the weights survived the
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
