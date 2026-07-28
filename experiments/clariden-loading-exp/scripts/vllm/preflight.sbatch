#!/bin/bash
# Preflight (vllm) - the cheap gate. Run this BEFORE either measured run.
#
# Two assumptions to kill here rather than inside a 1-hour 4-GPU allocation:
#
#   1. the NGC image runs on this node at all (arch, sm_90, 4 GPUs, a real bf16
#      kernel -- the upstream image passes an import check and still dies later,
#      so "imports" is not the test);
#   2. vLLM's ShardedStateLoader accepts the presharded checkpoint: filename
#      pattern from vLLM's own loader constant, one part per rank, and the
#      per-layer attention scale params present. That last check is why this
#      gate exists in its current form -- reusing SGLang's checkpoint looked
#      fine on pattern and key shape and still failed at load with 320 missing
#      `_k_scale`/`_v_scale`/`_q_scale`/`_prob_scale` keys (job 2918412).
#
# A mismatch found here costs 5 minutes. Found in the preshard run it costs an
# hour, or worse: silently wrong bytes.
#
# Run it AFTER ./submit.sh vllm <preset> save, which writes the checkpoint it
# checks.
#
#   cd experiments/clariden-loading-exp && ./submit.sh vllm llama70b preflight
#
#SBATCH --partition=debug
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=results/vllm/%x-%j.out

set -euo pipefail

ENGINE=vllm
source "${SLURM_SUBMIT_DIR}/scripts/shared/models.sh"

EDF="${SLURM_SUBMIT_DIR}/scripts/vllm/vllm-clariden.toml"

echo "job=${SLURM_JOB_ID} node=$(hostname)"
echo "preset=${MODEL_PRESET:-llama70b} model=${MODEL}"

INNER_CMD=$(cat <<EOF
set -euo pipefail

echo "--- node ---"
echo "arch=\$(uname -m)  nproc=\$(nproc)"
df -h /dev/shm
# No \`| head\`: SIGPIPE would trip the \`||\` and misreport numactl as missing.
numactl -H 2>/dev/null || echo "(numactl not present)"

echo "--- checkpoints ---"
ls -l "${MODEL}/config.json"
ls -l "${SHARDED_SRC}/model-rank-0-part-0.safetensors"
echo "sharded shards=\$(ls "${SHARDED_SRC}"/*.safetensors | wc -l)"

echo "--- engine ---"
# python3, not python: the vllm image has no \`python\` alias, unlike sglang's.
python3 - <<'PY'
import platform, sys
import torch, vllm

print("python  =", sys.version.split()[0])
print("machine =", platform.machine())
print("vllm    =", vllm.__version__)
print("torch   =", torch.__version__)
import vllm._custom_ops  # noqa: F401
print("custom ops:", hasattr(torch.ops, "_C"))
print("gpus    =", torch.cuda.device_count())
print("capab   =", torch.cuda.get_device_capability())
print("name    =", torch.cuda.get_device_name(0))
# Importable is not enough: a real kernel must run on this arch. The upstream
# image fails here-ish, not at import.
x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", bool((x @ x).isfinite().all()))

ok = True
if not hasattr(torch.ops, "_C"):
    print("FAIL: compiled ops did not register"); ok = False
if platform.machine() != "aarch64":
    print("FAIL: not aarch64"); ok = False
if torch.cuda.get_device_capability() != (9, 0):
    print("FAIL: capability %s, expected (9, 0)" % (torch.cuda.get_device_capability(),)); ok = False
if torch.cuda.device_count() != 4:
    print("FAIL: %d GPUs, expected 4" % torch.cuda.device_count()); ok = False
print("ENGINE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY

echo "--- sharded_state reuse gate ---"
SHARDED_SRC="${SHARDED_SRC}" TP_SIZE="${TP_SIZE}" python3 - <<'PY'
import glob, json, os, struct, sys

src = os.environ["SHARDED_SRC"]
tp = int(os.environ["TP_SIZE"])

# vLLM moved this class between releases; take whichever path this build has.
loader = None
for mod in ("vllm.model_executor.model_loader.sharded_state_loader",
            "vllm.model_executor.model_loader.loader"):
    try:
        loader = getattr(__import__(mod, fromlist=["ShardedStateLoader"]),
                         "ShardedStateLoader")
        print("loader  =", mod)
        break
    except (ImportError, AttributeError):
        continue

ok = True
if loader is None:
    print("FAIL: this vllm build has no ShardedStateLoader"); ok = False
else:
    pattern = loader.DEFAULT_PATTERN
    print("pattern =", pattern)
    want = pattern.format(rank=0, part=0)
    if not os.path.exists(os.path.join(src, want)):
        print("FAIL: vllm expects %s, which is not in the checkpoint" % want); ok = False
    for rank in range(tp):
        n = len(glob.glob(os.path.join(src, pattern.format(rank=rank, part="*"))))
        print("  rank %d: %d parts" % (rank, n))
        if n == 0:
            print("FAIL: rank %d has no parts under vllm's pattern" % rank); ok = False

# safetensors header only -- 8-byte length prefix then JSON. Cheap on Lustre.
path = os.path.join(src, "model-rank-0-part-0.safetensors")
with open(path, "rb") as f:
    n_hdr = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n_hdr))
keys = sorted(k for k in hdr if k != "__metadata__")
print("rank0 part0 tensors =", len(keys))
for k in keys[:5]:
    print("   ", k)
bad = [k for k in keys if k.startswith(("module.", "_orig_mod."))]
if bad:
    print("FAIL: %d keys carry a wrapper prefix, e.g. %s" % (len(bad), bad[0])); ok = False

# The check that an SGLang-written checkpoint fails. vLLM's model has four of
# these per layer and its loader treats every one as mandatory.
scales = [k for k in keys if k.endswith(("_k_scale", "_v_scale", "_q_scale", "_prob_scale"))]
print("attention scale params =", len(scales))
if not scales:
    print("FAIL: no scale params -- this checkpoint was not written by vllm"); ok = False

print("SHARDED_STATE", "PASS" if ok else "FAIL")
print()
print("NOTE: this proves the layout is addressable, NOT that the bytes are")
print("      right. Only errors=0 + correct output in the preshard run does.")
sys.exit(0 if ok else 1)
PY

# /dev/shm must hold the checkpoint plus headroom.
shm_free=\$(df --output=avail -B1 /dev/shm | tail -1 | tr -d ' ')
echo "shm_free_GB=\$(( shm_free / 1000000000 ))"
if (( shm_free < 160000000000 )); then
  echo "FAIL: /dev/shm has < 160 GB free"; exit 1
fi

echo "--- servekit ---"
python3 -m pip install -e "${SLURM_SUBMIT_DIR}/../../servekit" --no-build-isolation --no-deps -q
python3 -c "from servekit.profile import detect_framework; print('framework =', detect_framework(['vllm','serve','${MODEL}']).name)"
which servekit

echo "ALL CHECKS PASSED"
EOF
)

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" \
  bash -c "${INNER_CMD}"
