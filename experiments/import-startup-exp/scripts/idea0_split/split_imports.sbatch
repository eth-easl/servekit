#!/bin/bash
# Idea 0: what is the ~38-40 s of import time actually made of -- compile,
# cold I/O, or module exec?
#
# Import sglang.srt.server_args under -X importtime, three times in one
# container. No model, no TP, no server: the import graph is the workload.
#
#   run  page cache   .pyc          isolates
#   A    cold         absent        everything
#   B    warm         present (A)   exec only
#   C    warm         absent        exec + compile
#
#   compile = C - B      cold I/O = A - C      exec floor = B
#
#   sbatch experiments/import-startup-exp/scripts/idea0_split/split_imports.sbatch
#   sbatch --exclude=<nodes of previous runs> ... split_imports.sbatch
#
# Submit SERIALLY, three times, excluding the nodes already used. Never
# --dependency: it hands back the same node. Only run A can be cold, and only
# once per node -- page cache survives across container runs, so a node that
# already pulled the 30 GB image reads as ~0 cold I/O. Take the consistent
# high A; do not average in a pre-warmed node.
#
# Caveat, recorded not designed around: A also WRITES sglang's 1663 .pyc, so
# that write cost sits inside the cold-I/O bucket. It is the status quo.
#
#SBATCH --job-name=import-split
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=experiments/import-startup-exp/results/idea0_split/%x-%j.out

set -euo pipefail

EDF="${SLURM_SUBMIT_DIR}/profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml"
RES="${SLURM_SUBMIT_DIR}/experiments/import-startup-exp/results/idea0_split"
NODE="$(hostname)"
STEM="${SLURM_JOB_ID}-${NODE}"

mkdir -p "${RES}"
echo "job=${SLURM_JOB_ID} node=${NODE}"
echo "started=$(date --iso-8601=seconds)"

INNER=$(cat <<EOF
set -euo pipefail
SG=/sgl-workspace/sglang/python
RES="${RES}"
STEM="${STEM}"

# GPUs are requested so sgl_kernel/flashinfer import down the same path as a
# real launch, but nothing here touches a device.
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

# Where does CPython put the .pyc? The image tree may be read-only depending on
# how the overlay is set up; if it is, fall back to a cache prefix under
# HOME=/root (the container overlay), which is writable by definition.
if touch "\$SG/.wtest" 2>/dev/null; then
  rm -f "\$SG/.wtest"; PYC=""; echo "pyc destination: in-tree (\$SG)"
else
  PYC="PYTHONPYCACHEPREFIX=/root/pyc"; echo "pyc destination: /root/pyc (tree read-only)"
fi

pyc_count () { { find "\$SG" -name '*.pyc' 2>/dev/null; find /root/pyc -name '*.pyc' 2>/dev/null; } | wc -l; }
drop_pyc  () { find "\$SG" -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true; rm -rf /root/pyc; }

run () {
  name=\$1; shift
  t0=\$(date +%s.%N)
  env \$PYC "\$@" python -X importtime -c 'import sglang.srt.server_args' \
    >/dev/null 2>"\$RES/\$name-\$STEM.txt"
  t1=\$(date +%s.%N)
  awk -v s=\$t0 -v e=\$t1 -v n=\$name -v p="\$(pyc_count)" \
    'BEGIN{printf "%s wall_s=%.3f pyc_after=%s\n", n, e-s, p}' | tee -a "\$RES/walls-\$STEM.txt"
}

echo "pyc before anything: \$(pyc_count)"
drop_pyc                        # a previous job cannot have left any, but be sure
run A                           # cold cache, no pyc  -> everything
run B                           # warm cache, pyc from A -> exec only
drop_pyc
run C PYTHONDONTWRITEBYTECODE=1 # warm cache, no pyc -> exec + compile
EOF
)

srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task=4 \
  --environment="${EDF}" bash -c "${INNER}" || true

echo
echo "=== split (${STEM}) ==="
python3 - "${RES}" "${STEM}" <<'PY'
import collections, pathlib, re, sys

res, stem = pathlib.Path(sys.argv[1]), sys.argv[2]
runs = "ABC"
line = re.compile(r"^import time:\s+(\d+) \|\s+(\d+) \|\s+(\S+)")

wall = {}
for l in (res / f"walls-{stem}.txt").read_text().splitlines():
    n, w = l.split()[0], float(l.split("wall_s=")[1].split()[0])
    wall[n] = w

# Self time, not cumulative: cumulative overlaps, the root entry is "everything".
self_ms, totals = {}, {}
for r in runs:
    f = res / f"{r}-{stem}.txt"
    by_pkg = collections.Counter()
    for l in f.read_text().splitlines():
        m = line.match(l)
        if m:
            by_pkg[m.group(3).split(".")[0]] += int(m.group(1)) / 1000
    self_ms[r] = by_pkg
    totals[r] = sum(by_pkg.values())

print(f"{'wall (s)':<24}" + "".join(f"{r:>10}" for r in runs))
print("-" * 54)
print(f"{'':<24}" + "".join(f"{wall.get(r, float('nan')):>10.2f}" for r in runs))
print()
a, b, c = (wall.get(r, float("nan")) for r in runs)
print(f"  exec floor      B     = {b:6.2f} s")
print(f"  compile         C - B = {c - b:6.2f} s")
print(f"  cold I/O        A - C = {a - c:6.2f} s")
print(f"  status quo      A     = {a:6.2f} s")
print()

print(f"{'importtime self (s)':<24}" + "".join(f"{r:>10}" for r in runs))
print("-" * 54)
for pkg, _ in self_ms["A"].most_common(15):
    print(f"{pkg:<24}" + "".join(f"{self_ms[r][pkg]/1000:>10.2f}" for r in runs))
print(f"{'TOTAL':<24}" + "".join(f"{totals[r]/1000:>10.2f}" for r in runs))
n = sum(1 for l in (res / f"A-{stem}.txt").read_text().splitlines() if line.match(l))
print(f"\n({n} modules)")
PY
echo "finished=$(date --iso-8601=seconds)"
