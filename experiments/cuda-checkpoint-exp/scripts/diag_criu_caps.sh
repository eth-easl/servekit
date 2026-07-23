#!/bin/bash
# Diagnostic: what exactly blocks criu here? Runs in any environment (container,
# bare host, unshare) and reports which capability check fails first. Also tests
# whether criu --unprivileged (non-root mode) changes anything: it does not.
# criu supports non-root C/R since Linux 5.9 *if* CAP_CHECKPOINT_RESTORE is held.
# The open question is whether that mode avoids /proc/PID/map_files, which is what
# blocks us. `check` is not enough -- we attempt a real dump too.
set -uo pipefail
EXP_DIR="${EXP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CRIU="${EXP_DIR}/bin/criu"
export LD_LIBRARY_PATH="${EXP_DIR}/hostlibs:${LD_LIBRARY_PATH:-}"
W="$(mktemp -d /iopsstor/scratch/cscs/yboughizane/tmp.unpriv.XXXXXX)"

if grep -qi sles /etc/os-release 2>/dev/null; then WHERE="bare host"; else WHERE="pyxis container"; fi
echo "### env: ${WHERE}  uid=$(id -u)  Seccomp=$(grep -E '^Seccomp:' /proc/self/status | awk '{print $2}')"
echo "### CapEff=$(grep -E '^CapEff:' /proc/self/status | awk '{print $2}')"
echo "### criu: $("${CRIU}" --version 2>&1 | head -1)"

echo
echo "--- criu check (normal) ---"
"${CRIU}" check 2>&1 | tail -3
echo "rc=${PIPESTATUS[0]}"

echo
echo "--- criu check --unprivileged ---"
"${CRIU}" check --unprivileged 2>&1 | tail -6
echo "rc=${PIPESTATUS[0]}"

echo
echo "--- real dump attempt WITH --unprivileged ---"
( i=0; while :; do echo $i > "${W}/p.out"; i=$((i+1)); sleep 0.2; done ) & P=$!
sleep 1
mkdir -p "${W}/ck"
"${CRIU}" dump -t "$P" -D "${W}/ck" --shell-job --unprivileged -o d.log -v4 >/dev/null 2>&1
echo "dump --unprivileged rc=$?"
grep -iE '\)\s+Error' "${W}/ck/d.log" 2>/dev/null | tail -4
kill -9 "$P" 2>/dev/null

echo
echo "--- does --unprivileged still touch map_files? ---"
grep -icE 'map_files|mapfile' "${W}/ck/d.log" 2>/dev/null | xargs echo "  map_files mentions in dump log:"
rm -rf "${W}"
