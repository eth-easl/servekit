#!/usr/bin/env bash
# Gate 0 — environment probe for the local checkpoint/restore experiment.
#
# Records the facts that make this machine DIFFERENT from Bristen (where CRIU was
# blocked): whether the criu binary carries cap_checkpoint_restore as a file cap,
# and whether the bare host runs seccomp-free. Pure read-only; safe to re-run.
#
# Usage: scripts/probe_env.sh [output_file]
set -u
OUT="${1:-/dev/stdout}"
exec > >(tee "$OUT") 2>&1

echo "==== checkpoint-restore-local-exp :: environment probe ===="
echo "date(host clock): $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo

echo "== user / caps =="
id
grep -E 'Cap(Inh|Prm|Eff|Bnd|Amb)|Seccomp' /proc/self/status
echo

echo "== GPU / driver =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || echo "(nvidia-smi unavailable)"
echo

echo "== RAM =="
free -h | awk 'NR==1 || /Mem:/'
echo

echo "== criu =="
CRIU="$(command -v criu || echo /usr/sbin/criu)"
echo "path: $CRIU"
"$CRIU" --version 2>&1 | head -1
echo -n "file caps: "; getcap "$CRIU" 2>/dev/null || echo "(none)"
echo "-- criu check (feature probe; non-fatal) --"
"$CRIU" check 2>&1 | tail -3
echo

echo "== kernel knobs =="
echo "ptrace_scope        = $(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null)"
echo "unprivileged_userns = $(cat /proc/sys/kernel/unprivileged_userns_clone 2>/dev/null || sysctl -n kernel.unprivileged_userns_clone 2>/dev/null)"
echo "kernel              = $(uname -r)"
echo

echo "== interpretation =="
echo "Bristen blocker was: map_files open() needs CAP_CHECKPOINT_RESTORE in the INITIAL"
echo "userns, unobtainable there. Here criu should carry it as a FILE cap (see above),"
echo "so criu — not the shell — gets the cap effective in the initial userns."
echo "==== end probe ===="
