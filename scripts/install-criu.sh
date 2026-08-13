#!/usr/bin/env bash
# install-criu.sh — install criu with the checkpoint/restore capabilities our
# gate scripts need (see gate_criu_cpu.sh, gate_criu_gpu.sh).
#
# Installs criu 4.2.1 from ppa:criu/ppa (Ubuntu 24.04 "noble" — the version
# validated on the local test machine), grants the binary the file
# capabilities required for unprivileged dump/restore, and holds the package
# afterward. A routine `apt upgrade` replaces the binary and silently strips
# those capabilities — this already happened once and broke checkpoint work
# until someone noticed and reran setcap — so the hold prevents that class of
# breakage outright.
#
# Usage: sudo ./install-criu.sh

set -euo pipefail

CRIU_VERSION="4.2.1-1ppa1.24.04"
CRIU_BIN="/usr/sbin/criu"

if [[ $EUID -ne 0 ]]; then
  echo "error: must be run as root (sudo)" >&2
  exit 1
fi

if ! grep -q '^VERSION_CODENAME=noble' /etc/os-release 2>/dev/null; then
  codename="$(. /etc/os-release; echo "${VERSION_CODENAME:-unknown}")"
  echo "warning: validated on Ubuntu 24.04 (noble); this host reports '${codename}'" >&2
fi

echo "==> adding ppa:criu/ppa"
apt-get update
apt-get install -y software-properties-common
add-apt-repository -y ppa:criu/ppa
apt-get update

echo "==> installing criu=${CRIU_VERSION}"
apt-get install -y "criu=${CRIU_VERSION}"

echo "==> granting checkpoint/restore capabilities on ${CRIU_BIN}"
setcap cap_checkpoint_restore,cap_sys_ptrace+ep "${CRIU_BIN}"

echo "==> holding criu package so apt upgrade can't silently strip the capability"
apt-mark hold criu

echo "==> verifying"
"${CRIU_BIN}" --version
getcap "${CRIU_BIN}"
