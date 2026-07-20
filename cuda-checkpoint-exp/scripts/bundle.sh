#!/bin/bash
# STEP 1 (runs INSIDE the ubuntu/cuda container): produce artifacts that run on the
# bare SLES host. We build here only because the host has no nvcc / criu build deps.
#
#  - bin/criu        : criu 3.19 built from source WITHOUT libnftables (see below)
#  - hostlibs/       : the only shared lib the SLES host lacks (libnet.so.1)
#  - src/counter     : compiled with `-cudart static` so it needs nothing from the
#                      container at runtime except libcuda.so.1 (provided by the driver)
#
# Host is SLES15-SP6 / glibc 2.38, newer than jammy's 2.35, so jammy-built binaries
# run natively on it (glibc is backward compatible).
set -uo pipefail

EXP_DIR="${EXP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CRIU_VER="${CRIU_VER:-v3.19}"
mkdir -p "${EXP_DIR}/hostlibs" "${EXP_DIR}/bin"

echo "== bundle step: id=$(id -u) os=$(grep -oP 'VERSION_CODENAME=\K.*' /etc/os-release)"

# Everything below is cached on /iopsstor and survives between jobs, so on a warm
# tree there is nothing to do. Skip early: apt-get here costs minutes (and on some
# nodes has hung outright), which is pure waste once criu + counter are built.
if [ -x "${EXP_DIR}/bin/criu" ] && [ -x "${EXP_DIR}/src/counter" ] \
   && [ -f "${EXP_DIR}/hostlibs/libbsd.so.0" ]; then
  echo "all artifacts already built -- nothing to do:"
  echo "  criu    : $("${EXP_DIR}/bin/criu" --version 2>&1 | head -1)"
  echo "  counter : $(ls -l "${EXP_DIR}/src/counter" | awk '{print $5}') bytes"
  echo "  hostlibs: libnet.so.1"
  echo "(delete bin/criu or src/counter to force a rebuild)"
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>&1 | grep -iE '^E:'
# NOTE: libnftables-dev deliberately omitted. criu probes nftables during kernel
# feature detection; that probe fails on this host and criu<=3.16 makes it FATAL.
# Built without libnftables the probe is compiled out and criu starts cleanly.
apt-get install -y -qq git build-essential pkg-config \
  libprotobuf-dev libprotobuf-c-dev protobuf-c-compiler protobuf-compiler \
  python3-protobuf libnl-3-dev libnet1-dev libcap-dev libbsd-dev \
  2>&1 | grep -iE '^E:'

if [ ! -x "${EXP_DIR}/bin/criu" ]; then
  echo "-- building criu ${CRIU_VER} --"
  T=$(mktemp -d)
  git clone -q --depth 1 -b "${CRIU_VER}" https://github.com/checkpoint-restore/criu "${T}/criu"
  make -C "${T}/criu" -j"$(nproc)" criu >"${T}/build.log" 2>&1
  if [ -x "${T}/criu/criu/criu" ]; then cp "${T}/criu/criu/criu" "${EXP_DIR}/bin/criu"
  else echo "criu BUILD FAILED:"; tail -15 "${T}/build.log"; exit 1; fi
fi
echo "criu: $("${EXP_DIR}/bin/criu" --version 2>&1 | head -1)"

# Bundle criu's full non-glibc runtime deps, so the same binary runs BOTH on the
# bare SLES host (which lacks libnet) and inside the stock nvidia/cuda container
# (which lacks all of them -- earlier runs only worked because the build step had
# just apt-installed them into that same container instance).
# glibc itself is deliberately NOT bundled: the host's (2.38) and the container's
# (2.35) both satisfy a jammy-built binary, and shipping a libc would break things.
for lib in libnet.so.1 libbsd.so.0 libprotobuf-c.so.1 libnl-3.so.200 libmd.so.0; do
  src=$(find /usr/lib/x86_64-linux-gnu -name "${lib}" | head -1)
  if [ -n "${src}" ]; then cp -L "${src}" "${EXP_DIR}/hostlibs/${lib}"; echo "bundled ${lib}"
  else echo "WARN: ${lib} not found in container"; fi
done

echo "-- compiling counter (static cudart) --"
nvcc -cudart static -o "${EXP_DIR}/src/counter" "${EXP_DIR}/src/counter.cu" \
  && echo "counter compiled: $(ls -l "${EXP_DIR}/src/counter" | awk '{print $5}') bytes"
echo "bundle step done."
