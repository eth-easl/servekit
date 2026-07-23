#!/bin/bash
# Build the foundry-enabled SGLang image and export it as a squashfs for enroot.
# Run once, on a login node (podman graphroot is /dev/shm, ~176G free).
#
#   bash foundry-tp1/build_image.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG=foundry-sglang-apertus
BASE=docker.io/lmsysorg/sglang:v0.5.11   # keep in sync with Containerfile FROM
SQSH="${HERE}/apertus-8b-foundry.sqsh"

if [ ! -d "${HERE}/foundry" ]; then
  echo "==> cloning foundry"
  git clone --depth 1 https://github.com/foundry-org/foundry.git "${HERE}/foundry"
fi

echo "==> podman build (this pulls ~20G base; first run is slow)"
podman build -t "${TAG}" -f "${HERE}/Containerfile" "${HERE}"

# Boost/CMake are the two things most likely to have silently degraded, so
# assert on them rather than trusting the build to have failed loudly.
echo "==> verifying build products"
# The image carries dependencies only; the sglang patch is applied at job start,
# so do not look for foundry_shim/ServerArgs here.
podman run --rm "${TAG}" bash -c '
  set -e
  ls -l /opt/foundry/python/foundry/libcuda_hook.so
  python -c "import sglang, sys; print(\"sglang\", sglang.__version__)"
  python -c "from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig; \
             print(\"pool_configurator present (native)\")"
'

# The whole experiment rests on baseline and foundry runs sharing one engine
# build. An earlier attempt silently downgraded nccl and cudnn via foundry's
# torch dependency, so assert that the ONLY package changes are the ones we
# intended. Anything else means the engine moved and the comparison is invalid.
echo "==> asserting the build did not mutate the engine"
podman run --rm "${BASE}" pip freeze 2>/dev/null | sort > /tmp/pf-base.txt
podman run --rm "${TAG}"  pip freeze 2>/dev/null | sort > /tmp/pf-built.txt
UNEXPECTED=$(diff /tmp/pf-base.txt /tmp/pf-built.txt \
  | grep -E '^[<>]' \
  | grep -viE 'foundry|fastsafetensors|^[<>] cmake|ninja' || true)
if [ -n "${UNEXPECTED}" ]; then
  echo "FAIL: build changed packages beyond foundry/fastsafetensors/cmake/ninja:" >&2
  echo "${UNEXPECTED}" >&2
  exit 1
fi
echo "    engine packages unchanged"

echo "==> exporting squashfs -> ${SQSH}"
rm -f "${SQSH}"
# enroot exits non-zero on temp-dir cleanup noise even when the export is fine,
# so judge it by the artifact rather than the status code.
enroot import -x mount -o "${SQSH}" "podman://${TAG}" || true

echo "==> validating squashfs"
unsquashfs -s "${SQSH}" >/dev/null 2>&1 || { echo "FAIL: ${SQSH} is not a valid squashfs" >&2; exit 1; }
# grep -c, not grep -q: -q closes the pipe on first match, unsquashfs takes
# SIGPIPE, and `set -o pipefail` then reports a false failure. -c drains it.
HOOK_COUNT=$(unsquashfs -l "${SQSH}" 2>/dev/null | grep -c 'foundry/libcuda_hook.so$' || true)
[ "${HOOK_COUNT}" -ge 1 ] \
  || { echo "FAIL: libcuda_hook.so missing from the exported image" >&2; exit 1; }

echo "==> done: ${SQSH}"
ls -lh "${SQSH}"
