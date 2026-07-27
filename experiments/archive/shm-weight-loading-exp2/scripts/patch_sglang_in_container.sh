#!/bin/bash
# Patch SGLang source INSIDE the container, hermetically.
#
# Shared by phase3 (files-per-rank, 1 file) and phase5 (InstantTensor backport,
# 4 files). Depends on NOTHING in the working tree except the .patch you hand
# it: each job clones sglang fresh at the SHA pinned to the container image, so
# a run is reproducible from the repo alone and cannot be perturbed by whatever
# happens to be checked out at /iopsstor/.../sglang.
#
# The three steps, and why:
#   1. CLONE  sgl-project/sglang at the tag matching the image, then assert the
#      resolved SHA equals SGLANG_SHA. Fetching by tag always works; asserting
#      the SHA afterwards catches a moved/retagged release.
#   2. VERIFY every file the patch touches is byte-identical between the clone
#      and the sglang actually INSTALLED in the image. THIS DIFF IS THE VERSION
#      CHECK: if the pin drifts from the image, the job dies here instead of
#      quietly serving from different engine code and reporting a bogus number.
#   3. SWAP   apply the patch in the clone, then copy those files over the live
#      install. Everything else stays the image's blessed install; it all dies
#      with the container overlay.
#
# The file list is DERIVED FROM THE PATCHES ("+++ b/..." lines) - nothing to
# keep in sync by hand, and a 1-file or 4-file patch works identically.
#
# TAKES ALL PATCHES AT ONCE, and that is load-bearing. Calling this script twice
# does NOT compose: the second call re-clones a pristine tree and then diffs it
# against the INSTALLED file, which the first call already modified -> the
# version check trips and the job aborts. (It did exactly that, correctly,
# rather than serve a bogus number.) When two patches touch the same file
# (phase3's weight_utils.py knob + phase5's InstantTensor iterator), they must
# be applied to ONE clone and swapped ONCE.
#
# We never import sglang (importlib.util.find_spec / importlib.metadata
# instead): importing it drags in torch and would need a GPU, and this must
# also run in the GPU-less preflight job.
#
# Usage: patch_sglang_in_container.sh <patch-file> [<patch-file> ...]
set -euo pipefail

[[ $# -ge 1 ]] || { echo "usage: patch_sglang_in_container.sh <patch-file> [<patch-file> ...]" >&2; exit 1; }

# Pinned to the image in profile/llama-3.1-70b-bristen/llama-3.1-70b-sglang.toml:
#   image = "lmsysorg/sglang:v0.5.10"
SGLANG_TAG="${SGLANG_TAG:-v0.5.10}"
SGLANG_SHA="${SGLANG_SHA:-1519acf37c23f2189adb93f57ca9cd2db1bebf18}"
SGLANG_REPO="${SGLANG_REPO:-https://github.com/sgl-project/sglang.git}"
SRC="${SGLANG_SRC:-${HOME}/sglang-src}"   # HOME=/root -> ephemeral container overlay

command -v git >/dev/null || { echo "[patch] FATAL: git not in container" >&2; exit 1; }

PATCHES=()
for p in "$@"; do
  [[ -f "$p" ]] || { echo "[patch] FATAL: no patch file at $p" >&2; exit 1; }
  PATCHES+=("$(cd "$(dirname "$p")" && pwd)/$(basename "$p")")
done

# Union of files the patches touch, straight from the diff headers.
mapfile -t TOUCHED < <(grep -h '^+++ b/' "${PATCHES[@]}" | sed 's|^+++ b/||' | sort -u)
[[ ${#TOUCHED[@]} -gt 0 ]] || { echo "[patch] FATAL: patches touch no files?" >&2; exit 1; }

# --- locate the INSTALLED sglang without importing it (no torch, no GPU) ---
SITE="$(python -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("sglang").origin))')"
VERSION="$(python -c 'import importlib.metadata as m; print(m.version("sglang"))')"
echo "[patch] installed sglang ${VERSION} at ${SITE}"
echo "[patch] ${#PATCHES[@]} patch(es) -> ${#TOUCHED[@]} distinct file(s)"

# --- 1. clone at the pinned ref ---
rm -rf "$SRC"
git clone -q --depth 1 --branch "$SGLANG_TAG" "$SGLANG_REPO" "$SRC"
GOT_SHA="$(git -C "$SRC" rev-parse HEAD)"
if [[ "$GOT_SHA" != "$SGLANG_SHA" ]]; then
  echo "[patch] FATAL: tag ${SGLANG_TAG} resolved to ${GOT_SHA}, expected ${SGLANG_SHA}" >&2
  echo "[patch]        the upstream tag moved - re-pin SGLANG_SHA before trusting any number." >&2
  exit 1
fi
echo "[patch] cloned ${SGLANG_REPO} @ ${SGLANG_TAG} = ${GOT_SHA}"

# --- 2. verify the pin matches the image, for EVERY file the patch touches ---
for rel in "${TOUCHED[@]}"; do
  installed="${SITE}/${rel#python/sglang/}"
  [[ -f "$installed" ]] || { echo "[patch] FATAL: installed file missing: $installed" >&2; exit 1; }
  if ! diff -q "$installed" "${SRC}/${rel}" >/dev/null; then
    echo "[patch] FATAL: pinned source != sglang installed in the image, for ${rel}" >&2
    echo "[patch]        The image was not built from ${SGLANG_TAG}, or was patched on top of it." >&2
    echo "[patch]        Refusing to run: the swapped file would not be the code the image ships." >&2
    diff -u "$installed" "${SRC}/${rel}" | head -40 >&2
    exit 1
  fi
done
echo "[patch] verified: all ${#TOUCHED[@]} file(s) match the image byte-for-byte"

# --- 3. apply EVERY patch to the one clone, then swap the files in once ---
for p in "${PATCHES[@]}"; do
  git -C "$SRC" apply "$p"
  echo "[patch]   applied $(basename "$p")"
done
for rel in "${TOUCHED[@]}"; do
  installed="${SITE}/${rel#python/sglang/}"
  cp "${SRC}/${rel}" "$installed"
  python -m py_compile "$installed"
  echo "[patch]   swapped ${rel#python/sglang/}"
done

echo "[patch] done"
