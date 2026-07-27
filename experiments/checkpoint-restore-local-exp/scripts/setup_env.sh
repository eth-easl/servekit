#!/usr/bin/env bash
# One-time environment + model setup for the local Apertus-8B checkpoint experiment.
#
# We install SGLang NATIVELY in a venv (not the cluster container) on purpose: a
# container reintroduces the Seccomp:2 filter that blocked CRIU on Bristen, and the
# whole point here is bare-host checkpointing. Driver 610 runs torch 2.9.1 / cu12 fine.
#
# Matches the cluster pin: lmsysorg/sglang:v0.5.10  (torch 2.9.1, CUDA 12).
# Model: swiss-ai/Apertus-8B-Instruct-2509 (ApertusForCausalLM, needs --trust-remote-code).
#
# Phased + idempotent. Usage:
#   scripts/setup_env.sh venv       # create ~/ckpt-venv
#   scripts/setup_env.sh install    # pip install sglang[all]==0.5.10 + servekit + hf_hub
#   scripts/setup_env.sh validate   # import sglang/torch, confirm CUDA
#   scripts/setup_env.sh download   # hf download Apertus-8B to ~/models (~16 GB)
#   scripts/setup_env.sh all        # everything above, in order
set -u

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"          # repo root
SERVEKIT_DIR="$REPO/servekit"
VENV="${CKPT_VENV:-$HOME/ckpt-venv}"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.10}"
MODEL_ID="swiss-ai/Apertus-8B-Instruct-2509"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Apertus-8B-Instruct-2509}"

hr() { printf '\n========== %s ==========\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

free_gb() { df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9'; }

do_venv() {
  hr "venv -> $VENV"
  if [ -x "$PY" ]; then echo "exists, skipping"; else
    python3.12 -m venv "$VENV" || die "venv creation failed (need python3.12-venv?)"
    "$PIP" install -q --upgrade pip setuptools wheel || die "pip upgrade failed"
  fi
  "$PY" --version
}

do_install() {
  hr "install sglang[all]==$SGLANG_VERSION + servekit + huggingface_hub"
  echo "disk free before: $(free_gb) GiB"
  [ "$(free_gb)" -lt 25 ] && die "less than 25 GiB free; free space first"
  # sglang[all] pulls torch 2.9.1/cu12, flashinfer, sgl-kernel (prebuilt wheels).
  "$PIP" install -q "sglang[all]==$SGLANG_VERSION" || die "sglang install failed (see Risks: kernel/flashinfer resolution)"
  # ninja: SGLang JIT-compiles some kernels at first launch and needs the ninja binary.
  "$PIP" install -q ninja || die "ninja install failed"
  # servekit: pure-stdlib, editable, no deps. huggingface_hub for the download step.
  "$PIP" install -q -e "$SERVEKIT_DIR" || die "servekit install failed"
  "$PIP" install -q "huggingface_hub[cli,hf_transfer]" || die "huggingface_hub install failed"
  echo "disk free after: $(free_gb) GiB"
}

do_validate() {
  hr "validate imports + CUDA"
  "$PY" - <<'PY' || die "validation failed"
import torch, sglang
print("sglang :", sglang.__version__)
print("torch  :", torch.__version__)
print("cuda   :", torch.version.cuda, "| available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available to torch"
x = torch.ones(1024, 1024, device="cuda")
print("gpu    :", torch.cuda.get_device_name(0), "| matmul ok:", bool((x @ x).sum().item() > 0))
# servekit must import too (it wraps the launch)
import servekit.cli  # noqa
print("servekit: import OK")
PY
}

do_download() {
  hr "download $MODEL_ID -> $MODEL_DIR"
  if [ -d "$MODEL_DIR" ] && ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
    echo "weights present ($(du -sh "$MODEL_DIR" | cut -f1)), skipping"
    return 0
  fi
  echo "disk free before: $(free_gb) GiB"
  [ "$(free_gb)" -lt 20 ] && die "less than 20 GiB free; can't fit ~16 GB weights"
  mkdir -p "$MODEL_DIR"
  # hf_transfer for a faster pull; falls back automatically if unavailable.
  HF_HUB_ENABLE_HF_TRANSFER=1 "$VENV/bin/hf" download "$MODEL_ID" \
    --local-dir "$MODEL_DIR" || die "hf download failed"
  echo "downloaded: $(du -sh "$MODEL_DIR" | cut -f1)"
}

CMD="${1:-all}"
case "$CMD" in
  venv)     do_venv ;;
  install)  do_venv; do_install ;;
  validate) do_validate ;;
  download) do_download ;;
  all)      do_venv; do_install; do_validate; do_download ;;
  *) echo "usage: $0 {venv|install|validate|download|all}"; exit 2 ;;
esac
hr "done: $CMD"
