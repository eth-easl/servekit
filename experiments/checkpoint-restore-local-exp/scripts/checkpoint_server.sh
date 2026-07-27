#!/usr/bin/env bash
# Checkpoint a warmed SGLang server to disk (cuda-checkpoint + CRIU), so it can later be
# restored in ~18s instead of a ~59s cold launch (see restore_server.sh).
#
# Modes:
#   scripts/checkpoint_server.sh 3a          # de-risk: Python+torch.cuda round trip
#   scripts/checkpoint_server.sh checkpoint  # launch Apertus-8B, warm it, snapshot to disk
#   scripts/checkpoint_server.sh 3b          # checkpoint, then restore_server.sh --cold
#
# What it took to make a full SGLang server checkpointable, unprivileged, on the bare host
# (each discovered the hard way; see results/gate3b_server_checkpoint_restore/results.md):
#   1. cuda-checkpoint EVERY process holding /dev/nvidia VMAs (launcher + scheduler +
#      detokenizer), not just the nvidia-smi compute-app -- else criu hits an
#      unevicted device VMA.
#   2. No io_uring anywhere (criu can't dump it): the uvloop->epoll shim (shim/uvloop,
#      loaded via a .pth so it reaches SGLang's env-scrubbed scheduler) + USE_LIBUV=0
#      (torch TCPStore's libuv). ensure_shim() installs the .pth.
#   3. criu flags: --tcp-close (drop the torch TCPStore TCP conn; TCP_REPAIR needs
#      CAP_NET_ADMIN we lack) + --network-lock skip (iptables lock needs CAP_NET_ADMIN)
#      + --shell-job --file-locks --link-remap --ext-unix-sk.
# Keeping the process at Seccomp:0 is essential (a seccomp filter would need CAP_SYS_ADMIN
# for criu to suspend it) -- so we do NOT filter syscalls; we remove io_uring at the source.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${CKPT_VENV:-$HOME/ckpt-venv}"
PYBIN="$VENV/bin/python"
CRIU="$(command -v criu || echo /usr/sbin/criu)"
CCKPT="$HERE/bin/cuda-checkpoint"
export PATH="$VENV/bin:$PATH"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/criu-xdg}"; mkdir -p "$XDG_RUNTIME_DIR"

MODEL_DIR="${MODEL_DIR:-$HOME/models/Apertus-8B-Instruct-2509}"
SERVED_NAME="swiss-ai/Apertus-8B-Instruct-2509"
PORT="${PORT:-8080}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"; MEM_FRACTION="${MEM_FRACTION:-0.85}"; MAX_RUNNING="${MAX_RUNNING:-8}"
IMG="$HERE/results/gate3b_server_checkpoint_restore/gate3b-run/img"
SERVER_LOG="$HERE/results/gate3b_server_checkpoint_restore/ckpt_server.log"

# criu flags that make the full SGLang tree dumpable unprivileged (see header)
CRIU_DUMP_ARGS="--shell-job --tcp-close --network-lock skip --file-locks --link-remap --ext-unix-sk"

now_ms(){ date +%s%3N; }
tree_pids(){ local r=$1; echo "$r"; for k in $(pgrep -P "$r" 2>/dev/null); do tree_pids "$k"; done; }
gpu_vma_pids(){ local root=$1 p; for p in $(tree_pids "$root" | sort -un); do
    [ "$(grep -c /dev/nvidia "/proc/$p/maps" 2>/dev/null)" -gt 0 ] && echo "$p"; done; }
kill_servers(){ local p c; for p in $(pgrep -f "sglang.launch_server" 2>/dev/null); do
    c=$(cat "/proc/$p/comm" 2>/dev/null); case "$c" in python|sglang::*) kill -9 "$p" 2>/dev/null;; esac; done; }

# Install the uvloop->epoll shim via a .pth so it loads in EVERY venv process, including
# SGLang's env-scrubbed scheduler subprocess (a PYTHONPATH shim can't reach it). Also sets
# USE_LIBUV=0 there. Idempotent.
ensure_shim(){
  local sp pth
  sp="$("$PYBIN" -c 'import site;print(site.getsitepackages()[0])')"
  pth="$sp/zz_uvloop_epoll_shim.pth"
  printf "import os; os.environ.setdefault('USE_LIBUV','0'); import sys; sys.path.insert(0, '%s/shim')\n" "$HERE" > "$pth"
  # sanity: import uvloop must resolve to our shim
  env -u PYTHONPATH "$PYBIN" -c "import uvloop,sys; assert 'checkpoint-restore-local-exp/shim' in uvloop.__file__, uvloop.__file__" \
    || { echo "shim install FAILED"; return 1; }
  echo "shim active: $pth"
}

launch_and_warm(){   # -> writes launcher pid to stdout; server ready + warmed
  kill_servers; sleep 2
  TORCHINDUCTOR_COMPILE_THREADS=1 HF_HUB_OFFLINE=1 GLOO_SOCKET_IFNAME=lo \
  nohup "$PYBIN" -m sglang.launch_server \
    --model-path "$MODEL_DIR" --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" --tensor-parallel-size 1 --trust-remote-code \
    --context-length "$CONTEXT_LEN" --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$MAX_RUNNING" --enable-metrics \
    > "$SERVER_LOG" 2>&1 &
  local lp=$! i=0
  while [ $i -lt 180 ]; do grep -q "fired up and ready" "$SERVER_LOG" 2>/dev/null && break; sleep 1; i=$((i+1)); done
  grep -q "fired up and ready" "$SERVER_LOG" 2>/dev/null || { echo "server never became ready" >&2; tail -8 "$SERVER_LOG" >&2; return 1; }
  # warm (triggers JIT / first request path)
  "$PYBIN" -c "import json,urllib.request as u; u.urlopen(u.Request('http://127.0.0.1:$PORT/generate',data=json.dumps({'text':'Hello','sampling_params':{'temperature':0,'max_new_tokens':4}}).encode(),headers={'Content-Type':'application/json'}),timeout=30).read()" 2>/dev/null
  echo "$lp"
}

do_checkpoint(){   # $1 = launcher/root pid
  local root=$1
  local gpids; gpids="$(gpu_vma_pids "$root" | tr '\n' ' ')"
  echo "  GPU-VMA pids: $gpids"
  # audit: no io_uring may remain (criu can't dump it)
  local iou=0 p; for p in $(tree_pids "$root"); do iou=$((iou + $(ls -la /proc/$p/fd 2>/dev/null | grep -c io_uring))); done
  echo "  io_uring fds in tree: $iou $([ $iou -ne 0 ] && echo '(WARNING: criu will fail — shim not active?)')"
  echo "  gpu before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

  local t0; t0=$(now_ms)
  for p in $gpids; do "$CCKPT" --action lock       --pid "$p" >/dev/null || { echo "lock $p FAILED"; return 1; }; done
  for p in $gpids; do "$CCKPT" --action checkpoint --pid "$p" >/dev/null || { echo "checkpoint $p FAILED"; return 1; }; done
  local t1; t1=$(now_ms)
  echo "  cuda-checkpoint (GPU evict): $((t1-t0)) ms | gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

  rm -rf "$IMG"; mkdir -p "$IMG"
  "$CRIU" --unprivileged dump -t "$root" $CRIU_DUMP_ARGS -D "$IMG" -o dump.log
  local drc=$? t2; t2=$(now_ms)
  if [ $drc -ne 0 ]; then echo "  criu dump FAILED (rc=$drc):"; grep -iE "Error" "$IMG/dump.log" | tail -12; return 1; fi
  printf "gpu_pids=%s\nroot=%s\nport=%s\n" "$gpids" "$root" "$PORT" > "$IMG/ckpt_meta.txt"
  echo "  criu dump: $((t2-t1)) ms | snapshot: $(du -sh "$IMG"|cut -f1) | total checkpoint: $((t2-t0)) ms"
  echo "  meta: $IMG/ckpt_meta.txt"
}

# ------------------------------------------------------------ gate 3a (unchanged logic)
gate_3a(){
  local WORK="$HERE/results/gate3a_pytorch_cuda_roundtrip/gate3a-run" LOG IMG3; LOG="$WORK/py_counter.log"; IMG3="$WORK/img"
  rm -rf "$WORK"; mkdir -p "$IMG3"
  echo "==== Gate 3a :: Python+torch.cuda checkpoint/restore ===="
  "$PYBIN" "$HERE/src/python_cuda_counter.py" "$LOG" 1000 & local PID=$!
  local i=0; until grep -q "tick=" "$LOG" 2>/dev/null; do sleep 1; i=$((i+1)); [ $i -gt 60 ] && { echo "no tick"; kill $PID; return 1; }; done
  sleep 2; local FROZE; FROZE=$(grep -oE 'tick=[0-9]+' "$LOG"|tail -1|cut -d= -f2)
  "$CCKPT" --action lock --pid $PID >/dev/null; "$CCKPT" --action checkpoint --pid $PID >/dev/null
  "$CRIU" --unprivileged dump -t $PID --shell-job -D "$IMG3" -o dump.log || { echo "3a dump FAIL"; grep -i error "$IMG3/dump.log"|tail; return 1; }
  sleep 4
  "$CRIU" --unprivileged restore -D "$IMG3" --shell-job -d -o restore.log || { echo "3a restore FAIL"; return 1; }
  "$CCKPT" --action restore --pid $PID >/dev/null; "$CCKPT" --action unlock --pid $PID >/dev/null
  sleep 3
  local FIRST; FIRST=$(awk -F= -v f="$FROZE" '/tick=/{if($2>f){print $2;exit}}' "$LOG")
  kill $PID 2>/dev/null
  [ "$FIRST" = "$((FROZE+1))" ] && echo "GATE 3a: PASS (froze $FROZE, resumed $FIRST)" || { echo "GATE 3a: FAIL"; return 1; }
}

case "${1:-}" in
  3a) gate_3a ;;
  checkpoint)
    ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || { echo "weights missing — setup_env.sh download"; exit 2; }
    ensure_shim || exit 1
    echo "==== checkpoint Apertus-8B server ===="
    ROOT=$(launch_and_warm) || exit 1
    echo "  launcher pid=$ROOT (ready+warm)"
    do_checkpoint "$ROOT" || exit 1
    echo "==== done — restore with: scripts/restore_server.sh --cold ===="
    ;;
  3b)
    ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || { echo "weights missing — setup_env.sh download"; exit 2; }
    ensure_shim || exit 1
    ROOT=$(launch_and_warm) || exit 1
    echo "launcher pid=$ROOT"; do_checkpoint "$ROOT" || exit 1
    echo; bash "$HERE/scripts/restore_server.sh" --cold
    ;;
  *) echo "usage: $0 {3a|checkpoint|3b}"; exit 2 ;;
esac
