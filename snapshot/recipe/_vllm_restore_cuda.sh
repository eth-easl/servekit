#!/bin/bash
# _vllm_restore_cuda.sh — N5b Task 5/7: RESTORE-mode cold start. Reuses the
# .snap + entry.output meta recorded by _vllm_record_cuda.sh (same SNAP_ROOT /
# META_ROOT). Thin wrapper around _vllm_record_cuda.sh with MODE=restore — the
# launcher env-selects the restore path (VLLM_CG_RESTORE_META, skip forward).
set -uo pipefail
export MODE="${MODE:-restore}"
exec bash "$(dirname "$0")/_vllm_record_cuda.sh" "$@"
