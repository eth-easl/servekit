#!/bin/bash
# Status for the glm-47-flash SGLang deployment on bristen, routed through rcc.
#
# Runs entirely via `rcc`, so it reuses the shared SSH ControlMaster and
# resolves host + remote_dir from .rcc/config.toml (no hardcoded paths).
# Override the profile with RCC_PROFILE (or call `rcc` directly).
set -euo pipefail

PROFILE="${RCC_PROFILE:-glm-47-flash-bristen}"

exec rcc --profile "${PROFILE}" run bash -lc '
  set -euo pipefail
  if [[ -f last_service.env ]]; then
    cat last_service.env
  else
    echo "No last_service.env found in $(pwd)"
  fi
  echo
  squeue -u "$USER" -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
'
