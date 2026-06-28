#!/bin/bash
# HIP-free analyzer: compile + run on the login node (no ROCm needed).
set -euo pipefail
D=/capstor/scratch/cscs/xyao/kimi-k25-vllm
g++ -std=c++17 -O2 -I${D}/snapshot/include \
  ${D}/snapshot/csrc/core/record.cpp \
  ${D}/snapshot/csrc/core/serialize.cpp \
  ${D}/snapshot/csrc/core/hashing.cpp \
  ${D}/snapshot/csrc/core/graph_model.cpp \
  ${D}/snapshot/csrc/cli/analyze_tool.cpp \
  -o /tmp/analyze
echo "=== compiled OK ==="
/tmp/analyze "${D}/snapshot/record-vllm/graph-69920-0.snap"
