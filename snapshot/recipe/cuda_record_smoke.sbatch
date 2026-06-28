#!/bin/bash
#SBATCH --job-name=snapshot-cuda-record-smoke
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/xyao/snapshot-cuda/logs/%x-%j.out

# N5a consolidated record→restore driver.
# Sequence: build → bare-smoke determinism → record-interposer → Δ=0 record (×2) → restore.
# Emits: N5A_GATES: G1=PASS G2=PASS G3=PASS G4=PASS
#
# Gate map:
#   G1  bit-identical restored output (CHECKSUM_restore == CHECKSUM_record)
#   G2  capture genuinely skipped in restore run
#         (restored=1, fallthrough=0, real_begin=0, suppressed=2)
#   G3  Δ=0 determinism
#         (passthrough=0 / fixed_base_honored=1 + byte-identical .snap across two record runs)
#   G4  both identity paths resolved (blind=0; identity: 2 functions, 1 fatbin, 1 module)
#
# See snapshot/RESULTS.md §N5a for full gate narrative.

set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/snapshot-cuda}"
export DEPLOY_DIR
SCRIPT_DIR="${DEPLOY_DIR}/snapshot/recipe"
export EDF_PATH="${SCRIPT_DIR}:${EDF_PATH:-${HOME}/.edf}"
mkdir -p "${DEPLOY_DIR}/logs" "${DEPLOY_DIR}/snapshot/build-cuda"

echo "cuda-record-smoke started: $(date --iso-8601=seconds)"
echo "node: $(hostname)"

srun --environment=snapshot-cuda \
  bash -lc 'set -euo pipefail
    CMAKE_PREFIX="${DEPLOY_DIR}/cmake/cmake-3.30.8-linux-x86_64/bin"
    if [ -x "${CMAKE_PREFIX}/cmake" ]; then export PATH="${CMAKE_PREFIX}:${PATH}"; fi

    # A100 (sm_80): set CUDA_ARCHITECTURES so nvcc compiles the static k_add for the
    # right target.  Also passed to CMake so enable_language(CUDA) picks it up.
    cmake -S snapshot -B snapshot/build-cuda \
      -DSNAPSHOT_BACKEND=CUDA \
      -DCMAKE_CUDA_ARCHITECTURES=80

    # Build: smoke binary + record interposer + redirect interposer.
    cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" \
      --target cuda_record_smoke cuda_record_runtime_smoke cuda_record_full_smoke cuda_region_base_probe snapshot_record_cuda snapshot_redirect_cuda

    SMOKE="$PWD/snapshot/build-cuda/cuda_record_smoke"
    RT_SMOKE="$PWD/snapshot/build-cuda/cuda_record_runtime_smoke"
    FULL_SMOKE="$PWD/snapshot/build-cuda/cuda_record_full_smoke"
    PROBE="$PWD/snapshot/build-cuda/cuda_region_base_probe"
    RECORD_SO="$PWD/snapshot/build-cuda/libsnapshot_record_cuda.so"
    REDIRECT_SO="$PWD/snapshot/build-cuda/libsnapshot_redirect_cuda.so"
    ls -l "$SMOKE" "$RT_SMOKE" "$FULL_SMOKE" "$PROBE" "$RECORD_SO" "$REDIRECT_SO"

    # Persistent log paths — no /tmp (diskless compute nodes + reproducibility).
    LOG1="${DEPLOY_DIR}/logs/smoke-run1.log"
    LOG2="${DEPLOY_DIR}/logs/smoke-run2.log"
    LOG_RECORD="${DEPLOY_DIR}/logs/record-interposer.log"

    # -------------------------------------------------------------------
    # Bare smoke: two determinism-baseline runs (no interposer).
    # -------------------------------------------------------------------
    echo "--- run 1 ---"
    "$SMOKE" 2>&1 | tee "${LOG1}"; SMOKE_EXIT1="${PIPESTATUS[0]}"
    echo "smoke-run1 exit=${SMOKE_EXIT1}"

    echo "--- run 2 ---"
    "$SMOKE" 2>&1 | tee "${LOG2}"; SMOKE_EXIT2="${PIPESTATUS[0]}"
    echo "smoke-run2 exit=${SMOKE_EXIT2}"

    [ "${SMOKE_EXIT1}" -eq 0 ] && [ "${SMOKE_EXIT2}" -eq 0 ] || \
      { echo "SMOKE_BARE_FAIL (run1=${SMOKE_EXIT1} run2=${SMOKE_EXIT2})"; exit 1; }

    echo "--- determinism check ---"
    CS1=$(grep "^CHECKSUM" "${LOG1}" || echo "MISSING")
    CS2=$(grep "^CHECKSUM" "${LOG2}" || echo "MISSING")
    echo "run1: $CS1"
    echo "run2: $CS2"
    if [ "$CS1" = "$CS2" ] && [ "$CS1" != "MISSING" ]; then
      echo "CHECKSUM_DETERMINISTIC=1"
    else
      echo "CHECKSUM_DETERMINISTIC=0"
      exit 1
    fi

    # -------------------------------------------------------------------
    # Record interposer run (G4 identity gate).
    # Runs with record interposer only (no redirect) to confirm:
    #   (a) CHECKSUM unchanged — interposer is transparent.
    #   (b) SUMMARY: identity: 2 functions (1 fatbin, 1 module)
    #       i.e. k_add resolved via __cudaRegisterFunction (fatbin path) AND
    #            k_mul resolved via cuModuleGetFunction (module/nvrtc path).
    # -------------------------------------------------------------------
    SNAP_DIR="${DEPLOY_DIR}/snap-record-smoke"
    mkdir -p "${SNAP_DIR}"

    echo "--- record interposer run (G4 identity gate) ---"
    LD_PRELOAD="${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=record \
      SNAPSHOT_RECORD_CUDA_DIR="${SNAP_DIR}" \
      "$SMOKE" 2>&1 | tee "${LOG_RECORD}"
    RECORD_EXIT="${PIPESTATUS[0]}"

    echo "--- identity gate ---"
    SUMMARY=$(grep "\[record-cuda\].*SUMMARY" "${LOG_RECORD}" || echo "SUMMARY_MISSING")
    echo "summary line: ${SUMMARY}"

    # Gate (a): smoke still exits clean and prints the same CHECKSUM.
    CS_RECORD=$(grep "^CHECKSUM" "${LOG_RECORD}" || echo "MISSING")
    echo "checksum under interposer: ${CS_RECORD}"
    if [ "${CS_RECORD}" = "${CS1}" ] && [ "${CS_RECORD}" != "MISSING" ]; then
      echo "CHECKSUM_UNCHANGED=1"
    else
      echo "CHECKSUM_UNCHANGED=0 (expected ${CS1}, got ${CS_RECORD})"
      exit 1
    fi

    # Gate (b): SUMMARY reports both identities.
    if echo "${SUMMARY}" | grep -qF "identity: 2 functions (1 fatbin, 1 module)"; then
      echo "IDENTITY_GATE=1 (k_add via fatbin + k_mul via module: both resolved)"
      IDENTITY_OK=1
    else
      echo "IDENTITY_GATE=0 (expected \"identity: 2 functions (1 fatbin, 1 module)\" not found)"
      exit 1
    fi

    [ "${RECORD_EXIT}" -eq 0 ] || \
      { echo "RECORD_RUN_EXIT=${RECORD_EXIT} (smoke exited non-zero under interposer)"; exit 1; }

    echo "Identity gate PASSED"

    # -------------------------------------------------------------------
    # G3 gate: record the captured graph to .snap under redirect+record,
    # twice into two separate dirs, then prove byte-identical (Δ=0).
    # The smoke allocates d_a/d_out via the RUNTIME cudaMalloc (see
    # cuda_record_smoke.cpp), which snapshot_redirect_cuda hooks, so the
    # redirect serves them from its fixed-base region — that is what makes
    # the recorded kernarg pointers (and the .snap) byte-identical across
    # runs.
    # -------------------------------------------------------------------
    echo "--- G3: capture/record + Δ=0 ---"
    T3_DIR1="${DEPLOY_DIR}/snap-record-t3/run1"
    T3_DIR2="${DEPLOY_DIR}/snap-record-t3/run2"
    T3_LOG1="${DEPLOY_DIR}/snap-record-t3/r1.log"
    T3_LOG2="${DEPLOY_DIR}/snap-record-t3/r2.log"
    rm -rf "${T3_DIR1}" "${T3_DIR2}"
    mkdir -p "${T3_DIR1}" "${T3_DIR2}"

    echo "--- record run 1 (LD_PRELOAD=redirect:record) ---"
    LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=record \
      SNAPSHOT_RECORD_CUDA_DIR="${T3_DIR1}" \
      "$SMOKE" 2>&1 | tee "${T3_LOG1}"
    T3_EXIT1="${PIPESTATUS[0]}"

    echo "--- record run 2 (LD_PRELOAD=redirect:record) ---"
    LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=record \
      SNAPSHOT_RECORD_CUDA_DIR="${T3_DIR2}" \
      "$SMOKE" 2>&1 | tee "${T3_LOG2}"
    T3_EXIT2="${PIPESTATUS[0]}"

    if [ "${T3_EXIT1}" -ne 0 ] || [ "${T3_EXIT2}" -ne 0 ]; then
      echo "T3_RUN_EXIT=fail (run1=${T3_EXIT1} run2=${T3_EXIT2})"
      exit 1
    fi

    SNAP1="${T3_DIR1}/graph-0000.snap"
    SNAP2="${T3_DIR2}/graph-0000.snap"

    # (a) .snap files exist and are non-empty.
    if [ -s "${SNAP1}" ] && [ -s "${SNAP2}" ]; then
      echo "SNAP_WRITTEN=1"
      ls -l "${SNAP1}" "${SNAP2}"
    else
      echo "SNAP_WRITTEN=0 (expected ${SNAP1} and ${SNAP2})"
      exit 1
    fi

    # (b) record SUMMARY reports 1 graph, 2 nodes, 1 edge, 0 blind (run 1).
    REC_SUMMARY=$(grep "\[record-cuda\].*SUMMARY" "${T3_LOG1}" || echo "REC_SUMMARY_MISSING")
    echo "record summary: ${REC_SUMMARY}"
    if echo "${REC_SUMMARY}" | grep -qF "recorded: graphs=1 nodes=2 edges=1 blind=0"; then
      echo "RECORD_COUNTS=1 (1 graph, 2 kernel nodes, 1 dep edge, 0 blind)"
    else
      echo "RECORD_COUNTS=0 (expected \"recorded: graphs=1 nodes=2 edges=1 blind=0\")"
      exit 1
    fi

    # (c) smoke still prints the same CHECKSUM under redirect+record.
    CS_T3=$(grep "^CHECKSUM" "${T3_LOG1}" || echo "MISSING")
    echo "checksum under redirect+record: ${CS_T3}"
    if [ "${CS_T3}" = "${CS1}" ] && [ "${CS_T3}" != "MISSING" ]; then
      echo "T3_CHECKSUM_UNCHANGED=1"
    else
      echo "T3_CHECKSUM_UNCHANGED=0 (expected ${CS1}, got ${CS_T3})"
      exit 1
    fi

    # (d) redirect ENGAGED gate: confirm passthrough=0 (all allocs served from the
    # fixed region) and fixed_base_honored=1 (region reserved at the requested base
    # 0x600000000000).
    REDIR_SUMMARY=$(grep "\[redirect-cuda\].*SUMMARY" "${T3_LOG1}" || echo "ABSENT")
    REDIR_FIXED=$(grep "\[redirect-cuda\].*fixed_base_honored" "${T3_LOG1}" || echo "ABSENT")
    echo "redirect summary: ${REDIR_SUMMARY}"
    echo "redirect fixed-vmm: ${REDIR_FIXED}"
    if echo "${REDIR_SUMMARY}" | grep -qE "passthrough=0\("; then
      echo "REDIR_PASSTHROUGH_ZERO=1 (all allocs served from fixed region)"
    else
      echo "REDIR_PASSTHROUGH_ZERO=0 (redirect did not serve allocs from region)"
      exit 1
    fi
    if echo "${REDIR_FIXED}" | grep -qF "fixed_base_honored=1"; then
      echo "REDIR_FIXED_BASE_HONORED=1 (region at requested fixed base)"
    else
      echo "REDIR_FIXED_BASE_HONORED=0 (fixed base not honored)"
      exit 1
    fi

    # (e) DECISIVE Δ=0 gate: the two .snap files must be byte-identical.
    if cmp "${SNAP1}" "${SNAP2}"; then
      echo "SNAP_BYTE_IDENTICAL=1 (Δ=0: recorded kernarg pointers are deterministic)"
    else
      echo "SNAP_BYTE_IDENTICAL=0 (.snap differs between runs — Δ≠0)"
      echo "--- cmp -l (differing bytes: expected ONLY device-pointer slots) ---"
      cmp -l "${SNAP1}" "${SNAP2}" | head -40 || true
      exit 1
    fi

    echo "G3 gate PASSED"

    # -------------------------------------------------------------------
    # G1/G2/G3/G4 gate: BIT-IDENTICAL RESTORE (restore run).
    #
    # Replay the graph recorded into T3_DIR1 (graph-0000.snap) WITHOUT real
    # capture: LD_PRELOAD=redirect:record, MODE=restore, same fixed-base env,
    # same .snap dir. The restore shim fakes begin-capture, suppresses the two
    # kernel launches (so nothing runs eagerly), and rebuilds+returns the graph
    # at end-capture; the smoke then instantiates+launches the rebuilt graph as
    # its SOLE execution.
    #
    # Gate fields asserted:
    #   G1  CHECKSUM_restore == CHECKSUM_record (bit-identical output)
    #   G2  restored=1  fallthrough=0  real_begin=0  suppressed=2
    #   G3  redirect passthrough=0 + fixed_base_honored=1 (Δ=0 in restore run)
    #   G4  blind=0 (both kernels reverse-resolved) + IDENTITY_OK from above
    # -------------------------------------------------------------------
    echo "--- G1/G2/G3/G4: bit-identical restore ---"
    T4_LOG="${DEPLOY_DIR}/snap-record-t3/restore.log"
    ls -l "${SNAP1}"  # the .snap we restore from (recorded by T3 run 1)

    echo "--- restore run (LD_PRELOAD=redirect:record, MODE=restore) ---"
    LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=restore \
      SNAPSHOT_RECORD_CUDA_REBUILD=1 \
      SNAPSHOT_RECORD_CUDA_SUPPRESS=1 \
      SNAPSHOT_RECORD_CUDA_DIR="${T3_DIR1}" \
      "$SMOKE" 2>&1 | tee "${T4_LOG}"
    T4_EXIT="${PIPESTATUS[0]}"

    [ "${T4_EXIT}" -eq 0 ] || \
      { echo "T4_RUN_EXIT=${T4_EXIT} (smoke exited non-zero in restore mode)"; exit 1; }

    REST_SUMMARY=$(grep "\[record-cuda\].*SUMMARY" "${T4_LOG}" || echo "REST_SUMMARY_MISSING")
    echo "restore summary: ${REST_SUMMARY}"

    # G1: bit-identical output.
    CS_T4=$(grep "^CHECKSUM" "${T4_LOG}" || echo "MISSING")
    echo "restore checksum: ${CS_T4} (record was ${CS1})"
    if [ "${CS_T4}" = "${CS1}" ] && [ "${CS_T4}" != "MISSING" ]; then
      echo "G1_RESTORE_BIT_IDENTICAL=1"
    else
      echo "G1_RESTORE_BIT_IDENTICAL=0 (expected ${CS1}, got ${CS_T4})"
      exit 1
    fi

    # G2: exactly one graph restored, no fall-through, no real begin-capture,
    # both launches suppressed. Each field checked independently so column
    # reordering in the SUMMARY line cannot silently break the gate.
    echo "${REST_SUMMARY}" | grep -q "restored=1" || \
      { echo "G2_FAIL: missing restored=1 in: ${REST_SUMMARY}"; exit 1; }
    echo "${REST_SUMMARY}" | grep -q "fallthrough=0" || \
      { echo "G2_FAIL: missing fallthrough=0 in: ${REST_SUMMARY}"; exit 1; }
    echo "${REST_SUMMARY}" | grep -qE "real_begin=0( |$)" || \
      { echo "G2_FAIL: missing real_begin=0 in: ${REST_SUMMARY}"; exit 1; }
    echo "${REST_SUMMARY}" | grep -qE "suppressed=2( |$)" || \
      { echo "G2_FAIL: missing suppressed=2 in: ${REST_SUMMARY}"; exit 1; }
    echo "G2_NO_REAL_CAPTURE=1 (restored=1 fallthrough=0 real_begin=0 suppressed=2)"

    # G3: redirect still pinned the fixed base in the restore run (Δ=0 holds).
    REST_REDIR=$(grep "\[redirect-cuda\].*SUMMARY" "${T4_LOG}" || echo "ABSENT")
    REST_FIXED=$(grep "\[redirect-cuda\].*fixed_base_honored" "${T4_LOG}" || echo "ABSENT")
    echo "restore redirect summary: ${REST_REDIR}"
    echo "restore redirect fixed-vmm: ${REST_FIXED}"
    if echo "${REST_REDIR}" | grep -qE "passthrough=0\(" && \
       echo "${REST_FIXED}" | grep -qF "fixed_base_honored=1"; then
      echo "G3_REDIRECT_FIXED_BASE=1 (passthrough=0, fixed_base_honored=1)"
    else
      echo "G3_REDIRECT_FIXED_BASE=0 (redirect did not pin the fixed base in restore run)"
      exit 1
    fi

    # G4: both kernels reverse-resolved (no blind nodes); combined with the
    # identity gate above (IDENTITY_OK=1) this confirms both resolution paths.
    echo "${REST_SUMMARY}" | grep -q "blind=0" || \
      { echo "G4_FAIL: missing blind=0 in: ${REST_SUMMARY}"; exit 1; }
    [ "${IDENTITY_OK:-0}" -eq 1 ] || \
      { echo "G4_FAIL: IDENTITY_OK not set (identity gate did not pass)"; exit 1; }
    echo "G4_NO_BLIND=1 (k_add + k_mul both resolved at restore; 2 functions 1 fatbin 1 module)"

    echo "Restore gate PASSED"

    # -------------------------------------------------------------------
    # Consolidated gate summary.
    # All gates passed (set -e + exit 1 above would have aborted otherwise).
    # -------------------------------------------------------------------
    echo ""
    echo "N5A_GATES: G1=PASS G2=PASS G3=PASS G4=PASS"

    # -------------------------------------------------------------------
    # N5b Task 1: RUNTIME-API capture smoke (the PyTorch / vLLM path).
    # Same two kernels as the N5a smoke, but the graph is captured via the
    # RUNTIME cudaStream{Begin,End}Capture + cudaGraphInstantiate/Launch —
    # what torch.cuda.graph() uses. Exercises the runtime shims and the CUgraph
    # record dedupe (the runtime end-capture may route through the interposed
    # driver cuStreamEndCapture inside libcudart → walk twice for one CUgraph).
    # -------------------------------------------------------------------
    echo "--- N5b Task 1: runtime capture path (record → restore) ---"
    RT_DIR="${DEPLOY_DIR}/snap-record-rt"
    RT_LOG_R="${DEPLOY_DIR}/snap-record-rt/record.log"
    RT_LOG_S="${DEPLOY_DIR}/snap-record-rt/restore.log"
    rm -rf "${RT_DIR}"; mkdir -p "${RT_DIR}"

    echo "--- runtime record run (LD_PRELOAD=redirect:record) ---"
    LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=record \
      SNAPSHOT_RECORD_CUDA_DIR="${RT_DIR}" \
      "${RT_SMOKE}" 2>&1 | tee "${RT_LOG_R}"
    RT_EXIT_R="${PIPESTATUS[0]}"
    [ "${RT_EXIT_R}" -eq 0 ] || { echo "N5B_RT_GATE: FAIL (record exit=${RT_EXIT_R})"; exit 1; }

    RT_CS_R=$(grep "^RT_CHECKSUM" "${RT_LOG_R}" | head -1)
    echo "runtime record checksum: ${RT_CS_R}"
    RT_REC_SUMMARY=$(grep "\[record-cuda\].*SUMMARY" "${RT_LOG_R}" || echo MISSING)
    echo "runtime record summary: ${RT_REC_SUMMARY}"
    # CUgraph dedupe: the runtime end-capture must serialize exactly ONE graph
    # (a double-walk would produce graph-0000 + graph-0001 → graphs=2).
    echo "${RT_REC_SUMMARY}" | grep -qF "recorded: graphs=1 nodes=2 edges=1 blind=0" || \
      { echo "N5B_RT_GATE: FAIL (expected recorded: graphs=1 nodes=2 edges=1 blind=0; got ${RT_REC_SUMMARY})"; exit 1; }

    echo "--- runtime restore run (LD_PRELOAD=redirect:record, MODE=restore) ---"
    LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=restore \
      SNAPSHOT_RECORD_CUDA_REBUILD=1 \
      SNAPSHOT_RECORD_CUDA_SUPPRESS=1 \
      SNAPSHOT_RECORD_CUDA_DIR="${RT_DIR}" \
      "${RT_SMOKE}" 2>&1 | tee "${RT_LOG_S}"
    RT_EXIT_S="${PIPESTATUS[0]}"
    [ "${RT_EXIT_S}" -eq 0 ] || { echo "N5B_RT_GATE: FAIL (restore exit=${RT_EXIT_S})"; exit 1; }

    RT_CS_S=$(grep "^RT_CHECKSUM" "${RT_LOG_S}" | head -1)
    echo "runtime restore checksum: ${RT_CS_S}"
    RT_REST_SUMMARY=$(grep "\[record-cuda\].*SUMMARY" "${RT_LOG_S}" || echo MISSING)
    echo "runtime restore summary: ${RT_REST_SUMMARY}"

    # G1 (runtime): bit-identical restored output.
    if [ "${RT_CS_R}" = "${RT_CS_S}" ] && [ -n "${RT_CS_R}" ]; then
      echo "N5B_RT_BIT_IDENTICAL=1"
    else
      echo "N5B_RT_GATE: FAIL (record=${RT_CS_R} restore=${RT_CS_S})"; exit 1
    fi
    # G2 (runtime): restored via runtime shim, no real capture, rt_capture=1.
    echo "${RT_REST_SUMMARY}" | grep -q "restored=1"        || { echo "N5B_RT_GATE: FAIL (restored=1)";        exit 1; }
    echo "${RT_REST_SUMMARY}" | grep -q "fallthrough=0"    || { echo "N5B_RT_GATE: FAIL (fallthrough=0)";    exit 1; }
    echo "${RT_REST_SUMMARY}" | grep -qE "real_begin=0( |$)" || { echo "N5B_RT_GATE: FAIL (real_begin=0)";   exit 1; }
    echo "${RT_REST_SUMMARY}" | grep -qE "suppressed=2( |$)" || { echo "N5B_RT_GATE: FAIL (suppressed=2)";   exit 1; }
    echo "${RT_REST_SUMMARY}" | grep -q "blind=0"          || { echo "N5B_RT_GATE: FAIL (blind=0)";          exit 1; }
    echo "${RT_REST_SUMMARY}" | grep -qE "rt_capture=1( |$)" || { echo "N5B_RT_GATE: FAIL (rt_capture=1)";   exit 1; }

    # G3 (runtime): redirect pinned the fixed base in the restore run (Δ=0).
    RT_REDIR=$(grep "\[redirect-cuda\].*SUMMARY" "${RT_LOG_S}" || echo ABSENT)
    echo "${RT_REDIR}" | grep -qE "passthrough=0\(" || { echo "N5B_RT_GATE: FAIL (passthrough=0)"; exit 1; }
    grep "\[redirect-cuda\].*fixed_base_honored" "${RT_LOG_S}" | grep -qF "fixed_base_honored=1" || \
      { echo "N5B_RT_GATE: FAIL (fixed_base_honored=1)"; exit 1; }

    echo "N5B_RT_GATE: PASS (runtime path record→restore bit-identical, rt_capture=1)"

    # -------------------------------------------------------------------
    # N5b Task 2 (G1): FULL-like graph — non-kernel nodes + `extra` kernargs +
    # a diamond DAG. Exercises the v2 format (memcpy/memset/blind tags) and
    # the non-kernel edge resolution (the N5a gap) end to end.
    # -------------------------------------------------------------------
    echo "--- N5b Task 2 (G1): FULL-like graph (record → restore) ---"
    FULL_DIR1="${DEPLOY_DIR}/snap-record-full/run1"
    FULL_DIR2="${DEPLOY_DIR}/snap-record-full/run2"
    FULL_LOG_R="${DEPLOY_DIR}/snap-record-full/record.log"
    FULL_LOG_R2="${DEPLOY_DIR}/snap-record-full/record2.log"
    FULL_LOG_S="${DEPLOY_DIR}/snap-record-full/restore.log"
    rm -rf "${FULL_DIR1}" "${FULL_DIR2}"; mkdir -p "${FULL_DIR1}" "${FULL_DIR2}"

    echo "--- FULL record run 1 (LD_PRELOAD=redirect:record) ---"
    LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=record \
      SNAPSHOT_RECORD_CUDA_DIR="${FULL_DIR1}" \
      "${FULL_SMOKE}" 2>&1 | tee "${FULL_LOG_R}"
    FULL_EXIT_R="${PIPESTATUS[0]}"
    [ "${FULL_EXIT_R}" -eq 0 ] || { echo "N5B_FULL_GATE: FAIL (record exit=${FULL_EXIT_R})"; exit 1; }

    echo "--- FULL record run 2 (Δ=0 determinism) ---"
    LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=record \
      SNAPSHOT_RECORD_CUDA_DIR="${FULL_DIR2}" \
      "${FULL_SMOKE}" 2>&1 | tee "${FULL_LOG_R2}"
    FULL_EXIT_R2="${PIPESTATUS[0]}"
    [ "${FULL_EXIT_R2}" -eq 0 ] || { echo "N5B_FULL_GATE: FAIL (record2 exit=${FULL_EXIT_R2})"; exit 1; }

    FULL_CS_R=$(grep "^FULL_CHECKSUM" "${FULL_LOG_R}" | head -1)
    echo "FULL record checksum: ${FULL_CS_R}"
    FULL_REC_SUMMARY=$(grep "\[record-cuda\].*SUMMARY" "${FULL_LOG_R}" || echo MISSING)
    echo "FULL record summary: ${FULL_REC_SUMMARY}"
    # G1: every node resolved (memcpy/memset handled, no blind), 1 graph.
    echo "${FULL_REC_SUMMARY}" | grep -q "blind=0" || { echo "N5B_FULL_GATE: FAIL (blind=0 in record)"; exit 1; }
    echo "${FULL_REC_SUMMARY}" | grep -qE "recorded: graphs=1" || { echo "N5B_FULL_GATE: FAIL (recorded: graphs=1)"; exit 1; }
    # At least the 4 kernels + 1 memset + 2 memcpys = 7 nodes; print the count.
    FULL_NODES=$(echo "${FULL_REC_SUMMARY}" | grep -oE "nodes=[0-9]+" | head -1)
    echo "FULL node count: ${FULL_NODES}"
    echo "${FULL_NODES}" | grep -qE "nodes=([7-9]|[1-9][0-9]+)" || \
      { echo "N5B_FULL_GATE: FAIL (expected >=7 nodes for memset+2memcpy+4kernel diamond; got ${FULL_NODES})"; exit 1; }

    # Δ=0: the two FULL record dirs must be byte-identical.
    FULL_SNAP1="${FULL_DIR1}/graph-0000.snap"
    FULL_SNAP2="${FULL_DIR2}/graph-0000.snap"
    [ -s "${FULL_SNAP1}" ] && [ -s "${FULL_SNAP2}" ] || { echo "N5B_FULL_GATE: FAIL (.snap missing)"; exit 1; }
    if cmp "${FULL_SNAP1}" "${FULL_SNAP2}"; then
      echo "N5B_FULL_SNAP_BYTE_IDENTICAL=1"
    else
      echo "N5B_FULL_GATE: FAIL (.snap differs between two FULL record runs)"; exit 1
    fi

    echo "--- FULL restore run (LD_PRELOAD=redirect:record, MODE=restore) ---"
    LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=restore \
      SNAPSHOT_RECORD_CUDA_REBUILD=1 \
      SNAPSHOT_RECORD_CUDA_SUPPRESS=1 \
      SNAPSHOT_RECORD_CUDA_DIR="${FULL_DIR1}" \
      "${FULL_SMOKE}" 2>&1 | tee "${FULL_LOG_S}"
    FULL_EXIT_S="${PIPESTATUS[0]}"
    [ "${FULL_EXIT_S}" -eq 0 ] || { echo "N5B_FULL_GATE: FAIL (restore exit=${FULL_EXIT_S})"; exit 1; }

    FULL_CS_S=$(grep "^FULL_CHECKSUM" "${FULL_LOG_S}" | head -1)
    echo "FULL restore checksum: ${FULL_CS_S}"
    FULL_REST_SUMMARY=$(grep "\[record-cuda\].*SUMMARY" "${FULL_LOG_S}" || echo MISSING)
    echo "FULL restore summary: ${FULL_REST_SUMMARY}"

    # G1: bit-identical restored output, blind=0, restored=1/fallthrough=0.
    if [ "${FULL_CS_R}" = "${FULL_CS_S}" ] && [ -n "${FULL_CS_R}" ]; then
      echo "N5B_FULL_BIT_IDENTICAL=1"
    else
      echo "N5B_FULL_GATE: FAIL (record=${FULL_CS_R} restore=${FULL_CS_S})"; exit 1
    fi
    echo "${FULL_REST_SUMMARY}" | grep -q "restored=1"     || { echo "N5B_FULL_GATE: FAIL (restored=1)";     exit 1; }
    echo "${FULL_REST_SUMMARY}" | grep -q "fallthrough=0" || { echo "N5B_FULL_GATE: FAIL (fallthrough=0)"; exit 1; }
    echo "${FULL_REST_SUMMARY}" | grep -q "blind=0"       || { echo "N5B_FULL_GATE: FAIL (blind=0)";       exit 1; }

    echo "N5B_FULL_GATE: PASS (FULL-like graph record→restore bit-identical, blind=0)"

    # -------------------------------------------------------------------
    # N5b Task 3: region_base export + per-rank dir resolution + scale maps.
    # -------------------------------------------------------------------
    echo "--- N5b Task 3: region_base export + per-rank dir ---"
    # (a) the export symbol is present in the record .so.
    if nm -D "${RECORD_SO}" | grep -qE "snapshot_record_cuda_region_base"; then
      echo "N5B_REGION_BASE_SYMBOL=1"
    else
      echo "N5B_TASK3_GATE: FAIL (snapshot_record_cuda_region_base not exported)"; exit 1
    fi

    # (b) the export returns the redirect fixed base once redirect is engaged.
    PROBE_BASE=$(LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}" "${PROBE}" 2>/dev/null \
                 | grep "^REGION_BASE=" | head -1)
    echo "region_base probe: ${PROBE_BASE}"
    echo "${PROBE_BASE}" | grep -qE "^REGION_BASE=0x[0-9a-f]*[1-9a-f][0-9a-f]*" || \
      { echo "N5B_TASK3_GATE: FAIL (region_base returned 0 — redirect not engaged by probe?)"; exit 1; }

    # (c) per-rank %r substitution: RANK=2 → snaps land under .../rank2/, and
    # the literal "rank%r" dir is NOT created.
    RANK_DIR="${DEPLOY_DIR}/snap-record-rank"
    rm -rf "${RANK_DIR}"; mkdir -p "${RANK_DIR}"
    LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=record \
      SNAPSHOT_RECORD_CUDA_DIR="${RANK_DIR}/rank%r" \
      RANK=2 \
      "${RT_SMOKE}" > "${RANK_DIR}/probe.log" 2>&1 || true
    if [ -s "${RANK_DIR}/rank2/graph-0000.snap" ] && [ ! -e "${RANK_DIR}/rank%r" ]; then
      echo "N5B_PER_RANK_DIR=1 (RANK=2 → rank2/)"
    else
      echo "N5B_TASK3_GATE: FAIL (per-rank %r not substituted; expected ${RANK_DIR}/rank2/graph-0000.snap)"
      ls -R "${RANK_DIR}" | head -20
      exit 1
    fi
    # Per-rank record still complete + blind=0.
    RANK_REC_SUMMARY=$(grep "\[record-cuda\].*SUMMARY" "${RANK_DIR}/probe.log" || echo MISSING)
    echo "per-rank record summary: ${RANK_REC_SUMMARY}"
    echo "${RANK_REC_SUMMARY}" | grep -qF "recorded: graphs=1 nodes=2 edges=1 blind=0" || \
      { echo "N5B_TASK3_GATE: FAIL (per-rank record summary)"; exit 1; }

    echo "N5B_TASK3_GATE: PASS (region_base export non-zero + per-rank dirs)"
  '

# ---------------------------------------------------------------------------
# N5b Task 4: host-side unit test for cg_meta_cuda (DLPack kDLCUDA wrap +
# metadata round-trip). Needs torch+CUDA, so a SEPARATE srun under the
# torch-cuda EDF (the snapshot-cuda devel container has no torch). The recipe
# dir is on /capstor (mounted in both EDFs); the test imports cg_meta_cuda from
# its own dir (no vLLM needed — it exercises _serialize/_reconstruct/wrap only).
# ---------------------------------------------------------------------------
srun --environment=snapshot-torch-cuda \
  bash -lc 'set -euo pipefail
    cd "${DEPLOY_DIR}"
    python3 snapshot/recipe/cginst_cuda/_test_wrap.py 2>&1 \
      | tee "${DEPLOY_DIR}/logs/cg_meta_wrap.log"
    grep -q "CG_META_WRAP_OK=1" "${DEPLOY_DIR}/logs/cg_meta_wrap.log" || \
      { echo "N5B_CGMETA_GATE: FAIL (CG_META_WRAP_OK=1 not printed)"; exit 1; }
    echo "N5B_CGMETA_GATE: PASS (DLPack kDLCUDA wrap + metadata round-trip)"
  '

echo "cuda-record-smoke finished: $(date --iso-8601=seconds)"
