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

# N5a Task 1 gate: build cuda_record_smoke, run twice, confirm CHECKSUM is identical.
# N5a Task 2 gate: build libsnapshot_record_cuda.so, run smoke under the
#   interposer in record mode, confirm:
#     (a) CHECKSUM is unchanged (interposer is transparent)
#     (b) SUMMARY reports "identity: 2 functions (1 fatbin, 1 module)"
#         i.e. k_add resolved via __cudaRegisterFunction (fatbin path) AND
#              k_mul resolved via cuModuleGetFunction (module/nvrtc path).
# N5a Task 3 gate: build libsnapshot_redirect_cuda.so too, run the smoke under
#   LD_PRELOAD=redirect:record in record mode TWICE into two separate snap dirs,
#   confirm:
#     (a) graph-0000.snap is written with 2 kernel nodes + 1 dep edge
#         (record SUMMARY: recorded: graphs=1 nodes=2 edges=1 blind=0)
#     (b) the smoke still prints its normal CHECKSUM
#     (c) the two .snap files are BYTE-IDENTICAL (cmp) — the decisive Δ=0 proof
#         that recorded kernarg pointers are fixed-base-deterministic
#     (d) the redirect engaged and pinned the allocations: it prints
#         passthrough=0 and fixed_base_honored=1.
#         The smoke allocates d_a/d_out via the RUNTIME cudaMalloc (see
#         cuda_record_smoke.cpp), which snapshot_redirect_cuda hooks, so the
#         redirect serves them from its fixed-base region — that is what makes
#         the recorded kernarg pointers (and the .snap) byte-identical across
#         runs. (An earlier revision used driver cuMemAlloc, which the redirect
#         does NOT hook, leaving it inert and the pointer bytes non-deterministic;
#         the cudaMalloc switch fixed that and mirrors the PyTorch/vLLM caching
#         allocator path targeted in N5b.)

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

    # Task 1: build smoke binary.
    cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" \
      --target cuda_record_smoke

    # Task 2 + Task 3: build record interposer + the redirect interposer it
    # composes with (Task 3 runs the smoke under both .so via LD_PRELOAD).
    cmake --build snapshot/build-cuda -j"${SLURM_CPUS_PER_TASK:-8}" \
      --target snapshot_record_cuda snapshot_redirect_cuda

    SMOKE="$PWD/snapshot/build-cuda/cuda_record_smoke"
    RECORD_SO="$PWD/snapshot/build-cuda/libsnapshot_record_cuda.so"
    REDIRECT_SO="$PWD/snapshot/build-cuda/libsnapshot_redirect_cuda.so"
    ls -l "$SMOKE" "$RECORD_SO" "$REDIRECT_SO"

    echo "--- run 1 ---"
    "$SMOKE" | tee /tmp/r1.txt
    echo "exit $?"

    echo "--- run 2 ---"
    "$SMOKE" | tee /tmp/r2.txt
    echo "exit $?"

    echo "--- determinism check (Task 1 gate) ---"
    CS1=$(grep "^CHECKSUM" /tmp/r1.txt || echo "MISSING")
    CS2=$(grep "^CHECKSUM" /tmp/r2.txt || echo "MISSING")
    echo "run1: $CS1"
    echo "run2: $CS2"
    if [ "$CS1" = "$CS2" ] && [ "$CS1" != "MISSING" ]; then
      echo "CHECKSUM_DETERMINISTIC=1"
    else
      echo "CHECKSUM_DETERMINISTIC=0"
      exit 1
    fi

    # -----------------------------------------------------------------------
    # Task 2 gate: run smoke under the record interposer.
    # Capture stdout+stderr together so we can grep both the CHECKSUM and the
    # [record-cuda] SUMMARY line from the same file.
    # -----------------------------------------------------------------------
    SNAP_DIR="${DEPLOY_DIR}/snap-record-smoke"
    mkdir -p "${SNAP_DIR}"

    echo "--- record interposer run (Task 2 gate) ---"
    LD_PRELOAD="${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=record \
      SNAPSHOT_RECORD_CUDA_DIR="${SNAP_DIR}" \
      "$SMOKE" 2>&1 | tee /tmp/r_record.txt
    RECORD_EXIT="${PIPESTATUS[0]}"

    echo "--- identity gate ---"
    SUMMARY=$(grep "\[record-cuda\].*SUMMARY" /tmp/r_record.txt || echo "SUMMARY_MISSING")
    echo "summary line: ${SUMMARY}"

    # Gate (a): smoke still exits clean and prints the same CHECKSUM.
    CS_RECORD=$(grep "^CHECKSUM" /tmp/r_record.txt || echo "MISSING")
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
    else
      echo "IDENTITY_GATE=0 (expected \"identity: 2 functions (1 fatbin, 1 module)\" not found)"
      exit 1
    fi

    if [ "${RECORD_EXIT}" -ne 0 ]; then
      echo "RECORD_RUN_EXIT=${RECORD_EXIT} (smoke exited non-zero under interposer)"
      exit 1
    fi

    echo "Task 2 gate PASSED"

    # -----------------------------------------------------------------------
    # Task 3 gate: record the captured graph to .snap under redirect+record,
    # twice into two separate dirs, then prove byte-identical (Δ=0).
    # Use persistent paths under ${DEPLOY_DIR} (not /tmp) and PIPESTATUS.
    # -----------------------------------------------------------------------
    echo "--- Task 3: capture/record + Δ=0 ---"
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

    # (d) redirect ENGAGED gate: the smoke now allocs via runtime cudaMalloc, so
    # the redirect must initialize and pin those allocations — confirm its output
    # is PRESENT with passthrough=0 (all allocs served from the fixed region) and
    # fixed_base_honored=1 (region reserved at the requested base 0x600000000000).
    # (Reported BEFORE the decisive cmp so the log always carries this evidence.)
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

    echo "Task 3 gate PASSED"

    # -----------------------------------------------------------------------
    # Task 4 gate: BIT-IDENTICAL RESTORE (run #2).
    #
    # Replay the graph recorded into T3_DIR1 (graph-0000.snap) WITHOUT real
    # capture: LD_PRELOAD=redirect:record, MODE=restore, same fixed-base env,
    # same .snap dir. The restore shim fakes begin-capture, suppresses the two
    # kernel launches (so nothing runs eagerly), and rebuilds+returns the graph
    # at end-capture; the smoke then instantiates+launches the rebuilt graph as
    # its SOLE execution. Asserts:
    #   G1  restore CHECKSUM == record CHECKSUM (both == CS1)
    #   G2  SUMMARY restored=1 fallthrough=0 real_begin=0 (no real capture)
    #   G3  redirect SUMMARY passthrough=0 + fixed_base_honored=1 (Δ=0 in run #2)
    #   G4  SUMMARY blind=0 (both kernels reverse-resolved)
    #   SUPP SUMMARY suppressed=2 (k_add + k_mul intercepted → single execution)
    # -----------------------------------------------------------------------
    echo "--- Task 4: bit-identical restore ---"
    T4_LOG="${DEPLOY_DIR}/snap-record-t3/restore.log"
    ls -l "${SNAP1}"  # the .snap we restore from (recorded by T3 run 1)

    echo "--- restore run (LD_PRELOAD=redirect:record, MODE=restore) ---"
    LD_PRELOAD="${REDIRECT_SO}:${RECORD_SO}" \
      SNAPSHOT_RECORD_CUDA_MODE=restore \
      SNAPSHOT_RECORD_CUDA_DIR="${T3_DIR1}" \
      "$SMOKE" 2>&1 | tee "${T4_LOG}"
    T4_EXIT="${PIPESTATUS[0]}"

    if [ "${T4_EXIT}" -ne 0 ]; then
      echo "T4_RUN_EXIT=${T4_EXIT} (smoke exited non-zero in restore mode)"
      exit 1
    fi

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

    # G2: exactly one graph restored, no fall-through, no real begin-capture.
    if echo "${REST_SUMMARY}" | grep -qE "restored=1 fallthrough=0 "; then
      echo "G2_RESTORED_ONE_NO_FALLTHROUGH=1"
    else
      echo "G2_RESTORED_ONE_NO_FALLTHROUGH=0 (expected restored=1 fallthrough=0)"
      exit 1
    fi
    if echo "${REST_SUMMARY}" | grep -qE "real_begin=0( |$)"; then
      echo "G2_NO_REAL_BEGIN_CAPTURE=1 (real capture count is 0)"
    else
      echo "G2_NO_REAL_BEGIN_CAPTURE=0 (real begin-capture was called)"
      exit 1
    fi

    # G3: redirect still pinned the fixed base in run #2 (Δ=0 holds).
    REST_REDIR=$(grep "\[redirect-cuda\].*SUMMARY" "${T4_LOG}" || echo "ABSENT")
    REST_FIXED=$(grep "\[redirect-cuda\].*fixed_base_honored" "${T4_LOG}" || echo "ABSENT")
    echo "restore redirect summary: ${REST_REDIR}"
    echo "restore redirect fixed-vmm: ${REST_FIXED}"
    if echo "${REST_REDIR}" | grep -qE "passthrough=0\(" && \
       echo "${REST_FIXED}" | grep -qF "fixed_base_honored=1"; then
      echo "G3_REDIRECT_FIXED_BASE=1 (passthrough=0, fixed_base_honored=1)"
    else
      echo "G3_REDIRECT_FIXED_BASE=0 (redirect did not pin the fixed base in run #2)"
      exit 1
    fi

    # G4: both kernels reverse-resolved (no blind nodes).
    if echo "${REST_SUMMARY}" | grep -qE "blind=0 restored=1"; then
      echo "G4_NO_BLIND=1 (k_add + k_mul both resolved at restore)"
    else
      echo "G4_NO_BLIND=0 (a node was blind — reverse resolution failed)"
      exit 1
    fi

    # SUPPRESSION PROOF: both launches intercepted → no eager execution → G1 genuine.
    if echo "${REST_SUMMARY}" | grep -qE "suppressed=2( |$)"; then
      echo "SUPPRESS_BOTH_LAUNCHES=1 (k_add via cudaLaunchKernel + k_mul via cuLaunchKernel)"
    else
      echo "SUPPRESS_BOTH_LAUNCHES=0 (expected suppressed=2 — a kernel ran eagerly)"
      exit 1
    fi

    echo "Task 4 gate PASSED"
  '

echo "cuda-record-smoke finished: $(date --iso-8601=seconds)"
