#!/bin/bash
#SBATCH --job-name=restripe-variance
#SBATCH --partition=normal
#SBATCH --account=a-infra02
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/xyao/bench-loader/logs/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/xyao/bench-loader/logs/%x-%j.err
#
# restripe_variance.sbatch — characterize the VARIANCE of the restripe case study.
#
# The single-run case study (job 71670) regressed at 16 streams and timed out at
# 32 — clearly a contended OST window. The earlier clean run (71668) hit 3.49
# GB/s at 32. A 13x swing on identical files means OST contention, not stripe
# count, dominates this shared FS. This job quantifies that:
#   * source (stripe_count=1) full sweep incl. 32 (timeout-protected now)
#   * restriped (-c8 -S4m) sweep x3 back-to-back -> show the run-to-run spread
# Files already exist from the case-study copy; no re-copy.

set -uo pipefail
HERE="/capstor/scratch/cscs/xyao/bench-loader"
BC="$HERE/bench_core.sh"
export PER_STREAM_MB=512          # cap each stream at 512 MB so sweeps finish fast even under OST contention
export PER_STREAM_TIMEOUT=180
SRC="/capstor/store/cscs/swissai/infra01/hf_models/models/zai-org/GLM-4.7-Flash"
DST="/capstor/store/cscs/swissai/infra01/hf_models/models/restriped/GLM-4.7-Flash"

echo "############################################################"
echo "# RESTRIPED VARIANCE STUDY ($(date --iso-8601=seconds)) job=${SLURM_JOB_ID:-} host=$(hostname)"
echo "############################################################"

echo; echo "########## SOURCE (stripe_count=1), full sweep, 1x ##########"
STREAMS="1 2 4 8 16 32" bash "$BC" "SOURCE-stripe1" "$SRC"/model-*.safetensors

for i in 1 2 3; do
  echo; echo "########## RESTRIPED (-c8 -S4m), sweep run $i/3 ##########"
  STREAMS="1 2 4 8 16 32" bash "$BC" "RESTRIPED-run${i}" "$DST"/model-*.safetensors
done

echo; echo "############################################################"
echo "# VARIANCE STUDY DONE ($(date --iso-8601=seconds))"
echo "# Compare the 32-stream row across runs: the spread = OST contention noise."
echo "############################################################"
