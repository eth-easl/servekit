# Phase 5, first attempt — all 4 points aborted at the patch step

Cause: the sbatch called patch_sglang_in_container.sh TWICE (once per patch).
It does not compose — the second call re-clones a pristine sglang and diffs it
against the INSTALLED weight_utils.py, which the first call had already
patched. The version check tripped and the job exited 1.

The harness behaved CORRECTLY: it refused to run rather than serve a bogus
number from an engine build it could not vouch for.

Fix: the script now takes N patches, clones once, verifies the union of touched
files against the pristine install, applies all patches to that one clone, and
swaps once. Verified locally that phase3's knob and phase5's iterator co-apply
and compile.

No weights were ever read in these jobs, so the nodes were not warmed.
