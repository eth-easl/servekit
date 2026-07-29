"""Prove the dump is actually sharded across PP stages.

This is the check the round exists for, and it is independent of any timing.

NOTE the naive test -- "no two ranks share a tensor" -- is WRONG as soon as
TP>1, and reported a false failure on the TP=2 PP=2 dump (job 2924462). Within
one PP stage the TP peers legitimately hold identical copies of every
TP-replicated param (layernorms, and the TP-split lm_head halves are replicated
across stages), so content dedup correctly stores those once and points several
ranks at them. Sharing is expected; what must NOT happen is two PP stages owning
the same transformer layer.

So the invariant checked here is the layer partition, derived from the data
rather than assumed: group ranks by which `model.layers.N` they hold, and
require exactly PP distinct groups of TP ranks each, together tiling 0..L-1
without overlap.

Usage: inspect_dump.py <presharded_root> <tp_size> <pp_size>
"""

import collections
import json
import os
import re
import sys

root, tp, pp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
world = tp * pp

subdirs = [
    os.path.join(root, d)
    for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))
]
if len(subdirs) != 1:
    print(f"  FAIL: expected exactly one config subfolder under {root}, found {len(subdirs)}")
    print(f"        {[os.path.basename(d) for d in subdirs]}")
    sys.exit(1)

d = subdirs[0]
print(f"  subfolder: {os.path.basename(d)}   (tp={tp} pp={pp}, world={world})")

if not os.path.isfile(os.path.join(d, "READY")):
    print("  FAIL: no READY sentinel -- the dump did not complete")
    sys.exit(1)

plan = json.load(open(os.path.join(d, "checksum.json")))
reads = plan["rank_to_reads"]
files = plan["files"]

total_bytes = sum(
    os.path.getsize(os.path.join(d, f))
    for f in os.listdir(d)
    if f.endswith(".safetensor")
)
print(f"  world_size={plan['world_size']}  files={len(files)}  total={total_bytes / 1e9:.2f} GB")

ok = True
if plan["world_size"] != world:
    print(f"  FAIL: world_size={plan['world_size']}, expected {world}")
    ok = False
if sorted(int(r) for r in reads) != list(range(world)):
    print(f"  FAIL: rank_to_reads covers {sorted(reads)}, expected 0..{world - 1}")
    ok = False

LAYER = re.compile(r"model\.layers\.(\d+)\.")
rank_layers = {}
for r, items in reads.items():
    layers = set()
    for x in items:
        m = LAYER.search(x["name"])
        if m:
            layers.add(int(m.group(1)))
    rank_layers[int(r)] = layers

print("  per-rank:")
for r in sorted(rank_layers):
    nbytes = sum(
        os.path.getsize(os.path.join(d, f["filename"]))
        for f in files
        if f["writer_rank"] == r
    )
    ls = sorted(rank_layers[r])
    span = f"{ls[0]}-{ls[-1]}" if ls else "(none)"
    print(
        f"    rank {r}: {len(reads[str(r)]):4d} tensors, "
        f"{len(ls):3d} layers [{span}], writes {nbytes / 1e9:6.2f} GB"
    )

# Ranks holding the same layer set are one PP stage. Derived, not assumed, so
# this does not depend on how world rank maps to (pp_rank, tp_rank).
groups = collections.defaultdict(list)
for r, ls in rank_layers.items():
    groups[frozenset(ls)].append(r)

print(f"  distinct layer sets: {len(groups)} (expected pp={pp})")
for ls, ranks in sorted(groups.items(), key=lambda kv: min(kv[1])):
    s = sorted(ls)
    print(f"    ranks {sorted(ranks)} -> layers {s[0]}-{s[-1]} ({len(s)})")

if len(groups) != pp:
    print(f"  FAIL: {len(groups)} distinct layer sets, expected {pp} pipeline stages")
    ok = False
if any(len(ranks) != tp for ranks in groups.values()):
    print(f"  FAIL: a pipeline stage is not held by exactly tp={tp} ranks")
    ok = False

stage_sets = list(groups.keys())
for i in range(len(stage_sets)):
    for j in range(i + 1, len(stage_sets)):
        shared = stage_sets[i] & stage_sets[j]
        if shared:
            print(f"  FAIL: two pipeline stages both own layers {sorted(shared)}")
            ok = False

covered = set().union(*stage_sets) if stage_sets else set()
if covered != set(range(len(covered))):
    print(f"  FAIL: layers do not tile 0..N-1; got {len(covered)} distinct indices")
    ok = False
else:
    print(f"  OK: {pp} stages tile layers 0-{len(covered) - 1} with no overlap")

print("  DUMP CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
