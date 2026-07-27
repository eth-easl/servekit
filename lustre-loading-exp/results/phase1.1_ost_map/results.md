# Phase 1.1 — shard→OST map (native layout)

**Status:** done

## Goal

Before touching any loader/layout knob, establish how the model's 30
`.safetensors` shards are actually distributed across Lustre OSTs today, so
later striping experiments (Phase 2) test something real instead of an
assumption.

## Method

`scripts/phase1.1_ost_map/shard_ost_map.sh` runs `lfs getstripe` over every
shard of `Llama-3.1-70B-Instruct` (~132 GB, 30 shards) on capstor, and tallies
which OST each shard lands on.

## Result

Raw output: `phase1_shard_ost_map.txt`.

- Every shard: `stripe_count=1, stripe_size=1M` (default layout, no explicit
  striping).
- 30 shards land on **24 distinct OSTs** out of capstor's ≥150.
- **Hot OSTs** (hosting >1 shard — the concurrency-tail candidates):
  - OST 30 → **3** shards (~13 GB serialized on one OST if read serially)
  - OST 15, 46, 59, 93 → 2 shards each

## Verdict

Shards already scatter across 24 distinct OSTs at `stripe_count=1` — concurrent
multi-shard reads already exploit most of the filesystem width without any
striping change. This reframes what Phase 2 needs to test: the potential win
from striping is *not* "spread reads over more OSTs" (already true), it's
whether per-file striping kills the straggler tail from the hot OSTs above.
Phase 2 measured this directly and found no benefit (see
[`../phase2_stripe_sweep/results.md`](../phase2_stripe_sweep/results.md)).

## Caveats

None — this is a direct measurement, no inference involved.
