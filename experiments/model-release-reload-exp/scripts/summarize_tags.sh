#!/bin/bash
# Read the Apertus-8B tag matrix: for each point, did the model survive the cycle?
#
# The ONLY thing that matters here is whether the generated text is still coherent.
# Throughput, HTTP status and the endpoint's own `success` flag ALL reported healthy
# on the 70B run that produced pure garbage - so they are shown but never trusted.
#
# Run from the repo root:
#   bash experiments/model-release-reload-exp/scripts/summarize_tags.sh              # Apertus-8B
#   RES_GLOB=mrr70b bash experiments/model-release-reload-exp/scripts/summarize_tags.sh   # Llama-70B
set -euo pipefail

RES="${1:-experiments/model-release-reload-exp/results}"
RES_GLOB="${RES_GLOB:-mrr8b}"

python3 - "$RES" "$RES_GLOB" <<'PY'
import glob, json, sys, os

res, prefix = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(res, f"{prefix}-*-cycle.json"))):
    try:
        c = json.load(open(f))
    except Exception as e:
        print(f"  !! unreadable {f}: {e}")
        continue
    p = f.replace("-cycle.json", "-profile.json")
    prof = json.load(open(p)) if os.path.exists(p) else {}
    tag = os.path.basename(f).split("-")[1]

    before = ((prof.get("benchmark") or {}).get("correctness") or {}).get("results", [])
    after = ((c.get("bench_after_reload") or {}).get("correctness") or {}).get("results", [])
    t = (c.get("bench_after_reload") or {}).get("throughput") or {}

    # Coherence heuristic, deliberately crude: garbage in the 70B run was a single
    # token repeated ("QuestionQuestion...", "assistantassistant"). A healthy answer
    # has many distinct words. Always eyeball the sample below - never trust this.
    verdict, sample = "NO DATA", ""
    if after:
        txt = after[0]["output"]
        words = txt.split()
        uniq = len(set(words))
        verdict = "COHERENT?" if uniq > 5 else "GARBAGE"
        sample = " ".join(words)[:70]
    ident = c.get("correctness_identical")
    n_same = sum(ident) if ident else 0

    rows.append((tag, c.get("no_release"), ",".join(c.get("tags") or []) or "-",
                 c.get("timings_s", {}), verdict, f"{n_same}/{len(ident or [])}",
                 t.get("output_tok_per_s"), t.get("errors"), sample, c.get("error")))

if not rows:
    print(f"no {prefix}-*-cycle.json in {res} yet")
    sys.exit(0)

print(f"{'point':<11}{'no_rel':<8}{'tags':<28}{'VERDICT':<11}{'same/6':<8}{'tok/s':<9}{'err':<5}")
print("-" * 80)
for tag, norel, tags, tm, verdict, same, tok, err, sample, error in rows:
    print(f"{tag:<11}{str(bool(norel)):<8}{tags:<28}{verdict:<11}{same:<8}{str(tok):<9}{str(err):<5}")
print()
print("timings (s):")
for tag, _, _, tm, *_ in rows:
    if tm:
        print(f"  {tag:<11}" + "  ".join(f"{k}={v}" for k, v in tm.items()))
print()
print("first correctness answer per point ('The capital of France is ...') -- READ THIS:")
for tag, _, _, _, verdict, _, _, _, sample, error in rows:
    print(f"  [{verdict:<9}] {tag:<11} {sample!r}")
    if error:
        print(f"                  driver error: {error}")
PY
