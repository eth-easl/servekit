"""Six fixed greedy completions, written one per line for byte-exact diffing.

The correctness check this round actually rests on. Throughput and errors=0 do
not prove the shards landed on the right PP stage -- a model assembled from
mismatched slices still emits fluent text at full speed. Two runs that differ
only in the loader must produce identical bytes at temperature 0.

Usage: probe.py <base_url> <model_name> <out_path>
"""

import json
import sys
import urllib.request

PROMPTS = [
    "The capital of Switzerland is",
    "def fibonacci(n):",
    "Q: What is 17 times 23?\nA:",
    "Translate to French: The weather is cold today.",
    "The three primary colors are",
    "In one sentence, explain what a compiler does:",
]

base_url, model, out_path = sys.argv[1:4]

with open(out_path, "w") as out:
    for prompt in PROMPTS:
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": 48,
                "temperature": 0.0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = json.load(resp)["choices"][0]["text"]
        out.write(json.dumps({"prompt": prompt, "completion": text}) + "\n")

print(f"wrote {len(PROMPTS)} greedy probes to {out_path}")
