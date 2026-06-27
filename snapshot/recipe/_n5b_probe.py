#!/usr/bin/env python3
# _n5b_probe.py — N5b Task 7: send a FIXED prompt set to a vLLM completions
# endpoint (greedy/seeded) and print "<prompt>\t<generated_text>" per line.
#
# Used by both the record run (graph-mode serving under the transparent
# interposer == baseline reference) and the restore run; the restore sbatch
# diffs the two outputs for the token-identical gate (G4). TP aggregates the
# ranks, so matching final tokens across a prompt set implies the per-rank
# computation is consistent (a divergent rank would change the final tokens).
import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8821/v1/completions"

# A small, deterministic prompt set spanning prefill + short decode.
PROMPTS = [
    "The capital of France is",
    "Count from one to five:",
    "The reverse of the word 'hello' is",
    "2 plus 2 equals",
    "The primary colors are",
]


def main() -> int:
    for p in PROMPTS:
        body = json.dumps(
            {"model": "cs", "prompt": p, "max_tokens": 8, "temperature": 0}
        ).encode()
        req = urllib.request.Request(
            URL, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            raw = urllib.request.urlopen(req, timeout=60).read().decode()
            txt = json.loads(raw)["choices"][0]["text"]
        except Exception as e:  # noqa: BLE001
            txt = f"<error: {e!r}>"
        print(f"{p}\t{txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
