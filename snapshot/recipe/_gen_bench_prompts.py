#!/usr/bin/env python3
"""Generate a deterministic prompts JSONL for benchmaker `llm` serving benchmarks.

Emits N prompts spanning a realistic range of input lengths so the benchmark
exercises BOTH prefill (input-bound) and decode (output-bound) paths. Each row
is ``{"prompt": "<text>"}``, the default ``--prompt-field`` for ``benchmaker llm``.

Length tiers (approximate token counts, words ~= 0.75 tokens):
  short  ~ 15-40 tokens   (40% of prompts)  — chat-style Q&A
  medium ~ 80-220 tokens  (35% of prompts)  — paragraph reasoning
  long   ~ 400-1400 tokens(25% of prompts)  — document/context summarization

Deterministic (seed=0) so the A/B baseline-vs-restore runs see identical input.
"""
import json
import random
import sys

SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
    "Modern language models route tokens through sparse expert layers to scale capacity efficiently.",
    "A CUDA graph captures a sequence of kernel launches into a single replayable DAG.",
    "Distributed training balances communication overhead against per-step compute on each accelerator.",
    "Memory bandwidth, not raw FLOPS, usually limits autoregressive decoding throughput.",
    "Quantization reduces weight precision to shrink memory and widen the effective bandwidth budget.",
    "Paged attention partitions the KV cache into fixed blocks to avoid fragmentation under batching.",
    "Speculative decoding drafts tokens with a small model and verifies them in parallel chunks.",
    "The ROCm platform exposes HIP as a near-source-compatible interface to AMD accelerators.",
    "Continuous batching re-batches active sequences on every step to keep hardware saturated.",
]
QA = [
    "What is the capital of France?",
    "Explain gradient descent in one sentence.",
    "Name three sorting algorithms and their time complexity.",
    "How does a hash table handle collisions?",
    "What is the difference between TCP and UDP?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "Why is the sky blue during the day?",
    "What are the axioms of a vector space?",
    "Describe how a compiler optimizes a loop.",
    "What causes ocean tides?",
]
DOC = (
    "In recent years, the field of natural language processing has been transformed by large "
    "transformer models trained on internet-scale corpora. These models exhibit emergent abilities "
    "such as in-context learning, chain-of-thought reasoning, and code generation. However, their "
    "sheer size introduces significant engineering challenges at inference time: gigabytes of "
    "weights must be streamed through the compute units on every forward pass, the key-value cache "
    "grows linearly with context length, and batching many concurrent users requires careful memory "
    "management to avoid fragmentation. Serving systems address these challenges with techniques "
    "like paged attention, continuous batching, quantization, and CUDA graph capture, which together "
    "can raise hardware utilization from a few percent to over sixty percent of peak memory bandwidth."
)


def make_prompt(rng, tier):
    if tier == "short":
        q = rng.choice(QA)
        return f"{q} Answer concisely."
    if tier == "medium":
        n = rng.randint(2, 5)
        body = " ".join(rng.sample(SENTENCES, n))
        return f"{rng.choice(QA)} Context: {body}\n\nAnswer the question using the context."
    # long
    reps = rng.randint(6, 18)
    body = " ".join([DOC] * reps)
    return (
        f"Read the following document and then summarize the key engineering "
        f"challenges of large-model inference in a numbered list.\n\n"
        f"Document:\n{body}\n\nSummary:"
    )


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else "-"
    rng = random.Random(seed)
    tiers = (
        ["short"] * int(n * 0.40)
        + ["medium"] * int(n * 0.35)
        + ["long"] * (n - int(n * 0.40) - int(n * 0.35))
    )
    rng.shuffle(tiers)
    rows = [{"prompt": make_prompt(rng, t)} for t in tiers]
    fh = open(out, "w") if out != "-" else sys.stdout
    for r in rows:
        fh.write(json.dumps(r) + "\n")
    if fh is not sys.stdout:
        fh.close()
    # Report length distribution to stderr.
    words = [len(r["prompt"].split()) for r in rows]
    words.sort()
    sys.stderr.write(
        f"generated {len(rows)} prompts; words min={words[0]} "
        f"median={words[len(words)//2]} max={words[-1]}\n"
    )


if __name__ == "__main__":
    main()
