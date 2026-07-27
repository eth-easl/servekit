# import-startup-exp — experiment log

See `PLAN.md` for context and methodology. This log is populated as runs land.

| sub-experiment | question | verdict | detail |
|---|---|---|---|
| `idea0_split` | is import time compile, cold I/O, or module exec? | **cold I/O 15.7-21.0s, exec 8.9-9.0s, compile 0.03-0.41s** | [results](results/idea0_split/results.md) |

2026-07-27 — Idea 0 done (jobs 76482/76483/76484). Compile is ~1% of import
time, so idea 2 (precompile `.pyc`) is closed despite sglang shipping 0 `.pyc`:
only 117 modules are actually imported and they compile in milliseconds. Idea 3
(cold I/O off the 30 GB squashfs on Lustre) is ~65-70% and is the next thing to
run. Idea 1 (spawn→fork) still open, but its ceiling is now the ~9s exec floor,
not the whole worker import cost.
