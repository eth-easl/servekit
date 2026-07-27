# Gate 3a — Python + torch.cuda checkpoint/restore

**Status:** done — **PASS**

## Goal

De-risk the jump from Gate 2's minimal C/CUDA counter to a full Python SGLang server
(Gate 3b) in one smaller step. A bare-metal CUDA C program and a Python process running
`torch.cuda` are very different beasts for CRIU to dump (interpreter state, loaded
shared libraries, torch's own CUDA context management) — this gate isolates whether
*that* jump works before adding SGLang's full process tree, its subprocess workers, and
its network sockets on top.

## Method

`scripts/checkpoint_server.sh 3a` runs `src/python_cuda_counter.py` — a Python process
that starts CUDA via torch and increments a device-memory counter, logging `tick=N`.
Same round trip as Gate 2: `cuda-checkpoint` lock+checkpoint → `criu --unprivileged dump`
(kills the process) → `criu --unprivileged restore` → `cuda-checkpoint` restore+unlock →
confirm the tick sequence resumes correctly.

## Result

Raw output: `gate3a.txt`.

- Froze at `tick=3`, GPU memory correctly dropped to ~1 MiB (evicted) at checkpoint time.
- `criu dump` killed the process as expected; **649 MB snapshot** on disk.
- Restore succeeded; GPU state reattached; first tick after restore was exactly `4`
  (`froze+1`), sequence contiguous.

## Verdict

**GATE 3a: PASS.** A real Python + `torch.cuda` process survives the full
kill-and-restore-from-disk round trip, not just the minimal C counter from Gate 2. Snapshot
size scaling is worth tracking across gates: 291 MB (Gate 2's C counter, essentially just
the CUDA context) → 649 MB (this gate, adds the Python/torch interpreter state) → 26 GB
(Gate 3b, adds the actual 16 GB of model weights). This confirms the technique
generalizes to a Python+CUDA process before spending a full server-launch cycle on Gate 3b.

## Caveats

Still a single process, no subprocess tree, no network sockets, no io_uring — those are
exactly the complications Gate 3b (a real SGLang server) adds and had to solve for.
