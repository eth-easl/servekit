"""Drop-in no-op *uvloop* shim — forces plain asyncio (epoll) instead of uvloop.

Why: uvloop → libuv → io_uring on modern kernels, and CRIU cannot dump a live
io_uring instance. Disabling io_uring with a seccomp filter is self-defeating here
(a seccomp'd process needs CAP_SYS_ADMIN for CRIU to suspend the filter — the Bristen
wall), so we keep the process at Seccomp:0 and simply prevent uvloop from ever
installing its loop. Plain asyncio uses epoll, which CRIU can checkpoint.

How: this package shadows the real `uvloop` by being first on PYTHONPATH. SGLang and
uvicorn do `import uvloop` and then either `uvloop.EventLoopPolicy()` /
`uvloop.install()` — both resolve here and select the default asyncio policy. It is
inherited by every child (scheduler/detokenizer) because multiprocessing 'spawn'
re-execs python with the same PYTHONPATH, so all processes get epoll.

To use:  PYTHONPATH="<this-dir>/shim:$PYTHONPATH" python -m sglang.launch_server ...
To disable: just drop the shim dir from PYTHONPATH. Nothing is patched on disk.
"""
import asyncio

__version__ = "0.0.0+criu-epoll-shim"

# uvloop.EventLoopPolicy() -> the stock asyncio policy (SelectorEventLoop / epoll)
EventLoopPolicy = asyncio.DefaultEventLoopPolicy

# uvloop.Loop is referenced as a type in a few places; alias to the selector loop
Loop = asyncio.SelectorEventLoop


def install():
    """uvloop.install() sets the uvloop policy; we set the default asyncio one."""
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())


def new_event_loop():
    return asyncio.new_event_loop()


def run(main, *, debug=None, **_ignored):
    """Mirror uvloop.run(); ignore uvloop-only kwargs like loop_factory."""
    if debug is None:
        return asyncio.run(main)
    return asyncio.run(main, debug=debug)
