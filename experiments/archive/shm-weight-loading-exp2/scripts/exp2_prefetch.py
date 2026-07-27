"""Background registrar: overlap cudaHostRegister with model construction and
with the bounce copy, and split the work between the two paths.

The measurements this is built on (jobs 76369-76378, all NUMA-bound, 4 ranks,
real 35.3 GB shards):

  DMA out of a registered tmpfs mapping   1.33 s   (26.5 GB/s -- line rate)
  cudaHostRegister on 4 KB tmpfs pages    9.7 s    (3.6 GB/s, does NOT
                                                   parallelise: 8 threads
                                                   gave 9.3-10.5 s)
  bounce: mt gather -> pinned -> DMA      4.46 s   (7.95 GB/s, bandwidth-bound
                                                   on the node's 2 channels)
  model construction inside the window    2.12 s   (not a copy; untouchable)

Neither path alone fits the 2.88 s that is left of a 5 s budget after
construction. But they bottleneck on different resources -- registration is
serialised kernel page-pinning work, the bounce is DRAM bandwidth -- and
registration does not need the model to exist. So:

  * the registrar starts at the TOP of load_model, before _initialize_model,
    and claims files from the FRONT of the list;
  * the copier starts after construction (~2.12 s later) and claims files
    from the BACK, bouncing them;
  * they meet in the middle, each file handled exactly once by whichever
    path reaches it first;
  * whenever a registered file is waiting, the copier prefers it -- that is a
    line-rate DMA with no host work at all.

Solving 3.6*T + 7.95*(T - 2.12) = 35.3 gives T ~ 4.5 s, which is the whole
reason this design exists.

Every CUDA copy is issued from the copier thread. The background thread only
ever calls cudaHostRegister. That split is deliberate: mixing
cudaHostRegister and cudaMemcpyAsync across threads is the configuration the
hugepage experiment spent several jobs suspecting, and there is no reason to
re-enter that territory when the work divides cleanly anyway.
"""

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_state = None


class Registrar:
    def __init__(self, paths):
        self.paths = list(paths)
        self.lock = threading.Lock()
        self.claimed = [None] * len(self.paths)   # None | "reg" | "copy"
        self.ready = {}                           # idx -> (mm, base, nbytes)
        self.keepalive = []                       # mappings must outlive the DMA
        self.registered_bytes = 0
        self.t_start = time.perf_counter()
        self.thread = None
        self.error = None

    # -- called by the copier ------------------------------------------------

    def take_registered(self):
        """A file already registered and not yet copied, or None."""
        with self.lock:
            for idx in sorted(self.ready):
                if self.claimed[idx] == "reg":
                    self.claimed[idx] = "reg-taken"
                    return idx, self.ready[idx]
        return None

    def take_for_bounce(self):
        """Claim the last unclaimed file, so the copier works backwards while
        the registrar works forwards."""
        with self.lock:
            for idx in range(len(self.paths) - 1, -1, -1):
                if self.claimed[idx] is None:
                    self.claimed[idx] = "copy"
                    return idx
        return None

    def all_done(self):
        with self.lock:
            return all(c in ("copy", "reg-taken") for c in self.claimed)

    # -- background thread ---------------------------------------------------

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True, name="exp2-registrar")
        self.thread.start()

    def _run(self):
        import ctypes
        import mmap

        import exp2_cudart as cu

        for idx, path in enumerate(self.paths):
            with self.lock:
                if self.claimed[idx] is not None:
                    continue        # the copier already took it; stop competing
                self.claimed[idx] = "reg-inflight"
            try:
                fd = os.open(path, os.O_RDWR)
                try:
                    mm = mmap.mmap(fd, 0)
                finally:
                    os.close(fd)
                # Hold the mapping: if it is collected the kernel reuses the
                # address and the next registration returns 712 against the
                # stale range (job 76368).
                self.keepalive.append(mm)
                base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                nbytes = len(mm)
                rc = cu.libcudart.cudaHostRegister(
                    ctypes.c_void_p(base), ctypes.c_size_t(nbytes), 0
                )
                if rc != 0:
                    raise RuntimeError(f"cudaHostRegister rc={rc} on {path}")
            except Exception as exc:  # noqa: BLE001
                # Never fail the load because the fast path is unavailable --
                # hand the file back and let the bounce take it.
                self.error = exc
                logger.warning("exp2 registrar: giving up on %s (%r)", path, exc)
                with self.lock:
                    self.claimed[idx] = None
                return
            with self.lock:
                self.ready[idx] = (mm, base, nbytes)
                self.claimed[idx] = "reg"
                self.registered_bytes += nbytes
        logger.info(
            "exp2 registrar: registered %.1f GB in %.2f s",
            self.registered_bytes / 1e9, time.perf_counter() - self.t_start,
        )


def start(paths):
    """Called at the top of load_model, before the model is built."""
    global _state
    if not paths:
        return None
    _state = Registrar(paths)
    _state.start()
    logger.info("exp2 registrar: started on %d files", len(paths))
    return _state


def get():
    return _state


def stop():
    global _state
    st, _state = _state, None
    return st
