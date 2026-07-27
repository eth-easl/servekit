"""Share a staged hugepage memfd across separately-spawned processes.

TP ranks in this SGLang are spawned via multiprocessing.spawn, not forked
(see CLAUDE.md / project memory: each rank re-imports the full torch stack --
that is only possible/needed because ranks do NOT share a fork'd address
space). An anonymous memfd(MFD_HUGETLB) has no filesystem path, so it can't be
opened independently by each rank the way a /dev/shm path can -- it has to be
handed over explicitly. This uses the exact mechanism the reference blog post
uses (a Unix-socket + SCM_RIGHTS handoff), via stdlib
multiprocessing.reduction.sendfds/recvfds rather than raw sendmsg/recvmsg.

Wire format per client connection: sendfds() first (1 fd), then one
length-prefixed pickle of (total_len, files_meta).

    server = FdBrokerServer(sock_path, fd, total_len, files_meta)
    server.start()   # background thread, serves every connecting client
    ...
    fd, total_len, files_meta = fetch(sock_path)   # called from each rank
"""
import multiprocessing.reduction as reduction
import pickle
import socket
import struct
import threading


class FdBrokerServer:
    def __init__(self, sock_path, fd, total_len, files_meta):
        self.sock_path = sock_path
        self.fd = fd
        self.payload = pickle.dumps((total_len, files_meta))
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(sock_path)
        self._srv.listen(8)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._stop = False

    def start(self):
        self._thread.start()

    def _serve(self):
        # One thread per connection: with N ranks connecting near-simultaneously
        # (e.g. all 4 TP ranks right after spawn), a single-threaded
        # accept-handle-close loop would serialize their handoffs -- whichever
        # rank connects first gets served first and every later rank waits on
        # the ones ahead of it. Concurrent handling guarantees the broker adds
        # no ordering-dependent overhead regardless of connect timing.
        while not self._stop:
            try:
                self._srv.settimeout(1.0)
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            t = threading.Thread(target=self._handle, args=(conn,), daemon=True)
            t.start()

    def _handle(self, conn):
        try:
            reduction.sendfds(conn, [self.fd])
            conn.sendall(struct.pack("<Q", len(self.payload)) + self.payload)
        finally:
            conn.close()

    def stop(self):
        self._stop = True
        self._srv.close()


def fetch(sock_path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    try:
        fds = reduction.recvfds(sock, 1)
        header = b""
        while len(header) < 8:
            chunk = sock.recv(8 - len(header))
            if not chunk:
                raise EOFError("broker closed before sending length header")
            header += chunk
        n = struct.unpack("<Q", header)[0]
        payload = b""
        while len(payload) < n:
            chunk = sock.recv(n - len(payload))
            if not chunk:
                raise EOFError("broker closed mid-payload")
            payload += chunk
        total_len, files_meta = pickle.loads(payload)
        return fds[0], total_len, files_meta
    finally:
        sock.close()
