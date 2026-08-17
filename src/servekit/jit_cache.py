"""Where the engine's JIT caches live, so they outlive the job."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Optional

CACHE_DIR_NAME = "servekit-caches"

# Not /dev/shm, where the weights go: these hold .so files the engine dlopens
# and the container mounts /dev/shm noexec. And the same path every time --
# ninja records absolute paths, so a cache read anywhere other than where it was
# built rebuilds itself.
NODE_LOCAL_ROOT = Path("/tmp/servekit/caches")

CACHES: Dict[str, str] = {
    "TRITON_CACHE_DIR": "triton",
    "TVM_FFI_CACHE_DIR": "tvm-ffi",
    # FlashInfer has no cache-path variable; this stands in for $HOME and it
    # appends .cache/flashinfer itself.
    "FLASHINFER_WORKSPACE_BASE": "flashinfer",
}


def node_local(root: Path, name: str) -> Path:
    return root / name


def env_for(root: Path) -> Dict[str, str]:
    return {var: str(root / subdir) for var, subdir in CACHES.items()}


def create(root: Path) -> Dict[str, str]:
    for subdir in CACHES.values():
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return env_for(root)


def copy_into(src_root: Path, dest_root: Path) -> Optional[Path]:
    if not src_root.is_dir():
        return None
    shutil.copytree(src_root, dest_root, dirs_exist_ok=True)
    return dest_root
