"""Which node of which world an engine command describes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .engine_args import _find_flag
from .profile import FrameworkSpec

SHARDED_FORMAT = "sharded_state"
PRESHARDED_FORMAT = "presharded"
_LOAD_FORMAT_FLAGS = ["--load-format"]
LOADER_CONFIG_FLAGS = ["--model-loader-extra-config"]


@dataclass(frozen=True)
class Topology:
    """One node's place in the world an engine command asks for."""

    nnodes: int = 1
    node_rank: int = 0
    tp_size: int = 1
    pp_size: int = 1
    dist_init_addr: Optional[str] = None

    @property
    def is_multinode(self) -> bool:
        return self.nnodes > 1

    @property
    def is_head(self) -> bool:
        return self.node_rank == 0

    @property
    def ranks_per_node(self) -> int:
        # Exact, not approximate: SGLang asserts the divisibility before it starts.
        return self.tp_size * self.pp_size // self.nnodes

    @property
    def local_rank_range(self) -> "tuple[int, int]":
        """(first, last) global rank on this node, both inclusive."""
        lo = self.node_rank * self.ranks_per_node
        return lo, lo + self.ranks_per_node - 1


def _int_flag(command: List[str], flags: List[str], default: int) -> int:
    found = _find_flag(command, flags)
    if found is None:
        return default
    try:
        return int(found[1])
    except ValueError:
        raise ValueError(f"{flags[0]} needs an integer, got {found[1]!r}")


def read_topology(command: List[str], spec: FrameworkSpec) -> Topology:
    """The topology `command` asks for, or raise if it asks for an impossible one.

    Checked here because the engine catches these late and cryptically -- a
    missing --dist-init-addr surfaces as a rendezvous that never completes --
    and because servekit must decide what to stage before the engine starts.
    """
    if not spec.dist_flags:
        return Topology()

    nnodes = _int_flag(command, spec.dist_flags["nnodes"], 1)
    node_rank = _int_flag(command, spec.dist_flags["node_rank"], 0)
    found = _find_flag(command, spec.dist_flags["dist_init_addr"])
    dist_init_addr = found[1] if found else None

    tp_size = _int_flag(command, spec.parallel_flags.get("tp_size", []), 1)
    pp_size = _int_flag(command, spec.parallel_flags.get("pp_size", []), 1)

    topo = Topology(
        nnodes=nnodes,
        node_rank=node_rank,
        tp_size=tp_size,
        pp_size=pp_size,
        dist_init_addr=dist_init_addr,
    )

    if nnodes < 1:
        raise ValueError(f"--nnodes must be at least 1, got {nnodes}")
    if not topo.is_multinode:
        return topo

    if not 0 <= node_rank < nnodes:
        raise ValueError(f"--node-rank {node_rank} is outside the {nnodes} nodes asked for")
    if dist_init_addr is None:
        raise ValueError(
            "--nnodes > 1 needs --dist-init-addr: without it every node rendezvouses "
            "on its own 127.0.0.1 and the process group never forms"
        )
    if (tp_size * pp_size) % nnodes != 0:
        raise ValueError(
            f"tp_size * pp_size ({tp_size} * {pp_size}) is not divisible by --nnodes {nnodes}, "
            f"so the ranks do not split evenly across nodes"
        )
    return topo


def shard_glob(topo: Topology) -> str:
    """The presharded files this node's own ranks will read, and no others.

    SGLang's ShardedStateLoader globs only its own rank's files, so staging just
    this pattern loads with no engine change.
    """
    lo, hi = topo.local_rank_range
    if lo == hi:
        return f"model-rank-{lo}-part-*.safetensors"
    # The stager hands FILE_PATTERN to `find -name`, which is fnmatch: `[0-3]`
    # is a range, `{8,9,10}` matches nothing. Refuse rather than stage zero files.
    if hi > 9:
        raise ValueError(
            f"this node holds ranks {lo}-{hi}, which needs more than one glob; "
            f"servekit can only rank-slice the stage for worlds of at most 10 ranks per node"
        )
    return f"model-rank-[{lo}-{hi}]-part-*.safetensors"


def wants_sharded_state(command: List[str]) -> bool:
    """Whether the command loads a presharded checkpoint, i.e. one file set per rank."""
    found = _find_flag(command, _LOAD_FORMAT_FLAGS)
    return found is not None and found[1] == SHARDED_FORMAT


def wants_presharded(command: List[str]) -> bool:
    """Whether the command loads a `presharded` dump, the format that survives PP.

    `sharded_state` keys its filenames on the TP rank, identical on every pipeline
    stage, so at pp > 1 the stages overwrite each other. `presharded` keys on the
    world rank instead.
    """
    found = _find_flag(command, _LOAD_FORMAT_FLAGS)
    return found is not None and found[1] == PRESHARDED_FORMAT


def loader_config(command: List[str]) -> dict:
    found = _find_flag(command, LOADER_CONFIG_FLAGS)
    if found is None:
        return {}
    try:
        config = json.loads(found[1])
    except ValueError as e:
        raise ValueError(f"{LOADER_CONFIG_FLAGS[0]} is not valid JSON: {e}")
    if not isinstance(config, dict):
        raise ValueError(f"{LOADER_CONFIG_FLAGS[0]} must be a JSON object, got {type(config).__name__}")
    return config


def presharded_root(command: List[str]) -> Path:
    """Where the command expects the dump to live."""
    root = loader_config(command).get("presharded_path")
    if not root:
        raise ValueError(
            f"--load-format {PRESHARDED_FORMAT} needs "
            f"{LOADER_CONFIG_FLAGS[0]} '{{\"presharded_path\": \"...\"}}' so servekit knows "
            f"which dump to stage"
        )
    return Path(root)
