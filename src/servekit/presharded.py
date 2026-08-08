"""Reading SGLang's `presharded` dump layout.

The format `sharded_state` should have been: `PreshardedModelLoader` keys files on
the world rank rather than the TP rank, so pipeline stages holding different layers
no longer collide on one filename.

Pure: reads `checksum.json` and stats files, nothing else.

`checksum.json` names, for every world rank, the exact files that rank reads. A
node's file set is the union over the ranks it hosts -- which is the whole staging
rule, and why this path needs no globbing. Files read by several ranks (`-common`,
and TP-replicated tensors that content dedup stored once) appear in each of their
read lists, so a file two nodes both need is staged on both without a special case.
"""
from __future__ import annotations

import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

READY_NAME = "READY"
CHECKSUM_NAME = "checksum.json"
DUMP_GLOB = "TP-*-sig-*"

# What servekit can read off a launch command; shard_config carries more (eplb,
# moe_dense_tp_size, ...) that only the engine knows.
COMPARABLE = ("tp", "pp", "ep", "dp")

_LAYER = re.compile(r"model\.layers\.(\d+)\.")


@dataclass(frozen=True)
class ShardPlan:
    directory: Path
    world_size: int
    shard_config: Dict[str, Any]
    rank_to_files: Dict[int, List[str]]

    def files_for_ranks(self, lo: int, hi: int) -> List[str]:
        """Every file the ranks `lo..hi` read, deduplicated, in dump order."""
        missing = [r for r in range(lo, hi + 1) if r not in self.rank_to_files]
        if missing:
            raise ValueError(
                f"{self.directory.name} was dumped for {self.world_size} ranks and has "
                f"no file list for rank(s) {missing}; this node expects ranks {lo}-{hi}"
            )
        seen: Dict[str, None] = {}
        for rank in range(lo, hi + 1):
            for name in self.rank_to_files[rank]:
                seen.setdefault(name, None)
        return sorted(seen)

    def bytes_for_ranks(self, lo: int, hi: int) -> int:
        return sum((self.directory / f).stat().st_size for f in self.files_for_ranks(lo, hi))


def loader_available() -> Optional[bool]:
    """Whether the installed SGLang has the loader. None when it cannot be told.

    Read off disk rather than imported: `import sglang` costs tens of seconds,
    and this runs before the engine is worth starting.
    """
    try:
        spec = importlib.util.find_spec("sglang")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(list(spec.submodule_search_locations)[0]) / "srt" / "configs" / "load_config.py"
    if not path.is_file():
        return None
    return "PRESHARDED" in path.read_text()


MISSING_LOADER = (
    "the installed sglang has no PreshardedModelLoader, so --load-format presharded cannot work. "
    "It is absent from v0.5.10 through v0.5.16 and reached no release; use a build from main, or "
    "one carrying it as an overlay (lmsysorg/sglang:kimi-k3 does)"
)


def find_dumps(root: Path) -> List[Path]:
    """Completed dumps under `root`, newest first by nothing in particular.

    Several may coexist: the subdirectory is keyed by the full parallel and quant
    config, so one root can hold a tp4/pp8 dump beside a tp8/pp4 one.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(d for d in root.glob(DUMP_GLOB) if (d / READY_NAME).is_file())


def read_plan(dump: Path) -> ShardPlan:
    dump = Path(dump)
    if not (dump / READY_NAME).is_file():
        raise ValueError(
            f"{dump} has no {READY_NAME} sentinel, so the dump never finished; "
            f"loading it would silently miss and re-dump"
        )
    raw = json.loads((dump / CHECKSUM_NAME).read_text())
    rank_to_files: Dict[int, List[str]] = {}
    for rank, reads in raw["rank_to_reads"].items():
        names: Dict[str, None] = {}
        for entry in reads:
            names.setdefault(entry["filename"], None)
        rank_to_files[int(rank)] = sorted(names)
    return ShardPlan(
        directory=dump,
        world_size=int(raw["world_size"]),
        shard_config=raw.get("shard_config") or {},
        rank_to_files=rank_to_files,
    )


def select_dump(root: Path, wanted: Dict[str, int]) -> ShardPlan:
    """The dump under `root` matching `wanted`, or raise saying what is there.

    Raising is the point. The loader's own response to a config it has no dump for
    is to warn and re-dump -- a full HF load plus a rewrite of the whole checkpoint,
    which on a big model is not something to discover from a log line afterwards.
    """
    dumps = find_dumps(root)
    if not dumps:
        raise ValueError(f"no completed presharded dump under {root}; run `servekit prepare` first")

    plans = [read_plan(d) for d in dumps]
    matches = [p for p in plans if all(p.shard_config.get(k) == v for k, v in wanted.items())]
    if len(matches) == 1:
        return matches[0]

    asked = ", ".join(f"{k}={v}" for k, v in wanted.items())
    if not matches:
        have = "; ".join(
            f"{p.directory.name} ({', '.join(f'{k}={p.shard_config.get(k)}' for k in COMPARABLE)})"
            for p in plans
        )
        raise ValueError(
            f"no presharded dump under {root} was written for {asked}. Available: {have}. "
            f"The engine would treat this as a cache miss and re-dump the whole checkpoint"
        )
    raise ValueError(
        f"{len(matches)} dumps under {root} all match {asked}: "
        f"{', '.join(p.directory.name for p in matches)}"
    )


def check_shard_config(plan: ShardPlan, wanted: Dict[str, int]) -> List[str]:
    problems = []
    for key, value in wanted.items():
        stored = plan.shard_config.get(key)
        if stored is not None and stored != value:
            problems.append(
                f"{key}: {plan.directory.name} was dumped for {key}={stored}, "
                f"the command asks for {value}"
            )
    world = wanted.get("tp", 1) * wanted.get("pp", 1)
    if plan.world_size != world:
        problems.append(
            f"world_size: {plan.directory.name} holds {plan.world_size} ranks, "
            f"the command asks for tp*pp = {world}"
        )
    return problems


def verify(plan: ShardPlan, tp: int, pp: int, reads: Optional[Dict[int, Sequence[str]]] = None) -> List[str]:
    """Ways the dump is not a valid `tp` x `pp` split, in plain words.

    NOTE the obvious check -- no two ranks share a tensor -- is wrong as soon as
    tp > 1: TP peers within a stage legitimately hold identical copies of every
    replicated param, which content dedup stores once and points both ranks at.
    What must not happen is two pipeline stages owning the same layer. So group
    ranks by the layer set they hold and require exactly `pp` groups of `tp` ranks
    tiling 0..L-1. Derived from the data, so it assumes nothing about how SGLang
    maps a world rank to (pp_rank, tp_rank).
    """
    if reads is None:
        raw = json.loads((plan.directory / CHECKSUM_NAME).read_text())
        reads = {int(r): [e["name"] for e in entries] for r, entries in raw["rank_to_reads"].items()}

    problems = []
    world = tp * pp
    if plan.world_size != world:
        problems.append(f"world_size is {plan.world_size}, expected tp*pp = {world}")
    if sorted(reads) != list(range(world)):
        problems.append(f"rank_to_reads covers ranks {sorted(reads)}, expected 0..{world - 1}")
        return problems

    rank_layers = {
        rank: frozenset(int(m.group(1)) for m in map(_LAYER.search, names) if m)
        for rank, names in reads.items()
    }
    groups: Dict[frozenset, List[int]] = {}
    for rank, layers in rank_layers.items():
        groups.setdefault(layers, []).append(rank)

    if len(groups) != pp:
        problems.append(f"{len(groups)} distinct layer sets, expected {pp} pipeline stages")
    odd = sorted(r for ranks in groups.values() if len(ranks) != tp for r in ranks)
    if odd:
        problems.append(f"ranks {odd} are in a pipeline stage not held by exactly tp={tp} ranks")

    stages = list(groups)
    for i, first in enumerate(stages):
        for second in stages[i + 1 :]:
            shared = sorted(first & second)
            if shared:
                problems.append(f"two pipeline stages both own layers {shared}")

    covered = set().union(*stages) if stages else set()
    if covered != set(range(len(covered))):
        problems.append(f"layers do not tile 0..N-1: {len(covered)} distinct indices")
    return problems
