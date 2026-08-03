"""`servekit report <dir>`: one cold-start verdict from a multi-node run.

A node's report maxes each phase over the ranks in ITS log only, and the server
is ready when the slowest rank anywhere is done -- so every phase has to be
maxed again across nodes. Node 0's report alone quotes four ranks as eight.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .profile import Phase, ProfileReport, render_table

NODE_GLOB = "*.node*.json"


def find_reports(directory: Path) -> List[Path]:
    return sorted(directory.glob(NODE_GLOB))


def _load(paths: List[Path]) -> List[dict]:
    nodes = []
    for path in paths:
        try:
            nodes.append(json.loads(path.read_text()))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not readable as a report: {e}")
    return sorted(nodes, key=lambda n: n.get("node_rank") or 0)


def merge(nodes: List[dict]) -> Tuple[ProfileReport, dict]:
    """One report over every node, plus the per-node detail behind it."""
    if not nodes:
        raise ValueError("no per-node reports to merge")

    head = next((n for n in nodes if (n.get("node_rank") or 0) == 0), nodes[0])

    order: List[str] = []
    longest: Dict[str, Tuple[float, str]] = {}
    for node in [head] + [n for n in nodes if n is not head]:
        for phase in node.get("phases", []):
            name = phase["name"]
            if name not in longest:
                order.append(name)
                longest[name] = (phase["duration_s"], phase["source"])
            elif phase["duration_s"] > longest[name][0]:
                longest[name] = (phase["duration_s"], phase["source"])

    merged = ProfileReport(
        command=head.get("command", ""),
        started_at=head.get("started_at", 0.0),
        # Only the head sees the server accept traffic.
        ready_at=head.get("ready_at"),
        success=all(n.get("success") for n in nodes),
        framework=head.get("framework", "unknown"),
        phases=[Phase(name, longest[name][0], longest[name][1]) for name in order],
        benchmark=head.get("benchmark"),
        node_rank=None,
        nnodes=head.get("nnodes") or len(nodes),
    )

    detail = {
        "nodes_reporting": f"{len(nodes)}/{merged.nnodes}",
        "per_node": [
            {
                "node_rank": n.get("node_rank"),
                "success": n.get("success"),
                "total_duration_s": n.get("total_duration_s"),
                "phases": n.get("phases", []),
            }
            for n in nodes
        ],
    }
    return merged, detail


def render(merged: ProfileReport, detail: dict) -> str:
    lines = [render_table(merged), "", f"nodes reporting: {detail['nodes_reporting']}"]
    width = max([len(p["name"]) for n in detail["per_node"] for p in n["phases"]] + [len("phase")])
    for node in detail["per_node"]:
        lines.append("")
        lines.append(f"  node {node['node_rank']} ({'ready' if node['success'] else 'FAILED'})")
        for phase in node["phases"]:
            lines.append(f"    {phase['name']:<{width}}  {phase['duration_s']:>10.2f}  {phase['source']}")
    return "\n".join(lines)


def run_report(directory: Path, out: Optional[Path] = None) -> int:
    if not directory.is_dir():
        print(f"error: {directory} is not a directory", file=sys.stderr)
        return 2

    paths = find_reports(directory)
    if not paths:
        print(f"error: no per-node reports ({NODE_GLOB}) in {directory}", file=sys.stderr)
        print("       `servekit launch` writes those only when --nnodes > 1", file=sys.stderr)
        return 2

    try:
        merged, detail = merge(_load(paths))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(render(merged, detail))

    body = merged.to_dict()
    body.update(detail)
    if out is not None:
        out.write_text(json.dumps(body, indent=2))
        print(f"\nmerged report written to {out}", flush=True)

    expected = merged.nnodes or len(paths)
    if len(paths) < expected:
        # The numbers are real, they just do not cover the whole world, and a
        # phase maxed over too few ranks reads too fast.
        print(
            f"error: {len(paths)} of {expected} nodes reported; the phases above are "
            f"maxed over an incomplete world",
            file=sys.stderr,
        )
        return 1
    return 0 if merged.success else 1
