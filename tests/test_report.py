"""Merging per-node reports into one cold-start verdict."""
import json

from servekit.cli import main
from servekit.report import merge


def _node(rank, phases, nnodes=2, ready_at=None, success=True, benchmark=None):
    return {
        "command": "python -m sglang.launch_server --tp-size 8",
        "framework": "sglang",
        "started_at": 100.0,
        "ready_at": ready_at,
        "success": success,
        "total_duration_s": round((ready_at or 100.0) - 100.0, 2),
        "phases": [{"name": n, "duration_s": d, "source": "engine_reported"} for n, d in phases],
        "benchmark": benchmark,
        "node_rank": rank,
        "nnodes": nnodes,
    }


def _write(tmp_path, nodes):
    for node in nodes:
        (tmp_path / f"run.node{node['node_rank']}.json").write_text(json.dumps(node))


def test_a_phase_is_the_slowest_rank_anywhere_not_the_slowest_on_the_head():
    """The whole point: node 0's report only ever maxed over node 0's ranks."""
    merged, _ = merge([
        _node(0, [("weight_loading", 3.25), ("cuda_graph_capture", 21.0)], ready_at=220.3),
        _node(1, [("weight_loading", 9.10), ("cuda_graph_capture", 20.4)]),
    ])
    phases = {p.name: p.duration_s for p in merged.phases}
    assert phases["weight_loading"] == 9.10
    assert phases["cuda_graph_capture"] == 21.0


def test_the_total_comes_from_the_head_which_is_the_only_node_that_sees_ready():
    merged, _ = merge([
        _node(0, [("weight_loading", 3.0)], ready_at=223.4),
        _node(1, [("weight_loading", 9.0)], ready_at=190.0),
    ])
    assert merged.total_duration_s == 123.4


def test_phase_order_follows_the_head_and_keeps_a_workers_extra_phase():
    merged, _ = merge([
        _node(0, [("stage", 3.1), ("weight_loading", 3.2)]),
        _node(1, [("stage", 2.7), ("weight_loading", 3.3), ("tp_worker_spawn", 1.5)]),
    ])
    assert [p.name for p in merged.phases] == ["stage", "weight_loading", "tp_worker_spawn"]


def test_the_benchmark_rides_along_from_the_head():
    bench = {"throughput": {"output_tok_per_s": 478.1}}
    merged, _ = merge([_node(0, [("weight_loading", 3.0)], benchmark=bench), _node(1, [("weight_loading", 9.0)])])
    assert merged.benchmark == bench


def test_one_failed_node_fails_the_run():
    merged, _ = merge([_node(0, [("weight_loading", 3.0)]), _node(1, [], success=False)])
    assert not merged.success


def test_report_merges_a_directory_and_says_how_many_nodes_it_saw(tmp_path, capsys):
    _write(tmp_path, [
        _node(0, [("weight_loading", 3.25)], ready_at=223.4),
        _node(1, [("weight_loading", 9.10)]),
    ])

    rc = main(["report", str(tmp_path), "--out", str(tmp_path / "merged.json")])

    assert rc == 0
    assert "nodes reporting: 2/2" in capsys.readouterr().out
    body = json.loads((tmp_path / "merged.json").read_text())
    assert body["total_duration_s"] == 123.4
    assert {p["name"]: p["duration_s"] for p in body["phases"]}["weight_loading"] == 9.10
    # The per-node detail survives the merge, so the max can be checked.
    assert [n["node_rank"] for n in body["per_node"]] == [0, 1]


def test_a_missing_node_is_an_error_because_the_max_is_over_too_few_ranks(tmp_path, capsys):
    _write(tmp_path, [_node(0, [("weight_loading", 3.25)], ready_at=223.4)])

    rc = main(["report", str(tmp_path), "--out", str(tmp_path / "merged.json")])

    assert rc == 1
    assert "1 of 2 nodes reported" in capsys.readouterr().err
    # Still written: the numbers are real, they just do not cover the world.
    assert (tmp_path / "merged.json").is_file()


def test_a_directory_with_no_per_node_reports_says_so(tmp_path, capsys):
    (tmp_path / "run.json").write_text(json.dumps(_node(0, [], nnodes=1)))
    assert main(["report", str(tmp_path)]) == 2
    assert "no per-node reports" in capsys.readouterr().err
