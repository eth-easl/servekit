"""The `presharded` format reader, against plans cut from two real dumps.

The fixtures are the `checksum.json` of the Apertus-8B tp1/pp4 and tp2/pp2 dumps
from experiments/pp-loading-exp, with the per-tensor detail trimmed away. Real
plans matter here: the whole staging rule is "believe rank_to_reads", and a
hand-written plan would only prove servekit agrees with itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from servekit import presharded
from servekit.engine_args import parallel_sizes, with_presharded_loader
from servekit.profile import detect_framework
from servekit.topology import Topology, load_format

FIXTURES = Path(__file__).parent / "fixtures" / "presharded"


def build_dump(tmp_path: Path, tag: str, name: str = "TP-2-sig-a78ef1f60dd6a013") -> Path:
    """A dump directory with the real plan and correctly-sized empty files."""
    dump = tmp_path / name
    dump.mkdir(parents=True)
    (dump / presharded.CHECKSUM_NAME).write_text((FIXTURES / f"{tag}.json").read_text())
    for filename, size in json.loads((FIXTURES / f"{tag}.sizes.json").read_text()).items():
        with open(dump / filename, "wb") as f:
            f.truncate(size)
    (dump / presharded.READY_NAME).write_text("{}")
    return dump


# --- the plan --------------------------------------------------------------

@pytest.mark.parametrize("tag,tp,pp", [("tp1pp4", 1, 4), ("tp2pp2", 2, 2)])
def test_real_dumps_verify_as_their_own_split(tmp_path, tag, tp, pp):
    plan = presharded.read_plan(build_dump(tmp_path, tag))
    assert plan.world_size == tp * pp
    assert presharded.verify(plan, tp, pp) == []


def test_verify_rejects_the_wrong_pipeline_depth(tmp_path):
    plan = presharded.read_plan(build_dump(tmp_path, "tp2pp2"))
    problems = presharded.verify(plan, tp=1, pp=4)
    assert problems
    assert any("layer sets" in p or "exactly tp=1" in p for p in problems)


def test_a_dump_without_ready_is_refused(tmp_path):
    dump = build_dump(tmp_path, "tp2pp2")
    (dump / presharded.READY_NAME).unlink()
    with pytest.raises(ValueError, match="never finished"):
        presharded.read_plan(dump)
    assert presharded.find_dumps(tmp_path) == []


# --- the staging rule ------------------------------------------------------

def test_a_node_stages_the_union_of_its_ranks_reads(tmp_path):
    plan = presharded.read_plan(build_dump(tmp_path, "tp2pp2"))
    whole = plan.files_for_ranks(0, 3)
    assert set(whole) == {f.name for f in plan.directory.glob("*.safetensor")}

    first, second = plan.files_for_ranks(0, 1), plan.files_for_ranks(2, 3)
    assert set(first) | set(second) == set(whole)
    # Both halves are strictly smaller than the whole: that saving is the point.
    assert len(first) < len(whole) and len(second) < len(whole)


def test_files_two_nodes_both_read_land_on_both(tmp_path):
    """The case a partition would get wrong.

    At tp2/pp2 split over two nodes, `-common` files and the TP-replicated
    tensors that dedup stored once are read from both halves of the world. A
    stager that divided the files up would starve one node of them.
    """
    plan = presharded.read_plan(build_dump(tmp_path, "tp2pp2"))
    shared = set(plan.files_for_ranks(0, 1)) & set(plan.files_for_ranks(2, 3))
    assert shared
    assert any("-common" in f for f in shared)
    # Replication is real but small; it must not approach staging everything twice.
    both = plan.bytes_for_ranks(0, 1) + plan.bytes_for_ranks(2, 3)
    whole = plan.bytes_for_ranks(0, 3)
    assert whole < both < 1.5 * whole


def test_a_rank_outside_the_dump_is_named(tmp_path):
    plan = presharded.read_plan(build_dump(tmp_path, "tp2pp2"))
    with pytest.raises(ValueError, match="rank"):
        plan.files_for_ranks(4, 5)


# --- picking a dump --------------------------------------------------------

def test_select_dump_matches_on_shard_config(tmp_path):
    build_dump(tmp_path, "tp2pp2", "TP-2-sig-a78ef1f60dd6a013")
    build_dump(tmp_path, "tp1pp4", "TP-1-sig-bf6babb08ad0b807")
    assert presharded.select_dump(tmp_path, {"tp": 2, "pp": 2}).world_size == 4
    assert presharded.select_dump(tmp_path, {"tp": 1, "pp": 4}).shard_config["pp"] == 4


def test_no_matching_dump_says_what_is_there_rather_than_re_dumping(tmp_path):
    build_dump(tmp_path, "tp2pp2")
    with pytest.raises(ValueError) as e:
        presharded.select_dump(tmp_path, {"tp": 4, "pp": 8})
    assert "re-dump" in str(e.value)
    assert "TP-2-sig-a78ef1f60dd6a013" in str(e.value)


def test_a_dump_matching_on_tp_and_pp_but_not_ep_is_not_a_match(tmp_path):
    """tp/pp agreeing is not enough: the dump is keyed by the whole config.

    Every axis is matched in one pass, so an ep the dump was not written for is a
    miss like any other -- and the error names the dumps that do exist.
    """
    build_dump(tmp_path, "tp2pp2")
    assert presharded.select_dump(tmp_path, {"tp": 2, "pp": 2, "ep": 1, "dp": 1}).world_size == 4
    with pytest.raises(ValueError, match="re-dump"):
        presharded.select_dump(tmp_path, {"tp": 2, "pp": 2, "ep": 4, "dp": 1})


def test_an_axis_the_dump_does_not_record_cannot_disagree(tmp_path):
    """So a dump written before servekit read an axis stays usable."""
    dump = build_dump(tmp_path, "tp2pp2")
    raw = json.loads((dump / presharded.CHECKSUM_NAME).read_text())
    del raw["shard_config"]["ep"]
    (dump / presharded.CHECKSUM_NAME).write_text(json.dumps(raw))
    assert presharded.select_dump(tmp_path, {"tp": 2, "pp": 2, "ep": 4}).world_size == 4


def test_a_dump_whose_world_size_contradicts_its_config_is_refused(tmp_path):
    dump = build_dump(tmp_path, "tp2pp2")
    raw = json.loads((dump / presharded.CHECKSUM_NAME).read_text())
    raw["world_size"] = 7
    (dump / presharded.CHECKSUM_NAME).write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="corrupt"):
        presharded.read_plan(dump)


# --- the command -----------------------------------------------------------

def command(model: str = "/store/dump", **flags) -> list:
    """A serving command as a user writes it: no loader flags, servekit adds those."""
    out = ["python", "-m", "sglang.launch_server", "--model-path", model]
    for flag, value in flags.items():
        out += [f"--{flag}", str(value)]
    return out


def test_load_format_reports_only_what_the_command_names():
    assert load_format(command()) is None
    assert load_format(command(**{"load-format": "presharded"})) == "presharded"
    assert load_format(command(**{"load-format": "auto"})) == "auto"


def test_servekit_writes_the_loader_flags_rather_than_reading_them():
    """The dump lives under the model path, so a hand-written root could only differ."""
    after = with_presharded_loader(command(**{"load-format": "auto"}), "/dev/shm/servekit/dump")
    assert after.count("--load-format") == 1
    assert after[after.index("--load-format") + 1] == "presharded"
    assert json.loads(after[after.index("--model-loader-extra-config") + 1]) == {
        "presharded_path": "/dev/shm/servekit/dump"
    }


def test_a_caller_supplied_loader_config_is_dropped_in_either_spelling():
    before = command() + [
        "--model-loader-extra-config=" + json.dumps({"presharded_path": "/elsewhere"}),
    ]
    after = with_presharded_loader(before, "/dev/shm/d")
    assert "/elsewhere" not in " ".join(after)
    assert json.loads(after[after.index("--model-loader-extra-config") + 1]) == {
        "presharded_path": "/dev/shm/d"
    }


def test_parallel_sizes_speaks_shard_configs_vocabulary():
    cmd = command(**{"tp-size": 4, "pp-size": 8, "ep-size": 4})
    sizes = parallel_sizes(cmd, detect_framework(cmd))
    assert sizes == {"tp": 4, "pp": 8, "dp": 1, "ep": 4}


# --- what launch decides ---------------------------------------------------

def staging(tmp_path, node_rank, nnodes, tp, pp, shm=Path("/dev/shm/servekit")):
    """What launch decides for a command that never names a format."""
    from servekit.launch import _presharded_dump, _presharded_staging

    cmd = command(str(tmp_path), **{"tp-size": tp, "pp-size": pp, "nnodes": nnodes,
                                    "node-rank": node_rank, "dist-init-addr": "h:1"})
    spec = detect_framework(cmd)
    topo = Topology(nnodes=nnodes, node_rank=node_rank, tp_size=tp, pp_size=pp, dist_init_addr="h:1")
    dump = _presharded_dump(cmd, spec, topo, tmp_path)
    return _presharded_staging(dump, topo, shm / tmp_path.name), dump


def test_pipeline_parallelism_alone_selects_the_dump(tmp_path):
    """No --load-format anywhere: pp > 1 plus a prepared dump is the whole rule."""
    build_dump(tmp_path, "tp2pp2")
    plan, dump = staging(tmp_path, node_rank=0, nnodes=2, tp=2, pp=2)
    assert dump is not None
    assert plan.src == dump.directory
    assert plan.files == dump.files_for_ranks(0, 1)


def test_the_dump_keeps_its_subdirectory_under_the_staged_model_dir(tmp_path):
    """The staged copy is the model path now, so the dump sits inside it."""
    build_dump(tmp_path, "tp2pp2")
    plan, dump = staging(tmp_path, node_rank=0, nnodes=2, tp=2, pp=2)
    assert plan.dest == Path("/dev/shm/servekit") / tmp_path.name / dump.directory.name


def test_without_pipeline_parallelism_nothing_is_presharded(tmp_path):
    """Even with a dump sitting right there: tp-only keeps the established path."""
    from servekit.launch import _presharded_dump

    build_dump(tmp_path, "tp2pp2")
    cmd = command(str(tmp_path), **{"tp-size": 2})
    assert _presharded_dump(cmd, detect_framework(cmd), Topology(tp_size=2), tmp_path) is None


def test_pipeline_parallelism_with_no_dump_falls_back_rather_than_failing(tmp_path):
    """Serving PP off a plain checkpoint stays possible; it is just slower."""
    from servekit.launch import _presharded_dump

    cmd = command(str(tmp_path), **{"tp-size": 2, "pp-size": 2})
    assert _presharded_dump(cmd, detect_framework(cmd), Topology(tp_size=2, pp_size=2), tmp_path) is None


def test_an_explicit_presharded_format_with_no_dump_is_an_error(tmp_path):
    from servekit.launch import _presharded_dump

    cmd = command(str(tmp_path), **{"tp-size": 2, "pp-size": 2, "load-format": "presharded"})
    with pytest.raises(ValueError, match="run `servekit prepare` first"):
        _presharded_dump(cmd, detect_framework(cmd), Topology(tp_size=2, pp_size=2), tmp_path)


def test_each_node_gets_only_its_own_stages_files(tmp_path):
    build_dump(tmp_path, "tp2pp2")
    (head, _), (worker, _) = (
        staging(tmp_path, node_rank=0, nnodes=2, tp=2, pp=2),
        staging(tmp_path, node_rank=1, nnodes=2, tp=2, pp=2),
    )
    assert head.files != worker.files
    assert set(head.files) | set(worker.files) == {
        f for f in head.files + worker.files
    }


def test_the_plan_is_copied_first_and_ready_written_last(tmp_path):
    """Order is the loader's barrier: it reads checksum.json, and gates on READY."""
    build_dump(tmp_path, "tp2pp2")
    plan, _ = staging(tmp_path, node_rank=0, nnodes=1, tp=2, pp=2)
    assert plan.presync == (presharded.CHECKSUM_NAME,)
    assert plan.ready_marker == plan.dest / presharded.READY_NAME
    assert presharded.READY_NAME not in (plan.files or [])


def test_launch_refuses_a_dump_that_does_not_match_the_command(tmp_path):
    build_dump(tmp_path, "tp2pp2")
    with pytest.raises(ValueError, match="re-dump"):
        staging(tmp_path, node_rank=0, nnodes=1, tp=4, pp=8)


def test_launch_refuses_a_world_larger_than_the_dump(tmp_path):
    build_dump(tmp_path, "tp2pp2")
    # tp*pp matches the dump, but 8 nodes of it do not exist.
    with pytest.raises(ValueError):
        staging(tmp_path, node_rank=7, nnodes=8, tp=2, pp=2)


def test_ranks_per_node_splits_a_pipeline_across_nodes():
    """tp=4 pp=8 on 8 nodes is one stage per node; the same world on 4 is two."""
    assert Topology(nnodes=8, node_rank=3, tp_size=4, pp_size=8).local_rank_range == (12, 15)
    assert Topology(nnodes=4, node_rank=1, tp_size=4, pp_size=8).local_rank_range == (8, 15)
