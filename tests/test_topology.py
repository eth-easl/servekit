"""Reading a world out of an engine command."""
import pytest

from servekit.profile import SGLANG, VLLM
from servekit.topology import Topology, read_topology, shard_glob, wants_sharded_state


def sglang(*args):
    return ["python", "-m", "sglang.launch_server", "--model-path", "/store/m", *args]


def test_a_plain_command_is_one_node():
    topo = read_topology(sglang("--tp-size", "4"), SGLANG)
    assert not topo.is_multinode and topo.is_head
    assert topo.nnodes == 1 and topo.node_rank == 0
    # Every rank is local, which is what makes the single-node free trigger the
    # same code path as the multi-node one.
    assert topo.ranks_per_node == 4
    assert topo.local_rank_range == (0, 3)


def test_reads_the_engines_own_distributed_flags():
    topo = read_topology(
        sglang("--tp-size", "8", "--nnodes", "2", "--node-rank", "1", "--dist-init-addr", "nid007164:20000"),
        SGLANG,
    )
    assert topo.is_multinode and not topo.is_head
    assert topo.ranks_per_node == 4
    assert topo.local_rank_range == (4, 7)
    assert topo.dist_init_addr == "nid007164:20000"


def test_accepts_joined_flags_and_the_nccl_alias():
    topo = read_topology(
        sglang("--tp=8", "--nnodes=2", "--node-rank=1", "--nccl-init-addr=head:20000"),
        SGLANG,
    )
    assert topo.tp_size == 8 and topo.node_rank == 1
    assert topo.dist_init_addr == "head:20000"


def test_pipeline_parallel_counts_toward_the_world():
    topo = read_topology(
        sglang("--tp-size", "4", "--pp-size", "2", "--nnodes", "2", "--node-rank", "0", "--dist-init-addr", "h:1"),
        SGLANG,
    )
    assert topo.ranks_per_node == 4


def test_multinode_without_a_rendezvous_address_is_refused():
    """The engine would not fail here -- each node would quietly rendezvous with itself."""
    with pytest.raises(ValueError, match="dist-init-addr"):
        read_topology(sglang("--tp-size", "8", "--nnodes", "2", "--node-rank", "1"), SGLANG)


def test_a_rank_outside_the_world_is_refused():
    with pytest.raises(ValueError, match="node-rank"):
        read_topology(sglang("--tp-size", "8", "--nnodes", "2", "--node-rank", "2", "--dist-init-addr", "h:1"), SGLANG)


def test_a_world_that_does_not_split_evenly_is_refused():
    with pytest.raises(ValueError, match="divisible"):
        read_topology(sglang("--tp-size", "6", "--nnodes", "4", "--node-rank", "0", "--dist-init-addr", "h:1"), SGLANG)


def test_an_engine_with_no_dist_flags_is_always_one_node():
    """vLLM: servekit cannot read its topology, so it must not guess at one."""
    assert read_topology(["vllm", "serve", "/store/m"], VLLM) == Topology()


def test_shard_glob_covers_this_nodes_ranks_and_no_others():
    assert shard_glob(Topology(nnodes=2, node_rank=1, tp_size=8)) == "model-rank-[4-7]-part-*.safetensors"
    assert shard_glob(Topology(nnodes=4, node_rank=2, tp_size=8)) == "model-rank-[4-5]-part-*.safetensors"
    assert shard_glob(Topology(nnodes=8, node_rank=3, tp_size=8)) == "model-rank-3-part-*.safetensors"


def test_a_two_digit_rank_range_is_refused_rather_than_mismatched():
    """`find -name` is fnmatch: [8-15] is not a range, and would match nothing."""
    with pytest.raises(ValueError, match="more than one glob"):
        shard_glob(Topology(nnodes=2, node_rank=1, tp_size=16))


def test_sharded_state_is_what_makes_the_shards_per_rank():
    assert wants_sharded_state(sglang("--load-format", "sharded_state"))
    assert wants_sharded_state(sglang("--load-format=sharded_state"))
    assert not wants_sharded_state(sglang())
    assert not wants_sharded_state(sglang("--load-format", "auto"))
