from snapper import config, slurm


def _dep(deploy_tree):
    return config.load("glm-47-flash-bristen", root=deploy_tree)


def test_parse_jobid():
    assert slurm.parse_jobid("Submitted batch job 123456\n") == 123456


def test_parse_squeue_state_running_then_absent():
    assert slurm.parse_squeue_state("RUNNING\n") == "RUNNING"
    assert slurm.parse_squeue_state("\n") is None


def test_parse_rank_dump_and_count():
    dump = (
        "===RANK 0===\nstuff fired up and ready to roll\nvmagent_started\n"
        "===RANK 1===\nstill loading\n"
    )
    logs = slurm.parse_rank_dump(dump)
    assert set(logs) == {0, 1}
    markers = ["fired up and ready to roll", "vmagent_started"]
    assert slurm.count_ready_ranks(logs, markers) == 1


def test_submit_builds_command_with_env_and_nodes(deploy_tree):
    dep = _dep(deploy_tree)
    seen = {}

    class FakeRcc:
        @staticmethod
        def run(profile, command, **kw):
            seen["profile"] = profile
            seen["command"] = command
            from snapper.rcc import RunResult

            return RunResult(0, "Submitted batch job 999\n", "")

    jobid = slurm.submit(
        dep,
        nodes=5,
        serve_script=dep.serve_script,
        env={"TP_SIZE": "2"},
        rcc_mod=FakeRcc,
    )
    assert jobid == 999
    assert seen["profile"] == "glm-47-flash-bristen"
    assert (
        "TP_SIZE=2 sbatch --nodes=5 "
        "deploy/glm-47-flash-bristen/serve_glm_47_flash_sglang.sbatch"
    ) in seen["command"]


def test_wait_for_ready_returns_true_when_all_ranks_ready(deploy_tree):
    dep = _dep(deploy_tree)

    class FakeRcc:
        @staticmethod
        def run(profile, command, **kw):
            from snapper.rcc import RunResult

            return RunResult(
                0,
                "===RANK 0===\nfired up and ready to roll\nvmagent_started\n",
                "",
            )

    ok = slurm.wait_for_ready(
        dep,
        999,
        1,
        poll=0,
        timeout=1,
        rcc_mod=FakeRcc,
        sleep=lambda s: None,
    )
    assert ok is True


def test_wait_in_queue_returns_empty_when_job_leaves_queue():
    outputs = ["RUNNING\n", "\n"]

    class FakeRcc:
        @staticmethod
        def run(profile, command, **kw):
            from snapper.rcc import RunResult

            return RunResult(0, outputs.pop(0), "")

    state = slurm.wait_in_queue(
        "p",
        123,
        poll=0,
        timeout=1,
        rcc_mod=FakeRcc,
        sleep=lambda s: None,
    )
    assert state == ""
