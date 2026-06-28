from snapper import rcc


def test_push_builds_expected_argv(monkeypatch):
    seen = {}

    def fake(argv, *, stream=False):
        seen["argv"] = argv
        return rcc.RunResult(0, "", "")

    monkeypatch.setattr(rcc, "_run_argv", fake)
    rcc.push("glm-47-flash-bristen")
    assert seen["argv"] == ["rcc", "--profile", "glm-47-flash-bristen", "push"]


def test_run_wraps_command_in_bash_lc(monkeypatch):
    seen = {}

    def fake(argv, *, stream=False):
        seen["argv"] = argv
        return rcc.RunResult(0, "ok", "")

    monkeypatch.setattr(rcc, "_run_argv", fake)
    res = rcc.run("p", "squeue -j 5 -h -o %T")
    assert seen["argv"] == [
        "rcc",
        "--profile",
        "p",
        "run",
        "bash",
        "-lc",
        "squeue -j 5 -h -o %T",
    ]
    assert res.stdout == "ok"
