from snapper import config, service
from snapper.rcc import RunResult


def test_parse_last_service_env():
    text = "ENDPOINT_0=nid001:8080\nENDPOINT_1=nid002:8080\nJOBID=555\n"
    env = service.parse_last_service_env(text)
    assert env["JOBID"] == "555"
    assert service.endpoints(env) == ["nid001:8080", "nid002:8080"]


def test_endpoints_accepts_router_endpoint():
    assert service.endpoints({"ENDPOINT": "nid001:8090"}) == ["nid001:8090"]


def test_endpoints_derives_head_node_port_for_legacy_vllm_scripts():
    assert service.endpoints({"HEAD_NODE": "nid001", "PORT": "8000"}) == ["nid001:8000"]


def test_read_service_env_uses_cat():
    seen = {}

    class FakeRcc:
        @staticmethod
        def run(profile, command, **kw):
            seen["command"] = command
            return RunResult(0, "ENDPOINT_0=nid001:8080\nJOBID=7\n", "")

    env = service.read_service_env("p", rcc_mod=FakeRcc)
    assert "cat last_service.env" in seen["command"]
    assert env["JOBID"] == "7"


def test_verify_curls_models_and_chat(deploy_tree):
    dep = config.load("glm-47-flash-bristen", root=deploy_tree)
    cmds = []

    class FakeRcc:
        @staticmethod
        def run(profile, command, **kw):
            cmds.append(command)
            return RunResult(0, '{"data":[]}', "")

    service.verify(dep, "nid001:8080", rcc_mod=FakeRcc)
    joined = "\n".join(cmds)
    assert "nid001:8080/v1/models" in joined
    assert "nid001:8080/v1/chat/completions" in joined
    assert "zai-org/GLM-4.7-Flash" in joined
