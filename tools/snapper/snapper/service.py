from __future__ import annotations

import json
import shlex

from . import rcc


def parse_last_service_env(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def endpoints(env: dict[str, str]) -> list[str]:
    items = [(key, value) for key, value in env.items() if key.startswith("ENDPOINT_")]
    items.sort(key=lambda item: int(item[0].split("_", 1)[1]))
    out = [value for _, value in items]
    if env.get("ENDPOINT"):
        out.insert(0, env["ENDPOINT"])
    if not out and env.get("HEAD_NODE") and env.get("PORT"):
        out.append(f"{env['HEAD_NODE']}:{env['PORT']}")
    return out


def read_service_env(profile: str, *, rcc_mod=rcc) -> dict[str, str]:
    res = rcc_mod.run(profile, "cat last_service.env 2>/dev/null || true")
    return parse_last_service_env(res.stdout)


def verify(dep, endpoint: str, *, rcc_mod=rcc):
    base = f"http://{endpoint}"
    body = json.dumps(
        {
            "model": dep.served_model_name,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
        }
    )
    cmd = (
        "set -e; "
        f"curl -sS {shlex.quote(base + dep.verify_models_path)}; echo; "
        f"curl -sS {shlex.quote(base + dep.verify_chat_path)} "
        f"-H 'Content-Type: application/json' -d {shlex.quote(body)}"
    )
    return rcc_mod.run(dep.profile, cmd)
