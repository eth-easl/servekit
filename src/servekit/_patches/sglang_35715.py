"""Backport of sgl-project/sglang#35715 onto the installed sglang.

MLA models cannot round-trip through ShardedStateLoader without it:
DeepseekV2AttentionMLA registers kv_b_proj as a submodule of attn_mha on first
forward, so the weight reaches model.state_dict() under two names sharing one
storage. save_model's _filter_subtensors dedups by storage and keeps the
lexicographically smaller key, evicting the canonical name from the checkpoint;
a fresh model has never forwarded, so loading KeyErrors on the attn_mha name.
"""
import importlib.util
import sys
from pathlib import Path

OLD_INIT = "        self.attn_mha.kv_b_proj = None\n"
NEW_INIT = (
    "        # Alias for backends that only get the RadixAttention layer. Not registered\n"
    "        # as a submodule: a second state_dict name breaks sharded_state round-trips.\n"
    '        object.__setattr__(self.attn_mha, "kv_b_proj", self.kv_b_proj)\n'
)
OLD_FORWARD = (
    "        if self.attn_mha.kv_b_proj is None:\n"
    "            self.attn_mha.kv_b_proj = self.kv_b_proj\n"
    "\n"
)


def target() -> Path:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("error: sglang is not importable, nothing to patch")
    root = Path(list(spec.submodule_search_locations)[0])
    return root / "srt" / "models" / "deepseek_v2.py"


def main() -> int:
    path = target()
    text = path.read_text()

    if NEW_INIT in text:
        print(f"[PATCH] sglang#35715 already applied to {path}")
        return 0
    for anchor, what in ((OLD_INIT, "the __init__ assignment"), (OLD_FORWARD, "the forward_prepare fill-in")):
        if text.count(anchor) != 1:
            print(
                f"error: {what} is not in {path} exactly once; sglang#35715 does not "
                f"fit this build and the checkpoint would silently lose kv_b_proj",
                file=sys.stderr,
            )
            return 1

    path.write_text(text.replace(OLD_INIT, NEW_INIT, 1).replace(OLD_FORWARD, "", 1))
    print(f"[PATCH] applied sglang#35715 to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
