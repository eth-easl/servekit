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
from typing import Iterator, Optional

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


RELATIVE = Path("srt") / "models" / "deepseek_v2.py"


def roots() -> Iterator[Path]:
    """Every directory that could be the sglang package, best guess first.

    find_spec alone is not enough: the image's workdir is /opt and the sglang
    source tree sits at /opt/sglang, so with cwd on sys.path a namespace
    portion shadows the installed package and points a level above its own
    python/ directory. Whichever candidate actually holds the file wins.
    """
    seen = set()
    try:
        spec = importlib.util.find_spec("sglang")
    except Exception:
        spec = None
    locations = list(spec.submodule_search_locations or ()) if spec is not None else []
    for entry in sys.path:
        if entry:
            locations.append(str(Path(entry) / "sglang"))
    for location in locations:
        if location not in seen:
            seen.add(location)
            yield Path(location)


def _shadow_free_path() -> list:
    """sys.path without the entries that only offer a package missing the file.

    /opt/sglang is the source tree, not the package, so leaving it in place
    makes `import sglang` bind to an empty namespace and the submodule lookup
    below fail before it reaches the real install.
    """
    keep = []
    for entry in sys.path:
        directory = Path(entry or ".") / "sglang"
        if directory.is_dir() and not (directory / RELATIVE).is_file():
            continue
        keep.append(entry)
    return keep


def _module_origin() -> Optional[Path]:
    """Where the import system says the module lives.

    An editable install resolves through a meta-path finder rather than a
    directory on sys.path, so scanning cannot see it. This costs an import of
    sglang, which is why it is the fallback and not the first move.
    """
    saved = list(sys.path)
    sys.path[:] = _shadow_free_path()
    try:
        spec = importlib.util.find_spec("sglang.srt.models.deepseek_v2")
    except Exception:
        return None
    finally:
        sys.path[:] = saved
    return Path(spec.origin) if spec is not None and spec.origin else None


def target() -> Path:
    tried = []
    for root in roots():
        candidate = root / RELATIVE
        if candidate.is_file():
            return candidate
        tried.append(str(candidate))

    origin = _module_origin()
    if origin is not None and origin.is_file():
        return origin

    raise SystemExit(
        "error: no sglang holding {} was importable; tried:\n  {}".format(
            RELATIVE, "\n  ".join(tried) or "nothing"
        )
    )


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
