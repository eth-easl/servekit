import pathlib
import vllm

root = pathlib.Path(vllm.__file__).resolve().parent
print(root)

patterns = [
    "headless",
    "node_rank",
    "nnodes",
    "external_launcher",
    "master_addr",
]

for pattern in patterns:
    print(f"--- {pattern}")
    hits = []
    for path in root.rglob("*.py"):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if pattern in text:
            hits.append(str(path.relative_to(root)))
            if len(hits) >= 30:
                break
    print("\n".join(hits))
