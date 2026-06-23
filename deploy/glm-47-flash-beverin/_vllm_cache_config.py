import os, dataclasses, inspect
print("=== env relevant to compile caches ===")
for k in ["TRITON_CACHE_DIR","VLLM_CACHE_ROOT","TORCHINDUCTOR_CACHE_DIR","TORCHINDUCTOR_DIR",
          "INDUCTOR_CACHE_DIR","XDG_CACHE_HOME","VLLM_DISABLE_COMPILE_CACHE","HF_HOME","TMPDIR"]:
    print(f"  {k} = {os.environ.get(k, '<unset>')}")

print("\n=== vLLM CompilationConfig fields (cache/compile/inductor/cudagraph) ===")
from vllm.config import CompilationConfig
cc = CompilationConfig()
for f in dataclasses.fields(cc):
    try: v = getattr(cc, f.name)
    except Exception as e: v = f"<err {e}>"
    if any(s in f.name.lower() for s in ["cache","compile","inductor","level","mode","cudagraph","splitting"]):
        print(f"  {f.name} = {v!r}")

print("\n=== is_compile_cache_enabled()? ===")
try:
    from vllm.compilation.backends import is_compile_cache_enabled
    print("  is_compile_cache_enabled() =", is_compile_cache_enabled())
except Exception as e:
    print("  (could not call:", repr(e), ")")

print("\n=== where vLLM derives compile/inductor cache_dir (source refs) ===")
import vllm.compilation.backends as B
src = inspect.getsource(B)
for i, line in enumerate(src.splitlines()):
    if any(s in line for s in ["cache_dir","TORCHINDUCTOR","VLLM_CACHE","local_cache","def _get_inductor","inductor_config"]):
        print(f"  {i}: {line.strip()}")
