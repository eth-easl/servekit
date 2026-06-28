import torch
import torch.cuda

CG = torch.cuda.CUDAGraph
print("type:", type(CG))
print("module:", getattr(CG, "__module__", "?"))
print("capture_end:", CG.capture_end, "|", type(CG.capture_end))

# Can we monkeypatch it?
patched_marker = [False]
def fake(self, *a, **k):
    patched_marker[0] = True
    return "FAKE"
try:
    CG.capture_end = fake
    g = torch.cuda.CUDAGraph()
    # call capture_end via instance
    res = g.capture_end()
    print("assign OK; marker:", patched_marker[0], "res:", res)
except Exception as e:
    print("ASSIGN FAILED:", repr(e))
