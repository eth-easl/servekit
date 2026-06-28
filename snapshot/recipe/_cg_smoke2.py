import torch

g = torch.cuda.CUDAGraph()
side = torch.cuda.Stream()
with torch.cuda.stream(side):
    g.capture_begin()
    x = torch.ones(4, device="cuda")  # actual work so graph isn't empty
    x.add_(1.0)
    g.capture_end()
g.replay()
g.replay()
print("SMOKE_OK")
