import torch

REGION_BASE = 0x600000000000
REGION_END = REGION_BASE + (4 << 30)


def in_region(p):
    return REGION_BASE <= p < REGION_END


x = torch.arange(1000, device="cuda", dtype=torch.int32)
y = x * 2 + 1
torch.cuda.synchronize()
correct = bool((y == (x * 2 + 1)).all().item())
p = x.data_ptr()
print(f"data_ptr=0x{p:x} in_region={in_region(p)} correct={correct}")
