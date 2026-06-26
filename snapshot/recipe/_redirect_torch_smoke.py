import os, torch

print(f"[torch-smoke] ARENA={os.environ.get('SNAPSHOT_REDIRECT_ARENA')} "
      f"REGION_GIB={os.environ.get('SNAPSHOT_REDIRECT_REGION_GIB')} "
      f"torch={torch.__version__}", flush=True)

torch.cuda.init()
a = torch.zeros(1024, 1024, device='cuda')
b = torch.full((1024, 1024), 3.0, device='cuda')
s = (a + b).sum().item()
d = torch.zeros(8 << 20, device='cuda')   # 8 MiB
e = torch.zeros(16 << 20, device='cuda')  # 16 MiB
torch.cuda.synchronize()

print(f"[torch-smoke] PTRS a=0x{a.data_ptr():x} b=0x{b.data_ptr():x} "
      f"d=0x{d.data_ptr():x} e=0x{e.data_ptr():x}", flush=True)
expect = 3.0 * 1024 * 1024
ok = abs(s - expect) < 1.0
print(f"[torch-smoke] SUM={s} EXPECT={expect} COMPUTE={'OK' if ok else 'FAIL'}",
      flush=True)
raise SystemExit(0 if ok else 1)
