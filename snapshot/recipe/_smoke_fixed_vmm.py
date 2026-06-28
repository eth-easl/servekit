import os, sys, torch
print(f"[smoke] FIXED_BASE={os.environ.get('SNAPSHOT_REDIRECT_FIXED_BASE')} "
      f"REGION_GIB={os.environ.get('SNAPSHOT_REDIRECT_REGION_GIB')}", flush=True)
torch.cuda.init()
# init_device crashes here under the legacy per-block set_access (M2.3)
a = torch.zeros(1024, 1024, device='cuda')
b = torch.full((1024, 1024), 3.0, device='cuda')
c = (a + b).sum().item()
print(f"[smoke] a.ptr=0x{a.data_ptr():x} sum={c} (expected {3.0*1024*1024})", flush=True)
# second alloc exercises another sub-block (the intermittent M2.3 case)
d = torch.zeros(8 << 20, device='cuda')  # 8 MiB
e = torch.zeros(16 << 20, device='cuda') # 16 MiB
print(f"[smoke] d.ptr=0x{d.data_ptr():x} e.ptr=0x{e.data_ptr():x}", flush=True)
torch.cuda.synchronize()
print("[smoke] OK — fixed-base VMM survives multi-alloc torch init", flush=True)
