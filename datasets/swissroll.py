import math
import torch

@torch.no_grad()
def swiss_roll(n, noise=0.5, t_min=math.pi, t_max=4*math.pi, device="cpu", dtype=torch.float32, generator=None):
  t = torch.rand(n, device=device, dtype=dtype, generator=generator) * (t_max - t_min) + t_min
  x = t * torch.cos(t); y = t * torch.sin(t)
  z = torch.randn(n, 2, device=device, dtype=dtype, generator=generator) * noise
  return torch.stack([x, y], dim=1) + z
