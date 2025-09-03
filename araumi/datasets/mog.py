import math
import torch

LOG2PI = math.log(2.0 * math.pi)

def _to_device_dtype(x, device, dtype):
  return x.to(device=device, dtype=dtype, copy=False)

class IsoMoG:
  """
  等方的ガウス分布: Σ_k = σ_k^2 I.
  means:   [K, d]
  sigmas:  [K] or float  (standard deviation)
  weights: [K]  (nonnegative; will be normalized)
  """
  def __init__(self, means, sigmas, weights, eps=1e-6, device=None, dtype=None):
    if device is None: device = means.device
    if dtype is None: dtype = means.dtype
    self.device, self.dtype = device, dtype

    self.means = _to_device_dtype(means, device, dtype)  # [K,d]
    K, d = self.means.shape
    self.K, self.d = K, d

    sig = torch.as_tensor(sigmas, device=device, dtype=dtype)
    if sig.numel() == 1:
      sig = sig.expand(K)
    self.sigma2 = (sig * sig).clamp_min(1e-12)  # [K]

    # 重み正規化
    w = torch.clamp(weights.to(device=device, dtype=dtype), min=0)
    w = w / w.sum().clamp_min(1e-38)
    # 対数重み
    self.log_w = torch.log(w.clamp(min=1e-38))  # [K]

    # 事前計算: 各成分の -0.5 * (d * log(2πσ^2))
    # 正規分布の確率密度関数の対数計算の定数部分
    self.const = - 0.5 * (self.d * (LOG2PI + torch.log(self.sigma2)))  # [K]  

  @torch.no_grad()
  def sample(self, n, generator=None, return_comp_idx=False):
    idx = torch.multinomial(torch.exp(self.log_w), n, replacement=True, generator=generator)  # [n]
    mu  = self.means[idx]                  # [n,d]
    s2  = self.sigma2[idx].unsqueeze(-1)  # [n,1]
    z   = torch.randn(n, self.d, device=self.device, dtype=self.dtype, generator=generator)   # [n,d]
    x   = mu + torch.sqrt(s2) * z          # [n,d]
    return (x, idx) if return_comp_idx else x

  @torch.no_grad()
  def log_prob(self, x):
    # log N_k(x) = const_k - 0.5 * ||x-μ_k||^2 / σ_k^2
    x = x.to(device=self.device, dtype=self.dtype)                      # [N,d]
    delta2 = ((x[:, None, :] - self.means[None, :, :]) ** 2).sum(-1)    # [N,K], ブロードキャストで一括計算
    logN = self.const[None, :] - 0.5 * (delta2 / self.sigma2[None, :])  # [N,K]
    s = logN + self.log_w[None, :]                                      # [N,K]
    s_max = s.max(dim=1, keepdim=True).values  # for LogSumExp
    return (s_max.squeeze(1) + torch.log(torch.exp(s - s_max).sum(dim=1)))      # [N], LogSumExp

  @torch.no_grad()
  def responsibilities(self, x):
    x = x.to(device=self.device, dtype=self.dtype)
    delta2 = ((x[:, None, :] - self.means[None, :, :]) ** 2).sum(-1)    # [N,K]
    logN = self.const[None, :] - 0.5 * (delta2 / self.sigma2[None, :])  # [N,K]
    s = logN + self.log_w[None, :]
    s_max = s.max(dim=1, keepdim=True).values
    r = torch.exp(s - s_max)
    return r / r.sum(dim=1, keepdim=True).clamp_min(1e-38)  # [N,K]
  
  @torch.no_grad()
  def score(self, x):
    """
    ∇_x log p(x) = Σ_k r_k(x) * (μ_k - x) / σ_k^2
    """
    x = x.to(device=self.device, dtype=self.dtype)  # [N,d]
    r = self.responsibilities(x)                     # [N,K]
    num = (self.means[None, :, :] - x[:, None, :]) / self.sigma2[None, :].unsqueeze(-1)  # [N,K,d]
    return (r.unsqueeze(-1) * num).sum(dim=1)       # [N,d]
