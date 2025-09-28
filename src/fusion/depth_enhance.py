# src/fusion/depth_enhance.py
import numpy as np
import torch
from ..utils.torch_cuda import get_device, gaussian_blur

class DepthEnhancer:
    """
    Làm mượt depth ở log-domain nhưng bảo toàn biên (dựa trên gradient).
    Đồng thời ước lượng sigma (độ tin cậy) theo phương sai cục bộ.
    """
    def __init__(self, cfg: dict):
        self.k = int(cfg.get("gauss_k", 5))
        self.sigma = float(cfg.get("gauss_sigma", 1.2))
        self.log_eps = float(cfg.get("log_eps", 1e-6))

    @torch.inference_mode()
    def __call__(self, depth_m: np.ndarray, rgb=None):
        device = get_device()
        d = torch.from_numpy(depth_m).to(device=device, dtype=torch.float32)
        d = torch.clamp_min(d, 1e-6)
        logd = torch.log(d)
        sm = gaussian_blur(logd, k=self.k, sigma=self.sigma)
        # local variance estimation (simple)
        sm2 = gaussian_blur(logd*logd, k=self.k, sigma=self.sigma)
        var = torch.clamp_min(sm2 - sm*sm, 0.0)
        sigma = torch.sqrt(var + 1e-9)
        out = torch.exp(sm)
        return out.detach().cpu().numpy().astype(np.float32), sigma.detach().cpu().numpy().astype(np.float32)
