import numpy as np
import torch
from ..utils.torch_cuda import get_device, gaussian_blur

class DepthEnhancer:
    def __init__(self, config: dict):
        self.k = int(config.get("gauss_k", 5))
        self.sigma = float(config.get("gauss_sigma", 1.2))
        self.log_eps = float(config.get("log_eps", 0.000001))

    @torch.inference_mode()
    def __call__(self, depth_m: np.ndarray, rgb = None):        # Làm mượt depth tương đối thôi
        device = get_device()
        bien = torch.from_numpy(depth_m).to(device = device, dtype = torch.float32)
        bien = torch.clamp_min(bien, 0.000001)
        logBien = torch.log(bien)
        sm = gaussian_blur(logBien, k = self.k, sigma = self.sigma) 
        sm2 = gaussian_blur(logBien * logBien, k = self.k, sigma = self.sigma) 
        var = torch.clamp_min(sm2 - sm * sm, 0.0)
        sigma = torch.sqrt(var + 0.000000001)
        out = torch.exp(sm)
        return out.detach().cpu().numpy().astype(np.float32), sigma.detach().cpu().numpy().astype(np.float32)
