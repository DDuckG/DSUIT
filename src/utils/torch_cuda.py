import numpy as np
import torch
import torch.nn.functional as F

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_torch(x, device=None, dtype=None, non_blocking=True):
    if isinstance(x, torch.Tensor):
        t = x
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device, non_blocking=non_blocking)
        return t.contiguous()
    t = torch.from_numpy(np.ascontiguousarray(x))
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device, non_blocking=non_blocking)
    return t.contiguous()

def to_numpy(t: torch.Tensor):
    if isinstance(t, np.ndarray):
        return t
    return t.detach().contiguous().cpu().numpy()

def _gauss_1d(k: int, sigma: float, device, dtype):
    if k <= 1:
        g = torch.tensor([1.0], device=device, dtype=dtype)
        return g, 1
    r = (k - 1) // 2
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / max(1e-6, sigma)) ** 2)
    g = g / torch.clamp(g.sum(), min=1e-12)
    return g, r

@torch.no_grad()
def gaussian_blur(x: torch.Tensor, k: int = 5, sigma: float = 1.0):
    """
    x: [H,W] float
    separable Gaussian, chạy trên GPU (cudnn conv)
    """
    if k <= 1 or sigma <= 0:
        return x
    device, dtype = x.device, x.dtype
    g, r = _gauss_1d(k, sigma, device, dtype)

    # H direction
    x4 = x.unsqueeze(0).unsqueeze(0)
    g_row = g.view(1, 1, 1, k)
    y = F.pad(x4, (r, r, 0, 0), mode='reflect')
    y = F.conv2d(y, g_row)

    # V direction
    g_col = g.view(1, 1, k, 1)
    y = F.pad(y, (0, 0, r, r), mode='reflect')
    y = F.conv2d(y, g_col)
    return y[0, 0]

@torch.no_grad()
def sobel_grad(x: torch.Tensor):
    device, dtype = x.device, x.dtype
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    x4 = x.unsqueeze(0).unsqueeze(0)
    x4 = F.pad(x4, (1, 1, 1, 1), mode='reflect')
    gx = F.conv2d(x4, kx)[0, 0]
    gy = F.conv2d(x4, ky)[0, 0]
    return gx, gy

@torch.no_grad()
def _morph_core(bin_u8: torch.Tensor, k: int, hit_min: int):
    """
    bin_u8: [H,W] uint8 (0/1)
    k: kernel size (odd)
    hit_min: threshold for dilation/erosion/open/close
    return: [H,W] uint8
    """
    device = bin_u8.device
    x = bin_u8.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    ker = torch.ones((1,1,k,k), device=device, dtype=x.dtype)
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    y = F.conv2d(x, ker)  # sum in window
    y = (y >= hit_min).to(torch.uint8)
    return y[0,0]

@torch.no_grad()
def binary_morphology(bin_u8: torch.Tensor, op: str = "dilate", k: int = 3, it: int = 1):
    """
    bin_u8: [H,W] uint8 {0,1}
    """
    op = str(op).lower()
    out = bin_u8
    if op == "dilate":
        for _ in range(max(1, it)):
            out = _morph_core(out, k, 1)
        return out
    elif op == "erode":
        for _ in range(max(1, it)):
            out = _morph_core(out, k, k*k)
        return out
    elif op == "open":
        out = binary_morphology(out, "erode", k, it)
        out = binary_morphology(out, "dilate", k, it)
        return out
    elif op == "close":
        out = binary_morphology(out, "dilate", k, it)
        out = binary_morphology(out, "erode", k, it)
        return out
    else:
        return out