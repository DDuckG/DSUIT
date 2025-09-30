# src/fusion/center_depth.py
import numpy as np
import torch
from src.utils.torch_cuda import to_torch, get_device

@torch.inference_mode()
def depth_at_center_batch(depth_m, boxes_xyxy, q=0.65, center_frac=0.40,
                          clip_min=0.2, clip_max=80.0):
    dev = get_device()
    D = to_torch(depth_m, device=dev, dtype=torch.float32)
    B = torch.as_tensor(boxes_xyxy, device=dev, dtype=torch.float32).view(-1, 4)
    H, W = D.shape
    cx = (B[:, 0] + B[:, 2]) * 0.5
    cy = (B[:, 1] + B[:, 3]) * 0.5
    ww = (B[:, 2] - B[:, 0]) * center_frac
    hh = (B[:, 3] - B[:, 1]) * center_frac
    x1 = torch.clamp((cx - 0.5 * ww).long(), 0, W - 1)
    x2 = torch.clamp((cx + 0.5 * ww).long(), 0, W - 1)
    y1 = torch.clamp((cy - 0.5 * hh).long(), 0, H - 1)
    y2 = torch.clamp((cy + 0.5 * hh).long(), 0, H - 1)

    out = torch.empty((B.shape[0],), device=dev, dtype=torch.float32)
    for i in range(B.shape[0]):
        xs1, ys1, xs2, ys2 = int(x1[i]), int(y1[i]), int(x2[i]) + 1, int(y2[i]) + 1
        patch = D[ys1:ys2, xs1:xs2].reshape(-1)
        patch = patch[torch.isfinite(patch) & (patch > 0)]
        z = torch.quantile(patch, float(q)) if int(patch.numel()) > 0 else torch.tensor(float('nan'), device=dev)
        out[i] = z

    out = torch.clamp(out, min=float(clip_min), max=float(clip_max))
    return out.detach().cpu().numpy().astype(np.float32)
