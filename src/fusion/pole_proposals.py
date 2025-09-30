import numpy as np
import torch
import cupy as cp
from cupyx.scipy import ndimage as cpx_ndi
from torch.utils.dlpack import to_dlpack
from src.utils.torch_cuda import sobel_grad, gaussian_blur

def _torch_to_cupy(x: torch.Tensor) -> cp.ndarray:
    return cp.from_dlpack(to_dlpack(x.contiguous()))

@torch.no_grad()
def detect_poles_from_depth(depth_m: np.ndarray, cfg: dict):
    d = torch.from_numpy(depth_m).to(dtype = torch.float32, device = ("cuda" if torch.cuda.is_available() else "cpu"))
    d = torch.clamp_min(d, 0.000001)
    Dl = torch.log(d)
    gx, gy = sobel_grad(Dl)
    gy = torch.abs(gy)
    col_energy = gy.sum(dim = 0)
    thr = torch.quantile(col_energy, 0.90)
    mask = (gy >= (thr / (depth_m.shape[0]))).to(torch.uint8)

    mask = gaussian_blur(mask.float(), k = 3, sigma = 0.8) > 0.1
    labels_cp, _ = cpx_ndi.label(_torch_to_cupy(mask.to(torch.uint8)), structure = cp.ones((3,3), dtype = cp.uint8))

    height, width = depth_m.shape
    ys, xs = cp.where(labels_cp>0)
    if not bool((labels_cp>0).any()):
        return (np.zeros((0,4),np.float32), np.zeros((0,),np.float32), np.zeros((0,),np.float32))

    labs_full = labels_cp[ys, xs].astype(cp.int32)
    nlab = int(labs_full.max().item())
    binc = cp.bincount(labs_full, minlength = nlab + 1)
    counts = binc[1 : ]

    min_y = cp.full((nlab +1, ), height, dtype = cp.int32)
    min_x = cp.full((nlab +1, ), width, dtype = cp.int32)
    max_y = cp.zeros((nlab +1, ), dtype = cp.int32)
    max_x = cp.zeros((nlab +1, ), dtype = cp.int32)

    cp.minimum.at(min_y, labs_full, ys)
    cp.minimum.at(min_x, labs_full, xs)
    cp.maximum.at(max_y, labs_full, ys+1)
    cp.maximum.at(max_x, labs_full, xs+1)

    y1 = min_y[1:]
    x1 = min_x[1:]
    y2 = max_y[1:]
    x2 = max_x[1:]
    y1_h, x1_h, y2_h, x2_h = y1.get(), x1.get(), y2.get(), x2.get()

    boxes=[]
    dists=[]
    scores=[]
    min_len = int(cfg.get("min_len_px", 28))
    max_w = int(cfg.get("max_w_px", 16))

    D = d
    for i in range(len(y1_h)):
        xs_i, ys_i, xe_i, ye_i = int(x1_h[i]), int(y1_h[i]), int(x2_h[i]), int(y2_h[i])
        w = xe_i - xs_i
        h = ye_i - ys_i
        if h < min_len or w > max_w or w < 2:
            continue
        roi = D[ys_i : ye_i, xs_i : xe_i]
        z = torch.quantile(roi[torch.isfinite(roi)], 0.35).item()
        boxes.append([xs_i, ys_i, xe_i, ye_i])
        scores.append(min(1.0, h / float(height)))
        dists.append(z)

    if len(boxes) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros((0, ), np.float32), np.zeros((0, ), np.float32))
    return (np.asarray(boxes, dtype = np.float32), np.asarray(dists, dtype = np.float32), np.asarray(scores, dtype = np.float32))
