# src/fusion/planar_patch.py
import numpy as np
import torch
import cupy as cp
from cupyx.scipy import ndimage as cpx_ndi
from torch.utils.dlpack import to_dlpack
from src.utils.torch_cuda import get_device, to_torch, gaussian_blur, sobel_grad

_ALLOWED = {1,2,3,4,7,8,9,11,12,13,14,15,16,17,18}  # bỏ road/sky/pole nhỏ

def _torch_to_cupy(x: torch.Tensor) -> cp.ndarray:
    return cp.from_dlpack(to_dlpack(x.contiguous()))

@torch.no_grad()
def detect_planar_obstacles(depth_m_enh: np.ndarray,
                            sigma_m: np.ndarray,
                            seg: np.ndarray,
                            height_m: np.ndarray,
                            cfg: dict,
                            fx: float, fy: float):
    dev = get_device()
    D  = to_torch(depth_m_enh, device=dev, dtype=torch.float32)
    Sm = to_torch(sigma_m,       device=dev, dtype=torch.float32)
    Sg = to_torch(seg,           device=dev, dtype=torch.int16)
    Hm = to_torch(height_m,      device=dev, dtype=torch.float32)

    z_min = float(cfg.get("depth_min_m", 0.35))
    z_max = float(cfg.get("depth_max_m", 15.0))
    sigma_thr = float(cfg.get("sigma_thr", 0.05))
    grad_thr_in = float(cfg.get("grad_thr_in", 0.02))

    Dl = torch.log(torch.clamp(D, min=1e-3))
    gxx, gyy = sobel_grad(Dl)
    gmag = torch.sqrt(gxx*gxx + gyy*gyy)

    # interior: khá phẳng & trong dải Z
    mask = (Sm <= sigma_thr) & (D >= z_min) & (D <= z_max)
    allow = torch.tensor(sorted(_ALLOWED), device=dev, dtype=Sg.dtype)
    mask &= torch.isin(Sg, allow)

    # loại nội vùng nếu gradient quá lớn (không phải mảng phẳng)
    mask &= (gmag <= grad_thr_in)

    # morphology nhẹ
    m_u8 = mask.to(torch.uint8)
    from src.utils.torch_cuda import binary_morphology
    m_u8 = binary_morphology(m_u8, op="open",  k=3, it=1)
    m_u8 = binary_morphology(m_u8, op="close", k=5, it=1)

    labels_cp, _ = cpx_ndi.label(_torch_to_cupy(m_u8), structure=cp.ones((3,3), dtype=cp.uint8))
    H,W = D.shape
    ys, xs = cp.where(labels_cp>0)
    if not bool((labels_cp>0).any()):
        return (np.zeros((0,4),np.float32), np.zeros((0,),np.float32), np.zeros((0,),np.float32))

    labs_full = labels_cp[ys, xs].astype(cp.int32)
    nlab = int(labs_full.max().item())
    binc = cp.bincount(labs_full, minlength=nlab+1)
    counts = binc[1:]

    min_y = cp.full((nlab+1,), H, dtype=cp.int32)
    min_x = cp.full((nlab+1,), W, dtype=cp.int32)
    max_y = cp.zeros((nlab+1,), dtype=cp.int32)
    max_x = cp.zeros((nlab+1,), dtype=cp.int32)

    cp.minimum.at(min_y, labs_full, ys)
    cp.minimum.at(min_x, labs_full, xs)
    cp.maximum.at(max_y, labs_full, ys+1)
    cp.maximum.at(max_x, labs_full, xs+1)

    y1 = min_y[1:]; x1 = min_x[1:]; y2 = max_y[1:]; x2 = max_x[1:]
    counts_h = counts.get()
    y1_h, x1_h, y2_h, x2_h = y1.get(), x1.get(), y2.get(), x2.get()

    boxes=[]; dists=[]; scores=[]
    area_k = float(cfg.get("area_min_m2_k", 4.0e-4))
    for i in range(len(y1_h)):
        xs_i,ys_i,xe_i,ye_i = int(x1_h[i]), int(y1_h[i]), int(x2_h[i]), int(y2_h[i])
        w = xe_i - xs_i; h = ye_i - ys_i
        if w < int(cfg.get("min_w_px", 12)) or h < int(cfg.get("min_h_px", 12)):
            continue
        roi = D[ys_i:ye_i, xs_i:xe_i]
        z = torch.quantile(roi[torch.isfinite(roi)], 0.35).item()
        px = int(counts_h[i])
        if (float(px) / max(1, w*h)) < 0.65:
            continue
        if (w/h) < 0.2 or (h/w) < 0.2:
            continue
        if (float(px) * (z/fx)*(z/fy)) < area_k * (z**2):
            continue

        # height gating: median thay vì nanmedian để tương thích
        hroi = Hm[ys_i:ye_i, xs_i:xe_i]
        if float(torch.median(hroi).item()) < float(cfg.get("h_min_m", 0.05)):
            continue

        # score: mean thay vì nanmean
        sc_sigma = float(torch.mean(Sm[ys_i:ye_i, xs_i:xe_i]).item())
        boxes.append([xs_i,ys_i,xe_i,ye_i])
        dists.append(z)
        scores.append(0.6*(1.0 - z/z_max) + 0.4*(1.0 - sc_sigma))

    if len(boxes)==0:
        return (np.zeros((0,4),np.float32), np.zeros((0,),np.float32), np.zeros((0,),np.float32))

    return (np.asarray(boxes, dtype=np.float32),
            np.asarray(dists, dtype=np.float32),
            np.asarray(scores, dtype=np.float32))
