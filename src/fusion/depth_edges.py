import numpy as np
import torch
import torch.nn.functional as F
import cupy as cp
from cupyx.scipy import ndimage as cpx_ndi
from torch.utils.dlpack import to_dlpack

def _nanmean1d(x: torch.Tensor) -> torch.Tensor:
    m = torch.isfinite(x)
    if not bool(m.any()):
        return torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
    return x[m].mean()

def _nanmax1d(x: torch.Tensor) -> torch.Tensor:
    m = torch.isfinite(x)
    if not bool(m.any()):
        return torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
    return x[m].max()

from src.utils.torch_cuda import (
    get_device, to_torch,
    gaussian_blur, sobel_grad, binary_morphology
)

# Cityscapes: bỏ road/sky
_ALLOWED = {1,2,3,4,5,7,8,9,11,12,13,14,15,16,17,18}
ROAD_ID, SKY_ID = 0, 10

def _torch_to_cupy(x: torch.Tensor) -> cp.ndarray:
    return cp.from_dlpack(to_dlpack(x.contiguous()))

@torch.no_grad()
def _edge_maps_from_depth(depth_m: torch.Tensor, cfg: dict):
    # == checkpoint logic ==
    Dl = torch.log(torch.clamp(depth_m, min=1e-3))
    Dl = gaussian_blur(Dl, k=int(cfg.get("gauss_k", 5)), sigma=float(cfg.get("gauss_sigma", 1.15)))
    gx, gy = sobel_grad(Dl)
    mag = torch.sqrt(gx * gx + gy * gy)

    p_str  = float(cfg.get("edge_percentile_str", 92.0)) / 100.0
    p_weak = float(cfg.get("edge_percentile_weak", 85.0)) / 100.0
    flat = mag.flatten()
    t_str  = torch.quantile(flat, p_str)  * float(cfg.get("mag_thr_rel", 1.00))
    t_weak = torch.quantile(flat, p_weak) * float(cfg.get("mag_thr_rel_weak", 0.85))

    strong = (mag >= t_str).to(torch.uint8)
    weak   = (mag >= t_weak).to(torch.uint8)
    strong = binary_morphology(strong, op="dilate", k=3, it=1)
    weak   = binary_morphology(weak,   op="dilate", k=3, it=1)
    return mag, strong, weak, gx, gy, float(t_str.item()), float(t_weak.item())

def _area_px_to_m2(px: int, z_m: float, fx: float, fy: float) -> float:
    if not np.isfinite(z_m) or z_m <= 0.0:
        return 0.0
    return float(px) * (z_m / fx) * (z_m / fy)

def _compute_bboxes_from_labels(labels_cp: cp.ndarray):
    H, W = labels_cp.shape
    mask = labels_cp > 0
    if not bool(mask.any()):
        return (cp.empty((0,), dtype=cp.int32),)*6

    ys, xs = cp.where(mask)
    labs_full = labels_cp[ys, xs].astype(cp.int32)
    nlab = int(labs_full.max().item())

    binc = cp.bincount(labs_full, minlength=nlab+1)
    counts = binc[1:]  # bỏ nhãn 0

    min_y = cp.full((nlab+1,), H, dtype=cp.int32)
    min_x = cp.full((nlab+1,), W, dtype=cp.int32)
    max_y = cp.zeros((nlab+1,), dtype=cp.int32)
    max_x = cp.zeros((nlab+1,), dtype=cp.int32)

    cp.minimum.at(min_y, labs_full, ys)
    cp.minimum.at(min_x, labs_full, xs)
    cp.maximum.at(max_y, labs_full, ys+1)  # +1 để inclusive-exclusive
    cp.maximum.at(max_x, labs_full, xs+1)

    labs = cp.arange(1, nlab+1, dtype=cp.int32)
    y1 = min_y[1:]; x1 = min_x[1:]; y2 = max_y[1:]; x2 = max_x[1:]
    valid = counts > 0
    return labs[valid], y1[valid], x1[valid], y2[valid], x2[valid], counts[valid]

def _snap_edges_v2(boxes_np: np.ndarray, gx: torch.Tensor, gy: torch.Tensor,
                   max_dx: int = 24, max_dy: int = 24, min_w: int = 6, min_h: int = 6,
                   ignore_bottom_frac: float = 0.10):
    """
    Snap cạnh đứng theo |gy|, cạnh trên theo |gx| nhưng bỏ qua một phần đáy → tránh bị sàn kéo lệch.
    """
    if boxes_np is None or len(boxes_np) == 0:
        return boxes_np
    dev = gx.device
    H, W = gx.shape
    grad_v = torch.abs(gy)
    grad_h = torch.abs(gx)

    out = []
    boxes_t = torch.as_tensor(boxes_np, device=dev, dtype=torch.float32)
    for b in boxes_t:
        x1, y1, x2, y2 = [int(v.item()) for v in torch.round(b)]
        x1 = max(0, min(W-2, x1)); x2 = max(x1+1, min(W-1, x2))
        y1 = max(0, min(H-2, y1)); y2 = max(y1+1, min(H-1, y2))

        # bỏ 10% dải dưới để snap cạnh đứng
        cut = int(max(1, (y2-y1) * float(ignore_bottom_frac)))
        yr1, yr2 = y1, max(y1+1, y2 - cut)

        # RIGHT
        xs = torch.arange(max(0, x2-max_dx), min(W-1, x2+max_dx)+1, device=dev)
        x2n = int(xs[torch.argmax(grad_v[yr1:yr2, xs].sum(dim=0))].item()) if xs.numel() else x2
        # LEFT
        xs = torch.arange(max(0, x1-max_dx), min(W-1, x1+max_dx)+1, device=dev)
        x1n = int(xs[torch.argmax(grad_v[yr1:yr2, xs].sum(dim=0))].item()) if xs.numel() else x1

        if x2n - x1n < min_w: x1n, x2n = x1, x2

        # TOP
        ys = torch.arange(max(0, y1-max_dy), min(H-1, y1+max_dy)+1, device=dev)
        if x2n > x1n + 1 and ys.numel() > 0:
            y1n = int(ys[torch.argmax(grad_h[ys, x1n:x2n].sum(dim=1))].item())
        else:
            y1n = y1
        if (y2 - y1n) < min_h: y1n = max(0, y2 - min_h)
        out.append([x1n, y1n, x2n, y2])

    return np.asarray(out, dtype=float)

def _cut_bottom_by_height(boxes_np: np.ndarray, height_m_t: torch.Tensor,
                          h_free=0.07, band_px=12, min_drop=0.12):
    """
    Nếu dải đáy có median-height < h_free và phía trên khác biệt rõ rệt → cắt đáy lên.
    Bền với NaN (không cắt khi dữ liệu không chắc chắn).
    """
    if boxes_np is None or len(boxes_np) == 0:
        return boxes_np

    out=[]
    H, W = height_m_t.shape
    band_px = int(max(3, band_px))

    for x1,y1,x2,y2 in boxes_np.astype(int):
        x1=max(0,min(W-1,x1)); x2=max(x1+1,min(W,x2))
        y1=max(0,min(H-1,y1)); y2=max(y1+1,min(H,y2))
        roi = height_m_t[y1:y2, x1:x2]
        if roi.numel()==0:
            out.append([x1,y1,x2,y2]); continue

        # Profile theo trục dọc (median theo chiều ngang), dạng 1D
        prof = torch.nanmedian(roi, dim=1).values  # [H_roi]
        if prof.numel() <= band_px:
            out.append([x1,y1,x2,y2]); continue

        # Trung bình dải đáy và max toàn vùng (nan-safe)
        tail = _nanmean1d(prof[-band_px:])
        pmax = _nanmax1d(prof)
        if (not torch.isfinite(tail)) or (not torch.isfinite(pmax)):
            out.append([x1,y1,x2,y2]); continue

        # Tìm vị trí gần đáy nhất có trung bình cửa sổ < h_free
        prof_safe = prof.clone()
        prof_safe[~torch.isfinite(prof_safe)] = float('inf')   # NaN coi như rất cao → không cắt nhầm
        pooled = F.avg_pool1d(prof_safe.view(1,1,-1), kernel_size=band_px, stride=1)[0,0]  # [H_roi - band_px + 1]
        idx = torch.where(pooled < float(h_free))[0]

        if (float(tail.item()) < float(h_free)) and ((float(pmax.item()) - float(tail.item())) > float(min_drop)) and idx.numel() > 0:
            # cắt tại vị trí gần đáy nhất thỏa điều kiện
            cut_at = int(idx[-1].item())
            y2 = max(y1+1, y1 + cut_at)

        out.append([x1,y1,x2,y2])

    return np.asarray(out, dtype=float)

@torch.no_grad()
def detect_obstacles_by_depth_edges(depth_m_enh: np.ndarray,
                                    sigma_m: np.ndarray,
                                    seg: np.ndarray,
                                    cfg: dict,
                                    fx: float = 1000.0, fy: float = 1000.0,
                                    height_m: np.ndarray | None = None):
    """
    Trả về: ((boxes_xyxy, dists_m, scores), dbg)
    Gần như giữ nguyên checkpoint; chỉ thêm snap_v2 + cắt đáy (nếu có height_m).
    """
    dev = get_device()
    D = to_torch(depth_m_enh, device=dev, dtype=torch.float32)  # [H,W]
    S = to_torch(seg, device=dev, dtype=torch.int16)            # [H,W]

    mag, strong, weak, gx, gy, thr_s, thr_w = _edge_maps_from_depth(D, cfg)

    z_min = float(cfg.get("depth_min_m", 0.35))
    z_max = float(cfg.get("depth_max_m", 12.0))
    depth_ok = (D >= z_min) & (D <= z_max)

    # giữ nguyên nhưng chắc kiểu dtype cho is-in
    allow = torch.tensor(list(_ALLOWED), device=dev, dtype=S.dtype)
    seg_ok   = torch.isin(S, allow)
    edge_ok  = weak > 0

    cand = (depth_ok & seg_ok & edge_ok).to(torch.uint8)
    cand = binary_morphology(cand, op="open",  k=3, it=1)
    cand = binary_morphology(cand, op="close", k=5, it=1)

    labels_cp, _ = cpx_ndi.label(_torch_to_cupy(cand), structure=cp.ones((3,3), dtype=cp.uint8))
    labs, y1, x1, y2, x2, counts = _compute_bboxes_from_labels(labels_cp)
    if labs.size == 0:
        z = np.zeros((0,), dtype=np.float32)
        dbg = dict(num_edges=int((strong>0).sum().item()), thr_s=thr_s, thr_w=thr_w, pass_relax=0)
        return ((np.zeros((0,4),np.float32), z, z), dbg)

    y1_h, x1_h, y2_h, x2_h = y1.get(), x1.get(), y2.get(), x2.get()
    counts_h = counts.get()

    boxes = []; dists = []; scores=[]
    H, W = D.shape
    area_k = float(cfg.get("area_min_m2_k", 2.0e-4))
    for i in range(len(y1_h)):
        xs,ys,xe,ye = int(x1_h[i]), int(y1_h[i]), int(x2_h[i]), int(y2_h[i])
        if (xe-xs) < 6 or (ye-ys) < 6: continue
        roi = D[ys:ye, xs:xe]
        if not torch.isfinite(roi).any(): continue
        z = torch.quantile(roi[torch.isfinite(roi)], 0.25).item()
        if not np.isfinite(z) or z<z_min or z>z_max: continue
        px = int(counts_h[i])
        if _area_px_to_m2(px, z, fx, fy) < area_k * (z**2): continue
        near = max(0.0, min(1.0, (z_max - z) / max(1e-6, (z_max - z_min))))
        e_en = float(mag[ys:ye, xs:xe].mean().item())
        sc = 0.7*near + 0.3*(e_en/(thr_s+1e-6))

        boxes.append([xs,ys,xe,ye]); dists.append(z); scores.append(sc)

    if len(boxes)==0:
        z = np.zeros((0,), dtype=np.float32)
        dbg = dict(num_edges=int((strong>0).sum().item()), thr_s=thr_s, thr_w=thr_w, pass_relax=0)
        return ((np.zeros((0,4),np.float32), z, z), dbg)

    boxes = np.asarray(boxes, dtype=float)
    dists = np.asarray(dists, dtype=float)
    scores= np.asarray(scores, dtype=float)

    # (1) Snap cạnh: bỏ qua phần đáy nhỏ để không bị sàn kéo lệch
    boxes = _snap_edges_v2(
        boxes, gx, gy,
        max_dx=int(cfg.get("snap_dx_px", 24)),
        max_dy=int(cfg.get("snap_dy_px", 24)),
        min_w=int(cfg.get("min_w_px", 6)),
        min_h=int(cfg.get("min_h_px", 6)),
        ignore_bottom_frac=float(cfg.get("snap_ignore_bottom_frac", 0.10)),
    )
    # (2) Cắt đáy bằng height (nếu có)
    if height_m is not None:
        Ht = to_torch(height_m, device=dev, dtype=torch.float32)
        boxes = _cut_bottom_by_height(
            boxes, Ht,
            h_free=float(cfg.get("bottom_h_free_m", 0.07)),
            band_px=int(cfg.get("bottom_band_px", 12)),
            min_drop=float(cfg.get("bottom_min_drop_m", 0.12)),
        )

    dbg = dict(num_edges=int((strong>0).sum().item()), thr_s=thr_s, thr_w=thr_w, pass_relax=0)
    return ((boxes.astype(np.float32),
             dists.astype(np.float32),
             scores.astype(np.float32)), dbg)

