# src/fusion/depth_edges.py
import numpy as np
import torch
import torch.nn.functional as F
import cupy as cp
from cupyx.scipy import ndimage as cpx_ndi
from torch.utils.dlpack import to_dlpack, from_dlpack

from src.utils.torch_cuda import (
    get_device, to_torch, to_numpy,
    gaussian_blur, sobel_grad, binary_morphology
)

# Cityscapes: bỏ road/sky
_ALLOWED = {1,2,3,4,5,7,8,9,11,12,13,14,15,16,17,18}
ROAD_ID, SKY_ID = 0, 10


def _torch_to_cupy(x: torch.Tensor) -> cp.ndarray:
    # dùng API mới (tránh cảnh báo)
    return cp.from_dlpack(to_dlpack(x.contiguous()))


@torch.no_grad()
def _edge_maps_from_depth(depth_m: torch.Tensor, cfg: dict):
    """
    depth_m: [H,W] float32 (m). Trả về:
      mag (float32), strong_bin (uint8), weak_bin (uint8), gx, gy, thr_s, thr_w
    Tất cả trên GPU.
    """
    dev = depth_m.device
    # log depth để ổn định biên độ
    Dl = torch.log(torch.clamp(depth_m, min=1e-3))
    Dl = gaussian_blur(Dl, k=int(cfg.get("gauss_k", 5)), sigma=float(cfg.get("gauss_sigma", 1.15)))
    gx, gy = sobel_grad(Dl)
    mag = torch.sqrt(gx * gx + gy * gy)

    # Ngưỡng theo percentile
    p_str  = float(cfg.get("edge_percentile_str", 92.0)) / 100.0
    p_weak = float(cfg.get("edge_percentile_weak", 85.0)) / 100.0
    flat = mag.flatten()
    t_str  = torch.quantile(flat, p_str)  * float(cfg.get("mag_thr_rel", 1.00))
    t_weak = torch.quantile(flat, p_weak) * float(cfg.get("mag_thr_rel_weak", 0.85))

    strong = (mag >= t_str).to(torch.uint8)
    weak   = (mag >= t_weak).to(torch.uint8)

    # nở nhẹ để bền
    strong = binary_morphology(strong, op="dilate", k=3, it=1)
    weak   = binary_morphology(weak,   op="dilate", k=3, it=1)

    return mag, strong, weak, gx, gy, float(t_str.item()), float(t_weak.item())


def _area_px_to_m2(px: int, z_m: float, fx: float, fy: float) -> float:
    if not np.isfinite(z_m) or z_m <= 0.0:
        return 0.0
    return float(px) * (z_m / fx) * (z_m / fy)


def _compute_bboxes_from_labels(labels_cp: cp.ndarray):
    """
    Trả về:
      labs (cp.int32, shape [K])               -- các nhãn >0
      y1,x1,y2,x2 (cp.int32, shape [K])        -- bbox cho từng nhãn
      counts (cp.int32, shape [K])             -- số pixel mỗi nhãn
    Tất cả tính trên GPU (CuPy).
    """
    H, W = labels_cp.shape
    mask = labels_cp > 0
    if not bool(mask.any()):
        return (cp.empty((0,), dtype=cp.int32),)*6

    ys, xs = cp.where(mask)
    labs_full = labels_cp[ys, xs].astype(cp.int32)
    nlab = int(labs_full.max().item())

    # Đếm pixel theo nhãn (bao gồm 0 -> ta bỏ đi sau)
    binc = cp.bincount(labs_full, minlength=nlab+1)
    counts = binc[1:]  # bỏ nhãn 0

    # Khởi tạo min/max
    min_y = cp.full((nlab+1,), H, dtype=cp.int32)
    min_x = cp.full((nlab+1,), W, dtype=cp.int32)
    max_y = cp.zeros((nlab+1,), dtype=cp.int32)
    max_x = cp.zeros((nlab+1,), dtype=cp.int32)

    # Segment reduce (atomic) — tất cả GPU
    cp.minimum.at(min_y, labs_full, ys)
    cp.minimum.at(min_x, labs_full, xs)
    cp.maximum.at(max_y, labs_full, ys+1)  # +1 để inclusive-exclusive
    cp.maximum.at(max_x, labs_full, xs+1)

    # Lấy bbox cho nhãn >0
    labs = cp.arange(1, nlab+1, dtype=cp.int32)
    y1 = min_y[1:]; x1 = min_x[1:]; y2 = max_y[1:]; x2 = max_x[1:]

    # Lọc những nhãn không có pixel (counts==0)
    valid = counts > 0
    return labs[valid], y1[valid], x1[valid], y2[valid], x2[valid], counts[valid]


def _refine_boxes_with_edges(boxes_np: np.ndarray, gx: torch.Tensor, gy: torch.Tensor,
                             max_dx: int = 24, max_dy: int = 24, min_w: int = 6, min_h: int = 6):
    """
    Snap cạnh đứng theo |gy| (vertical energy), cạnh trên theo |gx| (horizontal energy).
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

        # RIGHT
        xs = torch.arange(max(0, x2-max_dx), min(W-1, x2+max_dx)+1, device=dev)
        if y2 > y1 + 1 and xs.numel() > 0:
            score_r = grad_v[y1:y2, xs].sum(dim=0)
            x2n = int(xs[torch.argmax(score_r)].item())
        else:
            x2n = x2

        # LEFT
        xs = torch.arange(max(0, x1-max_dx), min(W-1, x1+max_dx)+1, device=dev)
        if y2 > y1 + 1 and xs.numel() > 0:
            score_l = grad_v[y1:y2, xs].sum(dim=0)
            x1n = int(xs[torch.argmax(score_l)].item())
        else:
            x1n = x1

        if x2n - x1n < min_w:
            x1n, x2n = x1, x2

        # TOP
        ys = torch.arange(max(0, y1-max_dy), min(H-1, y1+max_dy)+1, device=dev)
        if x2n > x1n + 1 and ys.numel() > 0:
            score_t = grad_h[ys, x1n:x2n].sum(dim=1)
            y1n = int(ys[torch.argmax(score_t)].item())
        else:
            y1n = y1

        if (y2 - y1n) < min_h:
            y1n = max(0, y2 - min_h)

        out.append([x1n, y1n, x2n, y2])

    return np.asarray(out, dtype=float)


@torch.no_grad()
def detect_obstacles_by_depth_edges(depth_m_enh: np.ndarray,
                                    sigma_m: np.ndarray,
                                    seg: np.ndarray,
                                    cfg: dict,
                                    fx: float = 1000.0, fy: float = 1000.0):
    """
    Trả về: ((boxes_xyxy np.float32, dists_m np.float32, scores np.float32), dbg_dict)
    """
    dev = get_device()

    # to torch GPU
    D = to_torch(depth_m_enh, device=dev, dtype=torch.float32)  # [H,W]
    S = to_torch(seg, device=dev, dtype=torch.int16)            # [H,W]

    # Edge maps
    mag, strong, weak, gx, gy, thr_s, thr_w = _edge_maps_from_depth(D, cfg)

    # Candidate mask
    z_min = float(cfg.get("depth_min_m", 0.35))
    z_max = float(cfg.get("depth_max_m", 12.0))
    depth_ok = (D >= z_min) & (D <= z_max)

    # **FIX dtype** để tránh reject sạch
    allow = torch.tensor(list(_ALLOWED), device=dev, dtype=S.dtype)
    seg_ok = torch.isin(S, allow)
    edge_ok  = weak > 0

    cand = (depth_ok & seg_ok & edge_ok).to(torch.uint8)
    cand = binary_morphology(cand, op="open",  k=3, it=1)
    cand = binary_morphology(cand, op="close", k=5, it=1)

    # Connected components (GPU)
    labels_cp, num = cpx_ndi.label(_torch_to_cupy(cand), structure=cp.ones((3,3), dtype=cp.uint8))

    labs, y1, x1, y2, x2, counts = _compute_bboxes_from_labels(labels_cp)
    if labs.size == 0:
        dbg = dict(num_edges=int((strong>0).sum().item()),
                   thr_s=thr_s, thr_w=thr_w, pass_relax=0)
        z = np.zeros((0,), dtype=np.float32)
        return ((np.zeros((0,4), dtype=np.float32), z, z), dbg)

    # chuyển về host để loop nhẹ (số thành phần nhỏ)
    y1_h = y1.get(); x1_h = x1.get(); y2_h = y2.get(); x2_h = x2.get()
    counts_h = counts.get()

    boxes = []; dists = []; scores = []
    H, W = D.shape

    area_k = float(cfg.get("area_min_m2_k", 2.0e-4))
    for i in range(len(y1_h)):
        xs, ys, xe, ye = int(x1_h[i]), int(y1_h[i]), int(x2_h[i]), int(y2_h[i])
        if (xe - xs) < 6 or (ye - ys) < 6:
            continue

        # depth q25 trong ROI
        roi = D[ys:ye, xs:xe]
        if not torch.isfinite(roi).any():
            continue
        z = torch.quantile(roi[torch.isfinite(roi)], 0.25).item()
        if not np.isfinite(z) or z < z_min or z > z_max:
            continue

        # area gate theo z^2
        px = int(counts_h[i])
        area_m2 = _area_px_to_m2(px, z, fx, fy)
        if area_m2 < area_k * (z ** 2):
            continue

        # score: gần + năng lượng cạnh
        near = max(0.0, min(1.0, (z_max - z) / max(1e-6, (z_max - z_min))))
        e_en = float(mag[ys:ye, xs:xe].mean().item())
        sc = 0.7 * near + 0.3 * (e_en / (thr_s + 1e-6))

        boxes.append([xs, ys, xe, ye])
        dists.append(z)
        scores.append(sc)

    if len(boxes) == 0:
        dbg = dict(num_edges=int((strong>0).sum().item()),
                   thr_s=thr_s, thr_w=thr_w, pass_relax=0)
        z = np.zeros((0,), dtype=np.float32)
        return ((np.zeros((0,4), dtype=np.float32), z, z), dbg)

    boxes = np.asarray(boxes, dtype=float)
    dists = np.asarray(dists, dtype=float)
    scores = np.asarray(scores, dtype=float)

    # Snap theo mép
    boxes_ref = _refine_boxes_with_edges(
        boxes, gx, gy,
        max_dx=int(cfg.get("snap_dx_px", 24)),
        max_dy=int(cfg.get("snap_dy_px", 24)),
        min_w=int(cfg.get("min_w_px", 6)),
        min_h=int(cfg.get("min_h_px", 6))
    )

    dbg = dict(num_edges=int((strong>0).sum().item()),
               thr_s=thr_s, thr_w=thr_w, pass_relax=0)

    return ((boxes_ref.astype(np.float32),
             dists.astype(np.float32),
             scores.astype(np.float32)), dbg)
