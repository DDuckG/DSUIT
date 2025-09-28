# src/fusion/depth_utils.py
import numpy as np
import torch

ROAD_ID = 0


# ------------------------- helpers -------------------------

def _is_torch(x):
    return isinstance(x, torch.Tensor)

def _H_W(x):
    if _is_torch(x):
        h, w = x.shape[-2], x.shape[-1]
    else:
        h, w = x.shape[0], x.shape[1]
    return int(h), int(w)

def _to_int_scalar(v):
    if _is_torch(v):
        return int(float(v.item()))
    return int(round(float(v)))

def _to_int4(box_xyxy):
    return [_to_int_scalar(v) for v in box_xyxy]  # [x1,y1,x2,y2]

def _clip_xyxy(x1,y1,x2,y2, H, W):
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W,     x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H,     y2))
    return x1,y1,x2,y2

def _finite_pos_mask(x):
    if _is_torch(x):
        return torch.isfinite(x) & (x > 0)
    else:
        return np.isfinite(x) & (x > 0)

def _percentile_torch(x: torch.Tensor, q: float):
    if x.numel() == 0:
        return torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
    q01 = float(q) / 100.0
    q01 = min(1.0, max(0.0, q01))
    return torch.quantile(x, q01)

def _median_torch(x: torch.Tensor):
    if x.numel() == 0:
        return torch.tensor(float('nan'), device=x.device, dtype=x.dtype)
    return torch.median(x)


# ------------------------- stats -------------------------

def robust_depth_stat(roi, min_valid: int = 20):
    """
    Trả về (median, q25). Hỗ trợ numpy hoặc torch (CPU/GPU).
    """
    if roi is None:
        return (np.nan, np.nan)

    if _is_torch(roi):
        x = roi.reshape(-1)
        x = x[_finite_pos_mask(x)]
        if int(x.numel()) < int(min_valid):
            return (np.nan, np.nan)
        lo = _percentile_torch(x, 1.0)
        hi = _percentile_torch(x, 99.0)
        m  = (x >= lo) & (x <= hi)
        x  = x[m]
        if int(x.numel()) < int(min_valid):
            return (np.nan, np.nan)
        med = _median_torch(x)
        q25 = _percentile_torch(x, 25.0)
        return (float(med.item()), float(q25.item()))
    else:
        if roi.size == 0:
            return (np.nan, np.nan)
        x = np.asarray(roi, dtype=np.float32).reshape(-1)
        x = x[np.isfinite(x) & (x > 0)]
        if x.size < min_valid:
            return (np.nan, np.nan)
        lo, hi = np.percentile(x, [1.0, 99.0])
        x = x[(x >= lo) & (x <= hi)]
        if x.size < min_valid:
            return (np.nan, np.nan)
        med = float(np.median(x))
        q25 = float(np.percentile(x, 25.0))
        return (med, q25)


# ------------------------- ROIs -------------------------

def roi_bottom(depth_m, box_xyxy, frac: float = 0.25):
    """
    Lấy dải đáy bbox. Hỗ trợ numpy hoặc torch.
    """
    x1, y1, x2, y2 = _to_int4(box_xyxy)
    H, W = _H_W(depth_m)
    x1,y1,x2,y2 = _clip_xyxy(x1,y1,x2,y2,H,W)
    if x2 <= x1 or y2 <= y1:
        return depth_m[..., :0, :0] if _is_torch(depth_m) else depth_m[:0, :0]
    h = y2 - y1
    band = max(1, int(h * float(frac)))
    yb1 = int(max(y1, y2 - band))
    return depth_m[yb1:y2, x1:x2]

def roi_bottom_masked(depth_m, height_m, seg, box_xyxy, foot_h: float = 0.06, frac: float = 0.25):
    """
    Dùng đúng vùng đáy (rb) để tạo mask cùng kích thước -> tránh mismatch shape.
    """
    rb = roi_bottom(depth_m, box_xyxy, frac=frac)
    # empty?
    if (_is_torch(rb) and rb.numel() == 0) or (not _is_torch(rb) and rb.size == 0):
        return rb

    x1, y1, x2, y2 = _to_int4(box_xyxy)
    H, W = _H_W(depth_m)
    x1,y1,x2,y2 = _clip_xyxy(x1,y1,x2,y2,H,W)

    # LẤY CHIỀU CAO CỦA RB ĐỂ CẮT CHÍNH XÁC CÙNG VÙNG
    if _is_torch(rb):
        hb = int(rb.shape[-2]); wb = int(rb.shape[-1])
    else:
        hb = int(rb.shape[0]);   wb = int(rb.shape[1])
    if hb <= 0 or wb <= 0:
        return rb
    yb1 = max(y1, y2 - hb)

    if _is_torch(depth_m):
        hroi = height_m[yb1:y2, x1:x2]
        sroi = seg[yb1:y2,   x1:x2]
        # Bảo đảm shape khớp với rb (đề phòng sai số cắt)
        if hroi.shape != rb.shape:
            # cắt lại theo shape nhỏ hơn chung
            Hm, Wm = hroi.shape[-2], hroi.shape[-1]
            Hr, Wr = rb.shape[-2], rb.shape[-1]
            hs = min(Hm, Hr); ws = min(Wm, Wr)
            hroi = hroi[:hs, :ws]
            sroi = sroi[:hs, :ws]
            rb   = rb[:hs,   :ws]
        mask = (hroi >= 0.0) & (hroi <= float(foot_h)) & (sroi != int(ROAD_ID))
        if int(mask.sum().item()) < 10:
            return rb  # fallback
        return torch.where(mask, rb, torch.full_like(rb, float('nan')))
    else:
        hroi = height_m[yb1:y2, x1:x2]
        sroi = seg[yb1:y2,   x1:x2]
        if hroi.shape != rb.shape:
            Hm, Wm = hroi.shape[0], hroi.shape[1]
            Hr, Wr = rb.shape[0],  rb.shape[1]
            hs = min(Hm, Hr); ws = min(Wm, Wr)
            hroi = hroi[:hs, :ws]
            sroi = sroi[:hs, :ws]
            rb   = rb[:hs,   :ws]
        mask = (hroi >= 0.0) & (hroi <= float(foot_h)) & (sroi != ROAD_ID)
        if int(mask.sum()) < 10:
            return rb
        return np.where(mask, rb, np.nan)


# ------------------------- depth from box -------------------------

def depth_from_yolo_box(depth_m, box_xyxy, frac: float = 0.25):
    """
    q20 của dải đáy bbox. Hỗ trợ numpy/torch.
    """
    rb = roi_bottom(depth_m, box_xyxy, frac=frac)
    # empty?
    if (_is_torch(rb) and rb.numel() == 0) or (not _is_torch(rb) and rb.size == 0):
        return float('nan')

    if _is_torch(rb):
        z = rb[_finite_pos_mask(rb)]
        if int(z.numel()) < 20:
            med, q25 = robust_depth_stat(rb, min_valid=16)
            return float(q25 if np.isfinite(q25) else med)
        return float(_percentile_torch(z, 20.0).item())
    else:
        z = rb[np.isfinite(rb) & (rb > 0)]
        if z.size < 20:
            med, q25 = robust_depth_stat(rb, min_valid=16)
            return float(q25 if np.isfinite(q25) else med)
        return float(np.percentile(z, 20.0))

def depth_from_obs_box(depth_m, height_m, seg, box_xyxy, foot_h: float = 0.06, frac: float = 0.25):
    """
    Ưu tiên đo ở "bàn chân" (foot) dựa vào height & seg; fallback toàn bbox.
    """
    rb = roi_bottom_masked(depth_m, height_m, seg, box_xyxy, foot_h=foot_h, frac=frac)
    med, q25 = robust_depth_stat(rb, min_valid=16)
    z = float(q25 if np.isfinite(q25) else med)
    if np.isfinite(z):
        return z

    # fallback: toàn bbox
    x1, y1, x2, y2 = _to_int4(box_xyxy)
    H, W = _H_W(depth_m)
    x1,y1,x2,y2 = _clip_xyxy(x1,y1,x2,y2,H,W)
    roi = depth_m[y1:y2, x1:x2]
    med, q25 = robust_depth_stat(roi, min_valid=30)
    return float(q25 if np.isfinite(q25) else med)
