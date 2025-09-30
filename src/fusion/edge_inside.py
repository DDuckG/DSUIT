import numpy as np
import torch

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _quantile_t(x, q):
    super_x = x.reshape(-1)
    k = int(max(1, min(super_x.numel(), round((float(q) * (super_x.numel() - 1)) + 1))))
    k = max(1, min(super_x.numel(), k))
    val = torch.kthvalue(super_x, k).values
    return val

@torch.no_grad()
def _ribbons(depth_t: torch.Tensor, box, r = 3, side = "right"):
    x1, y1, x2, y2 = [int(round(float(item))) for item in box]
    height, width = depth_t.shape
    x1 = _clamp(x1, 0, width - 2)
    x2 = _clamp(x2, x1 + 1, width-1)
    y1 = _clamp(y1, 0, height-2)
    y2 = _clamp(y2, y1 + 1, height-1)
    r = int(max(1, r))

    if side == "right":
        Rin = depth_t[y1 : y2, _clamp(x2 - r, 0, width - 1) : x2]
        Rout = depth_t[y1 : y2, _clamp(x2, 0, width - 1) : _clamp(x2 + r, 0, width)]
    elif side == "left":
        Rin = depth_t[y1 : y2, _clamp(x1, 0, width - 1) : _clamp(x1 + r, 0, width)]
        Rout = depth_t[y1 : y2, _clamp(x1 - r, 0, width - 1) : _clamp(x1, 0, width)]
    elif side == "top":
        Rin = depth_t[_clamp(y1, 0, height - 1) : _clamp(y1 + r, 0, height), x1 : x2]
        Rout = depth_t[_clamp(y1 - r, 0, height - 1) : _clamp(y1, 0, height), x1 : x2]
    else: 
        Rin = depth_t[_clamp(y2 - r, 0, height - 1) : y2, x1 : x2]
        Rout = depth_t[_clamp(y2, 0, height - 1) : _clamp(y2 + r, 0, height), x1 : x2]
    return Rin, Rout

@torch.no_grad()
def _edge_ok(depth_t, box, side, ribbon_px, jump_min, var_in_max, fill_min):
    Rin, Rout = _ribbons(depth_t, box, r = ribbon_px, side = side)
    mask_in  = torch.isfinite(Rin) & (Rin > 0)
    mask_out = torch.isfinite(Rout) & (Rout > 0)
    if mask_in.sum().item() == 0 or mask_out.sum().item() == 0:
        return False, 0.0
    zin = _quantile_t(Rin[mask_in], 0.50)
    zout = _quantile_t(Rout[mask_out], 0.50)
    dz = torch.abs(zout - zin).item()
    vin = torch.var(Rin[mask_in]).item()
    fill = float(mask_in.float().mean().item())
    ok = (dz >= float(jump_min)) and (vin <= float(var_in_max)) and (fill >= float(fill_min)) and (zin <= zout)
    return bool(ok), dz

@torch.no_grad()
def snap_inside_box(box_np: np.ndarray, depth_t: torch.Tensor, cfg: dict):
    boxes = np.asarray(box_np, dtype = np.float32).reshape(4)
    x1, y1, x2, y2 = [int(round(float(item))) for item in boxes]
    ribbon_px = int(cfg.get("ribbon_px", 3))
    sdx = int(cfg.get("search_dx_px", 18))
    sdy = int(cfg.get("search_dy_px", 14))
    jmin = float(cfg.get("jump_min_m", 0.25))
    vinm = float(cfg.get("var_in_max", 0.08))
    fmin = float(cfg.get("fill_min", 0.55))
    H,W = depth_t.shape
    x1 = _clamp(x1, 0, W - 2)
    x2 = _clamp(x2, x1 + 1, W - 1)
    y1 = _clamp(y1, 0, H - 2)
    y2 = _clamp(y2, y1 + 1, H - 1)
    
    candidates = torch.arange(max(0, x2 - sdx), min(W - 1, x2 + sdx) + 1, device = depth_t.device)      # phải
    if candidates.numel() > 0:
        best = x2
        best_dz = -1.0
        for x in candidates.tolist():
            ok, dz = _edge_ok(depth_t, [x1, y1, x, y2], "right", ribbon_px, jmin, vinm, fmin)
            if ok and dz > best_dz:
                best = x
                best_dz = dz
        x2 = best

    candidates = torch.arange(max(0, x1 - sdx), min(W - 1, x1 + sdx) + 1, device = depth_t.device)            # trái
    if candidates.numel() > 0:
        best = x1
        best_dz = -1.0
        for x in candidates.tolist():
            ok, dz = _edge_ok(depth_t, [x, y1, x2, y2], "left", ribbon_px, jmin, vinm, fmin)
            if ok and dz > best_dz:
                best = x
                best_dz = dz
        x1 = best

    candidates = torch.arange(max(0, y1 - sdy), min(H - 1, y1 + sdy) + 1, device = depth_t.device)            # trên
    if candidates.numel() > 0:
        best = y1
        best_dz = -1.0
        for y in candidates.tolist():
            ok, dz = _edge_ok(depth_t, [x1, y, x2, y2], "top", ribbon_px, jmin, vinm, fmin)
            if ok and dz > best_dz:
                best = y
                best_dz = dz
        y1 = best

    return np.asarray([x1, y1, x2, y2], dtype = np.float32)

@torch.no_grad()
def edge_pair_ok(depth_t: torch.Tensor, box_np: np.ndarray, config: dict):
    ribbon_px = int(config.get("ribbon_px", 3))
    jmin = float(config.get("jump_min_m", 0.25))
    vinm = float(config.get("var_in_max", 0.08))
    fmin = float(config.get("fill_min", 0.55))
    okR,_ = _edge_ok(depth_t, box_np, "right", ribbon_px, jmin, vinm, fmin)
    okL,_ = _edge_ok(depth_t, box_np, "left", ribbon_px, jmin, vinm, fmin)
    okT,_ = _edge_ok(depth_t, box_np, "top", ribbon_px, jmin, vinm, fmin)
    okB,_ = _edge_ok(depth_t, box_np, "bottom",ribbon_px, jmin, vinm, fmin)
    count = int(okR) + int(okL) + int(okT) + int(okB)    # sao m cho nó chạy đc z =)))
    return bool(count >= 2)
