import numpy as np
import torch
import torch.nn.functional as F
import cupy as cp
from cupyx.scipy import ndimage as cpx_ndi
from torch.utils.dlpack import to_dlpack
from src.utils.torch_cuda import get_device, to_torch, gaussian_blur, sobel_grad, binary_morphology

_ALLOWED = {1,2,3,4,5,7,8,9,11,12,13,14,15,16,17,18} # bỏ road/sky
ROAD_ID, SKY_ID = 0, 10

def _torch_to_cupy(x: torch.Tensor) -> cp.ndarray:
    return cp.from_dlpack(to_dlpack(x.contiguous()))

@torch.no_grad()
def _edge_maps_from_depth_tiled(depth_m: torch.Tensor, cfg: dict):
    Dl = torch.log(torch.clamp(depth_m, min = 0.001))
    Dl = gaussian_blur(Dl, k = int(cfg.get("gauss_k", 5)), sigma = float(cfg.get("gauss_sigma", 1.15)))
    gx, gy = sobel_grad(Dl)
    mag = torch.sqrt(gx * gx + gy * gy)
    rows = int(cfg.get("tile_rows", 3))
    cols = int(cfg.get("tile_cols", 6))
    height, width = mag.shape
    rh = height // rows
    cw = width // cols
    p_str  = float(cfg.get("edge_percentile_str", 92.0)) / 100.0
    p_weak = float(cfg.get("edge_percentile_weak", 85.0)) / 100.0
    rel_s  = float(cfg.get("mag_thr_rel", 1.0))
    rel_w  = float(cfg.get("mag_thr_rel_weak", 0.85))
    strong = torch.zeros_like(mag, dtype=torch.uint8)
    weak   = torch.zeros_like(mag, dtype=torch.uint8)

    for i in range(rows):
        for j in range(cols):
            y1 = i * rh
            y2 = (i + 1) * rh if i < rows - 1 else height
            x1 = j * cw 
            x2 = (j + 1) * cw if j < cols - 1 else width
            tile = mag[y1 : y2, x1 : x2].reshape(-1)
            t_s = torch.quantile(tile, p_str)  * rel_s
            t_w = torch.quantile(tile, p_weak) * rel_w
            strong[y1 : y2, x1 : x2] = (mag[y1 : y2, x1 : x2] >= t_s).to(torch.uint8)
            weak[y1 : y2, x1 : x2] = (mag[y1 : y2, x1 : x2] >= t_w).to(torch.uint8)

    strong = binary_morphology(strong, op = "dilate", k = 3, it = 1)
    weak = binary_morphology(weak, op = "dilate", k = 3, it = 1)
    return mag, strong, weak, gx, gy

def _area_px_to_m2(px: int, z_m: float, fx: float, fy: float) -> float:
    if not np.isfinite(z_m) or z_m <= 0.0:
        return 0.0
    return float(px) * (z_m / fx) * (z_m / fy)

def _compute_bboxes_from_labels(labels_cp: cp.ndarray):
    height, width = labels_cp.shape
    mask = labels_cp > 0
    if not bool(mask.any()):
        return (cp.empty((0,), dtype = cp.int32),) * 6
    ys, xs = cp.where(mask)
    labs_full = labels_cp[ys, xs].astype(cp.int32)
    nlab = int(labs_full.max().item())

    binc = cp.bincount(labs_full, minlength=nlab+1)
    counts = binc[1:]

    min_y = cp.full((nlab + 1, ), height, dtype = cp.int32)
    min_x = cp.full((nlab + 1, ), width, dtype = cp.int32)
    max_y = cp.zeros((nlab + 1, ), dtype = cp.int32)
    max_x = cp.zeros((nlab + 1, ), dtype = cp.int32)

    cp.minimum.at(min_y, labs_full, ys)
    cp.minimum.at(min_x, labs_full, xs)
    cp.maximum.at(max_y, labs_full, ys+1)
    cp.maximum.at(max_x, labs_full, xs+1)

    labs = cp.arange(1, nlab+1, dtype=cp.int32)
    y1 = min_y[1:]
    x1 = min_x[1:]
    y2 = max_y[1:]
    x2 = max_x[1:]
    valid = counts > 0
    return labs[valid], y1[valid], x1[valid], y2[valid], x2[valid], counts[valid]

def _snap_edges_v2(boxes_np: np.ndarray, gx: torch.Tensor, gy: torch.Tensor, max_dx: int = 24, max_dy: int = 24, 
                   min_w: int = 6, min_h: int = 6, ignore_bottom_frac: float = 0.10):
    if boxes_np is None or len(boxes_np) == 0:
        return boxes_np
    devais = gx.device
    height, width = gx.shape
    grad_v = torch.abs(gy)
    grad_h = torch.abs(gx)

    out = []
    boxes_t = torch.as_tensor(boxes_np, device = devais, dtype = torch.float32)
    for b in boxes_t:
        x1, y1, x2, y2 = [int(v.item()) for v in torch.round(b)]
        x1 = max(0, min(width-2, x1))
        x2 = max(x1+1, min(width-1, x2))
        y1 = max(0, min(height-2, y1))
        y2 = max(y1+1, min(height-1, y2))
        cut = int(max(1, (y2-y1) * float(ignore_bottom_frac)))
        yr1, yr2 = y1, max(y1 + 1, y2 - cut)
        xs = torch.arange(max(0, x2 - max_dx), min(width - 1, x2 + max_dx) + 1, device = devais)
        x2n = int(xs[torch.argmax(grad_v[yr1 : yr2, xs].sum(dim = 0))].item()) if xs.numel() else x2
        xs = torch.arange(max(0, x1 - max_dx), min(width - 1, x1 + max_dx) + 1, device = devais)
        x1n = int(xs[torch.argmax(grad_v[yr1 : yr2, xs].sum(dim = 0))].item()) if xs.numel() else x1
        if x2n - x1n < min_w:
            x1n, x2n = x1, x2

        ys = torch.arange(max(0, y1 - max_dy), min(height - 1, y1 + max_dy) + 1, device = devais)
        if x2n > x1n + 1 and ys.numel() > 0:
            y1n = int(ys[torch.argmax(grad_h[ys, x1n : x2n].sum(dim = 1))].item())
        else:
            y1n = y1
        if (y2 - y1n) < min_h: 
            y1n = max(0, y2 - min_h)
        out.append([x1n, y1n, x2n, y2])
    return np.asarray(out, dtype=float)

def _cut_bottom_by_height(boxes_np: np.ndarray, height_m_t: torch.Tensor, h_free = 0.05, band_px = 12, min_drop = 0.10):
    if boxes_np is None or len(boxes_np) == 0:
        return boxes_np
    out = []
    height, width = height_m_t.shape
    band_px = int(max(3, band_px))

    for x1, y1, x2, y2 in boxes_np.astype(int):
        x1 = max(0, min(width - 1, x1)) 
        x2 = max(x1 + 1 ,min(width, x2))
        y1 = max(0, min(height - 1, y1)) 
        y2 = max(y1 + 1, min(height, y2))
        roi = height_m_t[y1 : y2, x1 : x2]
        if roi.numel() == 0:
            out.append([x1, y1, x2, y2])
            continue

        prof = torch.median(roi, dim = 1).values
        if prof.numel() <= band_px:
            out.append([x1, y1, x2, y2])
            continue

        tail = prof[-band_px : ].mean()
        pmax = torch.max(prof)
        pooled = F.avg_pool1d(prof.view(1, 1, -1), kernel_size = band_px, stride = 1)[0, 0]
        idx = torch.where(pooled < float(h_free))[0]

        if (float(tail.item()) < float(h_free)) and ((float(pmax.item()) - float(tail.item())) > float(min_drop)) and idx.numel() > 0:
            cut_at = int(idx[-1].item())
            y2 = max(y1 + 1, y1 + cut_at)

        out.append([x1, y1, x2, y2])

    return np.asarray(out, dtype=float)

def _two_edge_ok(box, gx: torch.Tensor, gy: torch.Tensor, k: int = 3, vthr: float = 0.0, hthr: float = 0.0):
    x1, y1, x2, y2 = [int(round(item)) for item in box]
    height, width = gx.shape
    x1 = max(0, min(width - 2, x1))
    x2 = max(x1 + 1, min(width-1, x2))
    y1 = max(0, min(height - 2, y1))
    y2 = max(y1 + 1, min(height - 1, y2))
    ks = int(max(1, k))
    vband = slice(y1, y2), slice(max(0, x1 - ks), min(width - 1, x1 + ks + 1)), slice(max(0, x2 - ks), min(width - 1, x2 + ks + 1))
    hband = slice(max(0, y1 - ks), min(height - 1, y1 + ks + 1)), slice(x1, x2)
    v_en = torch.abs(gy[vband[0], vband[1]]).mean() + torch.abs(gy[vband[0], vband[2]]).mean()
    h_en = torch.abs(gx[hband[0], hband[1]]).mean()
    return bool((float(v_en.item()) > vthr) and (float(h_en.item()) > hthr))

def two_edge_ok_array(boxes_np: np.ndarray, gx: torch.Tensor, gy: torch.Tensor, k: int = 3, vthr: float = 0.0, hthr: float = 0.0):
    if boxes_np is None or len(boxes_np) == 0:
        return np.zeros((0,), dtype = bool)
    out = []
    for b in boxes_np:
        out.append(_two_edge_ok(b, gx, gy, k = k, vthr = vthr, hthr = hthr))
    return np.asarray(out, dtype = bool)

@torch.no_grad()
def refine_box_by_edges_and_height(box_np: np.ndarray, gx: torch.Tensor, gy: torch.Tensor, height_m: torch.Tensor, cfg: dict):
    b = np.asarray(box_np, dtype = float).reshape(4)
    boxes = _snap_edges_v2(
        b[None,:], gx, gy,
        max_dx = int(cfg.get("snap_dx_px", 24)),
        max_dy = int(cfg.get("snap_dy_px", 24)),
        min_w = int(cfg.get("min_w_px", 6)),
        min_h = int(cfg.get("min_h_px", 6)),
        ignore_bottom_frac=float(cfg.get("snap_ignore_bottom_frac", 0.12)),
    )
    Ht = height_m
    boxes = _cut_bottom_by_height(
        boxes, Ht,
        h_free=float(cfg.get("bottom_h_free_m", 0.05)),
        band_px=int(cfg.get("bottom_band_px", 12)),
        min_drop=float(cfg.get("bottom_min_drop_m", 0.10)),
    )
    return boxes[0].astype(np.float32)

@torch.no_grad()
def detect_obstacles_by_depth_edges(depth_m_enh: np.ndarray, sigma_m: np.ndarray, seg: np.ndarray, cfg: dict, fx: float, fy: float, height_m: np.ndarray | None = None):
    dev = get_device()
    D = to_torch(depth_m_enh, device = dev, dtype = torch.float32)
    S = to_torch(seg, device = dev, dtype = torch.int16)

    mag, strong, weak, gx, gy = _edge_maps_from_depth_tiled(D, cfg)

    z_min = float(cfg.get("depth_min_m", 0.35))
    z_max = float(cfg.get("depth_max_m", 15.0))
    depth_ok = (D >= z_min) & (D <= z_max)

    allow = torch.tensor(list(_ALLOWED), device = dev, dtype = S.dtype)
    seg_ok = torch.isin(S, allow)
    edge_ok_map = weak > 0

    cand = (depth_ok & seg_ok & edge_ok_map).to(torch.uint8)
    cand = binary_morphology(cand, op = "open",  k = 3, it = 1)
    cand = binary_morphology(cand, op = "close", k = 5, it = 1)

    labels_cp, _ = cpx_ndi.label(_torch_to_cupy(cand), structure = cp.ones((3,3), dtype = cp.uint8))
    labs, y1, x1, y2, x2, counts = _compute_bboxes_from_labels(labels_cp)

    if labs.size == 0:
        z = np.zeros((0,), dtype = np.float32)
        return ((np.zeros((0,4),np.float32), z, z, np.zeros((0,), dtype = bool)), dict())

    y1_h, x1_h, y2_h, x2_h = y1.get(), x1.get(), y2.get(), x2.get()
    counts_h = counts.get()
    boxes = []
    dists = []
    scores = []
    edgeoks = []
    height, width = D.shape
    area_k = float(cfg.get("area_min_m2_k", 0.0003))
    for i in range(len(y1_h)):
        xs, ys, xe, ye = int(x1_h[i]), int(y1_h[i]), int(x2_h[i]), int(y2_h[i])
        if (xe - xs) < int(cfg.get("min_w_px", 8)) or (ye-ys) < int(cfg.get("min_h_px", 8)):
            continue
        roi = D[ys:ye, xs:xe]
        z = torch.quantile(roi[torch.isfinite(roi)], 0.30).item()
        px = int(counts_h[i])
        if _area_px_to_m2(px, z, fx, fy) < area_k * (z ** 2):
            continue

        boxes_np = np.asarray([[xs, ys, xe, ye]], dtype = float)
        boxes_np = _snap_edges_v2(
            boxes_np, gx, gy,
            max_dx = int(cfg.get("snap_dx_px", 24)),
            max_dy = int(cfg.get("snap_dy_px", 24)),
            min_w = int(cfg.get("min_w_px", 8)),
            min_h = int(cfg.get("min_h_px", 8)),
            ignore_bottom_frac = float(cfg.get("snap_ignore_bottom_frac", 0.12)),
        )
        if height_m is not None:
            Ht = to_torch(height_m, device = dev, dtype = torch.float32)
            boxes_np = _cut_bottom_by_height(
                boxes_np, Ht,
                h_free = float(cfg.get("bottom_h_free_m", 0.05)),
                band_px = int(cfg.get("bottom_band_px", 12)),
                min_drop = float(cfg.get("bottom_min_drop_m", 0.10)),
            )

        b = boxes_np[0]
        vthr = float(cfg.get("twoedge_v_min", 0.0025)) # phải có 2 cạnh (rìa)
        hthr = float(cfg.get("twoedge_h_min", 0.0025))
        ok2 = _two_edge_ok(b, gx, gy, k = int(cfg.get("twoedge_band_px", 3)), vthr = vthr, hthr = hthr)
        if not ok2:
            continue

        near = max(0.0, min(1.0, (z_max - z) / max(0.000001, (z_max - z_min))))
        e_en = float(mag[int(b[1]) : int(b[3]), int(b[0]) : int(b[2])].mean().item())
        sc = 0.7 * near + 0.3 * (e_en)

        boxes.append(b)
        dists.append(z)
        scores.append(sc)
        edgeoks.append(True)

    if len(boxes) == 0:
        z = np.zeros((0,), dtype = np.float32)
        return ((np.zeros((0, 4), np.float32), z, z, np.zeros((0, ), dtype = bool)), dict())

    boxes = np.asarray(boxes, dtype = np.float32)
    dists = np.asarray(dists, dtype = np.float32)
    scores = np.asarray(scores, dtype = np.float32)
    edgeoks = np.asarray(edgeoks, dtype = bool)
    return ((boxes, dists, scores, edgeoks), dict())
