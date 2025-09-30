import numpy as np

_EPS = 0.000001

def _as_np_f32(x, shape2 = False):
    a = np.asarray(x, dtype = np.float32)
    if shape2 and a.ndim == 1 and a.size == 0:
        return a.reshape(0, 4)
    return a

def _wh_ar(boxes):
    w = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    h = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    ar = h / (w + 1e-6)
    return w, h, ar

def _pairwise_intersection_area(A, B):
    Ax1, Ay1, Ax2, Ay2 = A[ : ,0,None], A[ : ,1,None], A[ : ,2,None], A[ : ,3,None]
    Bx1, By1, Bx2, By2 = B[None, : ,0], B[None, : ,1], B[None, : ,2], B[None, : ,3]
    iw = np.maximum(0.0, np.minimum(Ax2, Bx2) - np.maximum(Ax1, Bx1))
    ih = np.maximum(0.0, np.minimum(Ay2, By2) - np.maximum(Ay1, By1))
    return iw * ih

def _center_greedy_suppress(boxes, scores, dists_m, center_thr_px = 18.0, z_thr_m = 0.6):
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype = np.int32)
    cx = 0.5 * (boxes[:,0] + boxes[:,2])
    cy = 0.5 * (boxes[:,1] + boxes[:,3])
    order = np.argsort(-scores)
    keep = []
    removed = np.zeros((boxes.shape[0], ), dtype = bool)
    cthr2 = float(center_thr_px) ** 2
    for i in order:
        if removed[i]:
            continue
        keep.append(i)
        dx = cx[order] - cx[i]
        dy = cy[order] - cy[i]
        dist2 = dx * dx + dy * dy
        close = dist2 <= cthr2
        if np.isfinite(dists_m[i]):
            dz = np.abs(dists_m[order] - dists_m[i])
            close &= np.isfinite(dz) & (dz < float(z_thr_m))
        idx_close = order[close]
        removed[idx_close] = True
        removed[i] = False
    return np.asarray(keep, dtype = np.int32)

def prune_obs_slivers(boxes_xyxy, scores, dists = None, yolo_boxes = None, cfg = None, **kwargs):
    if dists is None and ("dists_m" in kwargs):
        dists = kwargs["dists_m"]

    cfg = cfg or {}
    b = _as_np_f32(boxes_xyxy, shape2 = True).reshape(-1, 4)
    s = _as_np_f32(scores).reshape(-1)
    if dists is None:
        z = np.full((b.shape[0],), np.nan, dtype=np.float32)
    else:
        z = _as_np_f32(dists).reshape(-1)

    n = b.shape[0]
    s = s[ : n]
    z = z[ : n]

    if n == 0:
        return b, s, z

    min_w = float(cfg.get("min_w_px", 8))
    min_h = float(cfg.get("min_h_px", 8))
    min_area = float(cfg.get("min_area_px", 64.0))
    s_min = float(cfg.get("score_min", 0.00))
    ar_min = float(cfg.get("slender_ar_min", 2.8))
    w_max_sl = float(cfg.get("slender_w_max_px", 20))
    h_min_sl = float(cfg.get("slender_h_min_px", 6))
    c_merge = float(cfg.get("center_merge_px", 22.0))
    z_merge = float(cfg.get("center_z_thr_m", 0.6))
    ycov_thr = float(cfg.get("yolo_cover_thr", 0.90))
    zmin_k = float(cfg.get("z_min_keep_m", 0.20))
    zmax_k = float(cfg.get("z_max_keep_m", 80.0))
    w, h, ar = _wh_ar(b)
    area = w * h
    keep = (w >= min_w) & (h >= min_h) & (area >= min_area) & (s >= s_min)
    keep &= ~(((ar >= ar_min) & (w <= w_max_sl)) | (h <= h_min_sl))
    keep &= (np.isnan(z) | ((z >= zmin_k) & (z <= zmax_k)))

    if not np.any(keep):
        return b[ : 0], s[ : 0], z[ : 0]

    b = b[keep]
    s = s[keep]
    z = z[keep]
    if yolo_boxes is not None:
        Y = _as_np_f32(yolo_boxes, shape2 = True).reshape(-1, 4)
        if Y.shape[0] > 0 and b.shape[0] > 0:
            inter = _pairwise_intersection_area(b, Y)
            area_obs = (np.maximum(0.0, b[ : ,2] - b[ : ,0]) * np.maximum(0.0, b[ : ,3]-b[ : ,1])) + _EPS
            max_cover = inter.max(axis = 1) / area_obs
            keep2 = max_cover < ycov_thr
            if not np.any(keep2):
                return b[ : 0], s[ : 0], z[ : 0]
            b = b[keep2]
            s = s[keep2]
            z = z[keep2]

    if b.shape[0] > 1:
        idx_keep = _center_greedy_suppress(b, s, z, center_thr_px = c_merge, z_thr_m = z_merge)
        b = b[idx_keep]
        s = s[idx_keep]
        z = z[idx_keep]

    return b.astype(np.float32, copy = False), s.astype(np.float32, copy = False), z.astype(np.float32, copy = False)
