# src/fusion/obstacle_general.py
from typing import Tuple
import numpy as np
import cv2

from .depth_utils import robust_depth_stat, mask_morphology

SKY_ID = 10

def _area_pixels_to_m2(pixel_count: int, z_med: float, fx: float, fy: float) -> float:
    """
    Diện tích xấp xỉ từ số pixel tại khoảng cách z (m).
    Một pixel đại diện ~ (z/fx) x (z/fy) m^2.
    """
    if not np.isfinite(z_med) or z_med <= 0:
        return 0.0
    return float(pixel_count) * (z_med / fx) * (z_med / fy)

def detect_obstacles_by_height(
    scaler,                       # GroundPlaneScaler (để lấy fx, fy & height map)
    depth_m: np.ndarray,          # HxW float32
    plane: Tuple[np.ndarray, float],
    seg: np.ndarray,              # HxW uint8 (dùng để loại sky)
    cfg: dict
):
    """
    Trích xuất obstacle từ height-map toàn khung, trả về (boxes, dists, scores).
    - height >= height_min_m
    - depth trong [depth_min_m, depth_max_m]
    - lọc sky
    - connected components trên ảnh downsample để nhanh hơn
    """
    H, W = depth_m.shape
    height = scaler.height_from_plane(depth_m, plane)  # HxW

    h_min = float(cfg.get("height_min_m", 0.30))
    z_min = float(cfg.get("depth_min_m", 0.4))
    z_max = float(cfg.get("depth_max_m", 10.0))
    foot_h = float(cfg.get("foot_height_m", 0.06))
    ds = float(cfg.get("downsample", 0.5))
    area_min = float(cfg.get("area_min_m2", 0.03))

    # mask ứng viên
    cand = (height >= h_min).astype(np.uint8)
    cand = np.where((depth_m >= z_min) & (depth_m <= z_max), cand, 0)
    cand = np.where(seg != SKY_ID, cand, 0)  # bỏ sky

    # morphology
    cand = mask_morphology(cand, k_close=5, k_open=3)

    # downsample để CC nhanh
    if ds != 1.0:
        newW = max(1, int(W * ds))
        newH = max(1, int(H * ds))
        small = cv2.resize(cand, (newW, newH), interpolation=cv2.INTER_NEAREST)
        scaleX, scaleY = W / newW, H / newH
    else:
        small = cand; scaleX = 1.0; scaleY = 1.0

    num, labels, stats, _ = cv2.connectedComponentsWithStats(small, connectivity=8)
    boxes = []; dists = []; scores = []

    for lab in range(1, num):
        x, y, w, h, area = stats[lab]
        if area < 20:
            continue
        # scale bbox về full-res
        x1 = int(x * scaleX); y1 = int(y * scaleY)
        x2 = int((x + w) * scaleX); y2 = int((y + h) * scaleY)
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        roi_depth = depth_m[y1:y2, x1:x2]
        med, q25 = robust_depth_stat(roi_depth, min_valid=30)
        z = q25 if np.isfinite(q25) else med
        if not np.isfinite(z) or z < z_min or z > z_max:
            continue

        # ước lượng diện tích thực để lọc vật nhỏ
        area_m2 = _area_pixels_to_m2(int(area * scaleX * scaleY), z, scaler.fx, scaler.fy)
        if area_m2 < area_min:
            continue

        # footpoint roi (sát đáy) để lấy khoảng cách ổn định hơn
        fh = max(1, int((y2 - y1) * 0.2))
        foot = depth_m[max(y1, y2 - fh):y2, x1:x2]
        med_f, q25_f = robust_depth_stat(foot, min_valid=20)
        z_foot = q25_f if np.isfinite(q25_f) else med_f
        if np.isfinite(z_foot):
            z = z_foot

        boxes.append([x1, y1, x2, y2])
        dists.append(float(z))

        # điểm số ưu tiên vật gần + cao hơn
        h_local = float(np.nanmedian(height[y1:y2, x1:x2]))
        near_score = max(0.0, min(1.0, (z_max - z) / (z_max - z_min)))
        h_score = max(0.0, min(1.0, h_local / (h_min * 2.0)))
        scores.append(0.6 * near_score + 0.4 * h_score)

    if len(boxes) == 0:
        return (np.zeros((0, 4), dtype=float),
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=float))
    return (np.asarray(boxes, dtype=float),
            np.asarray(dists, dtype=float),
            np.asarray(scores, dtype=float))
