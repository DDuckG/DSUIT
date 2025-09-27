# src/fusion/depth_utils.py
import numpy as np
import cv2

def robust_depth_stat(roi: np.ndarray, min_valid: int = 20):
    # Tính các thống kê robust cho ROI độ sâu (m).
    # - Bỏ NaN/Inf/0 và outlier quá lớn.
    # Trả về: (median, p25). Nếu không đủ điểm, trả về (NaN, NaN).
    if roi is None or roi.size == 0:
        return (np.nan, np.nan)
    x = np.asarray(roi, dtype=np.float32).reshape(-1)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size < min_valid:
        return (np.nan, np.nan)
    # clamp 1%–99% để chống outlier
    lo, hi = np.percentile(x, [1.0, 99.0])
    x = x[(x >= lo) & (x <= hi)]
    if x.size < min_valid:
        return (np.nan, np.nan)
    med = float(np.median(x))
    q25 = float(np.percentile(x, 25.0))
    return (med, q25)

def roi_bottom(depth_m: np.ndarray, box_xyxy, frac: float = 0.35):
    # Lấy dải đáy của bbox (phần sát mặt đất) để đo khoảng cách.
    # frac: tỉ lệ chiều cao bbox (0–1) tính từ đáy đi lên.
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    H, W = depth_m.shape
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return depth_m[:0, :0]
    h = y2 - y1
    yb1 = int(max(y1, y2 - max(1, int(h * frac))))
    return depth_m[yb1:y2, x1:x2]

def mask_morphology(mask: np.ndarray, k_close: int = 5, k_open: int = 3):
    # Close → Open để lấp lỗ nhỏ và loại nhiễu.
    m = (mask > 0).astype(np.uint8)
    if k_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if k_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m
