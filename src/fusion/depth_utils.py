# src/fusion/depth_utils.py
import numpy as np

ROAD_ID = 0
SKY_ID  = 10

def roi_bottom(depth_m, box_xyxy, frac=0.25):
    x1,y1,x2,y2 = [int(round(float(v))) for v in box_xyxy]
    H,W = depth_m.shape
    x1 = max(0, min(W-1, x1)); x2 = max(x1+1, min(W, x2))
    y1 = max(0, min(H-1, y1)); y2 = max(y1+1, min(H, y2))
    h = y2 - y1
    band = max(1, int(h*float(frac)))
    yb1 = max(y1, y2-band)
    return depth_m[yb1:y2, x1:x2]

def distance_from_box(depth_m, height_m, seg, box_xyxy, foot_h=0.06, frac=0.25):
    """
    Đo khoảng cách trực tiếp từ depth_m (m).
    - Nếu có đủ điểm 'grounded' (height<=foot_h) ở dải đáy -> dùng q20 trên band đáy (đã mask).
    - Ngược lại coi là elevated -> dùng q35 toàn ROI.
    """
    x1,y1,x2,y2 = [int(round(float(v))) for v in box_xyxy]
    H,W = depth_m.shape
    x1 = max(0, min(W-1, x1)); x2 = max(x1+1, min(W, x2))
    y1 = max(0, min(H-1, y1)); y2 = max(y1+1, min(H, y2))

    rb = roi_bottom(depth_m, box_xyxy, frac=frac)
    hr = height_m[y2-rb.shape[0]:y2, x1:x2]
    sr = seg[y2-rb.shape[0]:y2, x1:x2]

    mfoot = (hr <= float(foot_h)) & (sr != ROAD_ID)
    z_band = rb[mfoot]
    if z_band.size >= 20:
        return float(np.percentile(z_band, 15.0))
    roi = depth_m[y1:y2, x1:x2]
    mask = np.isfinite(roi) & (roi>0) & (seg[y1:y2, x1:x2] != SKY_ID)
    return float(np.percentile(roi[mask], 30.0))
