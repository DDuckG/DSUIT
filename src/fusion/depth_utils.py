import numpy as np

ROAD_ID = 0
SKY_ID  = 10

def roi_bottom(depth_m, box_xyxy, frac = 0.25):
    x1, y1, x2, y2 = [int(round(float(item))) for item in box_xyxy]
    height, width = depth_m.shape
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1  +  1, min(width, x2))
    y1 = max(0, min(height - 1, y1)) 
    y2 = max(y1 + 1, min(height, y2))
    h = y2 - y1
    band = max(1, int(h * float(frac)))
    yb1 = max(y1, y2 - band)
    return depth_m[yb1 : y2, x1 : x2]

def distance_from_box(depth_m, height_m, seg, box_xyxy, foot_h = 0.06, frac = 0.25):
    x1, y1, x2, y2 = [int(round(float(item))) for item in box_xyxy]
    height, width = depth_m.shape
    x1 = max(0, min(width - 1, x1)) 
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1)) 
    y2 = max(y1 + 1, min(height, y2))
    rb = roi_bottom(depth_m, box_xyxy, frac = frac)
    hr = height_m[y2 - rb.shape[0] : y2, x1 : x2]
    sr = seg[y2 - rb.shape[0] : y2, x1 : x2]
    mfoot = (hr <= float(foot_h)) & (sr != ROAD_ID)
    z_band = rb[mfoot]
    if z_band.size >= 20:
        return float(np.percentile(z_band, 15.0))
    roi = depth_m[y1 : y2, x1 : x2]
    mask = np.isfinite(roi) & (roi > 0) & (seg[y1 : y2, x1 : x2] != SKY_ID)
    return float(np.percentile(roi[mask], 30.0))
