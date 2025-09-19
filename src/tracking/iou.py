import numpy as np

def xyxy_to_xyah(box):
    x1, y1, x2, y2 = box
    w = max(0.000001, x2 - x1)
    h = max(0.000001, y2 - y1)
    cx = x1 + w / 2
    cy = y1 + h / 2
    s = w * h
    r = w / float(h)
    if not np.isfinite(cx + cy + s + r):
        return np.array([cx, cy, max(s, 0.000001), max(r, 0.000001)], dtype = float)
    return np.array([cx, cy, s, r], dtype = float)

def xyah_to_xyxy(xyah):
    cx, cy, s, r = map(float, xyah)
    eps = 0.000001
    s = max(s, eps)
    r = max(r, eps)
    val = max(0.0, s * r)
    w = np.sqrt(val)
    if w <= 0:
        w = eps
    h = max(eps, s / w)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def iou_batch(bb_test, bb_gt):
    bb_test = np.asarray(bb_test)
    bb_gt = np.asarray(bb_gt)
    if bb_test.size == 0 or bb_gt.size == 0:
        return np.zeros((bb_test.shape[0], bb_gt.shape[0]), dtype = float)
    area_test = (bb_test[:,2] - bb_test[:,0]) * (bb_test[:,3] - bb_test[:,1])
    area_gt = (bb_gt[:,2] - bb_gt[:,0]) * (bb_gt[:,3] - bb_gt[:,1])

    iou_mat = np.zeros((bb_test.shape[0], bb_gt.shape[0]), dtype = float)
    for i in range(bb_test.shape[0]):
        xx1 = np.maximum(bb_test[i,0], bb_gt[:,0])
        yy1 = np.maximum(bb_test[i,1], bb_gt[:,1])
        xx2 = np.minimum(bb_test[i,2], bb_gt[:,2])
        yy2 = np.minimum(bb_test[i,3], bb_gt[:,3])
        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)
        inter = w * h
        union = area_test[i] + area_gt - inter + 0.000000001
        iou_mat[i,:] = inter / union
    return iou_mat
