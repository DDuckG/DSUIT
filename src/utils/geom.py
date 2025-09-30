import numpy as np

def clip_xyxy(box, W, H):
    x1, y1, x2, y2 = [float(item) for item in box]
    x1 = max(0.0, min(W - 1.0, x1))
    y1 = max(0.0, min(H - 1.0, y1))
    x2 = max(0.0, min(W * 1.0, x2))
    y2 = max(0.0, min(H * 1.0, y2))
    return np.asarray([x1, y1, x2, y2], dtype = np.float32)

def iou_matrix(A, B):
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype = np.float32)
    Ax1, Ay1, Ax2, Ay2 = A[:,0], A[:,1], A[:,2], A[:,3]
    Bx1, By1, Bx2, By2 = B[:,0], B[:,1], B[:,2], B[:,3]
    iw = np.maximum(0.0, np.minimum(Ax2[:,None], Bx2[None,:]) - np.maximum(Ax1[: ,None], Bx1[None, :]))
    ih = np.maximum(0.0, np.minimum(Ay2[:,None], By2[None,:]) - np.maximum(Ay1[: ,None], By1[None, :]))
    inter = iw * ih
    areaA = np.maximum(0.0, (Ax2 - Ax1)) * np.maximum(0.0, (Ay2 - Ay1))
    areaB = np.maximum(0.0, (Bx2 - Bx1)) * np.maximum(0.0, (By2 - By1))
    union = areaA[:,None] + areaB[None,:] - inter + 0.000000001
    return (inter / union).astype(np.float32)

def nms_iou(xyxy, scores, iou_thr = 0.55):
    if xyxy is None or len(xyxy) == 0:
        return []
    boxes = np.asarray(xyxy, dtype = np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype = np.float32).reshape(-1)
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1 : ]
        ious = iou_matrix(boxes[i : i + 1], boxes[rest]).reshape(-1)
        mask = ious <= float(iou_thr)
        order = rest[mask]
    return keep

def pairwise_intersection_area(A, B):
    Ax1, Ay1, Ax2, Ay2 = A[:, 0, None], A[:, 1, None], A[:, 2, None], A[:, 3, None]
    Bx1, By1, Bx2, By2 = B[None, :, 0], B[None, :, 1], B[None, :, 2], B[None, :, 3]
    iw = np.maximum(0.0, np.minimum(Ax2, Bx2) - np.maximum(Ax1, Bx1))
    ih = np.maximum(0.0, np.minimum(Ay2, By2) - np.maximum(Ay1, By1))
    return iw * ih

def cover_fraction(A, B):
    inter = pairwise_intersection_area(A,B) 
    areaA  = (np.maximum(0.0, A[: , 2]-A[: , 0]) * np.maximum(0.0, A[: , 3] - A[: , 1]))[: , None] + 0.000000001
    return inter / areaA
