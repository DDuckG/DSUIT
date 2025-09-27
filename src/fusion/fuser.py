# src/fusion/fuser.py
import numpy as np

def iou_with(a, arr):
    x1 = np.maximum(a[0], arr[:, 0])
    y1 = np.maximum(a[1], arr[:, 1])
    x2 = np.minimum(a[2], arr[:, 2])
    y2 = np.minimum(a[3], arr[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (arr[:,2]-arr[:,0])*(arr[:,3]-arr[:,1]) - inter + 1e-9
    return inter / ua

def nms_iou(xyxy, scores, iou_thr=0.5):
    if len(xyxy) == 0:
        return []
    order = np.argsort(-scores)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = iou_with(xyxy[i], xyxy[order[1:]])
        inds = np.where(ious <= iou_thr)[0]
        order = order[inds + 1]
    return keep

def merge_yolo_obstacle(
    yolo_boxes, yolo_scores, yolo_cls, yolo_dists,
    obs_boxes, obs_scores, obs_dists,
    iou_merge=0.5
):
    """
    Hợp nhất hai nguồn:
    - Nếu obstacle trùng YOLO (IoU>=iou_merge) → giữ class YOLO, score=max, distance ưu tiên footpoint ổn định.
    - Nếu không trùng → để obstacle (class=-1).
    - Thêm YOLO còn lại (không trùng obstacle).
    Trả về: boxes, scores, classes, is_from_yolo(bool), dists
    """
    boxes=[]; scores=[]; clss=[]; flags=[]; dists=[]
    yolo_used = np.zeros(len(yolo_boxes), dtype=bool)

    for i, (b, s, d) in enumerate(zip(obs_boxes, obs_scores, obs_dists)):
        if len(yolo_boxes) == 0:
            boxes.append(b); scores.append(s); clss.append(-1); flags.append(False); dists.append(d); continue
        ious = iou_with(b, yolo_boxes)
        j = int(np.argmax(ious))
        if ious[j] >= iou_merge:
            yolo_used[j] = True
            # score kết hợp: ưu tiên YOLO conf, cộng điểm gần
            near = 1.0 - min(1.0, max(0.0, d) / 10.0) if np.isfinite(d) else 0.0
            sc = 0.7 * yolo_scores[j] + 0.3 * max(s, near)
            dist = yolo_dists[j] if np.isfinite(yolo_dists[j]) else d
            boxes.append(yolo_boxes[j]); scores.append(sc); clss.append(yolo_cls[j]); flags.append(True); dists.append(dist)
        else:
            near = 1.0 - min(1.0, max(0.0, d) / 10.0) if np.isfinite(d) else 0.0
            sc = max(s, near)
            boxes.append(b); scores.append(sc); clss.append(-1); flags.append(False); dists.append(d)

    for j in range(len(yolo_boxes)):
        if not yolo_used[j]:
            near = 1.0 - min(1.0, max(0.0, yolo_dists[j]) / 10.0) if np.isfinite(yolo_dists[j]) else 0.0
            sc = 0.7 * yolo_scores[j] + 0.3 * near
            boxes.append(yolo_boxes[j]); scores.append(sc); clss.append(yolo_cls[j]); flags.append(True); dists.append(yolo_dists[j])

    if len(boxes) == 0:
        return (np.zeros((0, 4), dtype=float),
                np.zeros((0,), dtype=float),
                np.zeros((0,), dtype=int),
                np.zeros((0,), dtype=bool),
                np.zeros((0,), dtype=float))
    return (np.asarray(boxes, dtype=float),
            np.asarray(scores, dtype=float),
            np.asarray(clss, dtype=int),
            np.asarray(flags, dtype=bool),
            np.asarray(dists, dtype=float))
