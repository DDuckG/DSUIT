import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

def read_mot_txt(path: str):
    by_frame = {}
    with open(path, 'r', encoding = 'utf8') as file:
        for raw in file:
            line = raw.strip()
            parts = [p.strip() for p in line.replace(',', ' ').split()]
            frame = int(float(parts[0]))
            if len(parts) >= 10:
                x = float(parts[2]); y = float(parts[3]); w = float(parts[4]); h = float(parts[5])
                score = float(parts[6])
                cls = int(float(parts[9])) if len(parts) >= 10 else -1
                by_frame.setdefault(frame, []).append(('det', (x,y,w,h,score,cls)))
            else:
                tid = int(float(parts[1]))
                x = float(parts[2]); y = float(parts[3]); w = float(parts[4]); h = float(parts[5])
                score = float(parts[6]) if len(parts) > 6 else None
                by_frame.setdefault(frame, []).append(('trk', (tid,x,y,w,h,score)))
    return by_frame

def iou_xyxy(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    xx1 = max(ax1, bx1) 
    yy1 = max(ay1, by1)
    xx2 = min(ax2, bx2) 
    yy2 = min(ay2, by2)
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    areaA = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    areaB = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = areaA + areaB - inter + 0.000000001
    return inter / union if union > 0 else 0.0

def xywh_to_xyxy(box):
    x, y, w, h = box
    return (x, y, x+w, y+h)

def find_depth_for_frame(depth_dir: str, frame_idx: int):
    path = Path(depth_dir) / f"depth_{frame_idx:05d}.npz"
    arr = np.load(path)
    k = arr.files[0]
    return arr[k]

def robust_box_depth(depth_map: np.ndarray, box_xyxy, percentile = 0.4):
    x1, y1, x2, y2 = [max(0, v) for v in box_xyxy]
    H, W = depth_map.shape[:2]
    x1, x2 = int(min(x1, W - 1)), int(min(max(x2, 0), W - 1))
    y1, y2 = int(min(y1, H - 1)), int(min(max(y2, 0), H - 1))
    patch = depth_map[y1 : y2, x1 : x2].reshape(-1)
    patch = patch[np.isfinite(patch)]
    return float(np.percentile(patch, percentile * 100.0))

def relative_to_metric(depth_value: float, frame_depth: Optional[np.ndarray] = None):
    median_scene = float(np.nanmedian(frame_depth))
    scale = 3.0 if (np.isfinite(median_scene) and median_scene > 0) else 1.0
    depth_value = max(depth_value, 0.000001)
    meters = scale * median_scene / depth_value
    return float(np.clip(meters, 0.2, 150.0))

def write_jsonl(path: str, entries: List[dict]):
    Path(path).parent.mkdir(parents = True, exist_ok = True)
    with open(path, 'w', encoding = 'utf8') as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii = False) + "\n")

def fuse(dets_path: str, tracks_path: str, depth_dir: str, out_path: str, fps: Optional[float] = None, video_path: Optional[str] = None, iou_thresh: float = 0.5):
    detections = read_mot_txt(dets_path)
    tracks = read_mot_txt(tracks_path)

    if fps is None and video_path:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
    fps = float(fps or 0.0)

    frames = sorted(set(list(detections.keys()) + list(tracks.keys())))
    out_entries = []
    for frame in frames:
        detect_list = [v for (t,v) in detections.get(frame,[]) if t == 'det']
        track_list = [v for (t,v) in tracks.get(frame,[]) if t == 'trk']

        det_boxes = [xywh_to_xyxy((x, y, w, h)) for (x, y, w, h, score, cid) in detect_list]
        det_scores = [score for (x, y, w, h, score, cid) in detect_list]
        det_cls = [int(cid) for (x, y, w, h, score, cid) in detect_list]

        frame_depth = None
        objects = []
        for track in track_list:
            tid, x, y, w, h, score_tr = track
            x1, y1, x2, y2 = x, y, x+w, y+h
            best_iou = 0.0
            best_idx = -1
            for idx, db in enumerate(det_boxes):
                i = iou_xyxy((x1,y1,x2,y2), db)
                if i > best_iou:
                    best_iou = i
                    best_idx = idx
            if best_iou >= iou_thresh and best_idx >= 0:
                cls_id = det_cls[best_idx]
                det_score = det_scores[best_idx]
            else:
                cls_id = -1
                det_score = score_tr or 0.0
            if frame_depth is None:
                frame_depth = find_depth_for_frame(depth_dir, frame)
            depth_rel = robust_box_depth(frame_depth, (x1,y1,x2,y2))
            dist_m = relative_to_metric(depth_rel, frame_depth)
            objects.append({
                'track_id': int(tid),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'class_id': int(cls_id),
                'score': float(det_score),
                'depth_rel': float(depth_rel),
                'distance_m': float(dist_m)
            })
        time_s = float(frame) / fps
        out_entries.append({'frame': int(frame), 'time_s': time_s, 'objects': objects})
    write_jsonl(out_path, out_entries)
    print(f"Wrote fusion jsonl -> {out_path} (frames: {len(out_entries)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detects', required = True)
    parser.add_argument('--tracks', required = True)
    parser.add_argument('--depth-dir', required = True)
    parser.add_argument('--out', required = True)
    parser.add_argument('--video', default = None)
    parser.add_argument('--fps', type = float, default = None)
    parser.add_argument('--iou', type = float, default = 0.5)
    argument = parser.parse_args()
    fuse(argument.detects, argument.tracks, argument.depth_dir, argument.out, fps = argument.fps, video_path = argument.video, iou_thresh = argument.iou)