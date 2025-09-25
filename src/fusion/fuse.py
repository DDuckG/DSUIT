# src/fusion/fuse.py
"""
Fuse YOLO detections + tracks + depth + segmentation + pseudo-obstacle candidates.

- Đọc detects & tracks (MOT-like).
- Gọi CandidateManager (depth-first).
- Ước lượng distance theo depth robust percentile.
- Gộp thành fusion.jsonl.

Author: assistant
"""
import argparse, os, json, sys
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --- Bootstrap sys.path để import "fusion.candidates" khi chạy: python src/fusion/fuse.py
ROOT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_SRC not in sys.path:
    sys.path.append(ROOT_SRC)
# ---

from candidates import CandidateManager  # noqa: E402


def read_mot_txt(path: str):
    by = {}
    if not os.path.exists(path): return by
    with open(path, 'r', encoding='utf8') as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            p = line.replace(',', ' ').split()
            frame = int(float(p[0]))
            if len(p) >= 10:
                x = float(p[2]); y=float(p[3]); w=float(p[4]); h=float(p[5])
                score = float(p[6]) if len(p)>6 else 0.0
                cls = int(float(p[9])) if len(p)>=10 else -1
                by.setdefault(frame, []).append(('det', (x,y,w,h,score,cls)))
            else:
                tid=int(float(p[1])); x=float(p[2]); y=float(p[3]); w=float(p[4]); h=float(p[5])
                score=float(p[6]) if len(p)>6 else 0.0
                by.setdefault(frame, []).append(('trk', (tid,x,y,w,h,score)))
    return by


def write_jsonl(path: str, entries: List[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def iou_xyxy(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    xx1=max(ax1,bx1); yy1=max(ay1,by1)
    xx2=min(ax2,bx2); yy2=min(ay2,by2)
    w=max(0.0,xx2-xx1); h=max(0.0,yy2-yy1)
    inter=w*h
    ua=max(0.0,ax2-ax1)*max(0.0,ay2-ay1)
    ub=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    u=ua+ub-inter+1e-9
    return inter/u if u>0 else 0.0


def xywh_to_xyxy(b):
    x,y,w,h=b; return (x,y,x+w,y+h)


def robust_box_depth(depth_map: np.ndarray, box_xyxy, percentile=0.4):
    x1,y1,x2,y2 = [int(max(0, v)) for v in box_xyxy]
    H,W = depth_map.shape[:2]
    x1=min(x1,W-1); x2=min(x2,W-1); y1=min(y1,H-1); y2=min(y2,H-1)
    if x2<=x1 or y2<=y1: return float('nan')
    patch = depth_map[y1:y2, x1:x2].reshape(-1)
    patch = patch[np.isfinite(patch)]
    if patch.size==0: return float('nan')
    return float(np.percentile(patch, percentile*100.0))


def relative_to_metric(depth_value: float, frame_depth: Optional[np.ndarray] = None):
    if frame_depth is None: return float(depth_value)
    finite = np.isfinite(frame_depth)
    med = float(np.nanmedian(frame_depth[finite])) if finite.any() else 1.0
    if not np.isfinite(med) or med<=0: med = 1.0
    depth_value = max(depth_value, 1e-6)
    meters = 3.0 * med / depth_value
    return float(np.clip(meters, 0.2, 80.0))


def read_seg_jsonl(path: Optional[str]):
    frames={}
    if not path or not os.path.exists(path): return frames
    with open(path,'r',encoding='utf8') as f:
        for line in f:
            js=json.loads(line)
            fid = int(js.get('frame_id', js.get('frame', 0)))
            frames[fid]=js
    return frames


def find_depth(depth_dir: str, frame_idx: int):
    p = Path(depth_dir) / f"depth_{frame_idx:05d}.npz"
    if not p.exists(): return None
    arr=np.load(str(p)); k=arr.files[0]; return arr[k]


def fuse(detects:str, tracks:str, depth_dir:str, out_path:str,
         segments: Optional[str]=None, video: Optional[str]=None,
         fps: Optional[float]=None):
    det = read_mot_txt(detects)
    trk = read_mot_txt(tracks)
    seg = read_seg_jsonl(segments) if segments else {}

    # metadata
    W=H=None
    if video and os.path.exists(video):
        cap=cv2.VideoCapture(video)
        if cap.isOpened():
            W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps is None: fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
    fps=float(fps or 0.0)

    frames = sorted(set(det.keys()) | set(trk.keys()) | set(seg.keys()))
    if not frames:
        print("[FUSE] No frames."); return

    if (W is None or H is None) and seg:
        anyf = next(iter(seg.values()))
        H = int(anyf.get('height',0)); W=int(anyf.get('width',0))
    image_area = int((W or 1)*(H or 1))

    cmgr = CandidateManager()

    out=[]
    for f in frames:
        det_list = [v for (t,v) in det.get(f,[]) if t=='det']
        tr_list  = [v for (t,v) in trk.get(f,[]) if t=='trk']
        depth = find_depth(depth_dir, f)
        seg_entry = seg.get(f, None)

        # pseudo obstacles
        pseudo_dets, pseudo_rows = cmgr.step_frame(f, seg_entry, depth, det_list, image_area)
        # add pseudo dets to list so có thể gán class khi match track
        for (x,y,w,h,score,cls,tid) in pseudo_dets:
            det_list.append((x,y,w,h,score,cls))
            # cũng thêm track giả để downstream thấy
            tr_list.append((tid,x,y,w,h,score))

        det_boxes=[xywh_to_xyxy((x,y,w,h)) for (x,y,w,h,_,_) in det_list]
        det_scores=[s for (_,_,_,_,s,_) in det_list]
        det_cls=[int(c) for (_,_,_,_,_,c) in det_list]

        objects=[]
        for tr in tr_list:
            if len(tr)==6:
                tid,x,y,w,h,ts=tr
            else:
                tid=int(tr[0]); x=float(tr[1]); y=float(tr[2]); w=float(tr[3]); h=float(tr[4]); ts=float(tr[5]) if len(tr)>5 else 0.0
            x1,y1,x2,y2 = x,y,x+w,y+h
            # match tới det class
            best_i, best_idx = 0.0, -1
            for i,db in enumerate(det_boxes):
                iou = iou_xyxy((x1,y1,x2,y2), db)
                if iou > best_i:
                    best_i, best_idx = iou, i
            cls = det_cls[best_idx] if best_i>=0.5 and best_idx>=0 else -1
            score = det_scores[best_idx] if best_idx>=0 else ts

            # depth distance
            dist_rel = float('nan')
            if depth is not None:
                try:
                    dist_rel = robust_box_depth(depth, (x1,y1,x2,y2))
                except Exception:
                    dist_rel = float('nan')
            dist_m = relative_to_metric(dist_rel, depth) if np.isfinite(dist_rel) else None

            objects.append({
                "track_id": int(tid),
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "class_id": int(cls),
                "class_name": ("obstacle" if cls==999 else None),
                "score": float(score),
                "depth_rel": float(dist_rel) if np.isfinite(dist_rel) else None,
                "distance_m": float(dist_m) if dist_m is not None and np.isfinite(dist_m) else None
            })
        t_s = (f / fps) if fps>0 else None
        out.append({"frame": int(f), "time_s": t_s, "objects": objects})

    write_jsonl(out_path, out)
    print(f"[FUSE] wrote {out_path} (frames={len(out)})")


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument('--detects', required=True)
    a.add_argument('--tracks', required=True)
    a.add_argument('--depth-dir', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--segments', default=None)
    a.add_argument('--video', default=None)
    a.add_argument('--fps', type=float, default=None)
    args = a.parse_args()
    fuse(args.detects, args.tracks, args.depth_dir, args.out,
         segments=args.segments, video=args.video, fps=args.fps)
