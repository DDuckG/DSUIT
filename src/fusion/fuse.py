import json
import glob
import sys
import math
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
from collections import Counter
import tqdm

def parse_mot_txt(path):
    frames = {}
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split(",")
            if len(parts) < 9:
                continue
            frame = int(parts[0].strip())
            x = float(parts[2].strip())
            y = float(parts[3].strip())
            w = float(parts[4].strip())
            h = float(parts[5].strip())
            score = float(parts[6].strip())
            cls = int(float(parts[7].strip()))
            det = {"bbox":[x,y,w,h], "score": score, "class_id": cls, "det_id": None}
            frames.setdefault(frame, []).append(det)
    return frames

def find_depth_path_for_index(depth_dir, frame_idx, patterns = None):
    if not patterns:
        patterns = [
            "frame_{:05d}.npz",
            "frame_{:06d}.npz",
            "{}_depth_{:05d}.npz",
            "*depth_{:05d}.npz",
            "*_{:05d}.npz",
            "{:05d}.npz",
            "test_depth_{:05d}.npz",
        ]
    p = Path(depth_dir)
    for pat in patterns:
        if "{}" in pat:
            continue
        name = pat.format(frame_idx)
        candidate = p / name
        if candidate.exists():
            return str(candidate)
        for g in p.glob(pat.replace("{:05d}", f"{frame_idx:05d}")):
            return str(g)
    for g in p.glob("*.npz"):
        if f"{frame_idx:05d}" in g.name or f"{frame_idx:06d}" in g.name or f"_{frame_idx}." in g.name:
            return str(g)
    return None

def load_depth_npz(path):
    try:
        arr = np.load(path)['depth']
        return arr.astype(np.float32)
    except Exception as e:
        return None

def bbox_stats_from_depth(depth, bbox):
    H, W = depth.shape
    x, y, w, h = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(W, int(round(x + w)))
    y2 = min(H, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = depth[y1:y2, x1:x2]
    vals = patch.flatten()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    vals = vals[vals > 0] if np.any(vals > 0) else vals
    if vals.size == 0:
        return None
    stats = {
        "median": float(np.median(vals)),
        "mean": float(np.mean(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "count": int(vals.size)
    }
    return stats

def pixel_to_cam(u, v, Z, fx, fy, cx, cy):
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    return [float(X), float(Y), float(Z)]

def load_mask_if_available(masks_dir, frame_idx, pattern="mask_{:06d}.png"):
    p = Path(masks_dir)
    candidate = p / pattern.format(frame_idx)
    if candidate.exists():
        try:
            im = Image.open(candidate)
            arr = np.array(im)
            return arr
        except Exception:
            return None
    for g in p.glob("*.png"):
        if f"{frame_idx:06d}" in g.name or f"{frame_idx:05d}" in g.name:
            try:
                return np.array(Image.open(g))
            except:
                pass
    return None

def compute_seg_info(mask_arr, bbox, top_k=3):
    H, W = mask_arr.shape[:2]
    x,y,w,h = bbox
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(W, int(round(x+w)))
    y2 = min(H, int(round(y+h)))
    if x2 <= x1 or y2 <= y1:
        return {}
    patch = mask_arr[y1:y2, x1:x2].ravel()
    if patch.size == 0:
        return {}
    counter = Counter(patch.tolist())
    total = sum(counter.values())
    most = counter.most_common(top_k)
    items = [{"class_id": int(k), "count": int(v), "ratio": float(v)/total} for (k,v) in most]
    return {"top": items, "total": int(total)}

def main(args):
    PATH_DETECTIONS = Path(args.detections)
    PATH_DEPTH_DIR = Path(args.depth_dir)
    PATH_OUT = Path(args.out)
    PATH_OUT.parent.mkdir(parents=True, exist_ok=True)

    # metadata
    meta = {}
    if args.meta and Path(args.meta).exists():
        try:
            with open(args.meta, "r") as f:
                meta = json.load(f)
        except:
            meta = {}
    width = meta.get("width", args.width)
    height = meta.get("height", args.height)
    fx = meta.get("fx", args.fx)
    fy = meta.get("fy", args.fy)
    cx = meta.get("cx", args.cx)
    cy = meta.get("cy", args.cy)
    fps = meta.get("fps", args.fps)

    # parse detections
    dets = parse_mot_txt(PATH_DETECTIONS)

    frames_list = sorted(dets.keys())
    if args.frame_ids:
        frames_list = [int(x) for x in args.frame_ids.split(",")]

    with open(PATH_OUT, "w") as fout:
        for frame in tqdm.tqdm(frames_list, desc="Frames"):
            rec = {"frame_id": frame, "detections": [], "ts": None}
            depth_path = find_depth_path_for_index(PATH_DEPTH_DIR, frame)
            depth = None
            if depth_path:
                depth = load_depth_npz(depth_path)
            # optional timestamp
            if fps:
                rec["ts"] = (frame - 1) / float(fps)

            # optional mask
            mask_arr = None
            if args.masks_dir:
                mask_arr = load_mask_if_available(args.masks_dir, frame)

            for det in dets.get(frame, []):
                bbox = det["bbox"]
                stats = None
                centroid_cam = None
                centroid_img = None
                if depth is not None:
                    stats = bbox_stats_from_depth(depth, bbox)
                    if stats is not None:
                        # centroid pixel
                        x,y,w,h = bbox
                        u = x + w/2.0
                        v = y + h/2.0
                        centroid_img = [float(u), float(v)]
                        centroid_cam = pixel_to_cam(u, v, stats["median"], fx, fy, cx, cy)
                seg_info = None
                seg_class_at_centroid = None
                if mask_arr is not None:
                    seg_info = compute_seg_info(mask_arr, bbox, top_k=3)
                    if centroid_img is not None:
                        cu, cv = int(round(centroid_img[0])), int(round(centroid_img[1]))
                        if 0 <= cv < mask_arr.shape[0] and 0 <= cu < mask_arr.shape[1]:
                            seg_class_at_centroid = int(mask_arr[cv, cu])

                fused = {
                    "bbox": bbox,
                    "score": det.get("score"),
                    "class_id": det.get("class_id"),
                    "depth_stats": stats,
                    "centroid_img": centroid_img,
                    "centroid_cam": centroid_cam,
                    "seg_info": seg_info,
                    "seg_class_at_centroid": seg_class_at_centroid
                }
                rec["detections"].append(fused)
            fout.write(json.dumps(rec) + "\n")
    print("Wrote fusion ->", PATH_OUT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type = str, required = True, help = "MOT txt detections")
    parser.add_argument("--depth-dir", type = str, required = True, help = "Directory containing depth npz files")
    parser.add_argument("--out", type = str, required = True, help = "Output fusion.jsonl path")
    parser.add_argument("--meta", type = str, default = "", help = "Optional meta.json with fx,fy,cx,cy,width,height,fps")
    parser.add_argument("--masks-dir", type = str, default = "", help = "Optional segmentation masks dir (png per frame)")
    parser.add_argument("--frame-ids", type = str, default = "", help = "Optional comma-separated frame ids to process")
    parser.add_argument("--frame-offset", type = int, default = 0, help = "If your depth indexing starts at 0 but detections at 1, set offset")
    parser.add_argument("--width", type = int, default = 0)
    parser.add_argument("--height", type = int, default = 0)
    parser.add_argument("--fx", type = float, default = 470.4)
    parser.add_argument("--fy", type = float, default = 470.4)
    parser.add_argument("--cx", type = float, default = 0.0)
    parser.add_argument("--cy", type = float, default = 0.0)
    parser.add_argument("--fps", type = float, default = 0.0)

    arguments = parser.parse_args()
    if arguments.cx == 0.0 and arguments.width:
        arguments.cx = arguments.width / 2.0
    if arguments.cy == 0.0 and arguments.height:
        arguments.cy = arguments.height / 2.0

    main(arguments)