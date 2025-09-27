# src/main.py
import os, sys
import numpy as np
import cv2
import torch

sys.path.append(os.path.abspath("."))

from src.config import load_config
from src.io.video_reader import VideoReader
from src.io.video_writer import VideoWriter
from src.viz.drawer import draw_boxes
from src.perception.detector_yolo import YOLODetector
from src.perception.segment_bisenet import BiSeNetSeg
from src.perception.depth_vda import DepthStreamVDA
from src.tracking.ocsort import OCSort
from src.fusion.ground_plane import GroundPlaneScaler
from src.fusion.depth_utils import roi_bottom, robust_depth_stat
from src.fusion.obstacle_general import detect_obstacles_by_height
from src.fusion.fuser import merge_yolo_obstacle, nms_iou

def _iou_with_all(a, arr):
    if len(arr) == 0: return np.array([])
    a = np.array(a, dtype=float); arr = np.array(arr, dtype=float)
    x1 = np.maximum(a[0], arr[:,0])
    y1 = np.maximum(a[1], arr[:,1])
    x2 = np.minimum(a[2], arr[:,2])
    y2 = np.minimum(a[3], arr[:,3])
    inter = np.maximum(0.0, x2-x1)*np.maximum(0.0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (arr[:,2]-arr[:,0])*(arr[:,3]-arr[:,1]) - inter + 1e-9
    return inter/ua

def _stable_persist_key(b):
    """Khoá ổn định theo (cx, cy, w, h) lượng tử hoá → giảm rung."""
    x1, y1, x2, y2 = [float(v) for v in b]
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w  = (x2 - x1)
    h  = (y2 - y1)
    q = lambda v, s: round(v / s) * s
    # lượng tử theo 4px → bền hơn giữa các khung
    return (q(cx, 4.0), q(cy, 4.0), q(w, 4.0), q(h, 4.0))

def process_video(src_path: str, out_path: str, cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    reader = VideoReader(src_path)
    outW, outH = cfg["io"]["output_size"]
    writer = VideoWriter(out_path, cfg["io"]["output_fps"], (outW, outH))

    # Models
    ycfg = cfg["models"]["yolo"]
    det = YOLODetector(
        ycfg["weights"], conf=ycfg["conf"], iou=ycfg["iou"], half=ycfg["half"],
        whitelist=ycfg.get("whitelist_classes", None), device=device
    )

    scfg = cfg["models"]["segmentation"]
    seg_model = BiSeNetSeg(scfg["weights"], half=scfg["half"], device=device)
    run_seg_every = max(1, int(scfg["run_every_k"]))

    dcfg = cfg["models"]["depth"]
    dep = DepthStreamVDA(
        encoder=dcfg["encoder"], checkpoint=dcfg["checkpoint"],
        input_size=dcfg["input_size"], device=device, fp32=dcfg["fp32"]
    )

    # Fusion helpers
    pf = cfg["fusion"]["plane_fit"]
    scaler = GroundPlaneScaler(
        reader.W, reader.H, cfg["camera"]["fov_deg_h"], cfg["camera"]["principal"],
        cam_h=cfg["camera"]["height_m"],
        sample_stride=pf.get("sample_stride", 4),
        ema_beta=pf.get("ema_beta", 0.9),
        use_sidewalk=pf.get("use_sidewalk", True),
        use_terrain=pf.get("use_terrain", True),
        method=pf.get("method", "ransac"),
        inlier_thr=pf.get("inlier_thr_m", 0.05),
        max_iter=pf.get("max_iter", 150),
    )

    thr_low, thr_high = cfg["fusion"]["distance_thresholds_m"]
    ocfg = cfg["fusion"]["obstacle"]

    # Tracker (gán ID ổn định cho các box YOLO)
    tracker = OCSort(max_age=cfg["tracking"]["max_age"], iou_thr=cfg["tracking"]["iou"])

    # stride để đạt output_fps
    stride = max(1, int(round(reader.fps / float(cfg["io"]["output_fps"]))))
    last_seg = None
    frame_id = 0
    persist_map = {}      # key -> count

    for packet in reader:
        frame_id += 1
        if (frame_id - 1) % stride != 0:
            continue

        rgb = packet.rgb
        H, W = packet.H, packet.W

        # 1) DEPTH
        depth_rel = dep.infer_one(rgb)  # HxW float32

        # 2) SEG (thưa khung)
        if (frame_id % run_seg_every) == 0 or last_seg is None:
            seg_id = seg_model(rgb)
            last_seg = seg_id
        else:
            seg_id = last_seg

        # 3) Plane scale (EMA) -> metric depth
        alpha, plane = scaler.estimate_scale(depth_rel, seg_id)
        depth_m = depth_rel * float(alpha)

        # 4) Obstacle từ height-map (toàn khung)
        obs_boxes, obs_dists, obs_scores = detect_obstacles_by_height(
            scaler, depth_m, plane, seg_id, ocfg
        )

        # 5) YOLO + gán khoảng cách theo dải đáy
        yolo_xyxy, yolo_scores, yolo_cls, _ = det(rgb, img_size=(640, 384))
        yolo_dists = []
        for b in yolo_xyxy:
            d_roi = roi_bottom(depth_m, b, frac=0.35)
            med, q25 = robust_depth_stat(d_roi, min_valid=20)
            d = float(q25 if np.isfinite(q25) else med)
            yolo_dists.append(d if np.isfinite(d) else np.nan)
        yolo_dists = np.asarray(yolo_dists, dtype=float)

        # filter xa > depth_max_m
        zmax = float(ocfg.get("depth_max_m", 10.0))
        keep_yolo = [i for i in range(len(yolo_xyxy)) if np.isfinite(yolo_dists[i]) and yolo_dists[i] <= zmax]
        yolo_xyxy = yolo_xyxy[keep_yolo] if len(keep_yolo) else np.zeros((0,4))
        yolo_scores = yolo_scores[keep_yolo] if len(keep_yolo) else np.zeros((0,))
        yolo_cls = yolo_cls[keep_yolo] if len(keep_yolo) else np.zeros((0,), dtype=int)
        yolo_dists = yolo_dists[keep_yolo] if len(keep_yolo) else np.zeros((0,))

        # 6) Hợp nhất YOLO ↔ obstacle
        boxes, scores, clss, flags, d_init = merge_yolo_obstacle(
            yolo_xyxy, yolo_scores, yolo_cls, yolo_dists,
            obs_boxes, obs_scores, obs_dists,
            iou_merge=float(ocfg.get("iou_merge_yolo", 0.5))
        )

        # 7) NMS + lọc xa
        if len(boxes) > 0:
            keep = [i for i in range(len(boxes)) if np.isfinite(d_init[i]) and d_init[i] <= zmax]
            boxes = boxes[keep]; scores = scores[keep]; clss = clss[keep]; flags = flags[keep]; d_init = d_init[keep]

        keep = nms_iou(boxes, scores, iou_thr=float(ocfg.get("nms_iou", 0.55)))
        boxes = boxes[keep]; scores = scores[keep]; clss = clss[keep]; flags = flags[keep]; d_init = d_init[keep]

        # Top-K gần nhất
        if len(boxes) > 0:
            order = np.argsort(d_init)  # gần trước
            boxes = boxes[order]; scores = scores[order]; clss = clss[order]; flags = flags[order]; d_init = d_init[order]
            K = int(ocfg.get("topk", 15))
            boxes = boxes[:K]; scores = scores[:K]; clss = clss[:K]; flags = flags[:K]; d_init = d_init[:K]

        # 8) Tracking cho YOLO (ID)
        tracks = tracker.update(yolo_xyxy.tolist(), yolo_scores.tolist(), yolo_cls.tolist())

        # 9) Nhãn hiển thị
        labels = []
        for i, b in enumerate(boxes):
            if clss[i] < 0:
                label = "obstacle"
            else:
                label = str(det.names.get(int(clss[i]), str(int(clss[i]))))
            labels.append(label)

        # 10) Persistence filter (ổn định hơn)
        persisted_idx = []
        pmin = int(ocfg.get("persistence_min", 2))
        if pmin <= 1:
            persisted_idx = list(range(len(boxes)))
        else:
            for i, b in enumerate(boxes):
                key = _stable_persist_key(b)
                persist_map[key] = persist_map.get(key, 0) + 1
                if persist_map[key] >= pmin:
                    persisted_idx.append(i)

        # Nếu vì lý do nào đó không có box sau persistence → để nguyên (tránh khung trống mãi)
        if len(persisted_idx) == 0:
            persisted_idx = list(range(len(boxes)))

        boxes = boxes[persisted_idx]
        labels = [labels[i] for i in persisted_idx]
        d_init = d_init[persisted_idx]

        # 11) Vẽ & ghi
        vis = draw_boxes(rgb, boxes, labels, d_init, thr=(thr_low, thr_high),
                         thickness=cfg["viz"]["thickness"], font_scale=cfg["viz"]["font_scale"])
        vis = cv2.resize(vis, (outW, outH), interpolation=cv2.INTER_AREA)
        writer.write_rgb(vis)

    writer.release()
    reader.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    process_video(args.src, args.out, cfg)
