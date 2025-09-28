# src/main.py
import os, sys
import numpy as np
import cv2
import torch

sys.path.append(os.path.abspath("."))

from src.config import load_config
from src.io.video_reader import VideoReader
from src.io.video_writer import VideoWriter
from src.viz.drawer import draw_boxes, draw_hud
from src.perception.detector_yolo import YOLODetector
from src.perception.segment_bisenet import BiSeNetSeg
from src.perception.depth_vda import DepthStreamVDA
from src.tracking.ocsort import OCSort
from src.fusion.ground_plane import GroundPlaneScaler
from src.fusion.depth_enhance import DepthEnhancer
from src.fusion.depth_utils import depth_from_yolo_box, depth_from_obs_box
from src.fusion.fuser import merge_yolo_obstacle, nms_iou
from src.fusion.depth_edges import detect_obstacles_by_depth_edges
from src.fusion.bev import BEVProjectorTorch, BEVConfig

def _stable_persist_key(b):
    x1,y1,x2,y2 = [float(v) for v in b]
    cx=(x1+x2)*0.5; cy=(y1+y2)*0.5; w=(x2-x1); h=(y2-y1)
    q=lambda v,s: round(v/s)*s
    return (q(cx,4.0), q(cy,4.0), q(w,4.0), q(h,4.0))

def _update_persistence(persist_map, boxes, dists, min_vote=3, decay=0.85, max_miss=8):
    seen = set()
    for i, b in enumerate(boxes):
        k = _stable_persist_key(b)
        st = persist_map.get(k, {"vote":0, "box":np.array(b, dtype=float), "z":float(dists[i]) if i<len(dists) else np.nan, "miss":0})
        st["vote"] = min(10, st["vote"] + 1)
        st["box"] = st["box"]*decay + np.array(b, dtype=float)*(1.0-decay)
        if i < len(dists) and np.isfinite(dists[i]):
            if not np.isfinite(st.get("z", np.nan)):
                st["z"] = float(dists[i])
            else:
                st["z"] = st["z"]*decay + float(dists[i])*(1.0-decay)
        st["miss"] = 0
        persist_map[k] = st
        seen.add(k)

    to_del=[]
    for k, st in persist_map.items():
        if k not in seen:
            st["miss"] = st.get("miss",0)+1
            st["vote"] = max(0, st["vote"] - 1)
            if st["miss"] > max_miss and st["vote"] == 0:
                to_del.append(k)
    for k in to_del:
        persist_map.pop(k, None)

    out_boxes=[]; out_dists=[]
    for k, st in persist_map.items():
        if st["vote"] >= min_vote and st["miss"] <= 1:
            out_boxes.append(st["box"])
            out_dists.append(st.get("z", np.nan))
    if len(out_boxes)==0:
        return np.zeros((0,4), dtype=float), np.zeros((0,), dtype=float)
    return np.vstack(out_boxes).astype(float), np.asarray(out_dists, dtype=float)

def process_video(src_path: str, out_path: str, cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    reader = VideoReader(src_path)
    outW, outH = cfg["io"]["output_size"]
    writer = VideoWriter(out_path, cfg["io"]["output_fps"], (outW, outH))

    # Models
    ycfg = cfg["models"]["yolo"]
    det = YOLODetector(ycfg["weights"], conf=ycfg["conf"], iou=ycfg["iou"],
                       half=ycfg["half"], whitelist=ycfg.get("whitelist_classes", None), device=device)

    scfg = cfg["models"]["segmentation"]
    seg_model = BiSeNetSeg(scfg["weights"], half=scfg["half"], device=device)
    run_seg_every = max(1, int(scfg["run_every_k"]))

    dcfg = cfg["models"]["depth"]
    dep = DepthStreamVDA(encoder=dcfg["encoder"], checkpoint=dcfg["checkpoint"],
                         input_size=dcfg["input_size"], device=device, fp32=dcfg["fp32"])

    # Fusion helpers
    pf = cfg["fusion"]["plane_fit"]
    scaler = GroundPlaneScaler(reader.W, reader.H, cfg["camera"]["fov_deg_h"], cfg["camera"]["principal"],
                               cam_h=cfg["camera"]["height_m"], sample_stride=pf.get("sample_stride",4),
                               ema_beta=pf.get("ema_beta",0.9), use_sidewalk=pf.get("use_sidewalk",True),
                               use_terrain=pf.get("use_terrain",True), method=pf.get("method","ransac"),
                               inlier_thr=pf.get("inlier_thr_m",0.06), max_iter=pf.get("max_iter",150))

    thr_low, thr_high = cfg["fusion"]["distance_thresholds_m"]
    ocfg  = cfg["fusion"]["obstacle"]
    ecfg  = cfg["fusion"].get("edge_enhance", {})
    eocfg = cfg["fusion"].get("edge_obstacle", {})
    bev_cfg = cfg["fusion"].get("bev", {})

    tracker = OCSort(max_age=cfg["tracking"]["max_age"], iou_thr=cfg["tracking"]["iou"])

    stride = max(1, int(round(reader.fps / float(cfg["io"]["output_fps"]))))
    last_seg = None
    frame_id = 0
    persist_obs = {}

    # Lazy init: enhancer + BEV projector
    if not hasattr(process_video, "_depth_enh"):
        from src.fusion.depth_enhance import DepthEnhancer
        process_video._depth_enh = DepthEnhancer(ecfg)
    enh = process_video._depth_enh

    bev_proj = None

    for packet in reader:
        frame_id += 1
        if (frame_id - 1) % stride != 0:
            continue

        rgb = packet.rgb  # RGB
        H, W = packet.H, packet.W

        # 1) Depth tương đối
        depth_rel = dep.infer_one(rgb).astype(np.float32)

        # 2) Seg (thưa khung)
        if (frame_id % run_seg_every) == 0 or last_seg is None:
            seg_id = seg_model(rgb); last_seg = seg_id
        else:
            seg_id = last_seg

        # 3) Ước lượng α, plane –> mét: Z = α / depth_rel
        alpha, plane = scaler.estimate_scale(depth_rel, seg_id)
        depth_m = alpha / (np.maximum(depth_rel, 1e-6))

        # 3b) Depth enhance (GPU trong DepthEnhancer)
        depth_m_enh, sigma_m = enh(depth_m, rgb)

        # 4) Height-map
        height_m = scaler.height_from_plane(depth_m, plane)

        # 5) BEV occupancy (bổ trợ, “thoáng tay”)
        if bev_proj is None:
            # build intrinsics (pixel principal point)
            fx, fy, cx, cy = scaler.fx, scaler.fy, scaler.cx, scaler.cy
            cfg_bev = BEVConfig(
                x_min = float(bev_cfg.get("x_min", -10.0)),
                x_max = float(bev_cfg.get("x_max",  10.0)),
                z_min = float(bev_cfg.get("z_min",   0.0)),
                z_max = float(bev_cfg.get("z_max",  40.0)),
                cell  = float(bev_cfg.get("cell",    0.20)),
                h_min_m = float(bev_cfg.get("h_min_m", 0.18))
            )
            bev_proj = BEVProjectorTorch(W, H, fx, fy, cx, cy, cfg_bev, device=device)

        occ_bev = bev_proj.splat(depth_m, seg_id, height_m)
        bev_proj.set_last_occ(occ_bev)  # để sample điểm

        # 6) Obstacle từ depth-edges (checkpoint)
        (obs_boxes, obs_dists, obs_scores), dbg = detect_obstacles_by_depth_edges(
            depth_m_enh, sigma_m, seg_id, eocfg, fx=scaler.fx, fy=scaler.fy
        )

        # 6b) Tăng/giảm score theo occupancy BEV gần (X,Z) của box — KHÔNG gate cứng
        if len(obs_boxes):
            bev_boost = bev_proj.occupancy_score_for_boxes(obs_boxes, obs_dists, weight_window=int(bev_cfg.get("win",2)))
            # normalize nhẹ (0..1) rồi cộng trọng số nhỏ
            lam = float(bev_cfg.get("boost_lambda", 0.20))
            obs_scores = obs_scores + lam * bev_boost.detach().cpu().numpy().astype(np.float32)

        # 7) YOLO + đo khoảng cách (q20 dải đáy)
        yolo_xyxy, yolo_scores, yolo_cls, _ = det(rgb, img_size=(640,384))
        yolo_dists = []
        for b in yolo_xyxy:
            d = depth_from_yolo_box(depth_m, b, frac=0.25)
            yolo_dists.append(d if np.isfinite(d) else np.nan)
        yolo_dists = np.asarray(yolo_dists, dtype=float)

        # Lọc xa > zmax
        zmax = float(ocfg.get("depth_max_m", 10.0))
        keep_y = [i for i in range(len(yolo_xyxy)) if (not np.isfinite(yolo_dists[i])) or (yolo_dists[i] <= zmax)]
        yolo_xyxy  = yolo_xyxy[keep_y] if len(keep_y) else np.zeros((0,4))
        yolo_scores= yolo_scores[keep_y] if len(keep_y) else np.zeros((0,))
        yolo_cls   = yolo_cls[keep_y] if len(keep_y) else np.zeros((0,), dtype=int)
        yolo_dists = yolo_dists[keep_y] if len(keep_y) else np.zeros((0,))

        # 8) Hợp nhất YOLO ↔ OBS (nhãn -1 cho OBS)
        boxes, scores, clss, flags, d_init = merge_yolo_obstacle(
            yolo_xyxy, yolo_scores, yolo_cls, yolo_dists,
            obs_boxes, obs_scores, obs_dists,
            iou_merge=float(ocfg.get("iou_merge_yolo", 0.5))
        )

        # 9) Re-measure cho OBS bằng ROI đáy masked
        if len(boxes):
            for i in range(len(boxes)):
                if clss[i] < 0:
                    d = depth_from_obs_box(depth_m, height_m, seg_id, boxes[i],
                                           foot_h=float(ocfg.get("foot_height_base_m",0.06)), frac=0.25)
                    if np.isfinite(d): d_init[i] = d

        # 10) NMS + Top-K (gần trước)
        if len(boxes):
            keep = nms_iou(boxes, scores, iou_thr=float(ocfg.get("nms_iou",0.55)))
            boxes = boxes[keep]; scores = scores[keep]; clss = clss[keep]; d_init = d_init[keep]
            order = np.argsort([ (np.inf if not np.isfinite(z) else z) for z in d_init ])
            boxes = boxes[order]; scores = scores[order]; clss = clss[order]; d_init = d_init[order]
            K = int(ocfg.get("topk",18))
            boxes = boxes[:K]; scores = scores[:K]; clss = clss[:K]; d_init = d_init[:K]

        # 11) Persistence cho OBS
        labels = []
        if len(boxes):
            obs_idx = [i for i in range(len(boxes)) if clss[i] < 0]
            if len(obs_idx):
                p_boxes, p_dists = _update_persistence(
                    persist_obs, boxes[obs_idx], d_init[obs_idx],
                    min_vote=int(ocfg.get("persistence_min",3))
                )
                keep_yolo_idx = [i for i in range(len(boxes)) if clss[i] >= 0]
                boxes_y = boxes[keep_yolo_idx]; dists_y = d_init[keep_yolo_idx]; clss_y = clss[keep_yolo_idx]
                if len(boxes_y):
                    labels.extend([str(det.names.get(int(c), str(int(c)))) for c in clss_y])
                if len(p_boxes):
                    boxes = np.vstack([boxes_y, p_boxes])
                    d_init = np.hstack([dists_y, p_dists])
                    clss = np.hstack([clss_y, np.full((len(p_boxes),), -1, dtype=int)])
                else:
                    boxes = boxes_y; d_init=dists_y; clss=clss_y
            else:
                labels = [str(det.names.get(int(c), str(int(c)))) if c>=0 else "OBS" for c in clss]

        # 12) Vẽ
        vis = draw_boxes(rgb, boxes, labels, d_init, thr=(thr_low,thr_high),
                         thickness=cfg["viz"]["thickness"], font_scale=cfg["viz"]["font_scale"])
        hud = [
            f"edge.num={len(obs_boxes)}  yolo={len(yolo_xyxy)}  merged={len(boxes)}",
        ]
        vis = draw_hud(vis, hud)
        vis = cv2.resize(vis, (outW,outH), interpolation=cv2.INTER_AREA)
        writer.write_rgb(vis)

    writer.release(); reader.release()

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
