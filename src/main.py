# src/main.py
import os, sys, gc
import numpy as np
import cv2
import torch

sys.path.append(os.path.abspath("."))

from src.config import load_config
from src.io.async_video import AsyncVideoReader, AsyncVideoWriter
from src.viz.drawer import draw_boxes, draw_hud
from src.perception.detector_yolo import YOLODetector
from src.perception.segment_bisenet import BiSeNetSeg
from src.perception.depth_vda import DepthStreamVDA
from src.tracking.ocsort import OCSort
from src.fusion.ground_plane import GroundPlaneScaler
from src.fusion.depth_enhance import DepthEnhancer
from src.fusion.fuser import merge_yolo_obstacle, nms_iou
from src.fusion.depth_edges import detect_obstacles_by_depth_edges
from src.fusion.bev import BEVProjectorTorch, BEVConfig
from src.fusion.obs_sliver import prune_obs_slivers

@torch.inference_mode()
def _warmup_pipeline(det, seg_model, dep, scaler, enh, W, H, y_imgsz):
    rgb0 = np.zeros((H, W, 3), dtype=np.uint8)
    depth_rel = dep.infer_one(rgb0).astype(np.float32)
    seg_id = seg_model(rgb0)
    alpha, plane = scaler.estimate_scale(depth_rel, seg_id)
    depth_m = alpha / (np.maximum(depth_rel, 1e-6))
    depth_m_enh, sigma_m = enh(depth_m, rgb0)
    _ = detect_obstacles_by_depth_edges(
        depth_m_enh, sigma_m, seg_id, {},
        fx=scaler.fx, fy=scaler.fy,
        height_m=scaler.height_from_plane(depth_m, plane)
    )
    _ = det(rgb0, img_size=y_imgsz)

def _iou_matrix_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    Ax1, Ay1, Ax2, Ay2 = A[:,0], A[:,1], A[:,2], A[:,3]
    Bx1, By1, Bx2, By2 = B[:,0], B[:,1], B[:,2], B[:,3]
    iw = np.maximum(0.0, np.minimum(Ax2[:,None], Bx2[None,:]) - np.maximum(Ax1[:,None], Bx1[None,:]))
    ih = np.maximum(0.0, np.minimum(Ay2[:,None], By2[None,:]) - np.maximum(Ay1[:,None], By1[None,:]))
    inter = iw * ih
    areaA = np.maximum(0.0, (Ax2 - Ax1)) * np.maximum(0.0, (Ay2 - Ay1))
    areaB = np.maximum(0.0, (Bx2 - Bx1)) * np.maximum(0.0, (By2 - By1))
    union = areaA[:,None] + areaB[None,:] - inter + 1e-9
    return (inter / union).astype(np.float32)

@torch.inference_mode()
def process_video(src_path: str, out_path: str, cfg: dict):
    # Hiệu năng CPU/GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    threads = int(cfg["io"].get("cpu_threads", 0))
    if threads <= 0:
        threads = os.cpu_count() or 8
    torch.set_num_threads(int(threads))

    # IO bất đồng bộ
    prefetch = int(cfg["io"].get("prefetch", 8))
    write_q = int(cfg["io"].get("write_queue", 64))
    reader = AsyncVideoReader(src_path, prefetch=prefetch)
    outW, outH = cfg["io"]["output_size"]
    writer = AsyncVideoWriter(out_path, cfg["io"]["output_fps"], (outW, outH), queue_size=write_q)

    # Models
    ycfg = cfg["models"]["yolo"]
    det = YOLODetector(
        ycfg["weights"], conf=ycfg["conf"], iou=ycfg["iou"],
        half=ycfg["half"], whitelist=ycfg.get("whitelist_classes", None), device=device
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
        cam_h=cfg["camera"]["height_m"], sample_stride=pf.get("sample_stride",4),
        ema_beta=pf.get("ema_beta",0.9), use_sidewalk=pf.get("use_sidewalk",True),
        use_terrain=pf.get("use_terrain",True), method=pf.get("method","ransac"),
        inlier_thr=pf.get("inlier_thr_m",0.06), max_iter=pf.get("max_iter",150)
    )

    thr_low, thr_high = cfg["fusion"]["distance_thresholds_m"]
    ocfg  = cfg["fusion"]["obstacle"]
    ecfg  = cfg["fusion"].get("edge_enhance", {})
    eocfg = cfg["fusion"].get("edge_obstacle", {})
    bev_cfg = cfg["fusion"].get("bev", {})
    slcfg = cfg["fusion"].get("obs_sliver", {})

    # --- tham số cho distance vẽ bằng depth_rel + median-ratio ---
    mcfg  = cfg["fusion"].get("depth_metric", {})
    dpcfg = cfg["fusion"].get("depth_percentile", {})
    scale_k     = float(mcfg.get("scale_k", 3.0))
    median_region = str(mcfg.get("median_region", "bottom"))   # bottom|full
    region_frac = float(mcfg.get("region_frac", 0.5))          # vùng đáy %
    clip_min_m  = float(mcfg.get("clip_min_m", 0.2))
    clip_max_m  = float(mcfg.get("clip_max_m", 80.0))
    yolo_q      = float(dpcfg.get("yolo_q", 0.40))
    obs_q       = float(dpcfg.get("obs_q", 0.35))
    foot_band_px= int(dpcfg.get("foot_band_px", 12))

    # Tracker
    tracker = OCSort(max_age=cfg["tracking"]["max_age"], iou_thr=cfg["tracking"]["iou"])

    # Warm-up
    y_imgsz = tuple(ycfg.get("imgsz", (640,384)))
    enh = DepthEnhancer(ecfg)
    _warmup_pipeline(det, seg_model, dep, scaler, enh, reader.W, reader.H, y_imgsz)
    print("[READY] Pipeline warmed up. Start streaming frames...")

    bev_proj = None
    stride = max(1, int(round(reader.fps / float(cfg["io"]["output_fps"]))))
    last_seg = None
    frame_id = 0
    mem_gc_every = int(cfg["io"].get("mem_gc_every", 0))

    for packet in reader:
        frame_id += 1
        if (frame_id - 1) % stride != 0:
            continue

        rgb = packet.rgb
        H, W = packet.H, packet.W

        # Depth tương đối
        depth_rel = dep.infer_one(rgb).astype(np.float32)

        # Seg thưa khung
        if (frame_id % run_seg_every) == 0 or last_seg is None:
            seg_id = seg_model(rgb); last_seg = seg_id
        else:
            seg_id = last_seg

        # Scale & plane
        alpha, plane = scaler.estimate_scale(depth_rel, seg_id)
        depth_m = alpha / (np.maximum(depth_rel, 1e-6))

        # Enhance
        depth_m_enh, sigma_m = enh(depth_m, rgb)

        # Height-map
        height_m = scaler.height_from_plane(depth_m, plane)

        # BEV occupancy (cache projector)
        if bev_proj is None:
            fx, fy, cx, cy = scaler.fx, scaler.fy, scaler.cx, scaler.cy
            cfg_bev = BEVConfig(
                x_min=float(bev_cfg.get("x_min", -10.0)),
                x_max=float(bev_cfg.get("x_max",  10.0)),
                z_min=float(bev_cfg.get("z_min",   0.0)),
                z_max=float(bev_cfg.get("z_max",  40.0)),
                cell=float(bev_cfg.get("cell",    0.20)),
                h_min_m=float(bev_cfg.get("h_min_m", 0.18))
            )
            bev_proj = BEVProjectorTorch(W, H, fx, fy, cx, cy, cfg_bev, device=device)

        occ_bev = bev_proj.splat(depth_m, seg_id, height_m)
        bev_proj.set_last_occ(occ_bev)

        # OBS từ depth-edges (checkpoint Y)
        (obs_boxes_0, obs_dists_raw_0, obs_scores_0), dbg = detect_obstacles_by_depth_edges(
            depth_m_enh, sigma_m, seg_id, eocfg, fx=scaler.fx, fy=scaler.fy, height_m=height_m
        )
        obs_boxes = obs_boxes_0.astype(np.float32)
        obs_scores = obs_scores_0.astype(np.float32)
        obs_dists_raw = obs_dists_raw_0.astype(np.float32)

        # YOLO
        yolo_xyxy, yolo_scores, yolo_cls, _ = det(rgb, img_size=y_imgsz)
        if yolo_xyxy.numel() > 0:
            yolo_xyxy_cpu = yolo_xyxy.detach().cpu().numpy().astype(np.float32)
            yolo_scores_np = yolo_scores.detach().cpu().numpy().astype(np.float32)
            yolo_cls_np    = yolo_cls.detach().cpu().numpy().astype(np.int32)
        else:
            yolo_xyxy_cpu = np.zeros((0,4), dtype=np.float32)
            yolo_scores_np = np.zeros((0,), dtype=np.float32)
            yolo_cls_np    = np.zeros((0,), dtype=np.int32)

        # Lọc YOLO theo zmax (giữ nguyên logic cũ: dùng dists cũ nếu cần)
        # Tính depth “cũ” cho YOLO để zmax filter giữ ổn định hành vi
        if yolo_xyxy_cpu.shape[0] > 0:
            # sử dụng depth metric (cũ) cho filter xa, giữ hành vi cũ
            # (nếu bạn muốn, có thể tắt filter này trong config bằng depth_max_m=1e9)
            from src.fusion.depth_utils import depth_from_yolo_box
            yolo_dists_filter = np.asarray(
                [depth_from_yolo_box(depth_m, b, frac=0.25) for b in yolo_xyxy_cpu], dtype=np.float32
            )
        else:
            yolo_dists_filter = np.zeros((0,), dtype=np.float32)

        zmax = float(ocfg.get("depth_max_m", 10.0))
        if yolo_xyxy_cpu.shape[0] > 0:
            keep_y = [i for i in range(yolo_xyxy_cpu.shape[0])
                      if (not np.isfinite(yolo_dists_filter[i])) or (yolo_dists_filter[i] <= zmax)]
            if len(keep_y) > 0:
                yolo_xyxy_cpu = yolo_xyxy_cpu[keep_y]
                yolo_scores_np = yolo_scores_np[keep_y]
                yolo_cls_np    = yolo_cls_np[keep_y]
            else:
                yolo_xyxy_cpu = np.zeros((0,4), dtype=np.float32)
                yolo_scores_np = np.zeros((0,), dtype=np.float32)
                yolo_cls_np    = np.zeros((0,), dtype=np.int32)

        # SLIVER + YOLO-PROTECT: giữ nguyên
        if len(obs_boxes) > 0:
            obs_boxes, obs_scores, obs_dists_raw = prune_obs_slivers(
                boxes_xyxy=obs_boxes,
                scores=obs_scores,
                dists_m=obs_dists_raw,
                yolo_boxes=yolo_xyxy_cpu,
                cfg=slcfg
            )

        # BEV boost nhẹ cho OBS còn lại
        if len(obs_boxes) > 0:
            bev_boost = bev_proj.occupancy_score_for_boxes(
                obs_boxes, obs_dists_raw, weight_window=int(bev_cfg.get("win",2))
            )
            lam = float(bev_cfg.get("boost_lambda", 0.20))
            obs_scores = obs_scores + bev_boost.detach().cpu().numpy().astype(np.float32) * lam

        # Dists “cũ” cho merge (giữ nguyên nhịp pipeline, nhưng sẽ bị override khi vẽ)
        from src.fusion.depth_utils import depth_from_obs_box
        obs_dists_merge = np.asarray([
            depth_from_obs_box(depth_m, height_m, seg_id, b,
                               foot_h=float(ocfg.get("foot_height_base_m",0.06)), frac=0.25)
            for b in obs_boxes
        ], dtype=np.float32) if len(obs_boxes) else np.zeros((0,), dtype=np.float32)
        yolo_dists_merge = yolo_dists_filter  # dùng cùng bộ dists với filter cho nhất quán merge

        # Merge YOLO ↔ OBS (nhãn -1 cho OBS)
        boxes, scores, clss, flags, d_init = merge_yolo_obstacle(
            yolo_xyxy_cpu,
            yolo_scores_np,
            yolo_cls_np,
            yolo_dists_merge,
            obs_boxes, obs_scores, obs_dists_merge,
            iou_merge=float(ocfg.get("iou_merge_yolo", 0.5))
        )

        # NMS + Top-K (giữ nguyên)
        if len(boxes):
            keep = nms_iou(boxes, scores, iou_thr=float(ocfg.get("nms_iou",0.55)))
            boxes = boxes[keep]; scores = scores[keep]; clss = clss[keep]; d_init = d_init[keep]
            order = np.argsort([(np.inf if not np.isfinite(z) else z) for z in d_init])
            boxes = boxes[order]; scores = scores[order]; clss = clss[order]; d_init = d_init[order]
            K = int(ocfg.get("topk",18))
            boxes = boxes[:K]; scores = scores[:K]; clss = clss[:K]; d_init = d_init[:K]
        else:
            boxes = np.zeros((0,4), dtype=np.float32)
            scores= np.zeros((0,), dtype=np.float32)
            clss  = np.zeros((0,), dtype=np.int32)
            d_init= np.zeros((0,), dtype=np.float32)

        # ================================
        # OVERRIDE DISTANCE ĐỂ VẼ (depth_rel + median-ratio)
        # ================================
        if boxes.shape[0] > 0:
            # median của scene (ưu tiên vùng đáy) trên depth_rel
            mask = np.isfinite(depth_rel)
            if isinstance(seg_id, np.ndarray) and seg_id.shape[:2] == depth_rel.shape:
                mask &= (seg_id != 10)  # loại sky
            if median_region == "bottom":
                y0 = int((1.0 - region_frac) * H)
                if y0 > 0:
                    mask[:y0, :] = False
            median_rel = np.nanmedian(depth_rel[mask]) if np.any(mask) else np.nanmedian(depth_rel)
            if not np.isfinite(median_rel) or median_rel <= 0:
                median_rel = np.nanmedian(depth_rel)

            rel_vals = np.zeros((boxes.shape[0],), dtype=np.float32)
            for i, b in enumerate(boxes):
                x1 = max(0, int(b[0])); y1 = max(0, int(b[1]))
                x2 = min(W, int(b[2])); y2 = min(H, int(b[3]))
                if x2 <= x1 + 1 or y2 <= y1 + 1:
                    rel_vals[i] = np.nan
                    continue
                is_obs = (clss[i] < 0)
                q = obs_q if is_obs else yolo_q
                if is_obs and foot_band_px > 0:
                    ys = max(y1, y2 - foot_band_px)
                    patch = depth_rel[ys:y2, x1:x2].reshape(-1)
                else:
                    patch = depth_rel[y1:y2, x1:x2].reshape(-1)
                patch = patch[np.isfinite(patch)]
                rel_vals[i] = np.percentile(patch, q*100.0) if patch.size > 0 else np.nan

            # fallback cho NaN
            if np.any(~np.isfinite(rel_vals)):
                rel_vals[~np.isfinite(rel_vals)] = median_rel

            dist_m_final = (scale_k * median_rel) / np.maximum(rel_vals, 1e-6)
            dist_m_final = np.clip(dist_m_final, clip_min_m, clip_max_m).astype(np.float32)

            # Ghi đè d_init chỉ để vẽ/track (không ảnh hưởng các bước trước)
            d_init = dist_m_final
        # ================================

        # Track & vẽ (OC-SORT y hệt bản cũ)
        tracks = tracker.update(boxes, scores, clss, d_init)

        draw_boxes_arr, draw_dists_arr, labels = [], [], []
        for tid, tb, ts, tc, tz, is_pred in tracks:
            draw_boxes_arr.append(tb)
            draw_dists_arr.append(tz if np.isfinite(tz) else np.nan)
            labels.append("OBS" if tc < 0 else str(det.names.get(int(tc), str(int(tc)))))

        draw_boxes_arr = np.asarray(draw_boxes_arr, dtype=np.float32) if len(draw_boxes_arr) else np.zeros((0,4), dtype=np.float32)
        draw_dists_arr = np.asarray(draw_dists_arr, dtype=np.float32) if len(draw_dists_arr) else np.zeros((0,), dtype=np.float32)

        vis = draw_boxes(rgb, draw_boxes_arr, labels, draw_dists_arr, thr=(thr_low,thr_high),
                         thickness=cfg["viz"]["thickness"], font_scale=cfg["viz"]["font_scale"])
        hud = [f"edge.num={len(obs_boxes)}  yolo={yolo_xyxy_cpu.shape[0]}  merged={len(boxes)}  tracks={len(tracks)}"]
        vis = draw_hud(vis, hud)
        vis = cv2.resize(vis, (outW,outH), interpolation=cv2.INTER_AREA)
        writer.write_rgb(vis)

        if mem_gc_every > 0 and (frame_id % mem_gc_every) == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

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
