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
from src.tracking.obs_track import ObsTracker
from src.fusion.ground_plane import GroundPlaneScaler
from src.fusion.depth_enhance import DepthEnhancer
from src.fusion.depth_edges import detect_obstacles_by_depth_edges, refine_box_by_edges_and_height, two_edge_ok_array
from src.fusion.planar_patch import detect_planar_obstacles
from src.fusion.pole_proposals import detect_poles_from_depth
from src.fusion.edge_inside import snap_inside_box
from src.fusion.obs_sliver import prune_obs_slivers
from src.fusion.depth_utils import distance_from_box
from src.utils.geom import iou_matrix

# ---------- helpers ----------
def _normalize_names(names):
    # trả về list/tuple các tên lớp
    if isinstance(names, (list, tuple)):
        return list(names)
    # YOLOv5/8 thường là dict {id:name}
    if hasattr(names, "keys"):
        # map thành list theo thứ tự id
        try:
            maxk = max(int(k) for k in names.keys())
            out = [""] * (maxk + 1)
            for k, v in names.items():
                ki = int(k)
                if 0 <= ki < len(out): out[ki] = str(v)
            return out
        except Exception:
            return [str(names.get(i, i)) for i in range(100)]
    # fallback
    return []

def _label_from_id(cid, names_list):
    cid = int(cid)
    if 0 <= cid < len(names_list):
        return str(names_list[cid])
    return str(cid)

def _match_tracks_to_dets(track_boxes, det_boxes):
    # trả về index detection tốt nhất cho mỗi track (hoặc -1 nếu không match)
    if track_boxes.shape[0] == 0 or det_boxes.shape[0] == 0:
        return np.full((track_boxes.shape[0],), -1, dtype=np.int32)
    M = iou_matrix(track_boxes.astype(np.float32), det_boxes.astype(np.float32))
    det_assigned = -np.ones((track_boxes.shape[0],), dtype=np.int32)
    used_d = np.zeros((det_boxes.shape[0],), dtype=bool)
    # greedy theo IoU
    for ti in range(track_boxes.shape[0]):
        di = int(np.argmax(M[ti]))
        if M[ti, di] <= 0.0 or used_d[di]:
            det_assigned[ti] = -1
        else:
            det_assigned[ti] = di
            used_d[di] = True
    return det_assigned

@torch.inference_mode()
def _warmup_pipeline(det, seg_model, dep, scaler, enh, W, H, y_imgsz):
    rgb0 = np.zeros((H, W, 3), dtype=np.uint8)
    depth_rel = dep.infer_one(rgb0).astype(np.float32)
    seg_id = seg_model(rgb0)
    alpha, plane = scaler.estimate_scale(depth_rel, seg_id)
    depth_m = alpha / (np.maximum(depth_rel, 1e-6))
    _ = enh(depth_m, rgb0)
    _ = det(rgb0, img_size=y_imgsz)

@torch.inference_mode()
def process_video(src_path: str, out_path: str, cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    threads = int(cfg["io"].get("cpu_threads", 0))
    if threads <= 0:
        threads = os.cpu_count() or 8
    torch.set_num_threads(int(threads))

    prefetch = int(cfg["io"].get("prefetch", 8))
    write_q = int(cfg["io"].get("write_queue", 64))
    reader = AsyncVideoReader(src_path, prefetch=prefetch)
    outW, outH = cfg["io"]["output_size"]
    writer = AsyncVideoWriter(out_path, cfg["io"]["output_fps"], (outW, outH), queue_size=write_q)

    # Models
    ycfg = cfg["models"]["yolo"]
    det = YOLODetector(
        ycfg["weights"], conf=ycfg["conf"], iou=ycfg["iou"],
        half=ycfg["half"], whitelist=ycfg.get("whitelist_classes", None), device=device,
        imgsz=tuple(ycfg.get("imgsz", (640,384)))
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
        inlier_thr=pf.get("inlier_thr_m",0.05), max_iter=pf.get("max_iter",180)
    )

    thr_low, thr_high = cfg["fusion"]["distance_thresholds_m"]
    eocfg = cfg["fusion"]["epp"]
    ppcfg = cfg["fusion"]["ppp"]
    slcfg = cfg["fusion"]["obs_sliver"]
    zmax_obs = float(cfg["fusion"].get("z_max_obs_m", 15.0))
    incfg = cfg["fusion"]["edge_inside"]
    polecfg= cfg["fusion"]["pole"]

    # Trackers
    tracker_y = OCSort(max_age=cfg["tracking"]["yolo"]["max_age"], iou_thr=cfg["tracking"]["yolo"]["iou"])
    tracker_o = ObsTracker(
        iou_thr=cfg["tracking"]["obs"]["iou"],
        max_age=cfg["tracking"]["obs"]["max_age"],
        z_max=zmax_obs,
        edge_fail_kill=2,
        staging=cfg["tracking"]["obs"].get("staging", {}),
        lock=cfg["tracking"]["obs"].get("lock", {})
    )

    # Warm-up
    y_imgsz = tuple(ycfg.get("imgsz", (640,384)))
    enh = DepthEnhancer(cfg["fusion"].get("edge_enhance", {"gauss_k":5,"gauss_sigma":1.2}))
    _warmup_pipeline(det, seg_model, dep, scaler, enh, reader.W, reader.H, y_imgsz)
    print("[READY] Pipeline D warmed up. Start streaming frames...")

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

        # Depth tương đối & segmentation
        depth_rel = dep.infer_one(rgb).astype(np.float32)
        if (frame_id % run_seg_every) == 0 or last_seg is None:
            seg_id = seg_model(rgb); last_seg = seg_id
        else:
            seg_id = last_seg

        # Scale & plane
        alpha, plane = scaler.estimate_scale(depth_rel, seg_id)
        depth_m = alpha / (np.maximum(depth_rel, 1e-6))
        g  = float(cfg["fusion"].get("depth_calib", {}).get("gamma", 1.0))
        s  = float(cfg["fusion"].get("depth_calib", {}).get("scale", 1.0))
        depth_m = (depth_m ** g) * s

        # Enhance & sigma
        depth_m_enh, sigma_m = enh(depth_m, rgb)

        # Height-map
        height_m = scaler.height_from_plane(depth_m, plane)
        height_t = torch.from_numpy(height_m).to(device=device, dtype=torch.float32)

        # ==== YOLO: detections + names ====
        yolo_xyxy, yolo_scores, yolo_cls, yolo_names = det(rgb, img_size=y_imgsz)
        y_names = _normalize_names(yolo_names)
        if yolo_xyxy.numel() > 0:
            yb = yolo_xyxy.detach().cpu().numpy().astype(np.float32)
            ys = yolo_scores.detach().cpu().numpy().astype(np.float32)
            yc = yolo_cls.detach().cpu().numpy().astype(np.int32)
        else:
            yb = np.zeros((0,4), dtype=np.float32); ys = np.zeros((0,), dtype=np.float32); yc = np.zeros((0,), dtype=np.int32)

        # ==== Tính distance cho YOLO detections để tô màu ====
        # dùng depth_m (m) + height_m + seg_id và logic trong depth_utils.distance_from_box
        if yb.shape[0] > 0:
            y_dists = np.empty((yb.shape[0],), dtype=np.float32)
            for i in range(yb.shape[0]):
                y_dists[i] = float(distance_from_box(depth_m, height_m, seg_id, yb[i], foot_h=0.06, frac=0.25))
        else:
            y_dists = np.zeros((0,), dtype=np.float32)

        # ==== OBS PROPOSALS ====
        (e_boxes, e_dists, e_scores, e_edgeok), _ = detect_obstacles_by_depth_edges(
            depth_m_enh, sigma_m, seg_id, eocfg, fx=scaler.fx, fy=scaler.fy, height_m=height_m
        )
        p_boxes, p_dists, p_scores = detect_planar_obstacles(
            depth_m_enh, sigma_m, seg_id, height_m, ppcfg, fx=scaler.fx, fy=scaler.fy
        )
        v_boxes, v_dists, v_scores = detect_poles_from_depth(depth_m_enh, polecfg)

        # Union proposals
        N_all = e_boxes.shape[0] + p_boxes.shape[0] + v_boxes.shape[0]
        if N_all > 0:
            if v_boxes.shape[0] > 0:
                boxes  = np.concatenate([e_boxes, p_boxes, v_boxes], axis=0)
                scores = np.concatenate([e_scores, p_scores, v_scores], axis=0)
                dists  = np.concatenate([e_dists, p_dists, v_dists], axis=0)
                edgeok = np.concatenate([e_edgeok, np.ones((p_boxes.shape[0],), dtype=bool), np.ones((v_boxes.shape[0],), dtype=bool)], axis=0)
            else:
                boxes  = np.concatenate([e_boxes, p_boxes], axis=0)
                scores = np.concatenate([e_scores, p_scores], axis=0)
                dists  = np.concatenate([e_dists, p_dists], axis=0)
                edgeok = np.concatenate([e_edgeok, np.ones((p_boxes.shape[0],), dtype=bool)], axis=0)
        else:
            boxes = np.zeros((0,4), dtype=np.float32)
            scores= np.zeros((0,), dtype=np.float32)
            dists = np.zeros((0,), dtype=np.float32)
            edgeok= np.zeros((0,), dtype=bool)

        # Prune slivers & YOLO-protect
        yolo_boxes_np = yb if yb.shape[0] > 0 else np.zeros((0,4), dtype=np.float32)
        if boxes.shape[0] > 0:
            boxes, scores, dists = prune_obs_slivers(
                boxes_xyxy=boxes, scores=scores, dists_m=dists, yolo_boxes=yolo_boxes_np, cfg=slcfg
            )

        # ==== OBS TRACK ====
        D = torch.from_numpy(depth_m_enh).to(device=device, dtype=torch.float32)
        Dl = torch.log(torch.clamp(D, min=1e-3))
        from src.utils.torch_cuda import sobel_grad
        gx, gy = sobel_grad(Dl)
        def _refine(bx_np):
            b1 = refine_box_by_edges_and_height(bx_np, gx, gy, height_t, eocfg)
            b2 = snap_inside_box(b1, D, cfg=incfg)
            return b2

        if boxes.shape[0] > 0:
            edge_ok_final = two_edge_ok_array(
                boxes, gx, gy,
                k=int(eocfg.get("twoedge_band_px",3)),
                vthr=float(eocfg.get("twoedge_v_min",0.0025)),
                hthr=float(eocfg.get("twoedge_h_min",0.0025))
            )
        else:
            edge_ok_final = np.zeros((0,), dtype=bool)

        obs_tracks  = tracker_o.update(boxes, scores, dists, _refine, (W,H), edge_ok_final, yolo_boxes_np)

        # ==== YOLO TRACK ====
        y_tracks = tracker_y.update(yb, ys, yc, None)

        # Gán label + distance cho YOLO tracks bằng IoU với detections hiện tại
        draw_boxes_arr, draw_dists_arr, labels = [], [], []

        if len(y_tracks) > 0:
            t_boxes = np.stack([tb for (_id, tb, _s, _c, _z, _p) in y_tracks], axis=0).astype(np.float32)
            det_idx = _match_tracks_to_dets(t_boxes, yb)
            for k, tr in enumerate(y_tracks):
                tid, tb, ts, tc, tz, is_pred = tr
                di = det_idx[k]
                if di >= 0:
                    lbl = _label_from_id(yc[di], y_names)
                    dz  = float(y_dists[di]) if np.isfinite(y_dists[di]) and y_dists[di] > 0 else np.nan
                else:
                    lbl = _label_from_id(tc, y_names)  # fallback theo class id trong track
                    dz  = np.nan
                draw_boxes_arr.append(tb)
                draw_dists_arr.append(dz)
                labels.append(lbl)

        # ==== OBS VIZ ====
        if len(obs_tracks):
            for tid, tb, ts, tc, tz, is_pred in obs_tracks:
                draw_boxes_arr.append(tb)
                draw_dists_arr.append(tz if np.isfinite(tz) else np.nan)
                labels.append("OBS")

        draw_boxes_arr = np.asarray(draw_boxes_arr, dtype=np.float32) if len(draw_boxes_arr) else np.zeros((0,4), dtype=np.float32)
        draw_dists_arr = np.asarray(draw_dists_arr, dtype=np.float32) if len(draw_dists_arr) else np.zeros((0,), dtype=np.float32)

        vis = draw_boxes(rgb, draw_boxes_arr, labels, draw_dists_arr, thr=(thr_low,thr_high),
                         thickness=cfg["viz"]["thickness"], font_scale=cfg["viz"]["font_scale"])
        hud = [f"OBS:epp={e_boxes.shape[0]} ppp={p_boxes.shape[0]} pole={v_boxes.shape[0]} out={len(obs_tracks)}  YOLO:{yb.shape[0]} trk={len(y_tracks)}  alpha={alpha:.3f}"]
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
