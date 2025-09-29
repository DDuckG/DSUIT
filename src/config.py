# src/config.py
import os, yaml

DEFAULT_CFG = {
    "io": {
        "output_fps": 7, 
        "output_size": [640,360], 
        "reader_backend": "opencv"
    },
    "camera": {
        "height_m": 1.35, 
        "fov_deg_h": 68.0, 
        "principal": [0.5, 0.5]
    },
    "models": {
        "yolo": {
            "weights": "models/yolov12/yolov12m.pt", 
            "conf": 0.40, 
            "iou": 0.60, 
            "half": False,
            "whitelist_classes": [0,1,2,3,5,7,8,9,10,11,13,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,66,67,68,69,70,71,72,73,74,75,76,77,78,79]
        },
        "segmentation": {
            "weights": "models/bisenet_v2/weights/bisenetv2_cityscapes.pth", 
            "half": True, 
            "run_every_k": 3
        },
        "depth": {
            "encoder": "vits", 
            "checkpoint": "models/video_depth_anything/checkpoints/video_depth_anything_vits.pth",
            "input_size": 518, 
            "stream": True, 
            "fp32": False, 
            "xformers": True
        }
    },
    "tracking": {
        "type": "ocsort", 
        "max_age": 30, 
        "iou": 0.3
    },
    "fusion": {
        "distance_thresholds_m": [1.8, 3.5],
        "plane_fit": {
            "method": "ransac",
            "inlier_thr_m": 0.06, 
            "max_iter": 150,
            "sample_stride": 4, 
            "ema_beta": 0.9, 
            "use_sidewalk": True, 
            "use_terrain": True
        },
        "edge_enhance": {
            "gauss_k": 5, 
            "gauss_sigma": 1.2, 
            "log_eps": 1e-6
        },
        "edge_obstacle": {
            "height_min_m": 0.30,
            "depth_min_m": 0.40, 
            "depth_max_m": 10.0,
            "mag_thr_rel": 0.22,
            "area_min_m2_k": 2.0e-4,
            "min_edge_len_px": 12
        },
        "obstacle": {
            "height_min_m": 0.30,
            "foot_height_m": 0.06,
            "depth_min_m": 0.40, 
            "depth_max_m": 10.0,
            "downsample": 0.5,
            "area_min_m2_k": 2.0e-4,
            "grad_thr": 0.035, 
            "grad_ds": 0.33,
            "gate_min_frac": 0.0008, 
            "trim_frac": 0.08,
            "iou_merge_yolo": 0.50, 
            "nms_iou": 0.55, 
            "topk": 15, 
            "persistence_min": 2
        }
    },
    "viz": {
        "thickness": 2, 
        "font_scale": 0.45, 
        "show_fps": True
    }
}

def load_config(path: str = "configs/config.yaml") -> dict:
    cfg = DEFAULT_CFG.copy()
    def merge(dst, src):
        for k,v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        merge(cfg, loaded)
    return cfg