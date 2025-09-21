import os
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

COCO_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
    7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
    12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
    18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
    35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
    45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
    57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
    62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
    68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

def load_config(default_path: str = "configs/config.yaml"):
    config = {}
    with open(default_path, "r", encoding = "utf-8") as file:
        config = yaml.safe_load(file) or {}
    return config


def get_whitelist(config: dict):
    white_list = config.get("src", {}).get("yolo", {}).get("whitelist_classes", None)
    wl_ints = set(int(x) for x in white_list)
    return wl_ints


def write_mot(frame_id: int, box: np.ndarray, score: float, class_id: int):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.3f},-1,-1,{class_id}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = False, help = "YOLO w8")
    parser.add_argument("--src", required = True, help = "input video")
    parser.add_argument("--out", required = True, help = "output txt")
    parser.add_argument("--conf", type = float, default = None, help = "confidence threshold")
    parser.add_argument("--stream", action = "store_true", help = "use stream reduce memory")
    parser.add_argument("--config", default = "configs/config.yaml", help = "path to config yaml")
    argument = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(argument.config)
    yolo_cfg = config.get("src", {}).get("yolo", {})
    model_path = argument.model or yolo_cfg.get("model", yolo_cfg.get("weights", None))
    conf_thresh = argument.conf if argument.conf is not None else float(yolo_cfg.get("conf_threshold", yolo_cfg.get("conf", 0.25)))
    stream_mode = argument.stream or bool(yolo_cfg.get("stream", False))

    whitelist = get_whitelist(config)

    model = YOLO(model_path)

    names = model.model.names if hasattr(model, "model") and getattr(model.model, "names", None) is not None else None
    if names is None:
        names = getattr(model, "names", None)
    if names is None:
        names = COCO_NAMES

    Path(os.path.dirname(argument.out) or ".").mkdir(parents = True, exist_ok = True)

    predictor_kwargs = dict(source = str(argument.src), conf = conf_thresh, device = device, verbose = False)
    predictor_kwargs["stream"] = stream_mode

    results_iterable = model.predict(**predictor_kwargs)

    write_lines = []
    frame_idx = 0
    for res in results_iterable:
        frame_idx += 1
        boxes_obj = getattr(res, "boxes", None)
        xyxy = boxes_obj.xyxy.cpu().numpy() if hasattr(boxes_obj, "xyxy") else np.zeros((0, 4))
        confs = boxes_obj.conf.cpu().numpy() if hasattr(boxes_obj, "conf") else np.zeros((0,))
        clsids = boxes_obj.cls.cpu().numpy().astype(int) if hasattr(boxes_obj, "cls") else np.zeros((0,), dtype=int)

        for i in range(len(xyxy)):
            cid = int(clsids[i])
            if (whitelist is not None) and (cid not in whitelist):
                continue
            box = xyxy[i]
            score = float(confs[i])
            line = write_mot(frame_idx, box, score, cid)
            write_lines.append(line)

    with open(argument.out, "w", encoding = "utf-8") as file:
        for line in write_lines:
            file.write(line + "\n")

    print(f"[YOLO] wrote {len(write_lines)} detections to {argument.out}")
    if whitelist is None:
        print("[YOLO] whitelist: ALL classes")
    else:
        mapped = {int(i): names.get(int(i), str(i)) for i in sorted(list(whitelist))}
        print(f"[YOLO] whitelist ids ({len(mapped)}): {mapped}")


if __name__ == "__main__":
    main()