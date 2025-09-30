import os
import yaml

COCO80_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

def load_config(path: str = "configs/config.yaml") -> dict:
    config = {}
    if os.path.exists(path):
        with open(path, "r", encoding = "utf-8") as file:
            y = yaml.safe_load(file) or {}
            if isinstance(y, dict):
                config = y

    models = config.setdefault("models", {})
    yolo   = models.setdefault("yolo", {})
    if not yolo.get("names"):
        yolo["names"] = COCO80_NAMES

    fusion = config.setdefault("fusion", {})
    fusion.setdefault("distance_thresholds_m", [1.8, 3.5])
    alerts = fusion.setdefault("alerts", {})
    alerts.setdefault("roi_frac", [0.25, 0.25, 0.5, 0.5])
    alerts.setdefault("roi_iou_min", 0.10)
    alerts.setdefault("beep_interval_s", 2.0)
    alerts.setdefault("debounce_frames", 2)
    alerts.setdefault("hysteresis_frames", 2)
    alerts.setdefault("log_texts", {"warn": "Be careful", "danger": "Danger"})
    alerts.setdefault("audio", {
        "warn_src": "src/utils/canthan.mp3",
        "danger_src": "src/utils/nguyhiem.mp3"
    })

    return config
