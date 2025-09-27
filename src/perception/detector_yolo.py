# src/perception/detector_yolo.py
import numpy as np, torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, weights: str, conf=0.25, iou=0.5, half=True, whitelist=None, device="cuda"):
        self.model = YOLO(weights)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.conf = float(conf); self.iou = float(iou)
        self.half = bool(half)
        self.whitelist = set(int(x) for x in whitelist) if whitelist is not None else None

    def __call__(self, rgb, img_size=None):
        # rgb: HxWx3 uint8
        h, w = rgb.shape[:2]
        pred = self.model.predict(source=rgb, verbose=False, imgsz=img_size, conf=self.conf,
                                  iou=self.iou, device=self.device, half=self.half, stream=False)[0]
        boxes = pred.boxes
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else np.zeros((0,4),dtype=np.float32)
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else np.zeros((0,),dtype=np.float32)
        cls  = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else np.zeros((0,),dtype=int)
        if self.whitelist is not None:
            keep = [i for i,c in enumerate(cls) if c in self.whitelist]
            xyxy = xyxy[keep]; conf = conf[keep]; cls = cls[keep]
        return xyxy, conf, cls, (w,h)

    @property
    def names(self):
        if hasattr(self.model, "model") and getattr(self.model.model, "names", None) is not None:
            return self.model.model.names
        return getattr(self.model, "names", {})
