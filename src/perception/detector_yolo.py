import numpy as np
import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, weights: str, conf = 0.25, iou = 0.5, half = False, whitelist = None, device = "cuda", imgsz = None):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weights)
        self.model.to(self.device)
        if hasattr(self.model, "model") and self.model.model is not None:
            self.model.model.float()
        fuse = getattr(self.model, "fuse", None)
        if callable(fuse):
            fuse()

        self.conf = float(conf)
        self.iou = float(iou)
        self.half = False
        self.imgsz = imgsz

        self._whitelist_tensor = (
            torch.as_tensor(sorted(set(int(class_name) for class_name in whitelist)), device = self.device, dtype = torch.int64)
            if whitelist is not None else None
        )

        height, width = 384, 640
        if isinstance(self.imgsz, (tuple, list)) and len(self.imgsz) == 2:
            width, height = int(self.imgsz[0]), int(self.imgsz[1])
        dummy = np.zeros((height, width, 3), dtype = "uint8")
        with torch.inference_mode():
            _ = self.model.predict(source = dummy, verbose = False, imgsz = (width, height), conf = self.conf, iou = self.iou, device = self.device, half = False, stream = False)

    @torch.inference_mode()
    def __call__(self, rgb, img_size = None):
        imgsz = img_size if img_size is not None else self.imgsz
        pred = self.model.predict(source = rgb, verbose = False, imgsz = imgsz, conf = self.conf, iou = self.iou, device = self.device, half = False, stream = False)[0]
        boxes = pred.boxes
        xyxy = boxes.xyxy
        conf = boxes.conf
        cls  = boxes.cls.to(torch.int64)

        if self._whitelist_tensor is not None and cls.numel() > 0:
            keep = torch.isin(cls, self._whitelist_tensor)
            xyxy = xyxy[keep]
            conf = conf[keep]
            cls  = cls[keep]

        height, width = rgb.shape[:2]
        names = self.names
        return xyxy, conf, cls, names

    @property
    def names(self):
        if hasattr(self.model, "model") and getattr(self.model.model, "names", None) is not None:
            return self.model.model.names
        return getattr(self.model, "names", {})
