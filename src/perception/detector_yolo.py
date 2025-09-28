# src/perception/detector_yolo.py
import torch
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, weights: str, conf=0.25, iou=0.5, half=False, whitelist=None, device="cuda"):
        self.model = YOLO(weights)
        self.device = device if torch.cuda.is_available() else "cpu"
        try:
            self.model.to(self.device)
            if hasattr(self.model, "model") and self.model.model is not None:
                self.model.model.float()
        except Exception:
            pass
        self.conf = float(conf); self.iou=float(iou)
        self.half=False
        self.whitelist = set(int(x) for x in whitelist) if whitelist is not None else None

    @torch.inference_mode()
    def __call__(self, rgb, img_size=None):
        try:
            if hasattr(self.model,"model") and self.model.model is not None:
                self.model.model.float()
        except Exception:
            pass
        pred = self.model.predict(source=rgb, verbose=False, imgsz=img_size,
                                  conf=self.conf, iou=self.iou, device=self.device,
                                  half=False, stream=False)[0]
        boxes = pred.boxes
        xyxy = boxes.xyxy
        conf = boxes.conf
        cls  = boxes.cls.to(torch.int64)

        if torch.cuda.is_available():
            if xyxy.device.type!="cuda": xyxy=xyxy.cuda(non_blocking=True)
            if conf.device.type!="cuda": conf=conf.cuda(non_blocking=True)
            if cls.device.type!="cuda":  cls =cls.cuda(non_blocking=True)

        if self.whitelist is not None and cls.numel()>0:
            keep_idx = [i for i,c in enumerate(cls.tolist()) if c in self.whitelist]
            if len(keep_idx):
                keep = torch.tensor(keep_idx, device=xyxy.device, dtype=torch.long)
                xyxy = xyxy.index_select(0, keep)
                conf = conf.index_select(0, keep)
                cls  = cls.index_select(0, keep)
            else:
                xyxy = torch.zeros((0,4), device=xyxy.device, dtype=xyxy.dtype)
                conf = torch.zeros((0,), device=xyxy.device, dtype=conf.dtype)
                cls  = torch.zeros((0,), device=xyxy.device, dtype=torch.int64)

        if hasattr(pred, "orig_img") and pred.orig_img is not None:
            H,W = pred.orig_img.shape[:2]
        else:
            try: H,W = rgb.shape[:2]
            except: H,W = 0,0
        return xyxy, conf, cls, (W,H)

    @property
    def names(self):
        if hasattr(self.model,"model") and getattr(self.model.model,"names",None) is not None:
            return self.model.model.names
        return getattr(self.model, "names", {})
