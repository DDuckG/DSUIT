# src/perception/segment_bisenet.py
import os, sys
import numpy as np, torch, torch.nn.functional as F

sys.path.append(os.path.abspath("."))  # repo root
from models.bisenet_v2.loader import load_bisenet_from_vendor

CITYSCAPES = {
    "road": 0, "sidewalk": 1, "building": 2, "wall": 3, "fence": 4, "pole": 5,
    "traffic light": 6, "traffic sign": 7, "vegetation": 8, "terrain": 9, "sky": 10,
    "person": 11, "rider": 12, "car": 13, "truck": 14, "bus": 15, "train": 16, "motorcycle": 17, "bicycle": 18
}

class BiSeNetSeg:
    def __init__(self, weights_path: str, half=True, device="cuda"):
        self.model, _ = load_bisenet_from_vendor(weights_path=weights_path, device=device, num_classes=19, verbose=False)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.half = bool(half)
        if self.half and (self.device.startswith("cuda")):
            self.model.half()

        self.mean = torch.tensor([0.3257, 0.3690, 0.3223], dtype=torch.float32, device=self.device).view(1,3,1,1)
        self.std  = torch.tensor([0.2112, 0.2148, 0.2115], dtype=torch.float32, device=self.device).view(1,3,1,1)
        if self.half and self.device.startswith("cuda"):
            self.mean = self.mean.half(); self.std = self.std.half()

    def preprocess(self, rgb):
        im = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(self.device)
        im = im.float() / 255.0
        im = (im - self.mean) / self.std
        return im

    def __call__(self, rgb, run_size_align=32):
        H, W = rgb.shape[0], rgb.shape[1]
        inp = self.preprocess(rgb)
        if self.half and self.device.startswith("cuda"):
            inp = inp.half()
        new_h = int(np.ceil(H/run_size_align)*run_size_align)
        new_w = int(np.ceil(W/run_size_align)*run_size_align)
        inp = F.interpolate(inp, size=(new_h, new_w), mode="bilinear", align_corners=False)
        with torch.no_grad():
            logits = self.model(inp)[0] if isinstance(self.model(inp), (list,tuple)) else self.model(inp)
            if logits.shape[-2:] != (H,W):
                logits = F.interpolate(logits, size=(H,W), mode="bilinear", align_corners=False)
            seg = torch.argmax(logits, dim=1)[0].detach().to("cpu").numpy().astype(np.uint8)
        return seg
