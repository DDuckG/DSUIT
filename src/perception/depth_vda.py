# src/perception/depth_vda.py
import os, sys
import torch, numpy as np
sys.path.append(os.path.abspath("."))
from models.video_depth_anything.video_depth_anything.video_depth_stream import VideoDepthAnything

_EMBED2ENC = {384:"vits", 768:"vitb", 1024:"vitl"}
_ENC2CFG = {
    "vits": dict(features=64,  out_channels=[48,96,192,384]),
    "vitb": dict(features=128, out_channels=[96,192,384,768]),
    "vitl": dict(features=256, out_channels=[256,512,1024,1024]),
}

def _clean_state_dict(sd):
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return { (k[7:] if k.startswith("module.") else k): v for k,v in sd.items() }

def _infer_encoder_from_sd(sd):
    embed_dim=None
    for k,v in sd.items():
        if k.endswith("pos_embed") and isinstance(v, torch.Tensor) and v.dim()==3:
            embed_dim=int(v.shape[-1]); break
    if embed_dim in _EMBED2ENC: return _EMBED2ENC[embed_dim]
    max_block=-1
    for k in sd.keys():
        if ".blocks." in k:
            try: max_block=max(max_block,int(k.split(".blocks.")[1].split(".")[0]))
            except: pass
    if max_block>=23: return "vitl"
    elif max_block>=11: return "vits"
    return "vits"

class DepthStreamVDA:
    def __init__(self, encoder="vits", checkpoint="", input_size=518, device="cuda", fp32=False):
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"Checkpoint không tồn tại: {checkpoint}")
        raw = torch.load(checkpoint, map_location="cpu")
        sd = _clean_state_dict(raw)
        enc = _infer_encoder_from_sd(sd)
        cfg = _ENC2CFG[enc]
        self.model = VideoDepthAnything(
            encoder=enc, features=cfg["features"], out_channels=cfg["out_channels"],
            num_frames=32, pe="ape",
        ).to(dev).eval()
        self.model.load_state_dict(sd, strict=True)
        self.dev = dev
        self.input_size = int(input_size)
        self.fp32 = bool(fp32)

    @torch.inference_mode()
    def infer_one(self, rgb: np.ndarray) -> np.ndarray:
        depth = self.model.infer_video_depth_one(
            rgb, input_size=self.input_size, device=str(self.dev), fp32=self.fp32
        )
        return depth.astype(np.float32)
