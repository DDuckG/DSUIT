import torch
import numpy as np
from dataclasses import dataclass

from src.utils.torch_cuda import get_device, to_torch

@dataclass
class BEVConfig:
    x_min: float = -10.0
    x_max: float =  10.0
    z_min: float =   0.0
    z_max: float =  40.0
    cell:  float =   0.20
    h_min_m: float = 0.18

class BEVProjectorTorch:
    """
    BEV occupancy projector (GPU-first). Không dính tới logic lọc—chỉ sinh occupancy và chấm điểm hỗ trợ.
    """
    def __init__(self, W:int, H:int, fx:float, fy:float, cx_pix:float, cy_pix:float, cfg:BEVConfig=None, device=None):
        self.dev = device or get_device()
        self.W, self.H = int(W), int(H)
        self.fx = float(fx); self.fy = float(fy)
        self.cx = float(cx_pix); self.cy = float(cy_pix)
        self.cfg = cfg or BEVConfig()

        uu = torch.arange(self.W, device=self.dev, dtype=torch.float32).view(1, self.W)
        self.x_factor = (uu - self.cx) / max(1e-6, self.fx)

        self.nx = int(np.ceil((self.cfg.x_max - self.cfg.x_min)/self.cfg.cell))
        self.nz = int(np.ceil((self.cfg.z_max - self.cfg.z_min)/self.cfg.cell))

        self.last_occ = torch.zeros((self.nz, self.nx), dtype=torch.uint8, device=self.dev)
        self._offs_cache = {}

    @torch.no_grad()
    def splat(self, depth_m, seg, height_m):
        """
        depth_m: [H,W] (m)
        seg:     [H,W] (int)
        height_m:[H,W] (m above plane)
        """
        D  = to_torch(depth_m, device=self.dev, dtype=torch.float32)
        Hh = to_torch(height_m, device=self.dev, dtype=torch.float32)
        S  = to_torch(seg, device=self.dev, dtype=torch.int16)

        m = (Hh >= float(self.cfg.h_min_m))
        m &= (D >= float(self.cfg.z_min)) & (D <= float(self.cfg.z_max))
        # loại sky/road nếu label tương ứng có (giữ nguyên như trước đây nếu cần)
        m &= (S != 0) & (S != 10)

        if int(m.sum().item()) == 0:
            return torch.zeros((self.nz, self.nx), dtype=torch.uint8, device=self.dev)

        ys, xs = torch.nonzero(m, as_tuple=True)
        z = D[ys, xs]
        x = self.x_factor[0, xs] * z

        ix = torch.floor((x - self.cfg.x_min) / self.cfg.cell).long().clamp_(0, self.nx-1)
        iz = torch.floor((z - self.cfg.z_min) / self.cfg.cell).long().clamp_(0, self.nz-1)

        occ = torch.zeros((self.nz, self.nx), dtype=torch.uint8, device=self.dev)
        lin = iz * self.nx + ix
        uniq = torch.unique(lin)
        occ.view(-1)[uniq] = 1
        return occ

    @torch.no_grad()
    def occupancy_score_for_boxes(self, boxes_xyxy: np.ndarray, dists_m: np.ndarray, weight_window:int=2):
        """
        Trả về điểm BEV [N] (torch.float32, device=self.dev). Chỉ là “boost” nhẹ, không lọc.
        """
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.dev)

        B = torch.as_tensor(boxes_xyxy, device=self.dev, dtype=torch.float32)
        Z = torch.as_tensor(dists_m,   device=self.dev, dtype=torch.float32)

        cx = 0.5*(B[:,0] + B[:,2])
        X  = ((cx - self.cx) / max(1e-6, self.fx)) * Z

        ok = torch.isfinite(Z) & (Z > 0)
        ix = torch.floor((X - self.cfg.x_min) / self.cfg.cell).long().clamp_(0, self.nx-1)
        iz = torch.floor((Z - self.cfg.z_min) / self.cfg.cell).long().clamp_(0, self.nz-1)

        w = int(max(1, weight_window))
        offs = self._offs_cache.get(w)
        if offs is None:
            offs = torch.stack(torch.meshgrid(
                torch.arange(-w, w+1, device=self.dev),
                torch.arange(-w, w+1, device=self.dev),
                indexing='ij'
            ), dim=-1).view(-1, 2)
            self._offs_cache[w] = offs
        K = offs.shape[0]

        cells_z = (iz.view(-1,1) + offs[:,0].view(1,-1)).clamp_(0, self.nz-1)
        cells_x = (ix.view(-1,1) + offs[:,1].view(1,-1)).clamp_(0, self.nx-1)
        lin = cells_z * self.nx + cells_x

        flat = self.last_occ.view(-1).float()
        scores = flat.gather(0, lin.view(-1).long()).view(-1, K).mean(dim=1)
        return torch.where(ok, scores, torch.zeros_like(scores))

    def set_last_occ(self, occ: torch.Tensor):
        self.last_occ = occ
