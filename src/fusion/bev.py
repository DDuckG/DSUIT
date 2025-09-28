# src/fusion/bev.py
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

from src.utils.torch_cuda import get_device, to_torch

@dataclass
class BEVConfig:
    x_min: float = -10.0  # trái (m)
    x_max: float =  10.0  # phải (m)
    z_min: float =   0.0  # trước camera
    z_max: float =  40.0
    cell:  float =   0.20
    h_min_m: float = 0.18  # threshold chiều cao so với mặt đường

class BEVProjectorTorch:
    """
    Splat occupancy từ (depth_m, height_m, seg) lên BEV. Tất cả trên GPU.
    Dùng cho gating nhẹ: tăng/giảm score OBS theo occupancy xung quanh (X,Z) ước lượng.
    """
    def __init__(self, W:int, H:int, fx:float, fy:float, cx_pix:float, cy_pix:float, cfg:BEVConfig=None, device=None):
        self.dev = device or get_device()
        self.W, self.H = int(W), int(H)
        self.fx = float(fx); self.fy=float(fy)
        self.cx = float(cx_pix); self.cy=float(cy_pix)
        self.cfg = cfg or BEVConfig()

        # precompute pixel-to-ray factor for X = (u - cx)/fx * Z
        uu = torch.arange(self.W, device=self.dev, dtype=torch.float32).view(1, self.W)
        self.x_factor = (uu - self.cx) / max(1e-6, self.fx)  # [1,W]

        # grid size
        self.nx = int(np.ceil((self.cfg.x_max - self.cfg.x_min)/self.cfg.cell))
        self.nz = int(np.ceil((self.cfg.z_max - self.cfg.z_min)/self.cfg.cell))

    @torch.no_grad()
    def splat(self, depth_m, seg, height_m):
        """
        depth_m: np.ndarray [H,W] float32 (m)
        seg:     np.ndarray [H,W] uint8
        height_m: np.ndarray [H,W] float32 (m) (cao hơn mặt đường)
        return: occ [nz, nx] uint8 in GPU
        """
        D = to_torch(depth_m, device=self.dev, dtype=torch.float32)  # [H,W]
        Hh = to_torch(height_m, device=self.dev, dtype=torch.float32)
        S = to_torch(seg, device=self.dev, dtype=torch.int16)

        # mask occupied
        m = (Hh >= float(self.cfg.h_min_m))
        m &= (D >= float(self.cfg.z_min)) & (D <= float(self.cfg.z_max))
        m &= (S != 0) & (S != 10)   # bỏ road/sky

        if m.sum().item() == 0:
            return torch.zeros((self.nz, self.nx), dtype=torch.uint8, device=self.dev)

        # get pixel coords of valid
        ys, xs = torch.nonzero(m, as_tuple=True)  # GPU
        z = D[ys, xs]  # [N]
        x = self.x_factor[0, xs] * z  # [N]

        # map to grid indices
        ix = torch.floor( (x - self.cfg.x_min) / self.cfg.cell ).long()
        iz = torch.floor( (z - self.cfg.z_min) / self.cfg.cell ).long()

        # clamp to grid
        ix = torch.clamp(ix, 0, self.nx-1)
        iz = torch.clamp(iz, 0, self.nz-1)

        occ = torch.zeros((self.nz, self.nx), dtype=torch.uint8, device=self.dev)
        # unique cells (avoid race)
        lin = iz * self.nx + ix
        uniq = torch.unique(lin)
        occ.view(-1)[uniq] = 1
        return occ

    @torch.no_grad()
    def occupancy_score_for_boxes(self, boxes_xyxy: np.ndarray, dists_m: np.ndarray, weight_window:int=2):
        """
        boxes_xyxy: Nx4 float (image coords)
        dists_m:    Nx1 float (estimated Z of box center), nếu NaN -> score 0
        return: torch.float32 [N] occupancy density (0..1) trong cửa sổ (2*win+1)^2
        """
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return torch.zeros((0,), dtype=torch.float32, device=self.dev)
        B = torch.as_tensor(boxes_xyxy, device=self.dev, dtype=torch.float32)
        Z = torch.as_tensor(dists_m,   device=self.dev, dtype=torch.float32)

        # center pixel (cx)
        cx = 0.5*(B[:,0] + B[:,2])  # [N]
        X = ((cx - self.cx) / max(1e-6, self.fx)) * Z  # [N]
        # invalid z -> score 0
        ok = torch.isfinite(Z) & (Z > 0)

        ix = torch.floor( (X - self.cfg.x_min) / self.cfg.cell ).long()
        iz = torch.floor( (Z - self.cfg.z_min) / self.cfg.cell ).long()

        # window sampling
        w = int(max(1, weight_window))
        scores = torch.zeros((B.shape[0],), dtype=torch.float32, device=self.dev)
        # clamp & mask
        ix = torch.clamp(ix, 0, self.nx-1)
        iz = torch.clamp(iz, 0, self.nz-1)

        # build offsets
        offs = torch.stack(torch.meshgrid(
            torch.arange(-w, w+1, device=self.dev),
            torch.arange(-w, w+1, device=self.dev),
            indexing='ij'
        ), dim=-1).view(-1,2)  # [(2w+1)^2, 2] -> (dz, dx)

        total = float((2*w+1)*(2*w+1))
        # vectorized gather per point
        for i in range(B.shape[0]):
            if not bool(ok[i].item()):
                scores[i] = 0.0
                continue
            cells_z = torch.clamp(iz[i] + offs[:,0], 0, self.nz-1)
            cells_x = torch.clamp(ix[i] + offs[:,1], 0, self.nx-1)
            lin = cells_z * self.nx + cells_x
            scores[i] = self._occ_flat[lin].float().mean()

        return scores

    @property
    def _occ_flat(self):
        # small helper to allow fast flat gather (expects self.last_occ set)
        return self.last_occ.view(-1)

    def set_last_occ(self, occ: torch.Tensor):
        # store last occupancy for sampling
        self.last_occ = occ
