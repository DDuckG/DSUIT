# src/fusion/ground_plane.py
import numpy as np
import torch
from ..utils.torch_cuda import get_device

ROAD_ID, SIDEWALK_ID, TERRAIN_ID, SKY_ID = 0,1,9,10
_EPS = 1e-6

def intrinsics_from_fov(W, H, fov_deg_h, principal_xy):
    fov = float(fov_deg_h) * np.pi/180.0
    fx = (W*0.5) / np.tan(fov*0.5)
    fy = fx  # giả định pixel square
    cx = principal_xy[0]*W
    cy = principal_xy[1]*H
    return float(fx), float(fy), float(cx), float(cy)

def backproject_grid(W, H, fx, fy, cx, cy, stride=1):
    ys, xs = torch.meshgrid(
        torch.arange(0,H,step=stride), torch.arange(0,W,step=stride), indexing="ij"
    )
    xs = xs.to(torch.float32); ys=ys.to(torch.float32)
    fx_t = torch.tensor(fx, dtype=torch.float32); fy_t=torch.tensor(fy, dtype=torch.float32)
    cx_t = torch.tensor(cx, dtype=torch.float32); cy_t=torch.tensor(cy, dtype=torch.float32)
    X = (xs - cx_t) / fx_t
    Y = (ys - cy_t) / fy_t
    Z = torch.ones_like(X)
    dirs = torch.stack([X,Y,Z], dim=-1)
    dirs = dirs / (torch.linalg.norm(dirs, dim=-1, keepdim=True) + 1e-9)
    return dirs, None

def fit_plane_ls(points: torch.Tensor):
    # points: [N,3]
    N = points.shape[0]
    ones = torch.ones((N,1), dtype=points.dtype, device=points.device)
    A = torch.cat([points, ones], dim=1)  # [N,4]
    # Solve via SVD
    U,S,Vh = torch.linalg.svd(A, full_matrices=False)
    p = Vh[-1,:]  # [4]
    n = p[:3]; d=p[3]
    norm = torch.linalg.norm(n) + 1e-9
    return n/norm, d/norm

def fit_plane_ransac(points: torch.Tensor, inlier_thr: float = 0.05, max_iter: int = 150):
    if points.shape[0] < 3:
        return torch.tensor([0,1,0], dtype=torch.float32, device=points.device), torch.tensor(0.0, dtype=torch.float32, device=points.device)
    best_in = -1
    best_n = torch.tensor([0,1,0], dtype=torch.float32, device=points.device); best_d = torch.tensor(0.0, dtype=torch.float32, device=points.device)
    N = points.shape[0]
    g = torch.Generator(device=points.device)
    g.manual_seed(12345)
    for _ in range(int(max_iter)):
        idx = torch.randint(0, N, (3,), generator=g, device=points.device)
        p1,p2,p3 = points[idx]
        n = torch.linalg.cross(p2-p1, p3-p1, dim=-1)
        if torch.linalg.norm(n) < 1e-6: 
            continue
        n = n / (torch.linalg.norm(n) + 1e-9)
        d = -torch.dot(n, p1)
        dist = torch.abs(points @ n + d)
        inliers = int((dist <= inlier_thr).sum().item())
        if inliers > best_in:
            best_in = inliers; best_n = n; best_d = d
    dist = torch.abs(points @ best_n + best_d)
    m = dist <= inlier_thr
    if int(m.sum().item()) >= 3:
        n,d = fit_plane_ls(points[m])
        return n,d
    return best_n, best_d

class GroundPlaneScaler:
    def __init__(self, W, H, fov_deg_h, principal_xy, cam_h=1.35,
                 sample_stride=4, ema_beta=0.9, use_sidewalk=True, use_terrain=True,
                 method="ransac", inlier_thr=0.05, max_iter=150):
        self.fx, self.fy, self.cx, self.cy = intrinsics_from_fov(W,H,fov_deg_h,principal_xy)
        self.cam_h = float(cam_h)
        self.sample_stride = int(sample_stride)
        self.beta = float(ema_beta)
        self.use_sidewalk = bool(use_sidewalk)
        self.use_terrain = bool(use_terrain)
        self.alpha = None
        self.method = str(method)
        self.inlier_thr = float(inlier_thr)
        self.max_iter = int(max_iter)

        self.dirs, _ = backproject_grid(W, H, self.fx, self.fy, self.cx, self.cy, stride=self.sample_stride)
        self._dirs_full = None
        self.n = torch.tensor([0,1,0], dtype=torch.float32, device=get_device())
        self.d = torch.tensor(0.0, dtype=torch.float32, device=get_device())

    def estimate_scale(self, depth_rel: np.ndarray, seg: np.ndarray):
        device = get_device()
        seg_t = torch.from_numpy(seg[::self.sample_stride, ::self.sample_stride]).to(device)
        r = torch.from_numpy(depth_rel[::self.sample_stride, ::self.sample_stride]).to(device=device, dtype=torch.float32)
        m = (seg_t == ROAD_ID)
        if self.use_sidewalk: m |= (seg_t == SIDEWALK_ID)
        if self.use_terrain:  m |= (seg_t == TERRAIN_ID)

        r = torch.where(r > _EPS, r, torch.tensor(float('nan'), device=device))
        m &= torch.isfinite(r)

        if int(m.sum().item()) < 100:
            Hs,Ws = r.shape
            m2 = torch.zeros_like(m, dtype=torch.bool); m2[Hs//2:, :] = True
            m = m2 & torch.isfinite(r) & (r > _EPS)

        invz = 1.0 / (r + _EPS)
        dirs = self.dirs.to(device=device, dtype=torch.float32)[m]
        zs = invz[m].unsqueeze(-1)
        pts = dirs * zs

        if pts.shape[0] >= 3:
            if self.method == "ransac":
                n,d = fit_plane_ransac(pts, inlier_thr=self.inlier_thr, max_iter=self.max_iter)
            else:
                n,d = fit_plane_ls(pts)
        else:
            n = torch.tensor([0,1,0], dtype=torch.float32, device=device); d = torch.tensor(0.0, dtype=torch.float32, device=device)

        Drel = torch.abs(d) + _EPS
        alpha_inst = self.cam_h / Drel
        self.alpha = float(alpha_inst.item()) if self.alpha is None else float(self.beta*self.alpha + (1-self.beta)*alpha_inst.item())

        self.n = (self.beta*self.n + (1-self.beta)*n)
        self.n = self.n / (torch.linalg.norm(self.n) + 1e-9)
        self.d = self.beta*self.d + (1-self.beta)*d

        return self.alpha, (self.n.detach().cpu().numpy().astype(np.float32), float(self.d.item()))

    def height_from_plane(self, depth_m: np.ndarray, plane):
        n_np, d_np = plane
        device = get_device()
        H,W = depth_m.shape
        if self._dirs_full is None or self._dirs_full.shape[:2] != (H,W):
            dirs_full, _ = backproject_grid(W, H, self.fx, self.fy, self.cx, self.cy, stride=1)
            self._dirs_full = dirs_full.to(device=device, dtype=torch.float32)
        v = self._dirs_full
        depth_t = torch.from_numpy(depth_m).to(device=device, dtype=torch.float32)
        n = torch.from_numpy(n_np).to(device=device, dtype=torch.float32)
        d = torch.tensor(d_np, device=device, dtype=torch.float32)
        h = (v[...,0]*depth_t)*n[0] + (v[...,1]*depth_t)*n[1] + (v[...,2]*depth_t)*n[2] + d
        return h.detach().cpu().numpy().astype(np.float32)
