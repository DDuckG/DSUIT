# src/fusion/ground_plane.py
import numpy as np
from ..utils.geometry import intrinsics_from_fov, backproject_grid

ROAD_ID, SIDEWALK_ID, TERRAIN_ID, SKY_ID = 0, 1, 9, 10

def fit_plane_ls(points: np.ndarray):
    """
    Fit phẳng bằng least squares (Ax+By+Cz+D=0) với ||n||=1.
    """
    A = np.c_[points, np.ones((points.shape[0], 1), dtype=points.dtype)]
    _, _, Vt = np.linalg.svd(A)
    p = Vt[-1, :]  # [a,b,c,d]
    n = p[:3]; d = p[3]
    norm = np.linalg.norm(n) + 1e-9
    return n / norm, d / norm

def fit_plane_ransac(points: np.ndarray, inlier_thr: float = 0.05, max_iter: int = 150):
    """
    RANSAC plane robust.
    points: Nx3 (m)
    """
    if points.shape[0] < 3:
        return fit_plane_ls(points) if points.shape[0] > 0 else (np.array([0, 1, 0], dtype=np.float32), 0.0)
    best_in = -1
    best_n = np.array([0, 1, 0], dtype=np.float32); best_d = 0.0
    N = points.shape[0]
    rnd = np.random.default_rng(12345)
    for _ in range(int(max_iter)):
        # chọn 3 điểm không thẳng hàng
        idx = rnd.choice(N, size=3, replace=False)
        p1, p2, p3 = points[idx]
        n = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(n) < 1e-6:
            continue
        n = n / (np.linalg.norm(n) + 1e-9)
        d = -np.dot(n, p1)
        # khoảng cách có dấu từ điểm tới plane
        dist = np.abs(points @ n + d)
        inliers = np.count_nonzero(dist <= inlier_thr)
        if inliers > best_in:
            best_in = inliers; best_n = n; best_d = d
    # refine bằng LS trên inliers
    dist = np.abs(points @ best_n + best_d)
    m = dist <= inlier_thr
    if np.count_nonzero(m) >= 3:
        n, d = fit_plane_ls(points[m])
        return n, d
    return best_n, best_d

class GroundPlaneScaler:
    """
    - Ước lượng scale từ depth tương đối → mét bằng plane (đường/ vỉa hè/ terrain).
    - Cung cấp tiện ích tính height-map so với mặt phẳng.
    """
    def __init__(self, W, H, fov_deg_h, principal_xy, cam_h=1.35,
                 sample_stride=4, ema_beta=0.9, use_sidewalk=True, use_terrain=True,
                 method="ransac", inlier_thr=0.05, max_iter=150):
        self.fx, self.fy, self.cx, self.cy = intrinsics_from_fov(W, H, fov_deg_h, principal_xy)
        self.cam_h = float(cam_h)
        self.sample_stride = int(sample_stride)
        self.beta = float(ema_beta)
        self.use_sidewalk = bool(use_sidewalk)
        self.use_terrain = bool(use_terrain)
        self.alpha = None  # EMA scale
        self.method = str(method)
        self.inlier_thr = float(inlier_thr)
        self.max_iter = int(max_iter)

        self.dirs, _ = backproject_grid(W, H, self.fx, self.fy, self.cx, self.cy, stride=self.sample_stride)
        # dirs: Hs x Ws x 3

    def estimate_scale(self, depth_rel: np.ndarray, seg: np.ndarray):
        """
        depth_rel: HxW (độ sâu tương đối, >0)
        seg:       HxW (id)
        Trả về: (alpha, (n,d)) với n chuẩn hoá.
        """
        m = (seg[::self.sample_stride, ::self.sample_stride] == ROAD_ID)
        if self.use_sidewalk:
            m = np.logical_or(m, (seg[::self.sample_stride, ::self.sample_stride] == SIDEWALK_ID))
        if self.use_terrain:
            m = np.logical_or(m, (seg[::self.sample_stride, ::self.sample_stride] == TERRAIN_ID))

        z = depth_rel[::self.sample_stride, ::self.sample_stride]
        m = np.logical_and(m, z > 1e-6)

        if np.count_nonzero(m) < 50:
            # fallback: lấy toàn ảnh vùng dưới nửa ảnh
            Hs, _ = z.shape
            m2 = np.zeros_like(m, dtype=bool)
            m2[Hs//2:, :] = True
            m = np.logical_and(m2, z > 1e-6)

        dirs = self.dirs[m]
        zs = z[m][:, None]
        pts = dirs * zs  # Nx3

        if pts.shape[0] < 3:
            n, d = np.array([0, 1, 0], dtype=np.float32), 0.0
        else:
            if self.method == "ransac":
                n, d = fit_plane_ransac(pts, inlier_thr=self.inlier_thr, max_iter=self.max_iter)
            else:
                n, d = fit_plane_ls(pts)

        Drel = abs(d)  # distance từ gốc tới plane (vì ||n||=1)
        alpha = self.cam_h / max(Drel, 1e-6)
        self.alpha = alpha if self.alpha is None else (self.beta * self.alpha + (1 - self.beta) * alpha)
        return self.alpha, (n, d)

    def height_from_plane(self, depth_m: np.ndarray, plane):
        # Tính height map (m) so với plane: h = n·(v*z) + d  (n chuẩn hoá, d theo mét)
        n, d = plane
        # sample stride như self.sample_stride để tiết kiệm — nhưng ở đây trả về full-res
        H, W = depth_m.shape
        # tạo lưới hướng ở full-res một lần (lazy)
        if not hasattr(self, "_dirs_full") or self._dirs_full.shape[:2] != (H, W):
            dirs_full, _ = backproject_grid(W, H, self.fx, self.fy, self.cx, self.cy, stride=1)
            self._dirs_full = dirs_full
        v = self._dirs_full  # HxWx3
        h = (v[..., 0] * depth_m + 0.0) * n[0] + (v[..., 1] * depth_m) * n[1] + (v[..., 2] * depth_m) * n[2] + d
        return h.astype(np.float32)
