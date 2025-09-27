# src/utils/geometry.py
import numpy as np

def intrinsics_from_fov(W, H, fov_deg_h, principal_xy=(0.5, 0.5)):
    fx = 0.5 * W / np.tan(0.5 * np.deg2rad(fov_deg_h))
    fy = fx * (H / W)
    cx = principal_xy[0] * W
    cy = principal_xy[1] * H
    return fx, fy, cx, cy

def backproject_grid(W, H, fx, fy, cx, cy, stride=1):
    us = np.arange(0, W, stride, dtype=np.float32)
    vs = np.arange(0, H, stride, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs)
    x = (uu - cx) / fx
    y = (vv - cy) / fy
    dirs = np.stack([x, y, np.ones_like(x)], axis=-1)  # HxWx3
    return dirs, (vs, us)
