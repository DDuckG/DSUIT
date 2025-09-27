# src/fusion/curb_boundary.py
import numpy as np
import cv2

ROAD_ID = 0

def curb_polylines(seg, stride=2, smooth_win=9):
    road = (seg == ROAD_ID).astype(np.uint8)
    H, W = road.shape
    xs_left = []; xs_right=[]
    ys = list(range(0, H, stride))
    for y in ys:
        row = road[y, :]
        if row.sum() == 0:
            xs_left.append(0); xs_right.append(W-1)
            continue
        idx = np.where(row>0)[0]
        xs_left.append(int(idx[0])); xs_right.append(int(idx[-1]))
    xs_left  = smooth1d(np.array(xs_left, dtype=np.float32), win=smooth_win)
    xs_right = smooth1d(np.array(xs_right, dtype=np.float32), win=smooth_win)
    ptsL = np.stack([xs_left, ys], axis=1).astype(np.int32)
    ptsR = np.stack([xs_right, ys], axis=1).astype(np.int32)
    return ptsL, ptsR

def smooth1d(x, win=9):
    if win <= 1: return x
    k = int(win)|1
    ker = np.ones(k, dtype=np.float32) / float(k)
    pad = k//2
    xp = np.pad(x, (pad,pad), mode="edge")
    y = np.convolve(xp, ker, mode="valid")
    return y
