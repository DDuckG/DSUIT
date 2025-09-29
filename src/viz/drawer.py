# src/viz/drawer.py
import cv2
import numpy as np
from typing import Iterable, Tuple

# RGB
RED    = (255, 0, 0)
YELLOW = (255, 255, 0)
GREEN  = (0, 200, 0)
GRAY   = (128, 128, 128)
BLACK  = (0, 0, 0)
WHITE  = (255, 255, 255)

def _choose_color_by_dist(d: float, thr: Tuple[float, float]) -> Tuple[int, int, int]:
    t1, t2 = thr
    if not np.isfinite(d) or d <= 0: return GRAY
    if d < t1:  return RED
    if d <= t2: return YELLOW
    return GREEN

def _text_color_for(bg: Tuple[int,int,int]):
    r, g, b = bg
    lum = 0.299*r + 0.587*g + 0.114*b
    return BLACK if lum > 140 else WHITE

def draw_boxes(img: np.ndarray, boxes: np.ndarray, labels: Iterable[str], dists: Iterable[float],
               thr: Tuple[float,float]=(1.8,3.5), thickness:int=10, font_scale:float=0.48) -> np.ndarray:
    if img is None: return img
    vis = img.copy()
    if boxes is None or len(boxes)==0: return vis

    boxes = np.asarray(boxes, dtype=float)
    dists = np.asarray(dists, dtype=float) if dists is not None else np.full((len(boxes),), np.nan, dtype=float)
    labels = list(labels) if labels is not None else [""]*len(boxes)

    H,W = vis.shape[:2]
    for i,b in enumerate(boxes):
        x1,y1,x2,y2 = [int(round(v)) for v in b]
        x1 = max(0,min(W-1,x1)); x2=max(0,min(W-1,x2))
        y1 = max(0,min(H-1,y1)); y2=max(0,min(H-1,y2))
        if x2<=x1 or y2<=y1: continue

        d = float(dists[i]) if i<len(dists) else np.nan
        color = _choose_color_by_dist(d, thr)

        # bbox
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,thickness, lineType=cv2.LINE_AA)

        # header
        label = labels[i] if i<len(labels) else ""
        text = f"{label} | {d:.1f}m" if np.isfinite(d) and d>0 else (label if label else "OBS")
        ((tw,th), bs) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        band_h = th + bs + 6
        bx1 = x1; by2 = max(0, y1-2); by1 = max(0, by2 - band_h); bx2 = min(W-1, bx1 + tw + 10)
        cv2.rectangle(vis, (bx1,by1), (bx2,by2), color, -1, lineType=cv2.LINE_AA)
        fg = _text_color_for(color)
        cv2.putText(vis, text, (bx1+5, by2-bs-3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, 1, cv2.LINE_AA)
    return vis

def draw_hud(img: np.ndarray, lines):
    if img is None: return img
    vis = img.copy()
    H,W = vis.shape[:2]
    y = 18
    for s in lines:
        cv2.putText(vis, str(s), (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        y += 16
    return vis