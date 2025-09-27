# src/viz/drawer.py
import cv2
import numpy as np
from typing import Iterable, Tuple, List

# BGR
RED    = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN  = (0, 255, 0)
GRAY   = (128, 128, 128)
BLACK  = (0, 0, 0)
WHITE  = (255, 255, 255)

def _choose_color_by_dist(d: float, thr: Tuple[float, float]) -> Tuple[int, int, int]:
    t1, t2 = thr
    if not np.isfinite(d) or d <= 0:
        return GRAY
    if d < t1:
        return RED
    if d <= t2:
        return YELLOW
    return GREEN

def _put_label(
    img: np.ndarray,
    text: str,
    tl: Tuple[int, int],
    font_scale: float = 0.5,
    txt_thickness: int = 1,
    fg: Tuple[int, int, int] = WHITE,
    bg: Tuple[int, int, int] = BLACK,
):
    """Vẽ nhãn có nền mờ dễ đọc."""
    ((tw, th), baseline) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_thickness)
    x, y = tl
    y = max(th + 2, y)  # tránh âm
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + 2), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, txt_thickness, cv2.LINE_AA)

def draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    labels: Iterable[str],
    dists: Iterable[float],
    thr: Tuple[float, float] = (1.8, 3.5),
    thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Vẽ bbox + màu theo khoảng cách:
      - đỏ  : d < thr[0]
      - vàng: thr[0] <= d <= thr[1]
      - xanh: d > thr[1]
    Text hiển thị: "<label> • <d>m" (nếu có d).
    """
    if img is None:
        return img
    vis = img.copy()
    if boxes is None or len(boxes) == 0:
        return vis

    boxes = np.asarray(boxes, dtype=float)
    dists = np.asarray(dists, dtype=float) if dists is not None else np.full((len(boxes),), np.nan, dtype=float)
    labels = list(labels) if labels is not None else [""] * len(boxes)

    H, W = vis.shape[:2]
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1 = max(0, min(W - 1, x1)); x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1)); y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        d = float(dists[i]) if i < len(dists) else np.nan
        color = _choose_color_by_dist(d, thr)

        # khung
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # nội dung nhãn
        base = labels[i] if i < len(labels) else ""
        if np.isfinite(d) and d > 0:
            text = f"{base} • {d:.1f} m" if base else f"{d:.1f} m"
        else:
            text = base if base else "obstacle"

        _put_label(vis, text, (x1, max(15, y1 - 6)), font_scale=font_scale, txt_thickness=1, fg=WHITE, bg=BLACK)

    return vis

def draw_polylines(img: np.ndarray, polys: List[np.ndarray], color=(255, 0, 0), thickness: int = 2) -> np.ndarray:
    """
    Vẽ polyline (không bắt buộc dùng trong bài toán này, giữ để tương thích).
    """
    if img is None: 
        return img
    vis = img.copy()
    if polys is None: 
        return vis
    for p in polys:
        p = np.asarray(p, dtype=int).reshape(-1, 1, 2)
        cv2.polylines(vis, [p], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return vis
