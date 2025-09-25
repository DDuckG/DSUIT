# src/fusion/candidates.py
"""
Depth-first candidate generator + simple tracklet manager
for generic obstacles (class_id = 999).

- Tạo ứng viên từ DISCONTINUITY của depth (ưu tiên).
- Dùng segmentation chỉ như bộ lọc & hợp khối (road/sidewalk = non-obstacle).
- Kiểm tra "ring depth contrast" và "planarity" để bỏ nền/lề.
- Loại trùng với YOLO (IoU).
- Giữ ổn định bằng EMA + chỉ xuất khi đã confirm >= 2 frames.

Author: assistant
"""
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import cv2
from pycocotools import mask as mask_util

# =================== Hyper-params (an toàn mặc định, có thể đổi qua fuse) ===================
OBSTACLE_CLASS_ID = 999
OBSTACLE_TRACK_START = 1_000_000

AREA_MIN_RATIO = 0.00035   # min area ứng viên / ảnh
IOU_SUPPRESS_YOLO = 0.50   # nếu IoU lớn hơn -> bỏ (đã có YOLO)
IOU_MATCH_TEMP = 0.30      # liên kết tracklet
DEPTH_DIFF_MATCH = 0.6     # m (khi đã quy đổi) hoặc tương đối, dùng nới lỏng
EMA_ALPHA = 0.55
MAX_TRACKLET_AGE = 20
MIN_CONFIRM_FRAMES = 2     # >= số khung hình mới “xuất” bbox

# Depth processing
DEPTH_EDGE_TH = 0.08       # ngưỡng Sobel(|∇depth_norm|)
DEPTH_MIN_AREA = 60        # tối thiểu số pixel thành phần cạnh
RING_PIXELS = 6            # bề rộng “vòng” để so sánh depth viền-vùng
RING_CONTRAST_MIN = 0.12   # tối thiểu chênh lệch depth_norm (out-in) để nhận vật cản
PLANAR_GRAD_TH = 0.03      # trung vị |∇depth_norm| trong bbox thấp -> mặt phẳng -> bỏ
ASPECT_MIN = 0.15          # h/W tối thiểu để tránh miếng mỏng sát đáy
BOTTOM_BAND = 0.92         # nếu đáy bbox > 92% H và h/H < 0.08 -> coi là vệt nền, bỏ

# Classes (Cityscapes) hữu ích để hợp khối; classes của YOLO nên tránh
GOOD_SEG_CLASSES = set([1,2,3,4,5,6,7,8,9])  # sidewalk..traffic sign..vegetation..terrain..
NON_OBSTACLE_CLASSES = set([0,1])            # road(0) + sidewalk(1) = non-obstacle

# ============================================================================================

def _rle_to_mask(rle_dict):
    dec = {"size": rle_dict["size"], "counts": rle_dict["counts"].encode("ascii")}
    m = mask_util.decode(dec)
    if m.ndim == 3: m = m[...,0]
    return m.astype(np.uint8)

def _mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0: return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return (float(x1), float(y1), float(x2 - x1), float(y2 - y1))

def _xywh_to_xyxy(b):
    x,y,w,h = b; return (x,y,x+w,y+h)

def _iou_xyxy(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    xx1=max(ax1,bx1); yy1=max(ay1,by1)
    xx2=min(ax2,bx2); yy2=min(ay2,by2)
    w=max(0.0,xx2-xx1); h=max(0.0,yy2-yy1)
    inter=w*h
    ua=max(0.0,ax2-ax1)*max(0.0,ay2-ay1)
    ub=max(0.0,bx2-bx1)*max(0.0,by2-by1)
    u=ua+ub-inter+1e-9
    return inter/u if u>0 else 0.0

def _normalize_depth(dm: np.ndarray):
    """chuẩn hóa depth tương đối theo median của frame (ổn định thang)."""
    d = np.array(dm, dtype=np.float32)
    finite = np.isfinite(d)
    if not finite.any():
        return np.zeros_like(d)
    med = float(np.nanmedian(d[finite]))
    if med <= 0: med = 1.0
    d[~finite] = med
    return d / med

def _ring_contrast(depth_n: np.ndarray, bbox, ring_pix=RING_PIXELS):
    """ so sánh median depth_norm vùng trong và 'vòng' bao quanh """
    H,W = depth_n.shape[:2]
    x,y,w,h = [int(round(v)) for v in bbox]
    x = max(0,x); y=max(0,y)
    x2 = min(W, x+w); y2=min(H, y+h)
    if x2-x < 4 or y2-y < 4: return 0.0, None, None
    inner = depth_n[y:y2, x:x2]
    m_in = float(np.nanmedian(inner))

    # vòng ngoài
    xs1 = max(0, x - ring_pix); xs2 = min(W, x2 + ring_pix)
    ys1 = max(0, y - ring_pix); ys2 = min(H, y2 + ring_pix)
    outer = depth_n[ys1:ys2, xs1:xs2].copy()
    outer[max(0,y-ys1):max(0,y-ys1)+(y2-y), max(0,x-xs1):max(0,x-xs1)+(x2-x)] = np.nan
    m_out = float(np.nanmedian(outer))
    if not np.isfinite(m_in) or not np.isfinite(m_out): return 0.0, m_in, m_out
    # object “đứng” trước background -> depth_norm nhỏ hơn (gần hơn) nên m_out - m_in > 0
    return (m_out - m_in), m_in, m_out

def _planar_score(depth_n: np.ndarray, bbox):
    """ trung vị độ lớn gradient trong bbox (thấp -> mặt phẳng/ground) """
    H,W = depth_n.shape[:2]
    x,y,w,h = [int(round(v)) for v in bbox]
    x = max(0,x); y=max(0,y)
    x2 = min(W, x+w); y2=min(H, y+h)
    if x2-x < 3 or y2-y < 3: return 0.0
    patch = depth_n[y:y2, x:x2]
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx*gx + gy*gy)
    return float(np.nanmedian(np.abs(g)))

def _suppress_with_yolo(cands, yolo_dets, iou_th=IOU_SUPPRESS_YOLO):
    if not yolo_dets: return cands
    yb = [ _xywh_to_xyxy((float(x),float(y),float(w),float(h))) for (x,y,w,h,_,_) in yolo_dets ]
    kept=[]
    for c in cands:
        bb = _xywh_to_xyxy(c['bbox'])
        if max((_iou_xyxy(bb,b) for b in yb), default=0.0) >= iou_th:
            continue
        kept.append(c)
    return kept

def _union_masks(masks: List[np.ndarray], H, W):
    if not masks: return np.zeros((H,W), np.uint8)
    out = np.zeros((H,W), np.uint8)
    for m in masks:
        if m is None: continue
        mm = m.astype(np.uint8)
        if mm.shape[:2] != (H,W):
            mm = cv2.resize(mm,(W,H), interpolation=cv2.INTER_NEAREST)
        out = np.logical_or(out, mm).astype(np.uint8)
    return out

# --------------------- Candidate builders ---------------------

def cands_from_depth(depth_map: np.ndarray, H: int, W: int,
                     area_min_ratio=AREA_MIN_RATIO,
                     edge_th=DEPTH_EDGE_TH, min_area=DEPTH_MIN_AREA):
    """Ứng viên từ gián đoạn depth (ưu tiên)."""
    dn = _normalize_depth(depth_map)
    gx = cv2.Sobel(dn, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dn, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx*gx + gy*gy)
    edge = (g > float(edge_th)).astype(np.uint8)

    # làm dày + đóng lỗ để thành mảng rõ
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    edge = cv2.morphologyEx(edge, cv2.MORPH_DILATE, k, iterations=1)
    edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, k, iterations=1)

    num, lab = cv2.connectedComponents(edge, connectivity=8)
    min_area = max(min_area, int(area_min_ratio * H * W))
    out=[]
    for i in range(1, num):
        comp = (lab==i).astype(np.uint8)
        if int(comp.sum()) < min_area: continue
        bb = _mask_to_bbox(comp)
        if bb is None: continue
        out.append({'bbox': bb, 'mask': comp, 'source':'depth', 'score':1.0})
    return out, dn

def cands_from_seg(seg_entry: Optional[Dict[str,Any]], H: int, W: int):
    """Dùng seg để hợp khối & làm non-obstacle mask."""
    if not seg_entry: 
        return [], np.zeros((H,W), np.uint8), np.zeros((H,W), np.uint8)
    masks = seg_entry.get('masks', []) or []
    good=[]
    non=[]
    for m in masks:
        try:
            mm = _rle_to_mask(m['rle'])
        except Exception:
            continue
        if mm.shape[:2] != (H,W):
            mm = cv2.resize(mm, (W,H), interpolation=cv2.INTER_NEAREST)
        cls = int(m.get('class_id', -1))
        if cls in NON_OBSTACLE_CLASSES:
            non.append(mm)
        if cls in GOOD_SEG_CLASSES:
            good.append(mm)
    non_mask = _union_masks(non, H, W)
    good_mask = _union_masks(good, H, W)
    if good_mask.sum() > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        good_mask = cv2.morphologyEx(good_mask, cv2.MORPH_CLOSE, k, iterations=1)
    return [], non_mask, good_mask  # không phát sinh bbox từ seg nữa

# --------------------- Tracklet ---------------------
@dataclass
class Trk:
    tid: int
    bbox: Tuple[float,float,float,float]
    score: float
    depth_hist: List[float]
    last_seen: int
    frames_seen: int
    confirmed: bool

class CandidateManager:
    def __init__(self,
                 iou_match_temp: float = IOU_MATCH_TEMP,
                 depth_diff_match: float = DEPTH_DIFF_MATCH,
                 ema_alpha: float = EMA_ALPHA,
                 max_age: int = MAX_TRACKLET_AGE,
                 min_confirm_frames: int = MIN_CONFIRM_FRAMES,
                 ring_contrast_min: float = RING_CONTRAST_MIN,
                 planar_grad_th: float = PLANAR_GRAD_TH,
                 bottom_band: float = BOTTOM_BAND,
                 aspect_min: float = ASPECT_MIN):
        self.tracks: Dict[int, Trk] = {}
        self.next_tid = OBSTACLE_TRACK_START
        self.iou_match_temp = iou_match_temp
        self.depth_diff_match = depth_diff_match
        self.ema_alpha = ema_alpha
        self.max_age = max_age
        self.min_confirm_frames = min_confirm_frames
        self.ring_contrast_min = ring_contrast_min
        self.planar_grad_th = planar_grad_th
        self.bottom_band = bottom_band
        self.aspect_min = aspect_min

    # ---------- core -----------
    def _quality_gate(self, bbox, depth_n, non_mask):
        """bỏ các bbox rác: overlap non-obstacle, planar, thiếu contrast, mỏng-sát-đáy"""
        H,W = depth_n.shape[:2]
        x,y,w,h = bbox
        # 1) loại non-obstacle (road/sidewalk)
        if non_mask is not None and non_mask.any():
            x1=int(max(0,round(x))); y1=int(max(0,round(y)))
            x2=int(min(W,round(x+w))); y2=int(min(H,round(y+h)))
            if x2>x1 and y2>y1:
                ov = float(non_mask[y1:y2, x1:x2].mean())  # tỉ lệ chồng
                if ov > 0.65:
                    return False

        # 2) mỏng & quá sát đáy -> thường là vệt nền
        if (h/max(1e-6,w)) < self.aspect_min:
            y2 = y + h
            if y2 > self.bottom_band * H:
                return False

        # 3) planar test
        pscore = _planar_score(depth_n, bbox)
        if pscore < self.planar_grad_th:
            # bề mặt phẳng tương đối -> thường là nền/lề
            # nhưng nếu có ring contrast mạnh thì vẫn giữ
            contrast, _, _ = _ring_contrast(depth_n, bbox)
            if contrast < self.ring_contrast_min * 1.2:
                return False

        # 4) ring contrast (out-in) phải dương & đủ lớn
        contrast, _, _ = _ring_contrast(depth_n, bbox)
        if not np.isfinite(contrast) or contrast < self.ring_contrast_min:
            return False
        return True

    def step_frame(self, frame_idx: int,
                   seg_entry: Optional[Dict[str,Any]],
                   depth_map: Optional[np.ndarray],
                   yolo_dets: List[Tuple[float,float,float,float,float,int]],
                   image_area: int):
        """
        Return:
          pseudo_dets: list[(x,y,w,h,score,OBSTACLE_CLASS_ID,tid)]
          rows:       list[(frame,tid,x,y,w,h,score,-1,-1)]
        """
        H = int(seg_entry.get('height', 0)) if seg_entry else (depth_map.shape[0] if depth_map is not None else 0)
        W = int(seg_entry.get('width', 0))  if seg_entry else (depth_map.shape[1] if depth_map is not None else 0)

        # segmentation masks (non-obstacle + good mask for union ops)
        _, non_mask, good_mask = cands_from_seg(seg_entry, H, W) if (seg_entry and H and W) else ([], np.zeros((H,W),np.uint8), np.zeros((H,W),np.uint8))
        # depth candidates
        cands = []
        dn = None
        if depth_map is not None:
            depth_c, dn = cands_from_depth(depth_map, H, W)
            cands += depth_c

        # mở rộng bbox một chút nếu nằm trong vùng good_mask để nuốt hết object mảnh
        if good_mask.any():
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            gm = cv2.morphologyEx(good_mask, cv2.MORPH_DILATE, k, iterations=1)
        else:
            gm = good_mask

        refined=[]
        for c in cands:
            x,y,w,h = c['bbox']
            # snap & clip
            x=max(0.0,x); y=max(0.0,y)
            w=max(1.0,w); h=max(1.0,h)
            if gm.any():
                # nếu bbox nằm trong vùng gm thì nới nhẹ theo gm để bọc đủ vật
                x1=int(max(0,round(x))); y1=int(max(0,round(y)))
                x2=int(min(W,round(x+w))); y2=int(min(H,round(y+h)))
                if x2>x1 and y2>y1:
                    sub = gm[y1:y2, x1:x2]
                    if sub.mean() > 0.15:
                        pad = 2
                        x = max(0, x - pad); y=max(0, y - pad)
                        w = min(W - x, w + 2*pad); h = min(H - y, h + 2*pad)
            c['bbox'] = (x,y,w,h)
            refined.append(c)

        # YOLO suppression
        refined = _suppress_with_yolo(refined, yolo_dets, IOU_SUPPRESS_YOLO)

        # chất lượng theo depth (gate mạnh để bỏ nền/lề)
        kept=[]
        if dn is None:
            dn = np.zeros((H,W), np.float32)
        for c in refined:
            if self._quality_gate(c['bbox'], dn, non_mask):
                kept.append(c)

        # ----- tracking (EMA) & xuất kết quả -----
        rows=[]
        used=set()
        # advance ages
        for tr in list(self.tracks.values()):
            tr.last_seen += 1

        # match
        for tid, tr in list(self.tracks.items()):
            best_i, best_idx = 0.0, -1
            tb = _xywh_to_xyxy(tr.bbox)
            for i,c in enumerate(kept):
                if i in used: continue
                iou = _iou_xyxy(tb, _xywh_to_xyxy(c['bbox']))
                if iou > best_i:
                    best_i, best_idx = iou, i
            if best_idx >= 0 and best_i >= self.iou_match_temp:
                c = kept[best_idx]; used.add(best_idx)
                # EMA update
                x,y,w,h = tr.bbox
                nx,ny,nw,nh = c['bbox']
                a = self.ema_alpha
                tr.bbox = (a*nx+(1-a)*x, a*ny+(1-a)*y, a*nw+(1-a)*w, a*nh+(1-a)*h)
                tr.score = max(tr.score, c.get('score',1.0))
                tr.last_seen = 0
                tr.frames_seen += 1
                if not tr.confirmed and tr.frames_seen >= self.min_confirm_frames:
                    tr.confirmed = True
                rows.append((frame_idx, tr.tid, tr.bbox[0], tr.bbox[1], tr.bbox[2], tr.bbox[3], tr.score, -1, -1))

        # new tracks
        for i,c in enumerate(kept):
            if i in used: continue
            tid = self.next_tid; self.next_tid += 1
            self.tracks[tid] = Trk(tid=tid, bbox=c['bbox'], score=c.get('score',1.0),
                                   depth_hist=[], last_seen=0, frames_seen=1, confirmed=False)
            rows.append((frame_idx, tid, c['bbox'][0], c['bbox'][1], c['bbox'][2], c['bbox'][3], c.get('score',1.0), -1, -1))

        # prune
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].last_seen > self.max_age:
                del self.tracks[tid]

        # xuất pseudo_detections cho fuse
        pseudo=[]
        for tr in self.tracks.values():
            if not tr.confirmed: 
                continue
            x,y,w,h = tr.bbox
            pseudo.append((float(x),float(y),float(w),float(h), float(tr.score), OBSTACLE_CLASS_ID, int(tr.tid)))
        return pseudo, rows
