# src/tracking/obs_track.py
import numpy as np
from typing import List, Tuple, Callable
from src.utils.geom import iou_matrix, clip_xyxy, cover_fraction

class ObsTrack:
    __slots__=("id","bbox","score","z","age","hits","time_since_update","edge_fail")
    def __init__(self, tid:int, bbox:np.ndarray, score:float, z:float):
        self.id   = int(tid)
        self.bbox = bbox.astype(np.float32)
        self.score= float(score)
        self.z    = float(z)
        self.age  = 1
        self.hits = 1
        self.time_since_update = 0
        self.edge_fail = 0

class Pending:
    __slots__=("bbox","score","z","frames","hits","edge_hits","z_hist","iou_sum")
    def __init__(self, bbox, score, z):
        self.bbox=bbox.astype(np.float32); self.score=float(score); self.z=float(z)
        self.frames = 1
        self.hits = 1
        self.edge_hits = 0
        self.z_hist = [float(z)]
        self.iou_sum = 0.0

class ObsTracker:
    def __init__(self,
                 iou_thr:float=0.5,
                 max_age:int=0,
                 z_max:float=15.0,
                 edge_fail_kill:int=2,
                 staging:dict|None=None,
                 lock:dict|None=None):
        self.iou_thr=float(iou_thr)
        self.max_age=int(max_age)
        self.z_max=float(z_max)
        self.edge_fail_kill=int(edge_fail_kill)
        self.tracks: List[ObsTrack]=[]
        self.pending: List[Pending]=[]
        self.next_id=1

        st = staging or {}
        self.win = int(st.get("window", 6))
        self.min_hits = int(st.get("min_hits", 4))
        self.iou_birth = float(st.get("iou_min", 0.35))
        self.twoedge_min = int(st.get("twoedge_min", 3))
        self.z_mad_max   = float(st.get("z_mad_max", 0.6))
        self.ycov_thr    = float(st.get("yolo_cover_thr", 0.90))
        self.yiou_thr    = float(st.get("yolo_iou_thr", 0.60))

        lk = lock or {}
        self.lock_dx = float(lk.get("search_dx_px", 8))
        self.lock_dy = float(lk.get("search_dy_px", 8))
        self.lock_beta = float(lk.get("ema_beta", 0.80))
        self.lock_scale = float(lk.get("max_scale_change", 0.15))

    def update(self,
               det_boxes: np.ndarray,
               det_scores: np.ndarray,
               det_dists: np.ndarray,
               refine_fn: Callable[[np.ndarray], np.ndarray],
               image_wh: Tuple[int,int],
               edge_ok: np.ndarray,
               yolo_boxes: np.ndarray):
        W,H = image_wh
        D = det_boxes.shape[0]
        T = len(self.tracks)

        # predict (giữ bbox; age++)
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        # Z gate
        if D>0:
            zmask = det_dists <= self.z_max
            det_boxes = det_boxes[zmask]; det_scores = det_scores[zmask]; det_dists=det_dists[zmask]; edge_ok=edge_ok[zmask]
            D = det_boxes.shape[0]

        # ------------------ match tracks ↔ detections ------------------
        if T>0 and D>0:
            t_boxes = np.stack([t.bbox for t in self.tracks], axis=0)
            M = iou_matrix(t_boxes, det_boxes)
            matches=[]; used_t=set(); used_d=set()
            while True:
                ti, di = divmod(np.argmax(M), D if D>0 else 1)
                if float(M[ti,di]) < self.iou_thr:
                    break
                matches.append((ti,di))
                used_t.add(ti); used_d.add(di)
                M[ti,:] = -1.0; M[:,di] = -1.0
                if len(used_t)==T or len(used_d)==D:
                    break
            # update matched + lock/hysteresis
            for (ti,di) in matches:
                tr = self.tracks[ti]
                b = refine_fn(det_boxes[di])
                # lock: giới hạn di chuyển mép & scale theo z
                cx = 0.5*(tr.bbox[0]+tr.bbox[2]); cy = 0.5*(tr.bbox[1]+tr.bbox[3])
                w0 = tr.bbox[2]-tr.bbox[0]; h0 = tr.bbox[3]-tr.bbox[1]
                s = tr.z / max(1e-6, float(det_dists[di]))
                s = max(1.0 - self.lock_scale, min(1.0 + self.lock_scale, s))
                w1 = max(2.0, w0 * s); h1 = max(2.0, h0 * s)
                bx = np.asarray([cx - w1*0.5, cy - h1*0.5, cx + w1*0.5, cy + h1*0.5], dtype=np.float32)
                # hạn chế delta mép
                d = b - bx
                d[0] = np.clip(d[0], -self.lock_dx, self.lock_dx)
                d[2] = np.clip(d[2], -self.lock_dx, self.lock_dx)
                d[1] = np.clip(d[1], -self.lock_dy, self.lock_dy)
                d[3] = np.clip(d[3], -self.lock_dy, self.lock_dy)
                b_locked = bx + d
                # EMA
                tr.bbox = (self.lock_beta*tr.bbox + (1.0-self.lock_beta)*b_locked).astype(np.float32)
                tr.score = max(tr.score, float(det_scores[di]))
                tr.z = float(det_dists[di])
                tr.time_since_update = 0
                tr.hits += 1
                tr.edge_fail = 0 if edge_ok[di] else (tr.edge_fail + 1)

            # unmatched detections → staging
            for di in range(D):
                if di in used_d:
                    continue
                self._add_or_update_pending(det_boxes[di], det_scores[di], det_dists[di], edge_ok[di])
        elif D>0:
            for di in range(D):
                self._add_or_update_pending(det_boxes[di], det_scores[di], det_dists[di], edge_ok[di])

        # ------------------ staging commit/drop ------------------
        self._staging_step(yolo_boxes, W, H, refine_fn)

        # ------------------ prune tracks ------------------
        self._age_and_prune(W,H)

        return self._dump()

    def _add_or_update_pending(self, b, s, z, edgeok):
        if len(self.pending) == 0:
            p = Pending(b, s, z); p.edge_hits += int(bool(edgeok))
            self.pending.append(p); return
        # match theo IoU với pending hiện có
        P = np.stack([p.bbox for p in self.pending], axis=0)
        ious = iou_matrix(P, b.reshape(1,4)).reshape(-1)
        j = int(np.argmax(ious))
        if float(ious[j]) >= self.iou_birth:
            p = self.pending[j]
            p.frames += 1; p.hits += 1; p.iou_sum += float(ious[j])
            p.bbox = 0.6*p.bbox + 0.4*b
            p.score = max(p.score, float(s))
            p.z_hist.append(float(z))
            p.z = float(z)
            p.edge_hits += int(bool(edgeok))
        else:
            p = Pending(b, s, z); p.edge_hits += int(bool(edgeok))
            self.pending.append(p)

    def _staging_step(self, yolo_boxes, W, H, refine_fn):
        if len(self.pending) == 0:
            return
        born=[]; keep=[]
        for p in self.pending:
            # cửa sổ đơn giản: nếu quá window mà chưa đạt min_hits thì drop
            if p.frames >= self.win:
                avg_iou = p.iou_sum / max(1.0, (p.hits-1))
                zh = np.asarray(p.z_hist, dtype=np.float32)
                zm = np.median(zh)
                mad = np.median(np.abs(zh - zm))
                # YOLO gate
                if yolo_boxes is not None and yolo_boxes.size > 0:
                    cov = cover_fraction(p.bbox.reshape(1,4), yolo_boxes).max()
                    iouy = iou_matrix(p.bbox.reshape(1,4), yolo_boxes).max()
                else:
                    cov = 0.0; iouy = 0.0
                cond = (p.hits >= self.min_hits) and (p.edge_hits >= self.twoedge_min) and (mad <= self.z_mad_max) and (avg_iou >= self.iou_birth) and (cov < self.ycov_thr) and (iouy <= self.yiou_thr)
                if cond:
                    b = refine_fn(p.bbox)
                    b = clip_xyxy(b, W, H)
                    tr = ObsTrack(self.next_id, b, p.score, p.z)
                    self.next_id += 1
                    self.tracks.append(tr)
                else:
                    pass
            else:
                keep.append(p)
        self.pending = keep

    def _age_and_prune(self, W, H):
        alive=[]
        for t in self.tracks:
            x1,y1,x2,y2 = t.bbox
            in_w = max(0.0, min(W*1.0, x2) - max(0.0, x1))
            in_h = max(0.0, min(H*1.0, y2) - max(0.0, y1))
            frac = (in_w * in_h) / max(1e-6, (x2-x1)*(y2-y1))
            if t.time_since_update > self.max_age:
                continue
            if t.edge_fail >= self.edge_fail_kill:
                continue
            if t.z > self.z_max:
                continue
            if frac < 0.15:
                continue
            alive.append(t)
        self.tracks = alive

    def _dump(self):
        out=[]
        for t in self.tracks:
            out.append((t.id, t.bbox.copy(), t.score, -1, float(t.z), t.time_since_update>0))
        return out
