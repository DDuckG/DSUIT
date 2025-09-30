# src/tracking/obs_track.py
import numpy as np
from typing import List, Tuple, Callable
from src.utils.geom import iou_matrix, clip_xyxy

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
    __slots__=("bbox","score","z","frames","edge_ok")
    def __init__(self, bbox, score, z, edge_ok):
        self.bbox=bbox.astype(np.float32); self.score=float(score); self.z=float(z)
        self.frames = 1
        self.edge_ok = bool(edge_ok)

class ObsTracker:
    def __init__(self, iou_thr:float=0.5, max_age:int=0, z_max:float=15.0, edge_fail_kill:int=2):
        self.iou_thr=float(iou_thr)
        self.max_age=int(max_age)
        self.z_max=float(z_max)
        self.edge_fail_kill=int(edge_fail_kill)
        self.tracks: List[ObsTrack]=[]
        self.pending: List[Pending]=[]
        self.next_id=1

    def update(self,
               det_boxes: np.ndarray,
               det_scores: np.ndarray,
               det_dists: np.ndarray,
               refine_fn: Callable[[np.ndarray], np.ndarray],
               image_wh: Tuple[int,int],
               edge_ok: np.ndarray):
        W,H = image_wh
        D = det_boxes.shape[0]
        T = len(self.tracks)

        # predict (giữ bbox; age++)
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        # Z-gate trước
        if D > 0:
            zmask = det_dists <= self.z_max
            det_boxes = det_boxes[zmask]; det_scores = det_scores[zmask]; det_dists = det_dists[zmask]; edge_ok = edge_ok[zmask]
            D = det_boxes.shape[0]

        if T==0 and D==0:
            self._age_and_prune(W,H)
            self._advance_pending()
            return self._dump()

        # match tracks ↔ detections
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
            # update matched
            for (ti,di) in matches:
                tr = self.tracks[ti]
                b = refine_fn(det_boxes[di])
                b = clip_xyxy(b, W, H)
                tr.bbox = 0.6*tr.bbox + 0.4*b
                tr.score = max(tr.score, float(det_scores[di]))
                tr.z = float(det_dists[di])
                tr.time_since_update = 0
                tr.hits += 1
                tr.edge_fail = 0 if edge_ok[di] else (tr.edge_fail + 1)

            # add unmatched detections to pending
            for di in range(D):
                if di in used_d:
                    continue
                p = Pending(det_boxes[di], det_scores[di], det_dists[di], edge_ok[di])
                if p.edge_ok:
                    self.pending.append(p)

        elif D>0:  # T==0
            for di in range(D):
                p = Pending(det_boxes[di], det_scores[di], det_dists[di], edge_ok[di])
                if p.edge_ok:
                    self.pending.append(p)

        # birth debounce: pending ↔ current dets/last frame
        self._promote_pending(W,H, refine_fn)

        # death tức thời + edge_fail kill + out-of-frame
        self._age_and_prune(W,H)

        # tiến tuổi pending
        self._advance_pending()

        return self._dump()

    def _promote_pending(self, W, H, refine_fn):
        if len(self.pending) == 0:
            return
        new_tracks=[]
        keep_pending=[]
        for p in self.pending:
            if p.frames >= 2:
                new_tracks.append(p)
                continue
            keep_pending.append(p)
        self.pending = keep_pending

        for p in new_tracks:
            b = refine_fn(p.bbox)
            b = clip_xyxy(b, W, H)
            tr = ObsTrack(self.next_id, b, p.score, p.z)
            self.next_id += 1
            self.tracks.append(tr)

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

    def _advance_pending(self):
        if len(self.pending)==0:
            return
        kept=[]
        for p in self.pending:
            p.frames += 1
            if p.frames <= 2:
                kept.append(p)
        self.pending = kept

    def _dump(self):
        out=[]
        for t in self.tracks:
            out.append((t.id, t.bbox.copy(), t.score, -1, float(t.z), t.time_since_update>0))
        return out
