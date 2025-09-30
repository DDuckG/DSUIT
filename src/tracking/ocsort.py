import torch
from typing import List

@torch.inference_mode()
def iou_matrix_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), device = a.device, dtype = torch.float32)
    ax1, ay1, ax2, ay2 = a[:, 0: 1], a[:, 1: 2], a[:, 2: 3], a[:, 3: 4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x1 = torch.maximum(ax1, bx1)
    y1 = torch.maximum(ay1, by1)
    x2 = torch.minimum(ax2, bx2)
    y2 = torch.minimum(ay2, by2)
    iw = torch.clamp_min(x2 - x1, 0.0)
    ih = torch.clamp_min(y2 - y1, 0.0)
    inter = iw * ih
    area_a = torch.clamp_min((ax2 - ax1) * (ay2 - ay1), 0.0)
    area_b = torch.clamp_min((bx2 - bx1) * (by2 - by1), 0.0)
    union = area_a + area_b - inter + 0.000000001
    return inter / union

class Track:
    __slots__=("bbox", "score", "clsid", "id", "age", "time_since_update", "hits", "max_age", "z")
    def __init__(self, bbox, score, clsid, z, tid, max_age):
        self.bbox = bbox.clone()
        self.score = float(score.item())
        self.clsid = int(clsid.item())
        self.z = float(z.item()) if torch.isfinite(z) else float("nan")
        self.id = int(tid)
        self.age = 1
        self.time_since_update = 0
        self.hits = 1
        self.max_age = int(max_age)

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox, score, z):
        alpha = 0.6
        self.bbox = self.bbox * alpha + bbox * (1.0 - alpha)
        self.score = float(max(self.score, float(score.item())))
        if torch.isfinite(z):
            if not (self.z == self.z):  # NaN 
                self.z = float(z.item())
            else:
                self.z = 0.7 * self.z + 0.3 * float(z.item())  # EMA
        self.time_since_update = 0
        self.hits += 1

    def dead(self) -> bool:
        return self.time_since_update > self.max_age

class OCSort:
    def __init__(self, max_age = 120, iou_thr = 0.3, device = None):
        self.max_age = int(max_age)
        self.iou_thr = float(iou_thr)
        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.tracks: List[Track] = []
        self.next_id = 1

    @torch.inference_mode()
    def update(self, det_xyxy, det_scores, det_clsids, det_dists = None):
        if not torch.is_tensor(det_xyxy):  
            det_xyxy = torch.as_tensor(det_xyxy)
        if not torch.is_tensor(det_scores):
            det_scores = torch.as_tensor(det_scores)
        if not torch.is_tensor(det_clsids):
            det_clsids = torch.as_tensor(det_clsids)
        det_xyxy = det_xyxy.to(self.device, dtype = torch.float32, non_blocking = True).contiguous().view(-1, 4)
        det_scores = det_scores.to(self.device, dtype = torch.float32, non_blocking = True).contiguous().view(-1)
        det_clsids = det_clsids.to(self.device, dtype = torch.int64, non_blocking = True).contiguous().view(-1)

        if det_dists is None:
            det_dists = torch.full((det_xyxy.shape[0],), float("nan"), device = self.device, dtype = torch.float32)
        else:
            if not torch.is_tensor(det_dists): 
                det_dists = torch.as_tensor(det_dists)
            det_dists = det_dists.to(self.device, dtype = torch.float32, non_blocking = True).contiguous().view(-1)

        for track in self.tracks: 
            track.predict()

        track = len(self.tracks)
        detect = int(det_xyxy.shape[0])

        if track == 0 and detect > 0:
            for i in range(detect):
                tr = Track(det_xyxy[i], det_scores[i], det_clsids[i], det_dists[i], self.next_id, self.max_age)
                self.next_id += 1
                self.tracks.append(tr)

        elif track > 0 and detect > 0:
            t_boxes = torch.stack([track.bbox for track in self.tracks], dim = 0)
            ious = iou_matrix_xyxy(t_boxes, det_xyxy)
            t_cls = torch.tensor([track.clsid for track in self.tracks], device = self.device, dtype = torch.int64)
            same = (t_cls.view(-1, 1) == det_clsids.view(1, -1))
            M = torch.where(same, ious, torch.zeros_like(ious))

            matches = []
            used_t = set()
            used_d = set()

            while M.numel() > 0:
                flat_idx = torch.argmax(M.view(-1))
                best = M.view(-1)[flat_idx]
                if float(best.item()) < self.iou_thr: 
                    break
                ti = int(torch.div(flat_idx, detect, rounding_mode = 'floor').item())
                di = int((flat_idx % detect).item())
                matches.append((ti, di))
                used_t.add(ti)
                used_d.add(di)
                M[ti, : ] = -1.0
                M[ : ,di] = -1.0
                if len(used_t) >= track or len(used_d) >= detect: 
                    break

            for (ti, di) in matches:
                self.tracks[ti].update(det_xyxy[di], det_scores[di], det_dists[di])

            for di in range(detect):
                if di not in used_d:
                    tr = Track(det_xyxy[di], det_scores[di], det_clsids[di], det_dists[di], self.next_id, self.max_age)
                    self.next_id += 1
                    self.tracks.append(tr)

            self.tracks=[track for track in self.tracks if not track.dead()]

        else:
            self.tracks=[track for track in self.tracks if not track.dead()]

        
        out = []          # Trả tất cả track còn sống (kể cả frame này chưa update)
        for track in self.tracks:
            is_pred = track.time_since_update > 0
            out.append((track.id, track.bbox.detach().float().cpu().numpy(), track.score, track.clsid, (float("nan") if not (track.z==track.z) else float(track.z)), is_pred))
        return out
