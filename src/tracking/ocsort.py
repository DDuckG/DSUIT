# src/tracking/ocsort.py
import torch
from typing import List

@torch.inference_mode()
def iou_matrix_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel()==0 or b.numel()==0:
        return torch.zeros((a.shape[0], b.shape[0]), device=a.device, dtype=torch.float32)
    ax1,ay1,ax2,ay2 = a[:,0:1],a[:,1:2],a[:,2:3],a[:,3:4]
    bx1,by1,bx2,by2 = b[:,0],b[:,1],b[:,2],b[:,3]
    x1 = torch.maximum(ax1,bx1); y1=torch.maximum(ay1,by1)
    x2 = torch.minimum(ax2,bx2); y2=torch.minimum(ay2,by2)
    iw = torch.clamp_min(x2-x1,0.0); ih=torch.clamp_min(y2-y1,0.0)
    inter = iw*ih
    area_a = torch.clamp_min((ax2-ax1)*(ay2-ay1),0.0)
    area_b = torch.clamp_min((bx2-bx1)*(by2-by1),0.0)
    union = area_a + area_b - inter + 1e-9
    return inter/union

class Track:
    __slots__=("bbox","score","clsid","id","age","time_since_update","hits","max_age","z")
    def __init__(self, bbox, score, clsid, z, tid, max_age):
        self.bbox = bbox.clone()
        self.score = float(score.item())
        self.clsid = int(clsid.item())
        self.z = float(z.item()) if torch.isfinite(z) else float("nan")
        self.id = int(tid)
        self.age=1; self.time_since_update=0; self.hits=1
        self.max_age=int(max_age)

    def predict(self):
        # (Đơn giản: giữ bbox; có thể thay bằng Kalman sau)
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox, score, z):
        alpha=0.6
        self.bbox = self.bbox*alpha + bbox*(1.0-alpha)
        self.score = float(max(self.score, float(score.item())))
        if torch.isfinite(z):
            if not (self.z == self.z):  # NaN check
                self.z = float(z.item())
            else:
                self.z = 0.7*self.z + 0.3*float(z.item())  # EMA
        self.time_since_update = 0
        self.hits += 1

    def dead(self) -> bool:
        return self.time_since_update > self.max_age

class OCSort:
    def __init__(self, max_age=120, iou_thr=0.3, device=None):
        self.max_age=int(max_age); self.iou_thr=float(iou_thr)
        self.device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.tracks: List[Track]=[]; self.next_id=1

    @torch.inference_mode()
    def update(self, det_xyxy, det_scores, det_clsids, det_dists=None):
        # Chuẩn hóa tensor & device
        if not torch.is_tensor(det_xyxy):  det_xyxy  = torch.as_tensor(det_xyxy)
        if not torch.is_tensor(det_scores):det_scores= torch.as_tensor(det_scores)
        if not torch.is_tensor(det_clsids):det_clsids= torch.as_tensor(det_clsids)
        det_xyxy  = det_xyxy.to(self.device, dtype=torch.float32, non_blocking=True).contiguous().view(-1,4)
        det_scores= det_scores.to(self.device, dtype=torch.float32, non_blocking=True).contiguous().view(-1)
        det_clsids= det_clsids.to(self.device, dtype=torch.int64,   non_blocking=True).contiguous().view(-1)

        if det_dists is None:
            det_dists = torch.full((det_xyxy.shape[0],), float("nan"), device=self.device, dtype=torch.float32)
        else:
            if not torch.is_tensor(det_dists): det_dists = torch.as_tensor(det_dists)
            det_dists = det_dists.to(self.device, dtype=torch.float32, non_blocking=True).contiguous().view(-1)

        # Dự đoán tất cả track
        for t in self.tracks: t.predict()

        T=len(self.tracks); D=int(det_xyxy.shape[0])

        if T==0 and D>0:
            for i in range(D):
                tr=Track(det_xyxy[i], det_scores[i], det_clsids[i], det_dists[i], self.next_id, self.max_age)
                self.next_id+=1; self.tracks.append(tr)

        elif T>0 and D>0:
            t_boxes = torch.stack([t.bbox for t in self.tracks], dim=0)
            ious = iou_matrix_xyxy(t_boxes, det_xyxy)
            # match theo class để tránh YOLO (>=0) bị OBS (-1) nuốt
            t_cls = torch.tensor([t.clsid for t in self.tracks], device=self.device, dtype=torch.int64)
            same = (t_cls.view(-1,1) == det_clsids.view(1,-1))
            M = torch.where(same, ious, torch.zeros_like(ious))

            matches=[]; used_t=set(); used_d=set()
            while M.numel()>0:
                flat_idx = torch.argmax(M.view(-1))
                best = M.view(-1)[flat_idx]
                if float(best.item()) < self.iou_thr: break
                ti = int(torch.div(flat_idx, D, rounding_mode='floor').item())
                di = int((flat_idx % D).item())
                matches.append((ti,di)); used_t.add(ti); used_d.add(di)
                M[ti,:] = -1.0; M[:,di] = -1.0
                if len(used_t)>=T or len(used_d)>=D: break

            for (ti,di) in matches:
                self.tracks[ti].update(det_xyxy[di], det_scores[di], det_dists[di])

            for di in range(D):
                if di not in used_d:
                    tr=Track(det_xyxy[di], det_scores[di], det_clsids[di], det_dists[di], self.next_id, self.max_age)
                    self.next_id+=1; self.tracks.append(tr)

            self.tracks=[t for t in self.tracks if not t.dead()]

        else:
            self.tracks=[t for t in self.tracks if not t.dead()]

        # Trả tất cả track còn sống (kể cả frame này chưa update)
        out=[]
        for t in self.tracks:
            is_pred = t.time_since_update > 0
            out.append((t.id,
                        t.bbox.detach().float().cpu().numpy(),
                        t.score, t.clsid, (float("nan") if not (t.z==t.z) else float(t.z)),
                        is_pred))
        return out
