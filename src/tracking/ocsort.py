# src/tracking/ocsort.py
import numpy as np

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2,bx2)-max(ax1,bx1)) * max(0.0, min(ay2,by2)-max(ay1,by1))
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter + 1e-9
    return inter / ua

def iou_matrix(tracks, dets):
    if len(tracks)==0 or len(dets)==0: 
        return np.zeros((len(tracks), len(dets)), dtype=float)
    M = np.zeros((len(tracks), len(dets)), dtype=float)
    for i,t in enumerate(tracks):
        for j,d in enumerate(dets):
            M[i,j] = iou_xyxy(t, d)
    return M

def xyxy_to_xyah(box):
    x1,y1,x2,y2 = box
    w = max(1e-6, x2-x1); h = max(1e-6, y2-y1)
    cx = x1 + w/2; cy = y1 + h/2; s = w*h; r = w/float(h)
    return np.array([cx,cy,s,r], dtype=float)

def xyah_to_xyxy(xyah):
    cx,cy,s,r = map(float, xyah)
    w = max(1e-6, np.sqrt(max(1e-6, s*r))); h = max(1e-6, s/w)
    return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dtype=float)

class KalmanFilter:
    def __init__(self):
        self.dim_x=7; self.dim_z=4
        self.F = np.eye(self.dim_x); dt=1.0
        for i in range(3): self.F[i,i+4]=dt
        self.H = np.zeros((self.dim_z,self.dim_x)); self.H[0,0]=self.H[1,1]=self.H[2,2]=self.H[3,3]=1.0
        self.P = np.eye(self.dim_x)*10.0
        self.R = np.eye(self.dim_z)*1.0
        self.Q = np.eye(self.dim_x)*0.01

    def initiate(self, z):
        x = np.zeros((self.dim_x,), dtype=float); x[:4]=z
        return x, self.P.copy()

    def predict(self, x, P):
        x = self.F@x; P = self.F@P@self.F.T + self.Q
        return x,P

    def update(self, x, P, z):
        S = self.H@P@self.H.T + self.R
        K = P@self.H.T@np.linalg.inv(S)
        y = z - self.H@x
        x = x + K@y
        P = (np.eye(self.dim_x)-K@self.H)@P
        return x,P

class Track:
    def __init__(self, bbox, score, clsid, tid, max_age=30):
        self.kf = KalmanFilter()
        self.mean, self.cov = self.kf.initiate(xyxy_to_xyah(bbox))
        self.bbox = np.array(bbox, dtype=float)
        self.score = float(score)
        self.clsid = int(clsid)
        self.id = int(tid)
        self.age=1; self.time_since_update=0; self.hits=1
        self.max_age = int(max_age)
        self.dist_ema = None

    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.bbox = xyah_to_xyxy(self.mean[:4])
        self.age+=1; self.time_since_update+=1
        return self.bbox

    def update(self, bbox, score):
        self.mean, self.cov = self.kf.update(self.mean, self.cov, xyxy_to_xyah(bbox))
        self.bbox = np.array(bbox, dtype=float)
        self.score = float(score)
        self.time_since_update=0; self.hits+=1

    def dead(self):
        return self.time_since_update > self.max_age

class OCSort:
    def __init__(self, max_age=30, iou_thr=0.3):
        self.max_age=int(max_age); self.iou_thr=float(iou_thr)
        self.tracks=[]; self.next_id=1

    def update(self, det_xyxy, det_scores, det_clsids):
        for t in self.tracks: t.predict()

        if len(self.tracks)==0:
            matches=[]; unmatched_tracks=[]; unmatched_dets=list(range(len(det_xyxy)))
        else:
            T = np.array([t.bbox for t in self.tracks], dtype=float)
            D = np.array(det_xyxy, dtype=float) if len(det_xyxy) else np.zeros((0,4))
            ious = iou_matrix(T, D)
            # invalidate different classes
            for ti,t in enumerate(self.tracks):
                for di,c in enumerate(det_clsids):
                    if t.clsid!=int(c): ious[ti,di]=0.0
            # greedy match
            matches=[]; unmatched_tracks=list(range(len(self.tracks))); unmatched_dets=list(range(len(det_xyxy)))
            while True:
                if ious.size==0: break
                ti,di = np.unravel_index(np.argmax(ious), ious.shape)
                if ious[ti,di] < self.iou_thr: break
                matches.append((ti,di))
                ious[ti,:]= -1; ious[:,di]= -1
                unmatched_tracks.remove(ti)
                unmatched_dets.remove(di)

        for ti,di in matches:
            self.tracks[ti].update(det_xyxy[di], det_scores[di])

        for di in unmatched_dets:
            tr = Track(det_xyxy[di], det_scores[di], det_clsids[di], self.next_id, max_age=self.max_age)
            self.next_id+=1; self.tracks.append(tr)

        survivors=[]
        for t in self.tracks:
            if not t.dead(): survivors.append(t)
        self.tracks = survivors

        out=[]
        for t in self.tracks:
            if t.time_since_update==0:
                out.append((t.id, t.bbox.copy(), t.score, t.clsid))
        return out
