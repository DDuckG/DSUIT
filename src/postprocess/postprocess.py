# src/postprocess/postprocess.py
"""
Postprocess v2 (rewrite):
- 1D Kalman smoothing on distance for each track
- Simple distance-based alerting (red/yellow)
- Outputs:
    * id_to_meta.json
    * alerts.csv  (level ∈ {red,yellow})
"""
from __future__ import annotations
import argparse, json, os, csv
from pathlib import Path
import numpy as np

class Kalman1D:
    def __init__(self, q=1.0, r=4.0, init=6.0):
        self.x = np.array([init, 0.0])   # dist, vel
        self.P = np.eye(2)*10.0
        self.Q = np.array([[q,0.0],[0.0,q]])
        self.R = r
    def predict(self, dt):
        F = np.array([[1.0, dt],[0.0,1.0]])
        self.x = F@self.x
        self.P = F@self.P@F.T + self.Q
    def update(self, z):
        H = np.array([[1.0, 0.0]])
        y = z - H@self.x
        S = H@self.P@H.T + self.R
        K = self.P@H.T/ S
        self.x = self.x + (K.flatten()*y)
        self.P = (np.eye(2)-K@H)@self.P
    @property
    def d(self): return float(self.x[0])
    @property
    def v(self): return float(self.x[1])

def parse_fusion(path):
    frames = []
    with open(path,'r',encoding='utf8') as f:
        for line in f:
            frames.append(json.loads(line))
    return frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fusion', required=True)
    ap.add_argument('--meta', default=None)
    ap.add_argument('--out-id-meta', required=True)
    ap.add_argument('--out-alerts', required=True)
    ap.add_argument('--red-distance-m', type=float, default=1.8)
    ap.add_argument('--distance-close-m', type=float, default=3.0)
    ap.add_argument('--cooldown-s', type=float, default=3.0)
    ap.add_argument('--fps', type=float, default=None)
    args = ap.parse_args()

    frames = parse_fusion(args.fusion)
    fps = float(args.fps or 0.0)
    if args.meta and os.path.exists(args.meta):
        try:
            with open(args.meta,'r',encoding='utf8') as f:
                meta = json.load(f)
                fps = float(fps or meta.get('fps', 0.0))
        except Exception:
            pass
    dt = 1.0/fps if fps>0 else 1.0

    bank = {}
    last_alert = {}
    id_stats = {}
    alerts = []

    for fr in frames:
        frame = int(fr.get('frame',0))
        time_s = fr.get('time_s', None)
        time_s = float(time_s) if time_s is not None else (frame*dt)
        for obj in fr.get('objects',[]):
            tid = int(obj.get('track_id'))
            dist = obj.get('distance_m')
            if dist is None or not np.isfinite(dist): 
                continue
            dist = float(dist)

            if tid not in bank: bank[tid] = Kalman1D()
            kf = bank[tid]; kf.predict(dt); kf.update(dist)
            est = kf.d; vel = kf.v

            # stats
            st = id_stats.setdefault(tid, {'first':frame,'last':frame,'n':0,'sum':0.0,'min':1e9,'max':0.0,'class':obj.get('class_name')})
            st['last']=frame; st['n']+=1; st['sum']+=dist; st['min']=min(st['min'],dist); st['max']=max(st['max'],dist)

            # alert by distance thresholds
            level = None
            if dist < args.red_distance_m: level='red'
            elif dist < args.distance_close_m: level='yellow'
            if level:
                prev = last_alert.get(tid,-1e9)
                if (time_s - prev) >= args.cooldown_s:
                    last_alert[tid] = time_s
                    alerts.append({'frame':frame,'time_s':time_s,'track_id':tid,'distance_m':float(dist),'class_name':obj.get('class_name'),'level':level})

    # id_to_meta.json
    meta = {}
    for tid,st in id_stats.items():
        avg = (st['sum']/st['n']) if st['n']>0 else None
        meta[int(tid)] = {
            'track_id': int(tid),
            'class_name': st['class'],
            'first_frame': st['first'],
            'last_frame': st['last'],
            'frames_seen': st['n'],
            'avg_distance_m': float(avg) if avg is not None else None,
            'min_distance_m': float(st['min']) if st['min']<1e9 else None,
            'max_distance_m': float(st['max'])
        }

    Path(os.path.dirname(args.out_id_meta) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out_id_meta,'w',encoding='utf8') as f:
        json.dump(meta,f,indent=2,ensure_ascii=False)

    Path(os.path.dirname(args.out_alerts) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out_alerts,'w',encoding='utf8',newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['frame','time_s','track_id','distance_m','class_name','level'])
        wr.writeheader()
        for r in alerts: wr.writerow(r)

    print("Wrote:", args.out_id_meta, "and", args.out_alerts)

if __name__ == '__main__':
    main()
