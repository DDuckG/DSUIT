from __future__ import annotations
import argparse, json, os
import numpy as np
import math
import csv
import time
from pathlib import Path

# Simple Kalman1D class (from repo snippet)
class Kalman1D:
    def __init__(self, process_var = 1.0, measure_var = 4.0, init_dist = 5.0):
        self.x = np.array([init_dist, 0.0])  # distance, velocity
        self.P = np.eye(2) * 10.0
        self.Q = np.array([[process_var, 0.0], [0.0,process_var]])
        self.R = measure_var
    def predict(self, dt):
        F = np.array([[1.0, dt], [0.0,1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
    def update(self, z):
        H = np.array([[1.0, 0.0]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S
        self.x = self.x + (K.flatten() * y)
        self.P = (np.eye(2) - K @ H) @ self.P
    @property
    def distance(self): 
        return float(self.x[0])
    @property
    def velocity(self): 
        return 0
        # return float(self.x[1])

class DistanceFilterBank:
    def __init__(self, process_var = 1.0, measure_var = 4.0, init_dist = 5.0):
        self.filters = {}
        self.process_var = process_var
        self.measure_var = measure_var
        self.init_dist = init_dist
    def step(self, track_id: int, dt: float, z: float | None):
        if track_id not in self.filters:
            self.filters[track_id] = Kalman1D(self.process_var, self.measure_var, self.init_dist)
        kf = self.filters[track_id]
        kf.predict(dt)
        if z is not None and np.isfinite(z):
            kf.update(float(z))
        return kf.distance, kf.velocity

def parse_fusion_jsonl(path):
    frames = []
    with open(path, 'r', encoding = 'utf8') as file:
        for line in file:
            frames.append(json.loads(line))
    return frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion', required = True)
    parser.add_argument('--meta', default = None)
    parser.add_argument('--out-id-meta', required = True)
    parser.add_argument('--out-alerts', required = True)
    parser.add_argument('--ttc-threshold', type = float, default = 3.0)
    parser.add_argument('--distance-close-m', type = float, default = 3.0)
    parser.add_argument('--cooldown-s', type = float, default = 3.0)
    parser.add_argument('--fps', type = float, default = None)
    argument = parser.parse_args()

    frames = parse_fusion_jsonl(argument.fusion)
    fps = argument.fps
    if argument.meta and os.path.exists(argument.meta):
        with open(argument.meta,'r',encoding = 'utf8') as file:
            meta = json.load(file)
            fps = fps or meta.get('fps', fps)
    fps = float(fps or 0.0)
    dt = 1.0 / fps if fps > 0 else 1.0

    filterbank = DistanceFilterBank(process_var = 1.0, measure_var = 4.0, init_dist = 5.0)
    last_alert_at = {}
    id_stats = {}
    alerts_rows = []

    for frame_entry in frames:
        frame = int(frame_entry.get('frame', 0))
        time_s = frame_entry.get('time_s', None)
        time_s = float(time_s) if time_s is not None else (frame * dt if fps > 0 else None)
        for obj in frame_entry.get('objects', []):
            tid = int(obj.get('track_id'))
            dist = obj.get('distance_m')
            dist = float(dist)
            
            # step filter
            z = dist
            est_dist, est_vel = filterbank.step(tid, dt, z)

            # velocity negative = closing
            ttc = float('inf')
            if est_vel < -0.000001:
                ttc = est_dist / (-est_vel)

            # record id_stats
            stat = id_stats.setdefault(tid, {'first_frame': frame, 'last_frame': frame, 'count': 0, 'sum_dist': 0.0, 'min_dist': float('inf'), 'max_dist': 0.0, 'class_name': obj.get('class_name')})
            stat['last_frame'] = frame
            stat['count'] += 1
            stat['sum_dist'] += z
            stat['min_dist'] = min(stat['min_dist'], z)
            stat['max_dist'] = max(stat['max_dist'], z)

            # decide alert
            do_alert = False
            if z is not None and np.isfinite(z):
                if z < argument.distance_close_m and ttc < argument.ttc_threshold:
                    do_alert = True
            if do_alert:
                last = last_alert_at.get(tid, -1000000000)
                now = time_s if time_s is not None else frame * dt
                if now - last >= argument.cooldown_s:
                    last_alert_at[tid] = now
                    alerts_rows.append({'frame': frame, 'time_s': now, 'track_id': tid, 'distance_m': float(z), 'ttc_s': (ttc), 'class_name': obj.get('class_name')})
    
    # write id_to_meta json
    id_meta = {}
    for tid, stat in id_stats.items():
        avg = (stat['sum_dist'] / stat['count'])
        id_meta[int(tid)] = {
            'track_id': int(tid),
            'class_name': stat.get('class_name'),
            'first_frame': stat['first_frame'],
            'last_frame': stat['last_frame'],
            'frames_seen': stat['count'],
            'avg_distance_m': float(avg),
            'min_distance_m': (stat['min_dist']),
            'max_distance_m': (stat['max_dist'])
        }
    Path(os.path.dirname(argument.out_id_meta) or ".").mkdir(parents = True, exist_ok = True)
    with open(argument.out_id_meta,'w',encoding = 'utf8') as file:
        json.dump(id_meta, file, indent = 2, ensure_ascii = False)

    # write alerts csv
    Path(os.path.dirname(argument.out_alerts) or ".").mkdir(parents = True, exist_ok = True)
    with open(argument.out_alerts, 'w', encoding = 'utf8', newline = '') as csvf:
        writer = csv.DictWriter(csvf, fieldnames = ['frame', 'time_s', 'track_id', 'distance_m', 'ttc_s', 'class_name'])
        writer.writeheader()
        for row in alerts_rows:
            writer.writerow(row)
    print("Wrote id_meta:", argument.out_id_meta, "alerts:", argument.out_alerts)

if __name__ == '__main__':
    main()