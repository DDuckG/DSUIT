import argparse, json, csv
import cv2
import numpy as np
from pathlib import Path

def load_fusion_jsonl(path):
    frames = {}
    with open(path, 'r', encoding = 'utf8') as file:
        for line in file:
            jsn = json.loads(line)
            frames[int(jsn['frame'])] = jsn
    return frames

def load_id_meta(path):
    path = Path(path)
    with open(path, 'r', encoding = 'utf8') as file:
        return json.load(file)

def load_alerts_csv(path):
    s = set()
    p = Path(path)
    with open(p, 'r', encoding = 'utf8') as file:
        rows = csv.DictReader(file)
        for row in rows:
            frame = int(float(row.get('frame', 0)))
            tid = int(float(row.get('track_id', -1)))
            s.add((frame, tid))
    return s

def draw_overlay(frame, objects, id_meta, alerts_set, dist_warn_m = 5.0):
    out = frame.copy()
    H, W = out.shape[:2]
    for obj in objects:
        tid = int(obj['track_id'])
        x1, y1, w, h = obj['bbox']
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x1 + w), int(y1 + h)
        cls_name = obj.get('class_name') or id_meta.get(str(tid), {}).get('class_name')
        dist = obj.get('distance_m')
        color = (0, 200, 0)
        text = f"ID {tid}"
        if cls_name:
            text += f" | {cls_name}"
        text += f" | {float(dist):.1f}m"
        if float(dist) < dist_warn_m:
            color = (0,200,200)  # yellowish

        # draw box
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), color, 2)

        # background for text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1i, max(0, y1i - th - 6)), (x1i + tw + 6, y1i), color, -1)
        cv2.putText(out, text, (x1i + 3, max(0, y1i - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out

def visualize(video_path, fusion_path, idmeta_path, alerts_path, out_video_path, fps_out = None):
    fusion = load_fusion_jsonl(fusion_path)
    idmeta = load_id_meta(idmeta_path)
    alerts = load_alerts_csv(alerts_path)

    cap = cv2.VideoCapture(str(video_path))
    fps = fps_out or cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_video_path), fourcc, float(fps), (W, H))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        entry = fusion.get(frame_idx)
        objs = []
        if entry:
            for obj in entry.get('objects', []):
                objs.append(obj)

        # mark alert color for specific objects
        # we will customize colors: if (frame_idx, tid) in alerts -> red
        out_frame = frame.copy()
        # draw each object: choose color per alert
        for obj in objs:
            tid = int(obj['track_id'])
            x1, y1, w, h = obj['bbox']
            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x1 + w), int(y1 + h)
            cls_name = obj.get('class_name') or idmeta.get(str(tid), {}).get('class_name')
            dist = obj.get('distance_m')
            is_alert = (frame_idx, tid) in alerts
            if is_alert:
                color = (0, 0, 255)  # red
            else:
                color = (0, 200, 200) if (dist is not None and dist < 5.0) else (0, 200, 0)
            cv2.rectangle(out_frame, (x1i,y1i), (x2i,y2i), color, 2)
            label = f"ID {tid}"
            if cls_name:
                label += f" | {cls_name}"
            if dist is not None:
                label += f" | {float(dist):.1f}m"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out_frame, (x1i, max(0, y1i - th - 6)), (x1i + tw + 6, y1i), color, -1)
            cv2.putText(out_frame, label, (x1i + 3, max(0, y1i - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # global overlay: frame/time
        txt = f"Frame: {frame_idx}"
        cv2.putText(out_frame, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255 ,255), 2, cv2.LINE_AA)
        out.write(out_frame)
    cap.release()
    out.release()
    print("Saved visualization to", out_video_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required = True)
    parser.add_argument('--fusion', required = True)
    parser.add_argument('--id-meta', default = None)
    parser.add_argument('--alerts', default = None)
    parser.add_argument('--out', required = True)
    parser.add_argument('--fps', type = float, default = None)
    argument = parser.parse_args()
    visualize(argument.video, argument.fusion, argument.id_meta, argument.alerts, argument.out, fps_out = argument.fps)
