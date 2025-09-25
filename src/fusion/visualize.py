# src/fusion/visualize.py
"""
Visualization for fusion.jsonl:
- Vẽ depth heatmap (tùy chọn) + bbox có màu theo khoảng cách.
- Hiển thị class_name nếu có (obstacle).
- Có thể overlay segmentation chỉ để kiểm tra.

Author: assistant
"""
import argparse, os, json
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any
from pycocotools import mask as mask_util

COLOR_RED   = (0, 0,255)
COLOR_YEL   = (0,255,255)
COLOR_GRN   = (0,200,  0)
COLOR_W     = (255,255,255)
COLOR_BG    = (50,50,50)

def load_jsonl(path: str):
    frames={}
    with open(path,'r',encoding='utf8') as f:
        for line in f:
            js=json.loads(line)
            frames[int(js['frame'])]=js
    return frames

def rle_to_mask(rle):
    dec={"size": rle["size"], "counts": rle["counts"].encode("ascii")}
    m=mask_util.decode(dec)
    if m.ndim==3: m=m[...,0]
    return m.astype(np.uint8)

def draw_label(im, text, xy):
    x,y=xy; fs=0.45; th=1
    (tw,tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cv2.rectangle(im,(x,max(0,y-tht-6)),(x+tw+6,y),COLOR_BG,-1)
    cv2.putText(im,text,(x+3,max(0,y-4)),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,0),th,cv2.LINE_AA)

def color_by_dist(d, red=1.8, yellow=3.0):
    if d is None or (isinstance(d,(int,float)) and not np.isfinite(d)):
        return (200,200,0), 'unknown'
    if d < red: return COLOR_RED, 'red'
    if d < yellow: return COLOR_YEL, 'yellow'
    return COLOR_GRN, 'green'

def visualize(video, fusion, out, segments=None, depth_dir=None,
              draw_seg=False, draw_depth=True, fps_out=None,
              red_distance=1.8, yellow_distance=3.0, max_vis_distance=30.0):
    fus = load_jsonl(fusion)
    seg_frames={}
    if segments and os.path.exists(segments):
        with open(segments,'r',encoding='utf8') as f:
            for line in f:
                js=json.loads(line); fid=int(js.get('frame_id', js.get('frame',0))); seg_frames[fid]=js

    cap=cv2.VideoCapture(video)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {video}")
    fps=float(fps_out or cap.get(cv2.CAP_PROP_FPS) or 25.0)
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outw=cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))

    fidx=0
    while True:
        ok, frame=cap.read()
        if not ok: break
        fidx += 1
        base=frame.copy()

        # depth heatmap
        if draw_depth and depth_dir:
            p=Path(depth_dir)/f"depth_{fidx:05d}.npz"
            if p.exists():
                arr=np.load(str(p)); k=arr.files[0]; d=arr[k].astype(np.float32)
                finite=np.isfinite(d)
                if finite.any():
                    dmin=float(np.nanpercentile(d[finite],2.0))
                    dmax=float(np.nanpercentile(d[finite],98.0))
                    if dmax-dmin>1e-6:
                        dn=np.clip((d-dmin)/(dmax-dmin),0,1)
                    else: dn=np.zeros_like(d)
                    hm=(255*(1.0-dn)).astype(np.uint8) # gần -> sáng
                    hm=cv2.applyColorMap(cv2.resize(hm,(W,H)), cv2.COLORMAP_JET)
                    base=cv2.addWeighted(base,0.6,hm,0.4,0.0)

        # seg overlay (optional)
        if draw_seg and (fidx in seg_frames):
            masks=seg_frames[fidx].get('masks',[])
            ov=np.zeros_like(base)
            for m in masks:
                try: mm=rle_to_mask(m['rle'])
                except Exception: continue
                if mm.shape[:2]!=(H,W):
                    mm=cv2.resize(mm,(W,H),interpolation=cv2.INTER_NEAREST)
                color=(int(30+7*m.get('class_id',0)%200),
                       int(80+13*m.get('class_id',0)%160),
                       int(40+17*m.get('class_id',0)%200))
                ov[mm>0]=color
            base=cv2.addWeighted(base,0.5,ov,0.5,0.0)

        outframe=base.copy()
        entry=fus.get(fidx, {})
        objs=entry.get('objects',[])
        for obj in objs:
            tid=int(obj['track_id'])
            x,y,w,h = obj['bbox']
            x1,y1,x2,y2 = int(round(x)), int(round(y)), int(round(x+w)), int(round(y+h))
            dist = obj.get('distance_m', None)
            # bỏ vẽ nếu quá xa cho sạch hình
            if dist is not None and np.isfinite(dist) and float(dist)>max_vis_distance:
                continue
            color, lvl = color_by_dist(dist, red_distance, yellow_distance)
            cv2.rectangle(outframe,(x1,y1),(x2,y2),color,2)
            label = f"ID {tid}"
            cn = obj.get('class_name', None)
            if cn: label += f" | {cn}"
            if dist is not None and (isinstance(dist,(int,float)) and np.isfinite(dist)):
                label += f" | {float(dist):.1f}m"
            label += f" | {lvl}"
            draw_label(outframe, label, (x1, y1))
        draw_label(outframe, f"Frame: {fidx}", (10, 24))
        draw_label(outframe, f"Red<{red_distance}m  Yellow<{yellow_distance}m", (140, 24))

        outw.write(outframe)
    cap.release(); outw.release()
    print("Saved visualization ->", out)

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument('--video', required=True)
    a.add_argument('--fusion', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--segments', default=None)
    a.add_argument('--depth-dir', default=None)
    a.add_argument('--draw-seg', action='store_true')
    a.add_argument('--draw-depth', action='store_true')
    a.add_argument('--fps', type=float, default=None)
    a.add_argument('--red-distance', type=float, default=1.8)
    a.add_argument('--yellow-distance', type=float, default=3.0)
    a.add_argument('--max-vis-distance', type=float, default=30.0)
    args=a.parse_args()
    visualize(args.video, args.fusion, args.out,
              segments=args.segments, depth_dir=args.depth_dir,
              draw_seg=args.draw_seg, draw_depth=args.draw_depth,
              fps_out=args.fps, red_distance=args.red_distance,
              yellow_distance=args.yellow_distance, max_vis_distance=args.max_vis_distance)
