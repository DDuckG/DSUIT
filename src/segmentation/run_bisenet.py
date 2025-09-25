# src/segmentation/run_bisenet.py
import os
import sys
import argparse
from pathlib import Path
import json
import math
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# loader expected at models/bisenet_v2/loader.py
from models.bisenet_v2.loader import load_bisenet_from_vendor

# Cityscapes names/palette (same order used in vendor config)
CITYSCAPES_CLASS_NAMES = [
    "road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
    "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"
]
CITYSCAPES_PALETTE = [
    (128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
    (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
    (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),
    (0,60,100),(0,80,100),(0,0,230),(119,11,32)
]

def colorize(segmap):
    H,W = segmap.shape
    vis = np.zeros((H,W,3), dtype=np.uint8)
    for i,c in enumerate(CITYSCAPES_PALETTE):
        vis[segmap==i] = c
    return vis

def mask_to_rle_np(bin_mask: np.ndarray):
    # expects binary numpy uint8 or bool with shape (H,W)
    if bin_mask.dtype != np.uint8:
        bin_mask = bin_mask.astype(np.uint8)
    fortran = np.asfortranarray(bin_mask)
    rle = mask_util.encode(fortran)
    # rle['counts'] is bytes
    return {"size": rle["size"], "counts": rle["counts"].decode("ascii")}

def frame_to_masks_from_segmap(segmap: np.ndarray, skip_background=True):
    """
    Convert segmap (H,W) int class ids -> list of mask dicts {class_id, rle, score}
    We'll output one mask per connected component for classes of interest.
    """
    masks = []
    H,W = segmap.shape
    unique = np.unique(segmap)
    for cls in unique:
        if skip_background and int(cls) == 0 and False:
            # keep road even (we may want to keep road/sidewalk)
            continue
        bin_mask = (segmap == int(cls)).astype(np.uint8)
        # find connected components to split separate objects
        num_labels, labels = cv2.connectedComponents(bin_mask, connectivity=8)
        for lab in range(1, num_labels):
            comp = (labels == lab).astype(np.uint8)
            if comp.sum() < 8:  # tiny
                continue
            rle = mask_to_rle_np(comp)
            masks.append({"class_id": int(cls), "rle": rle, "score": 1.0})
    return masks

def to_tensor_preprocess(frame_bgr, mean=(0.3257,0.3690,0.3223), std=(0.2112,0.2148,0.2115), to_rgb=True):
    # vendor demo uses RGB order and mean/std above (city)
    if to_rgb:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        img = frame_bgr.copy()
    im = img.astype(np.float32) / 255.0
    im = (im - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    im = np.transpose(im, (2,0,1))[None]
    return torch.from_numpy(im).float()

def write_jsonl(path: str, entries):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

def process_video(src, out_dir, weights, device='cuda', skip_frames=1, debug=False, visual=False, obstacle_classes=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = out_dir / "segmentation_maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    seg_jsonl = out_dir / "segmentation.jsonl"

    # load model
    model, info = load_bisenet_from_vendor(weights_path=weights, device=device, num_classes=len(CITYSCAPES_CLASS_NAMES), verbose=True)
    model.eval()

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_id = 0
    out_entries = []

    vis_writer = None
    if visual:
        vis_path = out_dir / (Path(src).stem + "_seg_vis.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vis_writer = cv2.VideoWriter(str(vis_path), fourcc, fps, (W,H))

    to_rgb = True
    mean=(0.3257,0.3690,0.3223)
    std=(0.2112,0.2148,0.2115)

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id % skip_frames != 0:
                continue
            H0, W0 = frame.shape[:2]
            inp = to_tensor_preprocess(frame, mean=mean, std=std, to_rgb=to_rgb)
            # make divisible by 32
            new_H = math.ceil(H0/32)*32
            new_W = math.ceil(W0/32)*32
            inp = F.interpolate(inp, size=(new_H,new_W), mode='bilinear', align_corners=False)
            inp = inp.to(next(model.parameters()).device)

            try:
                logits = model(inp)
                # vendor may return tuple (logits, aux...), or a single tensor
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                # upsample to original size
                if logits.shape[2] != H0 or logits.shape[3] != W0:
                    logits = F.interpolate(logits, size=(H0, W0), mode='bilinear', align_corners=False)
                probs = torch.softmax(logits, dim=1)
                segmap = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.uint8)
            except Exception as e:
                # fallback: try CPU forward (rare)
                print("[SEG][WARN] model forward failed on frame", frame_id, "->", e)
                raise

            # write segmentation map png (color)
            vis = colorize(segmap)
            map_path = maps_dir / f"seg_{frame_id:05d}.png"
            cv2.imwrite(str(map_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            # build masks array (per connected component per class)
            masks = frame_to_masks_from_segmap(segmap, skip_background=False)

            entry = {"frame_id": int(frame_id), "height": int(H0), "width": int(W0), "masks": masks}
            out_entries.append(entry)

            if visual:
                # overlay blended
                overlay = (0.6 * vis + 0.4 * frame[..., ::-1]).astype(np.uint8)  # vis is RGB; frame BGR -> convert
                vis_writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if debug and (frame_id % 10 == 0):
                print(f"[SEG] frame {frame_id} class_counts sample: {dict(zip(*np.unique(segmap, return_counts=True))) }")

    cap.release()
    if vis_writer:
        vis_writer.release()

    # write jsonl
    write_jsonl(seg_jsonl, out_entries)
    print(f"[SEG] wrote segmentation jsonl -> {seg_jsonl} (frames: {len(out_entries)})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--skip-frames', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--visual', action='store_true')
    parser.add_argument('--obstacle-classes', type=str, default="3,4,5,8,9")
    args = parser.parse_args()
    process_video(args.src, args.out, args.weights, device=args.device, skip_frames=args.skip_frames, debug=args.debug, visual=args.visual)
