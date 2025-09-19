import os, sys, time
import cv2
import json
import torch
import argparse
import numpy as np
from pycocotools import mask as mask_util
from PIL import Image, ImageOps
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.segformer_b1.segformer_model import model, processor, device
from src.segmentation.utils_segformer import seg_map_from_logits, get_video_rotation, rotate_cv2_frame

def to_serializable(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64, np.uint8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def color_for_class(cid):
    np.random.seed(int(cid) + 36363)
    c = tuple((np.random.randint(0, 255, size = (3,))).tolist())
    return (c[0], c[1], c[2])

def visualize_frame(frame_bgr, masks_info, alpha = 0.5):
    vis = frame_bgr.copy()
    height, width = vis.shape[:2]
    overlay = np.zeros_like(vis, dtype = np.uint8)
    for mi in masks_info:
        rle = mi["rle"]
        mask = mask_util.decode(rle)
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation = cv2.INTER_NEAREST)
        color = color_for_class(mi.get("class_id", 0))
        overlay[mask.astype(bool)] = color
    blended = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    return blended

def mask_to_rle(mask):
    bin_mask = (mask > 0).astype(np.uint8)
    fortran = np.asfortranarray(bin_mask)
    rle = mask_util.encode(fortran)
    return {"size": rle["size"], "counts": rle["counts"].decode("ascii")}

def frame_to_rle(seg_map, skip_background = True):
    masks = []
    unique = np.unique(seg_map)
    for cls in unique:
        if skip_background and cls == 0:
            continue
        bin_masks = (seg_map == cls).astype(np.uint8)
        rle = mask_to_rle(bin_masks)
        masks.append({"class_id": cls, "rle": rle, "score": 1.0})
    return masks

def image_to_rle(img):
    width, height = img.size
    inputs = processor(images = img, return_tensors = "pt")
    inputs = {k : v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    seg_map = seg_map_from_logits(outputs.logits, target_size = (height, width))
    masks = frame_to_rle(seg_map)
    return seg_map, masks, (height, width)

def process_video(src, out, skip_frames = 1, visual = False):
    out = Path(out)
    out.mkdir(parents = True, exist_ok = True)
    angle = get_video_rotation(src)
    cap = cv2.VideoCapture(src)
    writer = None
    id = 0

    if visual:
        vis_path = out / (Path(src).stem + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    with open(out / "segmentation.jsonl", "w", encoding = "utf8") as file:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if id % skip_frames != 0:
                id += 1
                continue
            if angle != 0:
                frame = rotate_cv2_frame(frame, angle)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            seg_map, masks, (H, W) = image_to_rle(img)
            entry = {"frame_id": id, "height": H, "width": W, "masks": masks}
            file.write(json.dumps(entry, default = to_serializable) + "\n")
            print(f"wrote frame {id} masks")
            if visual:
                vis_frame = visualize_frame(frame.copy(), masks, alpha = 0.5)
                if writer is None:
                    h_vis, w_vis = vis_frame.shape[:2]
                    writer = cv2.VideoWriter(vis_path, fourcc, cap.get(cv2.CAP_PROP_FPS) or 25.0, (w_vis, h_vis))
                writer.write(vis_frame)
            id += 1
    
    cap.release()
    if writer:
        writer.release()
        print("Visualizing done.")
    print(f"Writing RLE jsonl done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type = str, required = True)
    parser.add_argument('--out', type = str, required = True)
    parser.add_argument('--visual', action = 'store_true')
    argument = parser.parse_args()
    process_video(argument.src, argument.out, visual = argument.visual)
