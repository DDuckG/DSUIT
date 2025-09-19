import cv2
import re
import json
import subprocess
import numpy as np
import torch
import torch.nn.functional

def seg_map_from_logits(logits, target_size):
    logits_up = torch.nn.functional.interpolate(logits, size = target_size, mode = "bilinear", align_corners = False)
    labels = logits_up.argmax(dim = 1)
    seg = labels[0].detach().cpu().numpy().astype(np.uint8)
    return seg

def get_video_rotation(path):
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", path]
    run = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, text = True, check = False)
    text = json.loads(run.stdout)
    streams = text.get("streams")
    video_stream = None

    for stream in streams:
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        for stream in streams:
            if stream.get("side_data_list") or (stream.get("tags") and "rotate" in stream.get("tags")):
                video_stream = stream
                break

    tags = video_stream.get("tags") or {}
    if "rotate" in tags:
        angle = int(tags["rotate"]) % 360
        return angle if angle in (0, 90, 180, 270) else 0

    for side_data in video_stream.get("side_data_list", []) or []:
        if isinstance(side_data, dict):
            if "rotation" in side_data:
                angle = int(side_data["rotation"]) % 360
                return angle if angle in (0, 90, 180, 270) else 0
            display_matr = side_data.get("displaymatrix") or side_data.get("displaymatrix_string") or ""
            regex = re.search(r"(-?\d+)", str(display_matr))
            if regex:
                angle = int(regex.group(1)) % 360
                return angle if angle in (0, 90, 180, 270) else 0
    return 0

def rotate_cv2_frame(frame, angle):
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame