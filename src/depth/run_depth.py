import argparse
import numpy as np
import os
import torch
import cv2
import gc
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.video_depth_anything.video_depth_anything.video_depth import VideoDepthAnything
from models.video_depth_anything.utils.dc_utils import read_video_frames, save_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--src', type = str, required = True)
    parser.add_argument('--out', type = str, required = True)
    parser.add_argument('--encoder', type = str, default = 'vits', choices = ['vits', 'vitb', 'vitl'])
    parser.add_argument('--input-size', type = int, default = 518)
    parser.add_argument('--max-res', type = int, default = 1280)
    parser.add_argument('--target-fps', type = int, default = -1, help = 'target fps of the input video, -1 means the original fps')
    parser.add_argument('--visual', action = 'store_true')
    argument = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    vdanything = VideoDepthAnything(**model_configs[argument.encoder], metric = False)
    vdanything.load_state_dict(torch.load(f'models/video_depth_anything/checkpoints/video_depth_anything_{argument.encoder}.pth', map_location = device), strict = True)
    vdanything = vdanything.to(device).eval()

    frames, target_fps = read_video_frames(argument.src, -1, argument.target_fps, argument.max_res)
    fps = target_fps
    out_vid = os.path.join(argument.out, '0Depth0.mp4')
    os.makedirs(argument.out, exist_ok = True)

    CHUNK = 256   # tăng/giảm tuỳ RAM
    PLY_SKIP = 1 # (1 = mọi frame, 5 = 1 của 5 frame, ...)
    # process chunks, lưu depth per-frame
    depth_npz_paths = []
    for i in range(0, len(frames), CHUNK):
        chunk_frames = frames[i:i + CHUNK]
        print(f"Processing chunk frames {i}..{i + len(chunk_frames) - 1}")
        depths_chunk, _ = vdanything.infer_video_depth(chunk_frames, fps, input_size = argument.input_size, device = device, fp32 = True)

        for j, depth in enumerate(depths_chunk):
            id = i + j
            depth_npz = os.path.join(argument.out, f"depth_{id:05d}.npz")
            np.savez_compressed(depth_npz, depth = depth.astype(np.float32))
            depth_npz_paths.append(depth_npz)

        del depths_chunk
        torch.cuda.empty_cache()
        gc.collect()

    if argument.visual:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        sample = np.load(depth_npz_paths[0])['depth']
        H, W = sample.shape
        depth_writer = cv2.VideoWriter(out_vid, fourcc, float(fps), (W, H))
        # print(f"Writing depth visualization to {out_vid} (frames: {len(depth_npz_paths)})")
        for path in depth_npz_paths:
            depth = np.load(path)['depth']
            v = (255 * (depth - depth.min()) / (depth.max() - depth.min() + 0.00000001)).astype(np.uint8)
            v3 = np.stack([v, v, v], axis = -1)  # videowrite đòi 3 channel
            bgr = cv2.cvtColor(v3, cv2.COLOR_RGB2BGR)
            depth_writer.write(bgr)
        depth_writer.release()