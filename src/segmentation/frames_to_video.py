import sys
import cv2
import argparse
from pathlib import Path

def frames_to_video(frames_dir, output_path, fps = 25):
    frames = sorted(Path(frames_dir).glob("*.png"))
    first = cv2.imread(frames[0])
    H, W, _ = first.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    for f in frames:
        frame = cv2.imread(f)
        out.write(frame)
    out.release()
    # print("Video saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type = str, required = True)
    parser.add_argument('--out', type = str, required = True)
    argument = parser.parse_args()
    frames_to_video(argument.src, argument.out)
