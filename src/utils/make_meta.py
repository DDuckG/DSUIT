import cv2
import json
import argparse
import os
from pathlib import Path

def extract_meta(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = frame_count / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    meta = {
        "video_path": os.path.abspath(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": duration_s,
        "width": width,
        "height": height
    }

    Path(os.path.dirname(out_path) or ".").mkdir(parents = True, exist_ok = True)
    with open(out_path, "w", encoding = "utf8") as file:
        json.dump(meta, file, indent = 2, ensure_ascii = False)

    print(f"Wrote {out_path} for video {video_path}")
    cap.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required = True, help = "Input video")
    parser.add_argument("--out", required = True, help = "Output meta.json")
    argument = parser.parse_args()
    extract_meta(argument.src, argument.out)

if __name__ == "__main__":
    main()